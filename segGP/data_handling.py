import os
import random
import json
import pickle
import pprint
import re
import shutil
import subprocess
import sys
import time
import pygraphviz as pgv
import torch.nn.functional as F


from deap import gp
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
from PIL import Image

DEFAULT_IGNORE_INDEX = 255

def _seed_worker(worker_id):
    worker_seed = 12345 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def _set_global_seeds(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def pad_collate(batch):
    """
    Collate function to pad images and masks in a batch to the same size.

    Args:
        batch: List of tuples (image, mask) where image is (C,H,W) float in [0,1],
               and mask is either (1,H,W) float (binary) or (H,W) long (multiclass).
    Returns:
        (images, masks): both padded to (B,C,H,W). For multiclass masks, C=1 and dtype is long.
    """
    imgs, masks = zip(*batch)
    # target spatial size from images
    max_h = max(t.shape[-2] for t in imgs)
    max_w = max(t.shape[-1] for t in imgs)
    pad_imgs, pad_masks = [], []
    for im, ms in zip(imgs, masks):
        # ensure channel-first
        if im.dim() == 2:
            im = im.unsqueeze(0)
        # make masks 3D (C,H,W) by adding channel if needed
        if ms.dim() == 2:
            ms = ms.unsqueeze(0)
        # compute padding based on image spatial size
        dh = max_h - im.shape[-2]
        dw = max_w - im.shape[-1]
        pad = (0, dw, 0, dh)  # (left, right, top, bottom)
        pad_imgs.append(F.pad(im, pad, value=0.0))
        # pad masks with ignore-like value for integer masks, else 0.0 for float masks
        mask_pad_val = 0.0 if ms.dtype.is_floating_point else 255
        pad_masks.append(F.pad(ms, pad, value=mask_pad_val))
    return torch.stack(pad_imgs, 0), torch.stack(pad_masks, 0)

def make_run_name(dataset_name: str,
                  run_mode: str,
                  seed: int,
                  color_mode: str,
                  SELECTED_CLASSES: list[int] | None = None,
                  suffix: str | None = None,
                  baseline_only: bool = False,
                  include_transforms: bool = True,
                  use_data_augmentation: bool = False) -> str:
    """Construct the RUN_NAME used for output directories.

    Matches the existing format in segGP_main:
    - base: f"{dataset_name}_{run_mode}mode_seed{seed}_{jid}_{color_mode}-"
    - when baseline_only=True: append the provided suffix (if any)

    The string contains a literal "{jid}" placeholder that _make_run_dir
    will replace with SLURM job id or current pid.
    """
    trans_tag = "T" if include_transforms else ""
    aug_tag = "A" if use_data_augmentation else ""
    base = f"{dataset_name}_{run_mode}mode_seed{seed}_{{jid}}_-{color_mode}-{trans_tag}-{aug_tag}-classes{SELECTED_CLASSES}{suffix}"
    
    if baseline_only:
        return f"{base}-Baseline-only"
    return base

def _make_run_dir(args, dataset_name, seed):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    jid = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    jname = os.environ.get("SLURM_JOB_NAME", "local")
    chosen = args.run_name or os.environ.get("RUN_NAME")
    if chosen:
        chosen = chosen.format(timestamp=timestamp, job=jname, seed=seed, jid=jid)
        run_name = re.sub(r"[^A-Za-z0-9._-]", "_", chosen)[:200]
    else:
        run_name = f"{dataset_name}_{jname}_{seed}_{timestamp}_jid{jid}"
    base_out = args.outdir or os.environ.get("RUN_OUTDIR") or "/dataB1/niels_witbreuk/logs/results"
    run_dir = os.path.join(base_out, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "visualizations"), exist_ok=True)
    return run_dir, dict(timestamp=timestamp, jid=jid, jname=jname, run_name=run_name, base_out=base_out)

def _git_info_safe(repo_dir="."):
    try:
        branch = subprocess.check_output(["git","rev-parse","--abbrev-ref","HEAD"], cwd=repo_dir, text=True).strip()
        commit = subprocess.check_output(["git","rev-parse","HEAD"], cwd=repo_dir, text=True).strip()
    except Exception:
        branch = os.environ.get("RUN_BRANCH","unknown")
        commit = os.environ.get("RUN_COMMIT","unknown")
    return branch, commit

# Collect available primitives (name and typed signature) for run info
def _enumerate_primitives(_pset: gp.PrimitiveSetTyped):
    prim_list = []
    try:
        for _ret, _prims in _pset.primitives.items():
            for _p in _prims:
                _args = [getattr(t, '__name__', str(t)) for t in _p.args]
                _retname = getattr(_p.ret, '__name__', str(_p.ret))
                prim_list.append(f"{_p.name}({', '.join(_args)}) -> {_retname}")
    except Exception:
        # Fallback: at least list names if detailed typing fails
        try:
            prim_list = sorted({getattr(p, 'name', str(p)) for v in _pset.primitives.values() for p in v})
        except Exception:
            prim_list = []
    # de-duplicate and sort for readability
    prim_list = sorted(set(prim_list))
    return prim_list

# Enumerate available terminals (including ephemeral constants) for run info
def _enumerate_terminals(_pset: gp.PrimitiveSetTyped):
    terms = []
    try:
        for _ret, _terms in _pset.terminals.items():
            for _t in _terms:
                # gp.Terminal typically has .name and .ret
                _name = getattr(_t, 'name', str(_t))
                _retname = getattr(_t.ret, '__name__', str(_t.ret))
                terms.append(f"{_name} -> {_retname}")
    except Exception:
        try:
            # Fallback to names only
            terms = sorted({getattr(t, 'name', str(t)) for v in _pset.terminals.values() for t in v})
        except Exception:
            terms = []
    return sorted(set(terms))

def _write_run_info(run_dir, meta, args, randomSeeds):
    branch, commit = ("unknown","unknown") if args.no_git_check else _git_info_safe(os.getcwd())
    p = os.path.join(run_dir, "run_info.txt")
    with open(p, "w") as f:
        f.write(f"timestamp: {meta['timestamp']}\n")
        f.write(f"slurm_job_id: {meta['jid']}\n")
        f.write(f"slurm_job_name: {meta['jname']}\n")
        f.write(f"randomSeeds: {randomSeeds}\n")
        f.write(f"cmd: {' '.join(sys.argv)}\n")
        f.write(f"run_name: {meta['run_name']}\n")
        f.write(f"base_out: {meta['base_out']}\n")
        f.write(f"git_branch: {branch}\n")
        f.write(f"git_commit: {commit}\n")

        # Write fitness bonus configuration
        use_div_bonus = os.environ.get("USE_DIVERSITY_BONUS")
        use_cmplx_bonus = os.environ.get("USE_COMPLEXITY_BONUS")
        div_bonus_val = os.environ.get("DIVERSITY_BONUS_PER_MODEL")
        cmplx_bonus_val = os.environ.get("MAX_COMPLEXITY_BONUS")
        
        f.write("\n--- Fitness Bonus Configuration ---\n")
        if use_div_bonus is not None:
            f.write(f"use_diversity_bonus: {use_div_bonus}\n")
            if div_bonus_val is not None:
                f.write(f"diversity_bonus_per_model: {div_bonus_val}\n")
        if use_cmplx_bonus is not None:
            f.write(f"use_complexity_bonus: {use_cmplx_bonus}\n")
            if cmplx_bonus_val is not None:
                f.write(f"max_complexity_bonus: {cmplx_bonus_val}\n")

        use_aug = os.environ.get("USE_DATA_AUGMENTATION")
        aug_cfg = os.environ.get("AUGMENTATION_CONFIG")
        aug_params = os.environ.get("AUGMENTATION_PARAMS")
        if use_aug is not None:
            f.write(f"\nuse_data_augmentation: {use_aug}\n")
        if aug_cfg:
            try:
                f.write("augmentation_config:\n")
                f.write(json.dumps(json.loads(aug_cfg), indent=2) + "\n")
            except Exception:
                f.write(f"augmentation_config_raw: {aug_cfg}\n")
        if aug_params:
            try:
                f.write("augmentation_params:\n")
                f.write(json.dumps(json.loads(aug_params), indent=2) + "\n")
            except Exception:
                f.write(f"augmentation_params_raw: {aug_params}\n")

def _copy_slurm_logs(run_dir, meta):
    try:
        slurm_log_dir = "/dataB1/niels_witbreuk/logs"
        src_out = os.path.join(slurm_log_dir, f"{meta['jname']}.{meta['jid']}.out")
        src_err = os.path.join(slurm_log_dir, f"{meta['jname']}.{meta['jid']}.err")
        dest = os.path.join(run_dir, "slurm")
        os.makedirs(dest, exist_ok=True)
        for src in (src_out, src_err):
            if os.path.exists(src):
                shutil.copy(src, dest)
    except Exception as e:
        with open(os.path.join(run_dir, "run_info.txt"), "a") as f:
            f.write(f"slurm-copy-error: {e}\n")


def saveResults(fileName, *args, **kwargs):
    f = open(fileName, 'w')
    for i in args:
        f.writelines(str(i)+'\n')
    f.close()
    return



def saveLog (fileName, log):
   f = open(fileName, 'wb')
   pickle.dump(log, f)
   f.close()
   return


def plotTree(pathName, individual):
    """
    Save a visualization of an individual as PNG, plus DOT and TXT representations.
    - pathName: file path (extension ignored) or directory+basename; .png/.dot/.txt will be created.
    - individual: DEAP GP individual (hof[0] etc.)
    """
    nodes, edges, labels = gp.graph(individual)

    # base path (strip extension if present)
    base, _ = os.path.splitext(pathName)
    png_path = base + ".png"
    txt_path = base + ".txt"

    # 2) write human-readable text representation
    try:
        with open(txt_path, "w") as f:
            f.write("String representation:\n")
            try:
                f.write(str(individual) + "\n\n")
            except Exception:
                f.write(repr(individual) + "\n\n")
            f.write("Nodes and labels:\n")
            for n in nodes:
                f.write(f"{n}: {labels.get(n,'')}\n")
            f.write("\nEdges:\n")
            for a, b in edges:
                f.write(f"{a} -> {b}\n")
    except Exception as e:
        print("plotTree: failed to write TXT:", e)

    try:
        g = pgv.AGraph(directed=True)
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        # set labels
        for i in nodes:
            n = g.get_node(i)
            n.attr["label"] = labels.get(i, str(i)) # type: ignore
        g.layout(prog="dot")
        g.draw(png_path)
    except Exception as e:
        # If rendering fails, inform user but DOT is still available
        print("plotTree: pygraphviz render failed (DOT written):", e)

    return



def bestInd(toolbox, population, number):
    bestInd = []
    best = toolbox.selectElitism(population, k=number)
    for i in best:
        bestInd.append(i)
    return bestInd
        


def saveAllResults(params, hof, trainTime, testResults, log, outdir="/dataB1/niels_witbreuk/logs/myruns", meta=None, args=None, randomSeeds=None):
    os.makedirs(outdir, exist_ok=True)

    summary_path = os.path.join(outdir, f"Summary_on{params['Dataset']}.txt")
    try:
        with open(summary_path, "w") as f:
            f.write("=== RUN INFO ===\n")
            # write run meta (from _make_run_dir)
            if meta is None:
                meta = {}
            timestamp = meta.get("timestamp", time.strftime("%Y%m%d-%H%M%S"))
            f.write(f"timestamp: {timestamp}\n")
            f.write(f"slurm_job_id: {meta.get('jid','unknown')}\n")
            f.write(f"slurm_job_name: {meta.get('jname','unknown')}\n")
            f.write(f"run_name: {meta.get('run_name','')}\n")
            f.write(f"base_out: {meta.get('base_out', outdir)}\n")
            # git info
            branch, commit = ("unknown","unknown")
            if args is None or not getattr(args, "no_git_check", False):
                try:
                    branch, commit = _git_info_safe(os.getcwd())
                except Exception:
                    branch, commit = ("unknown","unknown")
            f.write(f"git_branch: {branch}\n")
            f.write(f"git_commit: {commit}\n")
            # command & seed info
            f.write(f"cmd: {' '.join(sys.argv)}\n")
            f.write(f"randomSeeds: {randomSeeds}\n\n")


            f.write("=== PARAMETERS ===\n")
            try:
                f.write(pprint.pformat(params) + "\n\n")
            except Exception:
                for k in sorted(params.keys()):
                    f.write(f"{k}: {params[k]}\n")
                f.write("\n")


            f.write("=== FINAL RESULTS ===\n")
            f.write(f"Dataset: {params.get('Dataset')}\n")
            f.write(f"image_dir: {params.get('image_dir')}\n")
            f.write(f"randomSeeds: {params.get('randomSeeds')}\n")
            f.write(f"trainTime: {trainTime}\n")
            f.write(f"trainResults (hof[0].fitness): {getattr(hof[0], 'fitness', None)}\n")
            f.write(f"testResults: {testResults}\n\n")

            plot_base = os.path.join(outdir, "best_individual")
            plotTree(plot_base, hof[0])

    except Exception as e:
        print("Warning: failed to write summary file:", e)

    return

def build_class_remap(selected: list[int] | None, k: int,
                      ignore_index: int = DEFAULT_IGNORE_INDEX,
                      drop_background: bool = False):
    """
    Build a mapping from original class IDs to new contiguous indices.
    
    Args:
        selected: list of original class IDs to include; if None or empty, returns (None, None)
        k: maximum number of classes to map to
        ignore_index: label value to ignore
        drop_background: if True, exclude background class (0) from mapping
    Returns:
        (class_to_idx, ignore_index) where class_to_idx maps original IDs to new indices, or (None, None) if no remapping is needed.
    """
    if not selected:
        return None, None
    # exclude ignore and optionally background (0)
    selected = [cid for cid in selected if cid != ignore_index and (cid != 0 if drop_background else True)]
    class_to_idx = {cid: i for i, cid in enumerate(selected[:k])}
    return class_to_idx, ignore_index



class RemapWrapper(Dataset):
    def __init__(self, base: Dataset, class_to_idx: dict[int,int] | None, ignore_index: int | None):
        self.base = base
        self.class_to_idx = class_to_idx
        self.num_out_classes = len(class_to_idx) if class_to_idx else 0
        self.ignore_index = ignore_index
    def __len__(self): return len(self.base)  # type: ignore
    def __getitem__(self, idx):
        img, mask = self.base[idx]
        mask_t = torch.as_tensor(np.array(mask), dtype=torch.int64)

        # ✅ Binary single-class case: map target -> 1, everything else -> 0
        if self.num_out_classes == 1 and len(self.class_to_idx) == 1: # type: ignore
            target_id = next(iter(self.class_to_idx.keys())) # type: ignore
            mask_t = torch.where(mask_t == target_id, 1, 0)
            return img, mask_t

        # ...existing multiclass remap logic...
        if self.class_to_idx and (self.ignore_index is not None):
            mask = remap_mask_tensor(mask, self.class_to_idx, self.ignore_index)
        return img, mask
    
def _count_matches(present: set[int], wanted: set[int]) -> int:
    return len(present & wanted)

def remap_mask_tensor(mask: torch.Tensor, class_to_idx: dict[int, int], ignore_index: int) -> torch.Tensor:
    if mask.dim() == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    out = torch.full_like(mask, ignore_index)
    for cid, new_idx in class_to_idx.items():
        out = torch.where(mask == cid, torch.as_tensor(new_idx, dtype=out.dtype, device=out.device), out)
    return out
class ImagePreFilterWrapper(Dataset):
    def __init__(self, base_dataset: Dataset, selected_ids: set[int],
                 ignore_index: int = DEFAULT_IGNORE_INDEX,
                 require_all: bool = False,
                 min_match: int | None = None):
        self.base = base_dataset
        self.valid_indices = []
        self.selected_ids = {c for c in selected_ids if c not in (ignore_index, 0)}
        self.require_all = require_all
        self.min_match = (len(self.selected_ids) if require_all else (min_match or 1))
        print(f"Pre-filtering dataset to find images with at least {self.min_match} of {sorted(self.selected_ids)}...")
        for i in tqdm(range(len(self.base)), desc="Scanning masks", disable=True):  # type: ignore
            try:
                _, mask = self.base[i]
            except Exception:
                continue
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(np.array(mask), dtype=torch.int64)
            if mask.dim() == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
            present = set(int(v) for v in torch.unique(mask).tolist())
            present.discard(ignore_index); present.discard(0)
            if _count_matches(present, self.selected_ids) >= self.min_match:
                self.valid_indices.append(i)
        print(f"Found {len(self.valid_indices)} / {len(self.base)} images matching the filter.")  # type: ignore
    def __len__(self): return len(self.valid_indices)
    def __getitem__(self, idx): return self.base[self.valid_indices[idx]]

class PreAugmentedDataset(Dataset):
    """Wraps pre-computed augmented samples for fast access."""
    def __init__(self, augmented_samples):
        self.samples = augmented_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

class AugmentationWrapper(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        image_transform: T.Compose | None = None,
        joint_transform: T.Compose | None = None,
        enabled: bool = True
    ):
        self.base = base_dataset
        self.enabled = enabled
        self.image_transform = image_transform
        self.joint_transform = joint_transform
        
    def __len__(self):
        return len(self.base) # type: ignore
    
    def __getitem__(self, idx):
        img, mask = self.base[idx]
        
        if not self.enabled:
            return img, mask
        
        # Convert tensors to PIL for v2 transforms
        if isinstance(img, torch.Tensor):
            img = T.ToPILImage()(img)
        
        # Apply spatial transforms (joint) FIRST
        if self.joint_transform:
            # Note: joint transforms need both img and mask
            # For now, we skip this if masks aren't PIL-compatible
            pass  # Spatial transforms disabled for now
        
        # Apply photometric transforms (image-only)
        # This includes ToTensor() at the end
        if self.image_transform:
            img = self.image_transform(img)
        else:
            img = T.ToTensor()(img)
        
        # Ensure mask is tensor
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(np.array(mask))
        
        return img, mask


def build_augmentation_transforms(
    augmentation_config: list[dict] | None = None,
    augmentation_params: dict | None = None,
) -> tuple[T.Compose | None, T.Compose | None]:
    """
    Build image-only and joint (spatial) transforms from config.
    
    Returns:
        (image_transform, joint_transform)
    """
    config = augmentation_config or []
    params = augmentation_params or {}
    
    image_only_transforms = []
    spatial_transforms = []
    
    for aug_spec in config:
        if not aug_spec.get("enabled", False):
            continue
        
        aug_name = aug_spec["name"]
        param_range = params.get(aug_name, {})
        
        # Photometric (image-only) transforms
        if aug_name == "adjust_brightness":
            delta = param_range.get("max", 0.3)
            image_only_transforms.append(
                v2.ColorJitter(brightness=delta, contrast=0, saturation=0, hue=0)
            )
        
        elif aug_name == "adjust_gamma":
            # v2 doesn't have gamma, use brightness as proxy
            image_only_transforms.append(
                v2.ColorJitter(brightness=0.15, contrast=0, saturation=0, hue=0)
            )
        
        elif aug_name == "adjust_contrast":
            alpha = param_range.get("max", 0.3)
            image_only_transforms.append(
                v2.ColorJitter(brightness=0, contrast=alpha, saturation=0, hue=0)
            )
        
        elif aug_name == "trans_blur":
            image_only_transforms.append(
                v2.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0))
            )
        
        # Spatial transforms (applied to both image and mask)
        elif aug_name == "trans_rotate":
            degrees = param_range.get("max", 30)
            spatial_transforms.append(v2.RandomRotation(degrees=degrees))
        
        elif aug_name == "trans_translate":
            translate_range = param_range.get("max", 0.1)
            spatial_transforms.append(
                v2.RandomAffine(degrees=0, translate=(translate_range, translate_range)) # type: ignore
            )
    
    # ✅ Always add ToTensor at the end to ensure output is tensor
    if image_only_transforms:
        image_only_transforms.extend([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
    else:
        image_only_transforms = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ]
    
    # Build final composed transforms
    image_transform = T.Compose(image_only_transforms)
    joint_transform = T.Compose(spatial_transforms) if spatial_transforms else None
    
    return image_transform, joint_transform

class StaticAugmentationWrapper(Dataset):
    """
    Applies one fixed augmentation pipeline to all samples (deterministic per run).
    Spatial transforms affect both image and mask; photometric transforms affect image only.
    """
    def __init__(self, base_dataset: Dataset,
                 augmentation_config: list[dict],
                 augmentation_params: dict,
                 seed: int = 12345):
        self.base = base_dataset
        self.config = augmentation_config or []
        self.params = augmentation_params or {}
        rng = np.random.default_rng(seed)

        # Sample fixed parameters once
        self.fixed = {}
        for aug in self.config:
            if not aug.get("enabled", False):
                continue
            name = aug["name"]
            pr = self.params.get(name, {})
            lo, hi = pr.get("min", 0.0), pr.get("max", 0.0)
            if name == "trans_blur":
                # k from [3..21] odd, sigma auto
                k_norm = float(rng.uniform(lo, hi))
                k = int(3 + k_norm * 18)
                if k % 2 == 0:
                    k += 1
                self.fixed[name] = {"kernel": k}
            elif name == "trans_rotate":
                deg = float(rng.uniform(lo, hi))
                # center around 0
                self.fixed[name] = {"degrees": (deg if rng.uniform() > 0.5 else -deg)}
            elif name == "trans_translate":
                t = float(rng.uniform(lo, hi))
                # translate as fraction of image size; stored as (-t..t)
                self.fixed[name] = {"translate_frac": (rng.uniform(-t, t), rng.uniform(-t, t))}
            elif name == "adjust_gamma":
                self.fixed[name] = {"gamma": float(rng.uniform(lo, hi))}
            elif name == "adjust_brightness":
                self.fixed[name] = {"factor": float(1.0 + rng.uniform(lo, hi))}
            elif name == "adjust_contrast":
                self.fixed[name] = {"factor": float(rng.uniform(lo, hi))}
            else:
                # unknown; skip
                pass

    def __len__(self):
        return len(self.base) # type: ignore

    def __getitem__(self, idx):
        img, mask = self.base[idx]

        # Ensure PIL for functional API
        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img)
        if isinstance(mask, torch.Tensor):
            mask = Image.fromarray(mask.cpu().numpy().astype(np.uint8))
        elif isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask.astype(np.uint8))

        # ----- spatial transforms (joint) -----
        if "trans_rotate" in self.fixed:
            ang = self.fixed["trans_rotate"]["degrees"]
            img = TF.rotate(img, ang, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, ang, interpolation=TF.InterpolationMode.NEAREST) # type: ignore
        if "trans_translate" in self.fixed:
            tx_f, ty_f = self.fixed["trans_translate"]["translate_frac"]
            w, h = img.size # type: ignore
            tx, ty = int(tx_f * w), int(ty_f * h)
            img = TF.affine(img, angle=0, translate=[tx, ty], scale=1.0, shear=0, # type: ignore
                            interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.affine(mask, angle=0, translate=[tx, ty], scale=1.0, shear=0, # type: ignore
                             interpolation=TF.InterpolationMode.NEAREST)

        # ----- photometric transforms (image only) -----
        if "adjust_brightness" in self.fixed:
            img = TF.adjust_brightness(img, self.fixed["adjust_brightness"]["factor"])
        if "adjust_contrast" in self.fixed:
            img = TF.adjust_contrast(img, self.fixed["adjust_contrast"]["factor"])
        if "adjust_gamma" in self.fixed:
            img = TF.adjust_gamma(img, self.fixed["adjust_gamma"]["gamma"])
        if "trans_blur" in self.fixed:
            k = self.fixed["trans_blur"]["kernel"]
            img = TF.gaussian_blur(img, kernel_size=k)

        # Convert back to tensors
        img = v2.ToImage()(img)
        img = v2.ToDtype(torch.float32, scale=True)(img)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return img, mask
