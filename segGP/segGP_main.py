import operator
import argparse
import random
import torch
from torch.utils.data import random_split, DataLoader
import os
import numpy as np
import time
import multiprocessing
from generic_seg_loader import GeneralSegDataset
import gp_restrict as gp_restrict
import algo_iegp as evalGP
import numpy as np
from deap import base, creator, tools, gp
import seggp_functions as felgp_fs
import seg_types
from typing import Any
from visualize import visualize_predictions
from data_handling import _enumerate_primitives, _enumerate_terminals, pad_collate
import data_handling as data_handling
from miou_eval import eval_dataset_miou  # dataset-level mIoU (no penalties integrated)
from algo_iegp import run_pretrained_baseline  # baseline eval without GP
import sys


# User-configurable options
COLOR_MODE = "rgb"  # "rgb" or "gray"
DATASET = "voc" # "voc" or "weizmann"
BASELINE_ONLY = "0" # "0" for no or "1" for yes, run only the pretrained NN and exit
RUN_MODE = "middle"  # "fast", "middle", "normal"
randomSeeds = 12
Run_title_SUFFIX = "test" # Optional suffix for run name

# region ==== GP Setup ====

toolbox: base.Toolbox  # type: ignore
creator.FitnessMax: Any  # type: ignore
creator.Individual: Any  # type: ignore
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Infer number of classes from dataset selection; allow optional override.
if DATASET == 'voc':
    inferred_classes = 21
elif DATASET == 'weizmann':
    inferred_classes = 1
else:
    raise ValueError(f"Unknown dataset option: {DATASET}")

NUM_CLASSES = inferred_classes


if DATASET == 'voc':
    dataSetName = 'Pascal_VOC'
    image_dir = "/dataB5/kieran_carrigg/VOC2012/VOC2012_train_val/VOC2012_train_val/JPEGImages"
    mask_dir = "/dataB5/kieran_carrigg/VOC2012/VOC2012_train_val/VOC2012_train_val/SegmentationClass"
elif DATASET == 'weizmann':
    dataSetName = 'weizmann_horse'
    image_dir = "data/weizmann_horse/horse"
    mask_dir = "data/weizmann_horse/mask"
else:
    raise ValueError(f"Unknown dataset option: {DATASET}")

dataset = GeneralSegDataset(image_dir=image_dir, mask_dir=mask_dir, mode="png", color_mode=COLOR_MODE, num_classes=NUM_CLASSES)
IGNORE_INDEX = getattr(dataset, 'ignore_index', None)
RUN_OUTDIR="/dataB1/niels_witbreuk/logs/myruns"



RUN_NAME = data_handling.make_run_name(
    dataset_name=dataSetName,
    run_mode=RUN_MODE,
    seed=randomSeeds,
    color_mode=COLOR_MODE,
    suffix=Run_title_SUFFIX,
    baseline_only=bool(int(BASELINE_ONLY)) if isinstance(BASELINE_ONLY, str) else bool(BASELINE_ONLY),
)

# Presets per mode
_PRESETS = {
    "fast":   {"pop_size": 10, "generation": 5,  "initialMaxDepth": 6, "maxDepth": 6,  "batch_size": 4, "cap_train": 80,  "cap_test": 32},
    "middle": {"pop_size": 25, "generation": 15, "initialMaxDepth": 7, "maxDepth": 7,  "batch_size": 6, "cap_train": 200, "cap_test": 80},
    "normal": {"pop_size": 50, "generation": 30, "initialMaxDepth": 8, "maxDepth": 8,  "batch_size": 8, "cap_train": 400, "cap_test": 160},
}
preset = _PRESETS[RUN_MODE]

# GP hyperparams (common values for all modes)
cxProb     = 0.8
mutProb    = 0.19
elitismProb= 0.01
initialMinDepth = 2

"""
Mode-dependent variables and data loaders will be initialized in __main__ after
parsing the main CLI flags, so users can switch datasets via main flags rather
than the early pre-parser.
"""
pop_size = preset["pop_size"]
generation = preset["generation"]
initialMaxDepth = preset["initialMaxDepth"]
maxDepth = preset["maxDepth"]
_MODE_BATCH_SIZE = preset["batch_size"]
_MODE_CAP_TRAIN = preset["cap_train"]
_MODE_CAP_TEST = preset["cap_test"]

# Split and optionally cap train/test sizes
full_train_size = int(0.8 * len(dataset))
full_test_size  = len(dataset) - full_train_size
train_dataset, test_dataset = random_split(
    dataset, [full_train_size, full_test_size],
    generator=torch.Generator().manual_seed(42)
)

# Optionally cap dataset sizes for faster modes
if _MODE_CAP_TRAIN is not None or _MODE_CAP_TEST is not None:
    from torch.utils.data import Subset
    max_train = _MODE_CAP_TRAIN or len(train_dataset)
    max_test  = _MODE_CAP_TEST or len(test_dataset)
    train_idx = list(range(min(max_train, len(train_dataset))))
    test_idx  = list(range(min(max_test, len(test_dataset))))
    train_dataset = Subset(train_dataset, train_idx)
  

# DataLoaders: increase workers and pin memory for CUDA
num_workers = min(4, os.cpu_count() or 1)
pin = (device == 'cuda')
train_loader = DataLoader(train_dataset, batch_size=_MODE_BATCH_SIZE,
                           shuffle=True, collate_fn=pad_collate,
                           num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers>0)
test_loader  = DataLoader(test_dataset, batch_size=_MODE_BATCH_SIZE,
                          shuffle=False, collate_fn=pad_collate,
                          num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers>0)

# Optional: run only the pretrained baseline and exit early
if BASELINE_ONLY:
    print(f"[Baseline] Evaluating pretrained_seg_nn on dataset '{dataSetName}' (classes={NUM_CLASSES})...")
    from types import SimpleNamespace
    args_ns = SimpleNamespace(run_name=RUN_NAME, outdir=RUN_OUTDIR, no_git_check=False)
    run_dir, meta = data_handling._make_run_dir(args_ns, dataSetName, randomSeeds)
    data_handling._write_run_info(run_dir, meta, args_ns, randomSeeds)
    # Perform baseline evaluation and save artifacts into run_dir
    _ = run_pretrained_baseline(
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=NUM_CLASSES,
        device=device,
        ignore_index=IGNORE_INDEX,
        run_dir=run_dir,
        dataset_name=dataSetName,
        randomSeeds=randomSeeds,
        params={
            "Dataset": dataSetName,
            "RUN_MODE": RUN_MODE,
            "color_mode": COLOR_MODE,
            "num_classes": NUM_CLASSES,
        },
        args=args_ns,
        meta=meta,
        copy_slurm_logs=True,
        save_visuals=True,
        vis_count=6,
    )
    sys.exit(0)

params = {
        "Dataset": dataSetName,
        "RUN_MODE": RUN_MODE,
        "randomSeeds": randomSeeds,
        "pop_size": pop_size,
        "generation": generation,
        "cxProb": cxProb,
        "mutProb": mutProb,
        "elitismProb": elitismProb,
        "initialMinDepth": initialMinDepth,
        "initialMaxDepth": initialMaxDepth,
        "maxDepth": maxDepth,
        "train_batch_size": train_loader.batch_size,
        "test_batch_size": test_loader.batch_size,
        "color_mode": COLOR_MODE,
        "num_classes": NUM_CLASSES,
        "image_dir": image_dir,
        "mask_dir": mask_dir,
    }

# endregion ==== GP Setup ====

# region ==== GP function and terminal set definitions ====

# Select input type based on color mode and desired output type
_INPUT_TYPE = seg_types.RGBImage if COLOR_MODE == "rgb" else seg_types.GrayImage
_OUTPUT_TYPE = seg_types.FeatureMap if NUM_CLASSES > 1 else seg_types.Mask
pset = gp.PrimitiveSetTyped('MAIN', [_INPUT_TYPE], _OUTPUT_TYPE, prefix='Image')

# Add Basic Math Operators
# Math operators operate on intermediate feature maps
pset.addPrimitive(felgp_fs.add, [seg_types.FeatureMap, seg_types.FeatureMap], seg_types.FeatureMap, name="Add")
pset.addPrimitive(felgp_fs.sub, [seg_types.FeatureMap, seg_types.FeatureMap], seg_types.FeatureMap, name="Sub")
pset.addPrimitive(felgp_fs.mul, [seg_types.FeatureMap, seg_types.FeatureMap], seg_types.FeatureMap, name="Mul")
pset.addPrimitive(felgp_fs.safe_div, [seg_types.FeatureMap, seg_types.FeatureMap], seg_types.FeatureMap, name="SafeDiv")
pset.addPrimitive(felgp_fs.abs_f, [seg_types.FeatureMap], seg_types.FeatureMap, name="Abs")
pset.addPrimitive(felgp_fs.sqrt_f, [seg_types.FeatureMap], seg_types.FeatureMap, name="Sqrt")
pset.addPrimitive(felgp_fs.log_f, [seg_types.FeatureMap], seg_types.FeatureMap, name="Log")
pset.addPrimitive(felgp_fs.exp_f, [seg_types.FeatureMap], seg_types.FeatureMap, name="Exp")
pset.addPrimitive(felgp_fs.sigmoid, [seg_types.FeatureMap], seg_types.FeatureMap, name="Sigmoid")
pset.addPrimitive(felgp_fs.tanh_f, [seg_types.FeatureMap], seg_types.FeatureMap, name="tanh")

# Add logical operators
pset.addPrimitive(felgp_fs.logical_not, [seg_types.FeatureMap], seg_types.FeatureMap, name="LogicalNot")
pset.addPrimitive(felgp_fs.logical_or, [seg_types.FeatureMap, seg_types.FeatureMap], seg_types.FeatureMap, name="LogicalOr")
pset.addPrimitive(felgp_fs.logical_and, [seg_types.FeatureMap, seg_types.FeatureMap], seg_types.FeatureMap, name="LogicalAnd")
pset.addPrimitive(felgp_fs.logical_xor, [seg_types.FeatureMap, seg_types.FeatureMap], seg_types.FeatureMap, name="LogicalXor")

# Add comparison and normalization
pset.addPrimitive(felgp_fs.gt, [seg_types.FeatureMap, float], seg_types.FeatureMap, name="Gt")
pset.addPrimitive(felgp_fs.lt, [seg_types.FeatureMap, float], seg_types.FeatureMap, name="Lt")
pset.addPrimitive(felgp_fs.normalize, [seg_types.FeatureMap], seg_types.FeatureMap, name="Normalize")

# Note: Do not add RGB/Gray converters; keep primitive set strictly within selected color mode

# Add filters and edge detection functions (color-mode specific)
if COLOR_MODE == "rgb":
    pset.addPrimitive(felgp_fs.sobel_x, [seg_types.RGBImage], seg_types.FeatureMap, name="SobelX")
    pset.addPrimitive(felgp_fs.sobel_y, [seg_types.RGBImage], seg_types.FeatureMap, name="SobelY")
    pset.addPrimitive(felgp_fs.laplacian, [seg_types.RGBImage], seg_types.FeatureMap, name="Laplacian")
    pset.addPrimitive(felgp_fs.gradient_magnitude, [seg_types.RGBImage], seg_types.FeatureMap, name="GradientMagnitude")
else:  # gray
    pset.addPrimitive(felgp_fs.sobel_x, [seg_types.GrayImage], seg_types.FeatureMap, name="SobelX_Gray")
    pset.addPrimitive(felgp_fs.sobel_y, [seg_types.GrayImage], seg_types.FeatureMap, name="SobelY_Gray")
    pset.addPrimitive(felgp_fs.laplacian, [seg_types.GrayImage], seg_types.FeatureMap, name="Laplacian_Gray")
    pset.addPrimitive(felgp_fs.gradient_magnitude, [seg_types.GrayImage], seg_types.FeatureMap, name="GradientMagnitude_Gray")

# Combination functions
pset.addPrimitive(felgp_fs.mix, [seg_types.FeatureMap, seg_types.FeatureMap, float], seg_types.FeatureMap, name="Mix")
pset.addPrimitive(felgp_fs.if_then_else, [seg_types.FeatureMap, seg_types.FeatureMap, seg_types.FeatureMap], seg_types.FeatureMap, name="IfElse")
if COLOR_MODE == "rgb":
    pset.addPrimitive(felgp_fs.gaussian_blur_param, [seg_types.RGBImage, float], seg_types.FeatureMap, name="Gauss")
else:
    pset.addPrimitive(felgp_fs.gaussian_blur_param, [seg_types.GrayImage, float], seg_types.FeatureMap, name="Gauss_Gray")

""" Pretrained segmentation NN and higher-level image->features (color-mode specific). """
if COLOR_MODE == "rgb":
    pset.addPrimitive(felgp_fs.pretrained_seg_nn, [seg_types.RGBImage], seg_types.FeatureMap, name="PretrainedSeg")
    pset.addPrimitive(felgp_fs.apply_depthwise_edge, [seg_types.RGBImage, float], seg_types.FeatureMap, name="EdgeFilter")
    pset.addPrimitive(felgp_fs.image_to_featuremap, [seg_types.RGBImage, float], seg_types.FeatureMap, name="ImageToFeat")
else:
    # gray mode: register pretrained as feature extractor
    pset.addPrimitive(felgp_fs.pretrained_seg_nn, [seg_types.GrayImage], seg_types.FeatureMap, name="PretrainedSeg")
    pset.addPrimitive(felgp_fs.apply_depthwise_edge, [seg_types.GrayImage, float], seg_types.FeatureMap, name="EdgeFilterGray")
    pset.addPrimitive(felgp_fs.image_to_featuremap, [seg_types.GrayImage, float], seg_types.FeatureMap, name="ImageToFeatGray")


pset.addPrimitive(felgp_fs.apply_aspp_cached, [seg_types.FeatureMap, float], seg_types.FeatureMap, name="ASPPCached")

if NUM_CLASSES == 1:
    # Only include single-channel collapse in binary mode
    pset.addPrimitive(felgp_fs.apply_feature_to_mask_cached, [seg_types.FeatureMap, float], seg_types.Mask, name="FeatToMask")

# Collect available primitives (name and typed signature) for run info
AVAILABLE_PRIMITIVES = _enumerate_primitives(pset)
try:
    params["available_primitives"] = AVAILABLE_PRIMITIVES
except Exception:
    pass

# Collect available terminals (name and typed signature) for run info
AVAILABLE_TERMINALS = _enumerate_terminals(pset)
try:
    params["available_terminals"] = AVAILABLE_TERMINALS
except Exception:
    pass


#Terminals
# Float thresholds for Gt/Lt (use a named function for multiprocessing pickling)
def rand_thresh():
    return float(np.random.uniform(0.05, 0.95))
pset.addEphemeralConstant('Thresh', rand_thresh, float)
def rand_channel():
    return float(np.random.choice([8,16,32,64,128]))
pset.addEphemeralConstant('Ch', rand_channel, float)
def rand_alpha(): return float(np.random.uniform(0.0, 1.0))
pset.addEphemeralConstant('Alpha', rand_alpha, float)

##
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax) # type: ignore

toolbox = base.Toolbox()
pool = multiprocessing.Pool()
toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=initialMinDepth, max_=initialMaxDepth)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr) # type: ignore
toolbox.register("population", tools.initRepeat, list, toolbox.individual) # type: ignore
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("mapp", pool.map)

# endregion ==== GP function and terminal set definitions ====

# region ==== GP helper and evaluation functions ====

def _binarize_from_logits(logits):
    # Robust: handle NaNs/Infs and non-float outputs
    if not logits.dtype.is_floating_point:
        logits = logits.float()
    logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
    # Reduce to single channel if multi-channel
    if logits.dim() == 4 and logits.shape[1] > 1:
        logits = logits.mean(dim=1, keepdim=True)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    return preds, probs

def _ensure_k_channels(x: torch.Tensor, k: int) -> torch.Tensor:
    # Project channel dimension to k using cached 1x1 conv in felgp_fs
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() == 4:
        c = x.shape[1]
        if c != k:
            try:
                return felgp_fs._project_channels_cached(x, k)
            except Exception:
                # fallback: simple repeat/truncate
                if c == 1:
                    return x.repeat(1, k, 1, 1)
                return x[:, :k, ...] if c > k else torch.cat([x, x[:, : (k - c), ...]], dim=1)
    return x

def _multiclass_preds(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    # logits: (B,C,H,W) or (B,H,W) or (C,H,W)
    if not logits.dtype.is_floating_point:
        logits = logits.float()
    logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
    if logits.dim() == 3:
        logits = logits.unsqueeze(0)
    if logits.dim() == 2:
        logits = logits.unsqueeze(0).unsqueeze(0)
    if logits.dim() != 4:
        raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")
    logits = _ensure_k_channels(logits, k)
    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)  # (B,H,W)
    return preds, probs

def evalTrain(toolbox, individual, hof):
    """
    Evaluate the fitness of an individual on the training data.
    If the individual is in the Hall of Fame, reuse its fitness.
    Otherwise, compile and execute the individual, compute predictions,
    and return classification accuracy as fitness.

    Args:
        toolbox: DEAP toolbox with compile method.
        individual: The GP individual to evaluate.
        hof: Hall of Fame list of individuals.
        trainData: Training data (features).
        trainLabel: Training labels.

    Returns:
        Tuple containing accuracy as a single-element tuple.
    """
    # Check if individual is in Hall of Fame
    for h in (hof or []):
        if individual == h:
            return h.fitness.values
    try:
        # Binary mode: keep existing Dice computation (dataset-level aggregation possible but left unchanged)
        if NUM_CLASSES == 1:
            func = toolbox.compile(expr=individual)
            inter_total = 0.0
            union_total = 0.0
            with torch.no_grad():
                for imgs, masks in train_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    out = func(imgs)
                    preds, probs = _binarize_from_logits(out)
                    inter_total += float((preds * masks).sum().item())
                    union_total += float(preds.sum().item() + masks.sum().item())
            dice = (2.0 * inter_total / (union_total + 1e-6)) if union_total > 0 else 0.0
            return (dice,)
        else:
            # Multiclass: dataset-level mIoU (background excluded by helper when NUM_CLASSES>1)
            _, miou = eval_dataset_miou(toolbox, individual, train_loader, NUM_CLASSES, device=device, ignore_index=IGNORE_INDEX)
            return (miou,)
    except Exception as e:
        print("Evaluation error (train):", e)
        return (0.0,)


toolbox.register("evaluate", evalTrain,toolbox)
toolbox.register("select", tools.selTournament,tournsize=7)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=2)
toolbox.register("mutate_eph", gp.mutEphemeral, mode='all')
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset) # type: ignore
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))

def GPMain(randomSeeds):
    """
    Main Genetic Programming loop.
    Initializes population, statistics, and runs the evolutionary algorithm.

    Args:
        randomSeeds: Seed for random number generator.

    Returns:
        pop: Final population.
        log: Logbook with statistics.
        hof: Hall of Fame individuals.
    """

    random.seed(randomSeeds)
   
    pop = toolbox.population(pop_size) # type: ignore
    hof = tools.HallOfFame(10)
    log = tools.Logbook()
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size_tree = tools.Statistics(key=len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size_tree=stats_size_tree)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    log.header = ["gen", "evals"] + mstats.fields # type: ignore

    pop, log = evalGP.eaSimple(pop, toolbox, cxProb, mutProb, elitismProb, generation,randomSeeds,
                    stats=mstats, halloffame=hof, verbose=True)

    return pop,log, hof


def evalTest(toolbox, individual, test_loader):
    """
    Evaluate the GP individual on the test set.
    Args:
        toolbox: DEAP toolbox with compile method.
        individual: The GP individual to evaluate.
        test_loader: DataLoader for the test dataset.
    Returns:
        Mean Dice score on the test set.
    """
    try:
        if NUM_CLASSES == 1:
            # Binary: dataset-level Dice
            func = toolbox.compile(expr=individual)
            inter_total = 0.0
            union_total = 0.0
            with torch.no_grad():
                for imgs, masks in test_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    out = func(imgs)
                    preds, probs = _binarize_from_logits(out)
                    inter_total += float((preds * masks).sum().item())
                    union_total += float(preds.sum().item() + masks.sum().item())
            dice = (2.0 * inter_total / (union_total + 1e-6)) if union_total > 0 else 0.0
            return dice
        else:
            _, miou = eval_dataset_miou(toolbox, individual, test_loader, NUM_CLASSES, device=device, ignore_index=IGNORE_INDEX)
            return miou
    except Exception as e:
        print("Evaluation error (test):", e)
        return 0.0
    
    # endregion ==== GP helper and evaluation functions ====


if __name__ == "__main__":
    os.environ["RUN_NAME"] = RUN_NAME
    os.environ["RUN_OUTDIR"] = RUN_OUTDIR

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--run-branch", type=str, default=None,
                        help="(optional) attempt to git checkout this branch before running")
    parser.add_argument("--no-git-check", action="store_true",
                        help="do not attempt to call git; use env vars or detect only")
    args, _unknown = parser.parse_known_args()

    start = time.process_time()
    pop, log, hof = GPMain(randomSeeds)
    trainTime = time.process_time() - start

    run_dir, meta = data_handling._make_run_dir(args, dataSetName, randomSeeds)

    testResults = evalTest(toolbox, hof[0], test_loader)
    # pass meta/args/randomSeeds so run info is embedded in the single summary file
    data_handling.saveAllResults(params, hof, trainTime, testResults, log, outdir=run_dir, meta=meta, args=args, randomSeeds=randomSeeds)
    # Visualize predictions for both binary and multiclass runs.
    visualize_predictions(toolbox, hof[0], test_dataset, num_samples=3, threshold=0.5,
                          save_dir=os.path.join(run_dir, "visualizations"), num_classes=NUM_CLASSES)

    # Persist primitive/terminal catalogs and best-individual primitive usage for quick inspection
    try:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "primitives.txt"), "w") as f:
            f.write("\n".join(AVAILABLE_PRIMITIVES) + "\n")
        with open(os.path.join(run_dir, "terminals.txt"), "w") as f:
            f.write("\n".join(AVAILABLE_TERMINALS) + "\n")

        from collections import Counter
        usage = Counter()
        try:
            for node in hof[0]:
                if isinstance(node, gp.Primitive):
                    usage[node.name] += 1
        except Exception:
            pass
        with open(os.path.join(run_dir, "primitive_usage_best.txt"), "w") as f:
            for name, cnt in sorted(usage.items(), key=lambda x: (-x[1], x[0])):
                f.write(f"{name} {cnt}\n")
    except Exception as _e:
        print(f"[warn] failed to write primitive/terminal catalogs: {_e}")

    data_handling._copy_slurm_logs(run_dir, meta)
    print("testResults", testResults)


    data_handling.saveAllResults(params, hof, trainTime, testResults, log, outdir=run_dir)

