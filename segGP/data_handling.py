import os
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
