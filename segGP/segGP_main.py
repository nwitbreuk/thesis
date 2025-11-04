import operator
import argparse
import random
import traceback
import torch
from torch.utils.data import random_split, DataLoader
import os
import numpy as np
import time
import multiprocessing
from weizmann_loader import WeizmannHorseDataset
import gp_restrict as gp_restrict
import algo_iegp as evalGP
import numpy as np
from deap import base, creator, tools, gp
import seggp_functions as felgp_fs
import seg_types
from typing import Any
from visualize import visualize_predictions
from data_handling import pad_collate
import data_handling as data_handling

# region ==== GP Setup ====

toolbox: base.Toolbox  # type: ignore
creator.FitnessMax: Any  # type: ignore
creator.Individual: Any  # type: ignore
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Early parse/env for color mode so dataset and GP can be configured before build
_DEFAULT_COLOR = os.environ.get("COLOR_MODE", "gray").lower() # change the default color mode here: "rgb" or "gray"
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--color", choices=["rgb", "gray"], default=_DEFAULT_COLOR,
                 help="Input color mode: rgb (3-channel) or gray (single-channel)")
_args_pre, _ = _pre.parse_known_args()
COLOR_MODE = _args_pre.color

dataSetName = 'Pascal_VOC'
image_dir = "data/PASCAL_VOC/VOC2012/images/train"
mask_dir = "data/pascal_voc/masks"
dataset = WeizmannHorseDataset(image_dir, mask_dir, color_mode=COLOR_MODE)
RUN_OUTDIR="/dataB1/niels_witbreuk/logs/myruns"

RUN_MODE = "normal"  # "fast", "middle", "normal"
randomSeeds = 10
RUN_NAME=f"{dataSetName}_{RUN_MODE}mode_seed{randomSeeds}_{{jid}}_-{COLOR_MODE}"
# _make_run_dir will replace {jid} with SLURM_JOB_ID (or process id if not running under SLURM)

# Presets per mode
_PRESETS = {
    "fast":   {"pop_size": 10, "generation": 5,  "initialMaxDepth": 6, "maxDepth": 6,  "batch_size": 4, "cap_train": 80,  "cap_test": 32},
    "middle": {"pop_size": 25, "generation": 15, "initialMaxDepth": 7, "maxDepth": 7,  "batch_size": 6, "cap_train": 200, "cap_test": 80},
    "normal": {"pop_size": 50, "generation": 30, "initialMaxDepth": 8, "maxDepth": 8,  "batch_size": 8, "cap_train": None, "cap_test": None},
}
preset = _PRESETS[RUN_MODE]

# GP hyperparams (common values for all modes)
cxProb     = 0.8
mutProb    = 0.19
elitismProb= 0.01
initialMinDepth = 2

# Mode-dependent
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
    }

# endregion ==== GP Setup ====

##GP
# Select input type based on color mode
_INPUT_TYPE = seg_types.RGBImage if COLOR_MODE == "rgb" else seg_types.GrayImage
pset = gp.PrimitiveSetTyped('MAIN', [_INPUT_TYPE], seg_types.Mask, prefix='Image')

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

# Converters between gray and RGB to allow cross-use of primitives
pset.addPrimitive(felgp_fs.ensure_rgb, [seg_types.GrayImage], seg_types.RGBImage, name="EnsureRGB")
pset.addPrimitive(felgp_fs.rgb_to_gray, [seg_types.RGBImage], seg_types.GrayImage, name="RGB2Gray")

# Add filters and edge detection functions
pset.addPrimitive(felgp_fs.sobel_x, [seg_types.RGBImage], seg_types.FeatureMap, name="SobelX")
pset.addPrimitive(felgp_fs.sobel_y, [seg_types.RGBImage], seg_types.FeatureMap, name="SobelY")
pset.addPrimitive(felgp_fs.laplacian, [seg_types.RGBImage], seg_types.FeatureMap, name="Laplacian")
pset.addPrimitive(felgp_fs.gradient_magnitude, [seg_types.RGBImage], seg_types.FeatureMap, name="GradientMagnitude")
pset.addPrimitive(felgp_fs.sobel_x, [seg_types.GrayImage], seg_types.FeatureMap, name="SobelX_Gray")
pset.addPrimitive(felgp_fs.sobel_y, [seg_types.GrayImage], seg_types.FeatureMap, name="SobelY_Gray")
pset.addPrimitive(felgp_fs.laplacian, [seg_types.GrayImage], seg_types.FeatureMap, name="Laplacian_Gray")
pset.addPrimitive(felgp_fs.gradient_magnitude, [seg_types.GrayImage], seg_types.FeatureMap, name="GradientMagnitude_Gray")

# Combination functions
pset.addPrimitive(felgp_fs.mix, [seg_types.FeatureMap, seg_types.FeatureMap, float], seg_types.FeatureMap, name="Mix")
pset.addPrimitive(felgp_fs.if_then_else, [seg_types.FeatureMap, seg_types.FeatureMap, seg_types.FeatureMap], seg_types.FeatureMap, name="IfElse")
pset.addPrimitive(felgp_fs.gaussian_blur_param, [seg_types.RGBImage, float], seg_types.FeatureMap, name="Gauss")
pset.addPrimitive(felgp_fs.gaussian_blur_param, [seg_types.GrayImage, float], seg_types.FeatureMap, name="Gauss_Gray")

# Pretrained segmentation NN
pset.addPrimitive(felgp_fs.pretrained_seg_nn, [seg_types.RGBImage], seg_types.FeatureMap, name="PretrainedSeg")
# high-level typed NN-like primitives
pset.addPrimitive(felgp_fs.apply_depthwise_edge, [seg_types.RGBImage, float], seg_types.FeatureMap, name="EdgeFilter")
pset.addPrimitive(felgp_fs.apply_depthwise_edge, [seg_types.GrayImage, float], seg_types.FeatureMap, name="EdgeFilterGray")
pset.addPrimitive(felgp_fs.image_to_featuremap, [seg_types.RGBImage, float], seg_types.FeatureMap, name="ImageToFeat")
pset.addPrimitive(felgp_fs.image_to_featuremap, [seg_types.GrayImage, float], seg_types.FeatureMap, name="ImageToFeatGray")
pset.addPrimitive(felgp_fs.apply_aspp_cached, [seg_types.FeatureMap, float], seg_types.FeatureMap, name="ASPPCached")
pset.addPrimitive(felgp_fs.apply_feature_to_mask_cached, [seg_types.FeatureMap, float], seg_types.Mask, name="FeatToMask")


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

AVAILABLE_PRIMITIVES = _enumerate_primitives(pset)
try:
    params["available_primitives"] = AVAILABLE_PRIMITIVES
except Exception:
    pass

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

_dbg_train_printed = False
_dbg_test_printed = False

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
        func = toolbox.compile(expr=individual)
        dice_scores = []
        with torch.no_grad():
            for imgs, masks in train_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                out = func(imgs)
                preds, probs = _binarize_from_logits(out)

                # one-time debug
                global _dbg_train_printed
                if not _dbg_train_printed:
                    print(f"[DBG train] imgs[min,max]=[{imgs.min().item():.3f},{imgs.max().item():.3f}] "
                          f"masks[min,max]=[{masks.min().item():.3f},{masks.max().item():.3f}] "
                          f"probs[min,max]=[{probs.min().item():.3f},{probs.max().item():.3f}] "
                          f"preds_mean={preds.mean().item():.4f} "
                          f"any_nan_out={torch.isnan(out).any().item()}")
                    _dbg_train_printed = True

                inter = (preds * masks).sum()
                union = preds.sum() + masks.sum()
                dice = (2.0 * inter / (union + 1e-6)).item()
                dice_scores.append(dice)
        mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    except Exception as e:
        print("Evaluation error:", e)
        mean_dice = 0.0
    return (mean_dice,)


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
    func = toolbox.compile(expr=individual)
    dice_scores = []
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            out = func(imgs)
            preds, probs = _binarize_from_logits(out)

            # one-time debug
            global _dbg_test_printed
            if not _dbg_test_printed:
                print(f"[DBG test] imgs[min,max]=[{imgs.min().item():.3f},{imgs.max().item():.3f}] "
                      f"masks[min,max]=[{masks.min().item():.3f},{masks.max().item():.3f}] "
                      f"probs[min,max]=[{probs.min().item():.3f},{probs.max().item():.3f}] "
                      f"preds_mean={preds.mean().item():.4f} "
                      f"any_nan_out={torch.isnan(out).any().item()}")
                _dbg_test_printed = True

            intersection = (preds * masks).sum()
            union = preds.sum() + masks.sum()
            dice = (2. * intersection / (union + 1e-6)).item()
            dice_scores.append(dice)
    return np.mean(dice_scores)


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
    visualize_predictions(toolbox, hof[0], test_dataset, num_samples=3, threshold=0.5,
                          save_dir=os.path.join(run_dir, "visualizations"))

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

