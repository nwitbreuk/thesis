import operator
import argparse
import random
import traceback
import torch
from torch.utils.data import random_split, DataLoader
import os
import torch.nn.functional as F
import numpy as np
import time
import multiprocessing
from weizmann_loader import WeizmannHorseDataset
import gp_restrict as gp_restrict
import algo_iegp as evalGP
import numpy as np
from deap import base, creator, tools, gp
import seggp_functions as felgp_fs
from typing import Any
from visualize import visualize_predictions
# defined by author
import data_handling as data_handling

toolbox: base.Toolbox  # type: ignore
creator.FitnessMax: Any  # type: ignore
creator.Individual: Any  # type: ignore
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataSetName = 'weizmann_horse'
image_dir = "data/weizmann_horse/horse"
mask_dir = "data/weizmann_horse/mask"
dataset = WeizmannHorseDataset(image_dir, mask_dir)
RUN_OUTDIR="/dataB1/niels_witbreuk/logs/myruns"

RUN_MODE = "fast"  # "fast", "middle", "normal"
randomSeeds = 12
RUN_NAME=f"{dataSetName}_{RUN_MODE}mode_seed{randomSeeds}_{{jid}}_structure-test"
# _make_run_dir will replace {jid} with SLURM_JOB_ID (or process id if not running under SLURM)


def pad_collate(batch):
    """
    Collate function to pad images and masks in a batch to the same size.
        Args:
    batch: List of tuples (image, mask).
        Returns:
    Padded images and masks as tensors.
    """
    # batch: List[Tuple[tensor(C,H,W), tensor(C,H,W)]]
    imgs, masks = zip(*batch)
    max_h = max(t.shape[1] for t in imgs)
    max_w = max(t.shape[2] for t in imgs)
    pad_imgs, pad_masks = [], []
    for im, ms in zip(imgs, masks):
        dh = max_h - im.shape[1]
        dw = max_w - im.shape[2]
        pad = (0, dw, 0, dh)  # (left, right, top, bottom)
        pad_imgs.append(F.pad(im, pad, value=0.0))
        pad_masks.append(F.pad(ms, pad, value=0.0))
    return torch.stack(pad_imgs, 0), torch.stack(pad_masks, 0)



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
        "test_batch_size": test_loader.batch_size
    }

##GP
pset = gp.PrimitiveSetTyped('MAIN', [torch.Tensor], torch.Tensor, prefix='Image')

# Add arithmetic
pset.addPrimitive(felgp_fs.add, [torch.Tensor, torch.Tensor], torch.Tensor, name="Add")
pset.addPrimitive(felgp_fs.sub, [torch.Tensor, torch.Tensor], torch.Tensor, name="Sub")
pset.addPrimitive(felgp_fs.mul, [torch.Tensor, torch.Tensor], torch.Tensor, name="Mul")
pset.addPrimitive(felgp_fs.safe_div, [torch.Tensor, torch.Tensor], torch.Tensor, name="SafeDiv")

# Add comparison and normalization
pset.addPrimitive(felgp_fs.gt, [torch.Tensor, float], torch.Tensor, name="Gt")
pset.addPrimitive(felgp_fs.lt, [torch.Tensor, float], torch.Tensor, name="Lt")
pset.addPrimitive(felgp_fs.normalize, [torch.Tensor], torch.Tensor, name="Normalize")

#Feature Extraction
#pset.addPrimitive(felgp_fs.global_hog_small, [Array1], Array3, name = 'F_HOG')
#pset.addPrimitive(felgp_fs.all_lbp, [Array1], Array3, name = 'F_uLBP')
#pset.addPrimitive(felgp_fs.all_sift, [Array1], Array3, name = 'F_SIFT')

# Add filters and logicals
pset.addPrimitive(felgp_fs.sobel_x, [torch.Tensor], torch.Tensor, name="SobelX")
pset.addPrimitive(felgp_fs.sobel_y, [torch.Tensor], torch.Tensor, name="SobelY")
pset.addPrimitive(felgp_fs.laplacian, [torch.Tensor], torch.Tensor, name="Laplacian")
pset.addPrimitive(felgp_fs.gradient_magnitude, [torch.Tensor], torch.Tensor, name="GradientMagnitude")


# Morphological and other image processing functions
pset.addPrimitive(felgp_fs.mix, [torch.Tensor, torch.Tensor, float], torch.Tensor, name="Mix")
pset.addPrimitive(felgp_fs.if_then_else, [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, name="IfElse")
pset.addPrimitive(felgp_fs.gaussian_blur_param, [torch.Tensor, float], torch.Tensor, name="Gauss")

# Pretrained segmentation NN
pset.addPrimitive(felgp_fs.pretrained_seg_nn, [torch.Tensor], torch.Tensor, name="PretrainedSeg")


#Terminals
# Float thresholds for Gt/Lt (use a named function for multiprocessing pickling)
def rand_thresh():
    return float(np.random.uniform(0.05, 0.95))
pset.addEphemeralConstant('Thresh', rand_thresh, float)
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

    data_handling._copy_slurm_logs(run_dir, meta)
    print("testResults", testResults)


    data_handling.saveAllResults(params, hof, trainTime, testResults, log, outdir=run_dir)

