import random
from deap import tools
from collections import defaultdict

# Baseline evaluation imports
from typing import Optional, Tuple, Dict, Any
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import seggp_functions as felgp_fs
from miou_eval import _accumulate_confmat, confmat_to_iou


def pop_compare(ind1, ind2):
    # List all available primitive types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)
    for idx, node in enumerate(ind1[1:],1):
        types1[node.ret].append(idx)
    for idx, node in enumerate(ind2[1:],1):
        types2[node.ret].append(idx)
    return types1==types2

def varAnd(population, toolbox, cxpb, mutpb):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param elitpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.

    The variation goes as follow. First, the parental population
    :math:`P_\\mathrm{p}` is duplicated using the :meth:`toolbox.clone` method
    and the result is put into the offspring population :math:`P_\\mathrm{o}`.  A
    first loop over :math:`P_\\mathrm{o}` is executed to mate pairs of
    consecutive individuals. According to the crossover probability *cxpb*, the
    individuals :math:`\\mathbf{x}_i` and :math:`\\mathbf{x}_{i+1}` are mated
    using the :meth:`toolbox.mate` method. The resulting children
    :math:`\\mathbf{y}_i` and :math:`\\mathbf{y}_{i+1}` replace their respective
    parents in :math:`P_\\mathrm{o}`. A second loop over the resulting
    :math:`P_\\mathrm{o}` is executed to mutate every individual with a
    probability *mutpb*. When an individual is mutated it replaces its not
    mutated version in :math:`P_\\mathrm{o}`. The resulting :math:`P_\\mathrm{o}`
    is returned.

    This variation is named *And* beceause of its propention to apply both
    crossover and mutation on the individuals. Note that both operators are
    not applied systematicaly, the resulting individuals can be generated from
    crossover only, mutation only, crossover and mutation, and reproduction
    according to the given probabilities. Both probabilities should be in
    :math:`[0, 1]`.
    """
    offspring = [toolbox.clone(ind) for ind in population]
    new_cxpb=cxpb/(cxpb+mutpb)

    #num_cx=int(new_cxpb*len(offspring))
    #num_mu=len(offspring)-num_cx
    #print(new_cxpb, new_mutpb)
    # Apply crossover and mutation on the offspring
    i = 1
    while i < len(offspring):
        if random.random() < new_cxpb:
            if (offspring[i - 1] == offspring[i]) or pop_compare(offspring[i - 1], offspring[i]):
                offspring[i - 1], = toolbox.mutate(offspring[i - 1])
                offspring[i], = toolbox.mutate(offspring[i])
            else:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
            i = i + 2
        else:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
            i = i + 1
    return offspring


def eaSimple(population, toolbox, cxpb, mutpb, elitpb, ngen, randomseed, stats=None,
             halloffame=None, verbose=__debug__):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param etilpb: The probability of elitism
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evalutions for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            elitismNum
            offspringE=selectElitism(population,elitismNum)
            population = select(population, len(population)-elitismNum)
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            offspring=offspring+offspringE
            evaluate(offspring)
            population = offspring.

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` and :meth::`toolbox.selectElitism`,
     aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else []) # type: ignore
    # Evaluate the individuals with an invalid fitness
    # invalid_ind = [ind for ind in population if not ind.fitness.valid]
    # print(len(invalid_ind))

    for i in population:
        i.fitness.values = toolbox.evaluate(individual=i, hof=[])

    if halloffame is not None:
        halloffame.update(population)
    hof_store = tools.HallOfFame(5 * len(population))
    hof_store.update(population)
    cop_po = population
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)

    for gen in range(1, ngen + 1):

        # Select the next generation individuals by elitism
        elitismNum = int(elitpb * len(population))
        population_for_eli = [toolbox.clone(ind) for ind in population]
        offspringE = toolbox.selectElitism(population_for_eli, k=elitismNum)

        # Select the next generation individuals for crossover and mutation
        offspring = toolbox.select(population, len(population) - elitismNum)
        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        # add offspring from elitism into current offspring
        # generate the next generation individuals

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # print(len(invalid_ind))
        for i in invalid_ind:
            i.fitness.values = toolbox.evaluate(individual=i, hof=cop_po)

        offspring[0:0] = offspringE

        # Update the hall of fame with the generated
        if halloffame is not None:
            halloffame.update(offspring)
        cop_po = offspring.copy()
        hof_store.update(offspring)
        for i in hof_store:
            cop_po.append(i)
        population[:] = offspring
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        # print(record)
        logbook.record(gen=gen, nevals=len(offspring), **record)
        # print(record)
        if verbose:
            print(logbook.stream)
    return population, logbook


# ===== Baseline: evaluate pretrained_seg_nn without GP =====
@torch.no_grad()
def eval_pretrained_on_loader(
    data_loader,
    num_classes: int,
    device: Optional[str] = None,
    ignore_index: Optional[int] = 255,
    max_batches: Optional[int] = None,
    horse_class_idx: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """Evaluate DeepLabV3 baseline directly on a DataLoader.

    - For multiclass (num_classes>1): returns (per_class_ious, mIoU).
    - For binary (num_classes==1): returns (np.array([dice]), dice).

    Args:
      data_loader: yields (images, masks). Images (B,C,H,W), masks (B,1,H,W) long for multiclass or float for binary.
      num_classes: number of classes including background (use 21 for VOC).
      device: 'cuda' or 'cpu'. Defaults to CUDA if available.
      ignore_index: label to ignore for multiclass (e.g., 255 for VOC).
      max_batches: optional limit for quick tests.
      horse_class_idx: optional override for VOC horse class index. Defaults to internal felgp_fs._HORSE_IDX.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if num_classes > 1:
        conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    else:
        total_inter = 0.0
        total_sum = 0.0

    processed = 0
    horse_idx = felgp_fs._HORSE_IDX if horse_class_idx is None else int(horse_class_idx)

    for imgs, masks in data_loader:
        imgs = imgs.to(device)
        # masks shape: (B,1,H,W); long for multiclass, float for binary
        # Use original dtype for comparisons below
        with torch.no_grad():
            logits = felgp_fs.pretrained_seg_nn(imgs)  # (B,21,H,W)

        if num_classes > 1:
            pred_ids = logits.argmax(dim=1)  # (B,H,W)
            # target ids: squeeze channel if present
            targ_ids = masks
            if targ_ids.dim() == 4 and targ_ids.shape[1] == 1:
                targ_ids = targ_ids.squeeze(1)
            targ_ids = targ_ids.to(torch.long)

            # Flatten and accumulate confusion matrix with ignore handling
            p_np = pred_ids.view(-1).cpu().numpy().astype(np.int64)
            t_np = targ_ids.view(-1).cpu().numpy().astype(np.int64)
            conf = _accumulate_confmat(conf, p_np, t_np, num_classes=num_classes, ignore_index=ignore_index)
        else:
            # Binary: derive horse mask from argmax; compute dataset-level Dice
            pred_ids = logits.argmax(dim=1)  # (B,H,W)
            pred_bin = (pred_ids == horse_idx).to(torch.float32)
            # target expected (B,1,H,W) float {0,1}
            targ_bin = masks
            if targ_bin.dim() == 4 and targ_bin.shape[1] == 1:
                targ_bin = targ_bin.squeeze(1)
            targ_bin = targ_bin.to(torch.float32)

            inter = (pred_bin * targ_bin).sum().item() * 2.0
            summ = (pred_bin + targ_bin).sum().item() + 1e-8
            total_inter += inter
            total_sum += summ

        processed += 1
        if max_batches is not None and processed >= max_batches:
            break

    if num_classes > 1:
        ious, miou = confmat_to_iou(conf, exclude_background=(num_classes > 1))
        return ious, miou
    else:
        if total_sum > 0.0:
            dice = float(total_inter) / float(total_sum)
        else:
            dice = 0.0
        return np.array([dice], dtype=float), dice


def run_pretrained_baseline(
    train_loader,
    test_loader,
    num_classes: int,
    device: Optional[str] = None,
    ignore_index: Optional[int] = 255,
    max_batches: Optional[int] = None,
    horse_class_idx: Optional[int] = None,
    # Saving/output integration params
    run_dir: Optional[str] = None,
    dataset_name: Optional[str] = None,
    randomSeeds: Optional[int] = None,
    params: Optional[Dict[str, Any]] = None,
    args: Optional[Any] = None,
    meta: Optional[Dict[str, Any]] = None,
    copy_slurm_logs: bool = True,
    save_visuals: bool = True,
    vis_count: int = 6,
) -> Dict[str, Any]:
    """Run baseline DeepLabV3 evaluation on train and test loaders.

    Also saves outputs similar to GP runs when `run_dir` is provided:
    - Writes a `Summary_on{Dataset}.txt` with metrics
    - Saves a few prediction visualizations under `visualizations/`
    - Copies Slurm logs into the run dir if `meta` is provided and `copy_slurm_logs=True`
    """
    train_scores = eval_pretrained_on_loader(train_loader, num_classes, device, ignore_index, max_batches, horse_class_idx)
    test_scores  = eval_pretrained_on_loader(test_loader,  num_classes, device, ignore_index, max_batches, horse_class_idx)

    train_per, train_mean = train_scores
    test_per,  test_mean  = test_scores
    metric_name = "mIoU" if num_classes > 1 else "Dice"
    print(f"[Baseline pretrained_seg_nn] Train {metric_name}: {train_mean:.4f} | Test {metric_name}: {test_mean:.4f}")

    # Save artifacts if run_dir provided
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        vis_dir = os.path.join(run_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # 1) Summary file similar naming
        ds_name = dataset_name or params.get("Dataset", "dataset") if params else (dataset_name or "dataset")
        summary_path = os.path.join(run_dir, f"Summary_on{ds_name}.txt")
        with open(summary_path, "w") as f:
            f.write(f"Mode: NN Only (pretrained_seg_nn)\n")
            if params:
                f.write(f"RUN_MODE: {params.get('RUN_MODE','n/a')}\n")
                f.write(f"color_mode: {params.get('color_mode','n/a')}\n")
                f.write(f"num_classes: {params.get('num_classes','n/a')}\n")
            if randomSeeds is not None:
                f.write(f"randomSeeds: {randomSeeds}\n")
            f.write(f"Metric: {metric_name}\n")
            f.write(f"Train {metric_name}: {train_mean:.6f}\n")
            f.write(f"Test  {metric_name}: {test_mean:.6f}\n")
            if num_classes > 1:
                f.write("Per-class IoU (train):\n")
                f.write(", ".join([f"{v:.4f}" if np.isfinite(v) else "nan" for v in train_per.tolist()]) + "\n")
                f.write("Per-class IoU (test):\n")
                f.write(", ".join([f"{v:.4f}" if np.isfinite(v) else "nan" for v in test_per.tolist()]) + "\n")

        # 2) Save a few visualization images from test set
        if save_visuals:
            saved = 0
            with torch.no_grad():
                for imgs, masks in test_loader:
                    imgs = imgs.to(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
                    logits = felgp_fs.pretrained_seg_nn(imgs)
                    if num_classes > 1:
                        preds = logits.argmax(dim=1, keepdim=True).float()
                        cmap = plt.get_cmap('tab20', max(2, num_classes))
                    else:
                        pred_ids = logits.argmax(dim=1, keepdim=True)
                        preds = (pred_ids == (felgp_fs._HORSE_IDX if horse_class_idx is None else horse_class_idx)).float()
                        cmap = 'gray'

                    # Loop samples in this batch
                    B = imgs.shape[0]
                    for b in range(B):
                        if saved >= vis_count:
                            break
                        img_t = imgs[b].detach().cpu()
                        mask_t = masks[b].detach().cpu()
                        pred_t = preds[b].detach().cpu()

                        # Prepare arrays for display
                        img_np = img_t.permute(1,2,0).numpy() if img_t.shape[0] in (1,3,4) else img_t[0].numpy()
                        if img_np.shape[-1] == 1:
                            img_np = img_np.squeeze(-1)
                        if mask_t.dim() == 3 and mask_t.shape[0] == 1:
                            mask_np = mask_t.squeeze(0).numpy()
                        else:
                            mask_np = mask_t.numpy()
                        pred_np = pred_t.squeeze(0).numpy()

                        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
                        axs[0].imshow(img_np, cmap='gray' if img_np.ndim==2 else None)
                        axs[0].set_title("Input")
                        if num_classes > 1:
                            axs[1].imshow(pred_np, cmap=cmap, vmin=0, vmax=max(1, num_classes-1))
                            axs[2].imshow(mask_np, cmap=cmap, vmin=0, vmax=max(1, num_classes-1))
                        else:
                            axs[1].imshow(pred_np, cmap='gray')
                            axs[2].imshow(mask_np, cmap='gray')
                        axs[1].set_title("Prediction")
                        axs[2].set_title("Ground Truth")
                        for ax in axs:
                            ax.axis('off')
                        fig.tight_layout()
                        out_path = os.path.join(vis_dir, f"baseline_vis_{saved:02d}.png")
                        fig.savefig(out_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        saved += 1
                    if saved >= vis_count:
                        break

        # 3) Copy Slurm logs into run dir if requested
        try:
            if copy_slurm_logs and meta is not None:
                from data_handling import _copy_slurm_logs  # local import to avoid cycles
                _copy_slurm_logs(run_dir, meta)
        except Exception:
            pass

    return {
        "train": {"per_class": train_per, "mean": train_mean},
        "test":  {"per_class": test_per,  "mean": test_mean},
        "metric": metric_name,
        "run_dir": run_dir,
    }

