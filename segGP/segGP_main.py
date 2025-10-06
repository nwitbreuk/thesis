#python packages
import operator
import random
import time
import multiprocessing
import gp_restrict as gp_restrict
import algo_iegp as evalGP
import numpy as np
from deap import base, creator, tools, gp
import seggp_functions as felgp_fs
from typing import Any
from strongGPDataType import Int1, Int2, Int3, Int4, Int5, Int6
from strongGPDataType import Float1, Float2, Float3
from strongGPDataType import Array1, Array2, Array3, Array4, Array5, Array6
# defined by author
import saveFile
import sys

toolbox: base.Toolbox  # type: ignore
creator.FitnessMax: Any  # type: ignore
creator.Individual: Any  # type: ignore

randomSeeds = 12
dataSetName = 'f1'


x_train = np.load(dataSetName+'_train_data.npy')/255.0
y_train = np.load(dataSetName+'_train_label.npy')
x_test = np.load(dataSetName+'_test_data.npy')/255.0
y_test = np.load(dataSetName+'_test_label.npy')

def stratified_sample(x, y, n_samples):
    """Randomly select n_samples, ensuring both classes are present."""
    classes = np.unique(y)
    idxs = []
    # Ensure at least one sample from each class
    for c in classes:
        class_idxs = np.where(y == c)[0]
        idxs.append(np.random.choice(class_idxs, 1, replace=False)[0])
    # Fill the rest randomly
    remaining = list(set(range(len(y))) - set(idxs))
    if n_samples > len(idxs):
        idxs += list(np.random.choice(remaining, n_samples - len(idxs), replace=False))
    np.random.shuffle(idxs)
    return x[idxs], y[idxs]

x_train, y_train = stratified_sample(x_train, y_train, 50)
x_test, y_test = stratified_sample(x_test, y_test, 20)

print(x_train.shape,y_train.shape, x_test.shape,y_test.shape)
print(x_train.max())
#parameters:
num_train = x_train.shape[0]
pop_size=50
generation=30
cxProb=0.8
mutProb=0.19
elitismProb=0.01
totalRuns = 1
initialMinDepth=2
initialMaxDepth=8
maxDepth=8

##GP
pset = gp.PrimitiveSetTyped('MAIN', [Array1, Array2], Array6, prefix = 'Image')
# Combination, use 'Combine' for increasing the depth of the GP tree
pset.addPrimitive(felgp_fs.combine, [Array6, Array6, Array6], Array6, name='Combine')
pset.addPrimitive(felgp_fs.combine, [Array5, Array5, Array5], Array6, name='Combine3')
pset.addPrimitive(felgp_fs.combine, [Array5, Array5, Array5, Array5, Array5], Array6, name='Combine5')
pset.addPrimitive(felgp_fs.combine, [Array5, Array5, Array5, Array5, Array5, Array5, Array5], Array6, name='Combine7')
#Classification
pset.addPrimitive(felgp_fs.linear_svm, [Array4, Array2, Int4], Array5, name='SVM')
pset.addPrimitive(felgp_fs.lr, [Array4, Array2, Int4], Array5, name='LR')
pset.addPrimitive(felgp_fs.randomforest, [Array4, Array2, Int5, Int6], Array5, name='RF')
pset.addPrimitive(felgp_fs.erandomforest, [Array4, Array2, Int5, Int6], Array5, name='ERF')
###Feature Concatenation
pset.addPrimitive(felgp_fs.FeaCon2, [Array4, Array4], Array4, name ='FeaCon')
pset.addPrimitive(felgp_fs.FeaCon2, [Array3, Array3], Array4, name ='FeaCon2')
pset.addPrimitive(felgp_fs.FeaCon3, [Array3, Array3, Array3], Array4, name ='FeaCon3')
pset.addPrimitive(felgp_fs.FeaCon4, [Array3, Array3, Array3, Array3], Array4, name ='FeaCon4')
#Feature Extraction
pset.addPrimitive(felgp_fs.global_hog_small, [Array1], Array3, name = 'F_HOG')
pset.addPrimitive(felgp_fs.all_lbp, [Array1], Array3, name = 'F_uLBP')
pset.addPrimitive(felgp_fs.all_sift, [Array1], Array3, name = 'F_SIFT')
##Filtering and Pooling
pset.addPrimitive(felgp_fs.maxP, [Array1, Int3, Int3], Array1,name='MaxP')
pset.addPrimitive(felgp_fs.gau, [Array1, Int1], Array1, name='Gau')
pset.addPrimitive(felgp_fs.gauD, [Array1, Int1, Int2, Int2], Array1, name='GauD')
pset.addPrimitive(felgp_fs.gab, [Array1, Float1, Float2], Array1, name='Gabor')
pset.addPrimitive(felgp_fs.laplace, [Array1], Array1, name='Lap')
pset.addPrimitive(felgp_fs.gaussian_Laplace1, [Array1], Array1, name='LoG1')
pset.addPrimitive(felgp_fs.gaussian_Laplace2, [Array1], Array1, name='LoG2')
pset.addPrimitive(felgp_fs.sobelxy, [Array1], Array1, name='Sobel')
pset.addPrimitive(felgp_fs.sobelx, [Array1], Array1, name='SobelX')
pset.addPrimitive(felgp_fs.sobely, [Array1], Array1, name='SobelY')
pset.addPrimitive(felgp_fs.lbp, [Array1], Array1, name='LBP')
pset.addPrimitive(felgp_fs.hog_feature, [Array1], Array1, name='HoG')
pset.addPrimitive(felgp_fs.medianf, [Array1], Array1,name='Med')
pset.addPrimitive(felgp_fs.maxf, [Array1], Array1,name='Max')
pset.addPrimitive(felgp_fs.minf, [Array1], Array1,name='Min')
pset.addPrimitive(felgp_fs.meanf, [Array1], Array1,name='Mean')
pset.addPrimitive(felgp_fs.sqrt, [Array1], Array1, name='Sqrt')
pset.addPrimitive(felgp_fs.mixconadd, [Array1, Float3, Array1, Float3], Array1, name='W_Add')
pset.addPrimitive(felgp_fs.mixconsub, [Array1, Float3, Array1, Float3], Array1, name='W_Sub')
pset.addPrimitive(felgp_fs.relu, [Array1], Array1, name='Relu')
#Terminals
pset.renameArguments(ARG0='grey')
def rand_sigma():
    return random.randint(1, 4)

def rand_order():
    return random.randint(0, 3)

def rand_theta():
    return random.randint(0, 8)

def rand_frequency():
    return random.randint(0, 5)

def rand_n():
    return round(random.random(), 3)

def rand_kernel_size():
    return random.randrange(2, 5, 2)

def rand_c():
    return random.randint(-2, 5)

def rand_num_tree():
    return random.randrange(50, 501, 10)

def rand_tree_depth():
    return random.randrange(10, 101, 10)

pset.addEphemeralConstant('Singma', rand_sigma, Int1)
pset.addEphemeralConstant('Order', rand_order, Int2)
pset.addEphemeralConstant('Theta', rand_theta, Float1)
pset.addEphemeralConstant('Frequency', rand_frequency, Float2)
pset.addEphemeralConstant('n', rand_n, Float3)
pset.addEphemeralConstant('KernelSize', rand_kernel_size, Int3)
pset.addEphemeralConstant('C', rand_c, Int4)
pset.addEphemeralConstant('num_Tree', rand_num_tree, Int5)
pset.addEphemeralConstant('tree_Depth', rand_tree_depth, Int6)
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


def evalTrain(toolbox, individual, hof, trainData, trainLabel):
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
    if len(hof) != 0 and individual in hof:
        ind = 0
        while ind < len(hof):
            if individual == hof[ind]:
                accuracy, = hof[ind].fitness.values
                ind = len(hof)
            else: ind+=1
    else:
        try:
            func = toolbox.compile(expr=individual)
            output = np.asarray(func(trainData, trainLabel))
            y_predict  = np.argmax(output, axis=1)
            accuracy = 100*np.sum(y_predict == trainLabel) / len(trainLabel)
        except:
            accuracy=0
    return accuracy,


toolbox.register("evaluate", evalTrain,toolbox, trainData=x_train,trainLabel=y_train)
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

def evalTest(toolbox, individual, trainData, trainLabel, test, testL):
    """
    Evaluate the best individual on the test set.
    Concatenates train and test data, compiles and executes the individual,
    and computes classification accuracy on the test set.

    Args:
        toolbox: DEAP toolbox with compile method.
        individual: The GP individual to evaluate.
        trainData: Training data (features).
        trainLabel: Training labels.
        test: Test data (features).
        testL: Test labels.

    Returns:
        accuracy: Classification accuracy on the test set (percentage).
    """
    x_train = np.concatenate((trainData, test), axis=0)
    func = toolbox.compile(expr=individual)
    output = np.asarray(func(x_train, trainLabel))
    print(output.shape)
    y_predict = np.argmax(output, axis=1)
    accuracy = 100*np.sum(y_predict==testL)/len(testL)
    return accuracy

if __name__ == "__main__":
    beginTime = time.process_time()
    pop, log, hof = GPMain(randomSeeds)
    endTime = time.process_time()
    trainTime = endTime - beginTime

    testResults = evalTest(toolbox, hof[0], x_train, y_train,x_test, y_test)
    saveFile.saveAllResults(randomSeeds, dataSetName, hof, trainTime, testResults, log)

    testTime = time.process_time() - endTime
    print('testResults ', testResults)

