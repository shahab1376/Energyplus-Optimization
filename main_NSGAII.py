
from ObjFun.py import *
from MOGWO.py import graph_results
dim = 22
N_Outputs = 2
popnum = 10
maxiter = 20
ub = 1
lb = 0

# optimize using NSGAII
NFE = popnum * maxiter

from platypus.problems import Problem
from platypus.algorithms import NSGAII
from platypus.types import Real

problem = Problem(dim, N_Outputs)

problem.types[:] = Real(lb, ub)
problem.function = EplusObjf

algorithm = NSGAII(problem, population_size = popnum)
algorithm.run(NFE)

pareto = [[s.objectives[0],s.objectives[1]] for s in algorithm.result]

print(pareto)

with open('pareto.txt' ,"w", encoding="utf-8") as text_file:
    text_file.write(f'Pareto is {pareto}')

import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

graph_results('NSGAII')

