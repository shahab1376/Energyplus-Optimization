
from ObjFun.py import *
dim = 22
N_Outputs = 2
popnum = 10
maxiter = 20
ub = 1
lb = 0

# optimze using MOGWO
from MOGWO.py import *

s = GWO(EplusObjf,lb,ub,dim,popnum,maxiter)

import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

graph_results('MOGWO')