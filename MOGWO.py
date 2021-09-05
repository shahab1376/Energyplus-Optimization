import random
import numpy as np
import math
import time


class solution:
    def __init__(self):
        self.best = 0
        self.bestIndividual=[]
        self.convergence = []
        self.optimizer=""
        self.objfname=""
        self.startTime=0
        self.endTime=0
        self.executionTime=0
        self.lb=0
        self.ub=0
        self.dim=0
        self.popnum=0
        self.maxiters=0
        self.N_Outputs = 1
        self.pareto = []

# Benchmark objf
def F(x):
    dim=len(x)

    w=[i for i in range(len(x))]
    for i in range(0,dim):
        w[i]=i+1
    o = 0.1*abs(np.sum(w*(x**4))+np.random.uniform(0,1))
    u = 70*abs(np.sum(w*x))
    y = 0.6*abs(np.sum(np.sqrt(w + x**2)))
    z = 20*abs(np.sum(w*np.sin(3.14*x + w)))
    return [o,u,y,z]

lb = -1.28
ub = 1.28
dim = 30
PopulationSize = 200
Iterations= 50

def GWO(objf,lb,ub,dim,SearchAgents_no,Max_iter):
    
    # initialize alpha, beta, and delta_pos
    Alpha_pos=np.zeros(dim)
    Alpha_score=float("inf")
    
    Beta_pos=np.zeros(dim)
    Beta_score=float("inf")
    
    Delta_pos=np.zeros(dim)
    Delta_score=float("inf")

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    
    # Defining domination between two points
    def dom(x,y):
        g = []
        ge = []
        for i in range(len(x)):
            g.append(x[i] > y[i])
            ge.append(x[i] >= y[i])
        if all(ge) and any(g):
            return True
        elif (not any(ge)) and (not all(g)):
            return False
        else:
            return None
    # Defining Crowd Distance Calculator Function
    def CrowdDist(f,fmin,fmax):
        F = f.copy()
        F.insert(0,[fmin for i in range(len(f[0]))])
        F.insert(-1,[fmax for i in range(len(f[0]))])
        O = []
        for i in range(len(f)):
            Sum = 0
            for dim in range(len(f[0])):
                Sum += (F[i][dim] - F[i+2][dim]) / (fmax - fmin)
            O.append(Sum)
        return O
    
    #Initialize the positions of search agents
    Positions = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = np.random.uniform(0,1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
    
    s=solution()
    
    Out, c = objf(Positions[0])
    if  isinstance(Out , list):
        s.N_Outputs = len(Out)
        Convergence_curve= np.array([[0 for i in range(s.N_Outputs)] for j in range(Max_iter)])
    else:
        s.N_Outputs = 1
        Convergence_curve = np.zeros(Max_iter)
    
     # Loop counter
    print("GWO is optimizing  \""+objf.__name__+"\"")    
    with open('Results/Log.txt',"w", encoding="utf-8") as text_file:
        text_file.write("GWO is optimizing  \""+objf.__name__+"\" \n")
    
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    # Main loop
    for l in range(0,Max_iter):
        F = []
        for i in range(0,SearchAgents_no):
            
            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dim):
                Positions[i,j]=np.clip(Positions[i,j], lb[j], ub[j])

            # Calculate objective function for each search agent
            fitness,cost = objf(Positions[i,:])
            
            
            assert fitness is not None,'None value for fitness!'
            
            if s.N_Outputs == 1:
            
                # Update Alpha, Beta, and Delta
                if fitness<Alpha_score :
                    Alpha_score =fitness # Update alpha
                    Alpha_pos=Positions[i,:].copy()


                if (fitness>Alpha_score and fitness<Beta_score ):
                    Beta_score=fitness  # Update beta
                    Beta_pos=Positions[i,:].copy()


                if (fitness>Alpha_score and fitness>Beta_score and fitness<Delta_score): 
                    Delta_score=fitness # Update delta
                    Delta_pos=Positions[i,:].copy()
                
            else:
                
                F.append(fitness)
                Fitness = F.copy()
                
        if s.N_Outputs > 1:
            Rank = []
            loop = True
            k = 0
            while loop:
                S = [[] for i in range(len(Fitness))]
                N = [0 for i in range(len(Fitness))]
                for i in range(len(Fitness)):
                    for j in range(len(Fitness)):
                        if dom(Fitness[i],Fitness[j]):
                            N[i] += 1
                        if dom(Fitness[j],Fitness[i]):
                            S[i].append(Fitness[j])
                Rank.append([Fitness[i] for i in range(len(N)) if N[i] == 0])
                if len(Rank[0]) >= 3:
                    loop = False
                if Rank[k] == []:
                    loop = False
                for i in range(len(Rank[k])):
                    Fitness.remove(Rank[k][i])
                k+=1
            if len(Rank[0]) < 3:
                Rank.remove([])
            Pareto = []
            for i in range(len(Rank)):
                for j in range(len(Rank[i])):
                    Pareto.append(Rank[i][j])

            Alpha_score =Pareto[0] # Update alpha
            Alpha_pos=Positions[F.index(Pareto[0]),:].copy()

            Beta_score=Pareto[1]  # Update beta
            Beta_pos=Positions[F.index(Pareto[1]),:].copy()

            Delta_score=Pareto[2] # Update delta
            Delta_pos=Positions[F.index(Pareto[2]),:].copy()

            if len(Rank[0]) >= 3:
                # determine dominators by crowding distance 
                cd = CrowdDist(Rank[0],fmin = 0, fmax = 1)

                pareto = zip(cd,Rank[0])
                pareto = sorted(pareto, key=lambda x: x[1])

                Pareto = [v for (k, v) in iter(pareto)]
                CD = [k for (k, v) in iter(pareto)]

                Alpha_score = Pareto[-1] # Update alpha
                Alpha_pos=Positions[F.index(Pareto[-1]),:].copy()

                Beta_score= Pareto[-2]  # Update beta
                Beta_pos=Positions[F.index(Pareto[-2]),:].copy()

                Delta_score= Pareto[-3] # Update delta
                Delta_pos=Positions[F.index(Pareto[-3]),:].copy()

            s.pareto = Pareto
        
        a=2-l*((2)/Max_iter)  # a decreases linearly fron 2 to 0
        
        # Update the Position of search agents including omegas
        for i in range(0,SearchAgents_no):
            for j in range (0,dim):     
                           
                r1=random.random() # r1 is a random number in [0,1]
                r2=random.random() # r2 is a random number in [0,1]
                
                A1=2*a*r1-a  # Equation (3.3)
                C1=2*r2  # Equation (3.4)
                
                D_alpha=abs(C1*Alpha_pos[j]-Positions[i,j])  # Equation (3.5)-part 1
                X1=Alpha_pos[j]-A1*D_alpha  # Equation (3.6)-part 1
                           
                r1=random.random()
                r2=random.random()
                
                A2=2*a*r1-a  # Equation (3.3)
                C2=2*r2  # Equation (3.4)
                
                D_beta=abs(C2*Beta_pos[j]-Positions[i,j])  # Equation (3.5)-part 2
                X2=Beta_pos[j]-A2*D_beta  # Equation (3.6)-part 2       
                
                r1=random.random()
                r2=random.random() 
                
                A3=2*a*r1-a  # Equation (3.3)
                C3=2*r2  # Equation (3.4)
                
                D_delta=abs(C3*Delta_pos[j]-Positions[i,j])  # Equation (3.5)-part 3
                X3=Delta_pos[j]-A3*D_delta  # Equation (3.5)-part 3             
                
                Positions[i,j]=(X1+X2+X3)/3  # Equation (3.7)
                
            
        
        if s.N_Outputs == 1:
            Convergence_curve[l]=Alpha_score
        else:
            for i in range(s.N_Outputs):
                Convergence_curve[l,i] = Alpha_score[i]
                
        if (l%1==0):
            log = 'At iteration '+ str(l+1)+ ' the best fitness is '+ str(Alpha_score)
            print(log) 
            with open('Results/Log.txt',"a", encoding="utf-8") as text_file:
                text_file.write(log + '\n')
                
    timerEnd=time.time()  
    s.best = Alpha_score
    s.bestIndividual = Alpha_score
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=Convergence_curve
    s.optimizer="GWO"
    s.objfname=objf.__name__
    s.maxiters = Max_iter
    
    print(f'Time elapsed is {int(s.executionTime/3600)} hours, {int(s.executionTime/60)} min and {s.executionTime} sec')
    with open('Results/Log.txt',"a", encoding="utf-8") as text_file:
        text_file.write(f'Time elapsed is {int(s.executionTime/3600)} hours, {int(s.executionTime/60)} min and {s.executionTime} sec')
    
    return s

s = GWO(F,lb,ub,dim,PopulationSize,Iterations)
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
def graph_results(name):
    df = pd.DataFrame(s.pareto, columns = [f'Output {i+1}' for i in range(s.N_Outputs)])
    df.to_excel('Results/'+name+'.xlsx')
    sb.pairplot(df)
    plt.savefig('Results/'+name+'.jpg')
    plt.show()
    plt.plot(np.array(range(s.maxiters)),np.sort(s.convergence))
    plt.savefig('Results/Convergence of '+name+'.jpg')
    plt.legend((f'Output {i+1}' for i in range(s.N_Outputs)))
    plt.title('Convergence Plot')
    plt.xlabel('Iterations')
    plt.ylabel('Output Value')
    df = pd.DataFrame(s.convergence, columns = [f'Output {i+1}' for i in range(s.N_Outputs)])
    df.to_csv('Results/Convergence of '+name+'.csv')
if s.N_Outputs == 1:
    plt.plot(np.array(range(Iterations)),s.convergence)
    df = pd.DataFrame(s.convergence, columns = ['Output 1'])
    df.to_excel('Results/Convergense.xlsx')
    plt.savefig('Results/Convergence.jpg')
else: 
    graph_results('F')
    
    