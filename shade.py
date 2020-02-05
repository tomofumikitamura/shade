import math
import numpy as np
import joblib
import multiprocessing
from multiprocessing import Pool
from numba import njit,vectorize,prange,jit
np.set_printoptions(precision=3)
np.set_printoptions(linewidth=200)

import sys
args = sys.argv
from importlib import import_module
problem_name = 'cec14'
func = import_module(problem_name)

def evaluate(trial,low_b,diff,func_num,dim,trial_f):
    trial_denorm = low_b + trial * diff
    for i,j in enumerate(func.cec14_test(func_num, dim, trial_denorm.reshape(-1))):
        if j<1e-8+func_num*100:
            trial_f[i]=func_num*100
            return 1
        else:
            trial_f[i]=j
    return 0

@njit
def select(trial,trial_f,psize,pop,fitness):
    for j in range(psize):
        if trial_f[j] < fitness[j]:
            fitness[j] = trial_f[j]
            pop[j] = trial[j]

@njit
def store(trial_f,fitness,pop,A,psize,archive_idx,archive,dfs):
    for j in range(psize):
        if trial_f[j]<fitness[j]:
            dfs[j]=fitness[j]-trial_f[j]
            if A>1:
                if archive_idx >= A:
                    archive[np.random.randint(A)] = pop[j]
                else:
                    archive[archive_idx] = pop[j]
            archive_idx+=1
        else:
            dfs[j]=0
    return archive_idx

@njit
def pam_update(dfs,memory_F,memory_C):
    FH = np.sum(lehmer(memory_F,dfs)) / np.sum(multiple(memory_F,dfs))
    if np.sum(multiple(memory_C,dfs))==0:
        CH = 0
    else:
        CH = np.sum(lehmer(memory_C,dfs)) / np.sum(multiple(memory_C,dfs))
    return FH, CH

@njit
def pam_generate(history_F,history_C,k,H,memory_F,memory_C):
    for i in range(memory_F.shape[0]):
        recall_idx = k - np.random.randint(H)
        mean_F = history_F[recall_idx]
        mean_C = history_C[recall_idx]
        while True:
            memory_F[i] = 0.1*np.tan(np.pi*(np.random.rand()-0.5))+mean_F
            if memory_F[i]>0: break
        if memory_F[i]>1: memory_F[i]=1
        memory_C[i] = 0.1*np.random.randn()+mean_C
        if memory_C[i]<0: memory_C[i]=0
        if memory_C[i]>1: memory_C[i]=1

@vectorize(nopython=True)
def lehmer(a,b):
    return (a**2)*b

@vectorize(nopython=True)
def multiple(a,b):
    return a*b

@njit
def generate(pop,archive,memory_F,memory_C,archive_idx,prate,psize,dim,trial):
    for j in range(psize):
        '''setting F and C'''
        F = memory_F[j]
        C = memory_C[j]

        '''Binomial crossover'''
        cross_points = np.random.rand(dim) < C
        if np.sum(cross_points)==0:
            cross_points[np.random.randint(dim)] = True
        
        '''current-to-pbest/1'''
        pbest = np.random.randint(max(int(psize*prate),1))
        d1 = pop[pbest]-pop[j]
        
        idxs = np.arange(psize)
        idxs = idxs[idxs!=j]
        r1 = idxs[np.random.randint(psize-1)]
        idxs = np.hstack((idxs[idxs!=r1],np.arange(psize,psize+archive_idx)))
        r2 = idxs[np.random.randint(psize-2+archive_idx)]
        if r2 >= psize:
            d2 = pop[r1]-archive[r2-psize]
        else:
            d2 = pop[r1]-pop[r2]
        
        mutant = pop[j] + F*d1 + F*d2
        for x in range(dim):
            if mutant[x]>1 or mutant[x]<0:
                mutant[x] = (mutant[x] + pop[j][x]) / 2
        trial[j] = np.where(cross_points, mutant, pop[j])

def de(func_num, bounds, psize, maxevals, prate=0.1, Fini=0.5, Cini=0.5, H=10, alpha=1, beta=10):
    gen, nofe = 0, 0
    initial_psize = psize
    dim = len(bounds)
    low_b, up_b = np.asarray(bounds).T
    diff = np.fabs(up_b - low_b)
    '''Initialize individuals and Evaluate'''
    pop, fitness =  np.random.rand(psize, dim), np.zeros(psize)
    trial, trial_f, dfs = np.zeros((psize, dim)), np.zeros(psize), np.zeros(psize)
    ret = evaluate(pop,low_b,diff,func_num,dim,fitness)
    nofe += psize
    '''Initialize archive'''
    arc_rate = 2
    A=psize*arc_rate
    archive = np.zeros((A,dim)); archive_idx = 0
    '''Initialize parameter adaptation arrays'''
    k = 0
    memory_F, memory_C  = np.full(psize,Fini), np.full(psize,Cini)
    history_F,history_C = np.full(H,Fini), np.full(H,Cini)
    pam_generate(history_F,history_C,k,H,memory_F,memory_C)
    while nofe<maxevals:
        gen += 1
        generate(pop,archive,memory_F,memory_C,archive_idx,prate,psize,dim,trial)
        '''Evaluation'''
        ret = evaluate(trial,low_b,diff,func_num,dim,trial_f)
        nofe += psize
        if np.min(trial_f) < fitness[0]:
            yield 1,nofe, np.min(trial_f)-func_num*100
            if ret==1: break
        '''Archive'''
        ret = store(trial_f,fitness,pop,A,psize,archive_idx,archive,dfs)
        nof_success_trials = ret - archive_idx
        if ret<=A: archive_idx = ret
        '''Parameter Adaptation'''
        if nof_success_trials>0:
            if history_C[k]==0:
                history_F[k],_ = pam_update(dfs,memory_F,memory_C)
            else:
                history_F[k],history_C[k] = pam_update(dfs,memory_F,memory_C)
            #yield 0,history_F[k],history_C[k]
            k = (k+1)%H
        pam_generate(history_F,history_C,k-alpha,beta,memory_F,memory_C)
        #pam_generate(history_F,history_C,k-1,H,memory_F,memory_C)  # SHADE
        '''Selection and Sort'''
        select(trial,trial_f,psize,pop,fitness)
        f_sorted = np.argsort(fitness)
        pop = pop[f_sorted]
        fitness = fitness[f_sorted]
        '''Populationsize reduction'''
        #psize = int(initial_psize - nofe*(initial_psize-4)/maxnofe)
        trial,trial_f,dfs = trial[:psize], trial_f[:psize], dfs[:psize]
        memory_F,memory_C = memory_F[:psize], memory_C[:psize]

def solve(args):
    func_num,evals,seed = args
    np.random.seed(seed=100*seed)
    maxevals=problem_size*evals
    bounds=[(-100,100)]*problem_size
    kind,nofe,val=[],[],[]
    for f,g,v in de(func_num=func_num, bounds=bounds, psize=pop_size, maxevals=maxevals,
                    H=H, alpha=alpha, beta=beta):
        kind.append(f)
        nofe.append(g)
        val.append(v)
    return np.array(kind),np.array(nofe),np.array(val)

paralell = 48; pop_size=100; evals=1e4
problem_list=list(range(1,31))

output=[]
problem_size=30; H=problem_size//2; alpha=1; beta=H
p = Pool(processes=multiprocessing.cpu_count())
for func_num in problem_list:
    results = p.map_async(solve, [(func_num,evals,j) for j in range(paralell)]).get(9999999)
    output.append(list(zip(*results)))
p.close()
