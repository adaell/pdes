#
# Solves the classic 1D Stefan problem using a monte carlo that simulates 
# Brownian motion and compares the output to the well known Neumann solution, 
# i.e. solves the problem:
#
#                                        2
#                 d                     d
#                 -- (T(x, t)) = alpha (--- (T(x, t)))
#                 dt                      2
#                                       dx
#
#
#                       d               d            |
#                L rho (-- (s(t))) = k (-- (T(x, t)))|
#                       dt              dx           |x=s(t)
#
#               T(0,t) = 1
#
#               s(0) =  0
#
#               T(x,0) = 0
#
# where T(x,t) is the temperature at (x,t), s(t) is the location of the moving
# boundary, L is the latent heat, rho is the density, k is the thermal 
# conductivity and alpha is the thermal diffusivity.
#
# The moving boundary is modelled using the method proposed by Daniel Stoor[1].
#
# The monte carlo solution is then compared to the Neumann solution.
#
# [1] https://www.diva-portal.org/smash/get/diva2:1325632/FULLTEXT02
#

import numpy as np
import math
import dask.array as da

#NUM_RAND=1000000
NUM_RAND=500000
class rand_stack():
    def __init__(self):
        self.reset_counter=0
        self.reset()
    #@profile
    def reset(self):
        self.R=np.random.randint(2,size=NUM_RAND)
        self.counter=0
        self.reset_counter+=1
    def pop(self,n):
        if self.counter + n > NUM_RAND:
            self.reset()
        self.counter+=n
        return self.R[self.counter-n:self.counter]

#np.random.seed(1)
#@profile
def mc():
    l=334e3
    rho=1e-6
    K=0.6e-3
    c=4.2e3
    alpha=K/(rho*c)
    beta=c/l
    L=2 # domain length
    t_max=1 #i.e. endtime
    #t_max=5
    T_0=1
    
    n=10000 #number of iterations
    dx=0.005
    dt=(dx*dx)/(2*alpha)
    ds=dx/(n*beta)
    N_x=math.ceil(L/dx)+1
    N_t=math.ceil(t_max/dt)+1
    T=np.zeros((N_x,N_t),dtype=int)
    s_vector=np.zeros((N_t,1),dtype=float)
    t_j=0
    s_i=0
    s=dx
    
    obstime=0.1
    obscounter=0
    obs=[]
    
    RR=rand_stack()  
    
    # t = 0
    T[0,0] = T_0*n # boundary
    s_vector[0] = s
    for x_i in range(0,s_i+1):
            R_vec = RR.pop(T[x_i,t_j])
            for k in range(0,T[x_i,t_j],1): 
                # if not on boundaries
                if x_i+R_vec[k] > 0 and x_i+R_vec[k] <= s_i:
                    T[x_i+R_vec[k],t_j+1] += 1
                elif x_i+R_vec[k] == s_i+1:
                    s += ds
                    s_i=int(s/dx)
    t_j=t_j+1
    
    while t_j < (N_t-1) and s_i < N_x:
        T[0,t_j] = T_0*n # dirichlet boundary
        s_vector[t_j] = s
        
        # x == 0
        R_vec = RR.pop(T[0,t_j])
        # Move all agents at position (x_i)
        num_r = np.sum(R_vec)
        T[1,t_j+1] += num_r
        
        # x == s_i
        R_vec = RR.pop(T[s_i,t_j])
        for k in range(0,T[s_i,t_j],1): 
            if s_i+R_vec[k] <= s_i:
                T[s_i+R_vec[k],t_j+1] += 1
            elif s_i+R_vec[k] == s_i+1:
                s += ds
                s_i=int(s/dx)
        
        # x > 0 and x < s_i
        for x_i in range(1,s_i):
            #R_vec = np.random.randint(2,size=T[x_i,t_j])
            R_vec = RR.pop(T[x_i,t_j])
            # Move all agents at position (x_i)
            num_r = np.sum(R_vec)
            T[x_i+1,t_j+1] += num_r
            T[x_i-1,t_j+1] += T[x_i,t_j] - num_r
            
            
        t_j=t_j+1
        
        # Save observation data
        if t_j*dt >= obscounter:
            obs.append((t_j*dt,s))
            #print("t=%s\ts=%s"%(t_j*dt,s))
            obscounter+=obstime
        
    T=T/n
    return obs

# Used to solve the Neumann equation 
def f(x,beta,T_0):
    return math.sqrt(math.pi)*beta*x*math.exp(x*x)*math.erf(x)-T_0


def df(x,beta,T_0):
    return beta*(math.sqrt(math.pi)*math.exp(x*x)*math.erf(x)*(2*x*x+1)+2*x*x)


def solve_neumann_lambda(beta, T_0):
    tol=1e-6
    x0=1
    while abs(f(x0,beta,T_0)) > tol:
        x0=x0-f(x0,beta,T_0)/df(x0,beta,T_0) #Newton-Raphson
    return x0

def neumann():
    l=334e3
    rho=1e-6
    K=0.6e-3
    c=4.2e3
    alpha=K/(rho*c)
    beta=c/l
    L=2 # domain length
    t_max=1 #i.e. endtime
    #t_max=5
    T_0=1
    n=1e3 #number of iterations
    dx=0.01
    dt=(dx*dx)/(2*alpha)
    
    obstime=0.1
    obscounter=0
    obs=[]
    
    lamb=solve_neumann_lambda(beta,T_0)
    n=math.ceil(t_max/obstime)+1
    for t_i in range(0,n):
        t=t_i*obstime
        analytic=dx+2*lamb*math.sqrt(alpha*t)
        obs.append((t,analytic))
        #print("%s\t%s" % (t,analytic))
    
    return obs

def print_results(n,m):
    print("%s\t%s\t%s" % ("Time","Neumann s(t)","Monte Carlo s(t)"))
    for i in range(0,len(n)):
        ntup=n[i]
        mtup=m[i]
        print("%s\t%s\t\t%s" % (round(ntup[0],4),round(ntup[1],4),round(mtup[1],4)))

n=neumann()
m=mc()
print_results(n,m)








