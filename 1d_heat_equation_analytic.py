# Calculates the analytical solution to the heat equation
#
#             d                 d
#             -- (u(x, t)) = D (-- (u(x, t))) ,
#             dt                dx
#
# on a finite domain 0 < x < L, where D is the heat constant,
# with boundary conditions
#
#             d            |
#          a (-- (u(x, t)))| + b u(0, t) = c , and
#             dx           |
#                          |x=0
#
#             d            |
#          d (-- (u(x, t)))| + e u(L, t) = f,
#             dx           |
#                          |x=L
#
# where a,b,c,d,e,f are real constants, and initial condition 
# u(x,0) = u0(x). 
#
# Assumes that a, b, d, e are non-zero. Also assumes that the 
# user has input reasonable values for these constants.


import math
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit()
def get_alpha(a,b,c,d,e,f,L):
    numer=L*b*f-a*f+c*d
    denom=b*(L*L*e+L*d)-L*a*e
    return numer/denom

@njit()
def get_beta(a,b,c,d,e,f,L):
    numer=c*(L*e+d)-a*f
    denom=b*(L*L*e+L*d)-L*a*e
    return numer/denom

@njit()
def f_lambda(a,b,c,d,e,f,L,lambd):
    term0=(a*d*lambd*lambd-b*e)*math.sin(lambd*L)
    term1=(a*e*lambd-b*d*lambd)*math.cos(lambd*L)
    return term0 + term1

@njit()
def df_lambda(a,b,c,d,e,f,L,lambd):
    term0=-L*(a*e*lambd-b*d*lambd)*math.sin(lambd*L)
    term1=2*a*d*lambd*math.sin(lambd*L)
    term2=L*(a*d*lambd-b*e)*math.cos(lambd*L)
    term3=(a*e-b*d)*math.cos(lambd*L)
    return term0+term1+term2+term3

@njit()
def get_lambda_raphson(a,b,c,d,e,f,L,lamdb0,TOL):
    ERROR=1000
    MAX_ITERATIONS=1000
    ITERATION_COUNTER=0
    lambd=lamdb0
    while ERROR > TOL:
        f=f_lambda(a,b,c,d,e,f,L,lambd)
        df=df_lambda(a,b,c,d,e,f,L,lambd)
        lambd1=lambd-f/df
        ERROR=abs(f_lambda(a,b,c,d,e,f,L,lambd1))
        lambd=lambd1
        ITERATION_COUNTER+=1
        if ITERATION_COUNTER > MAX_ITERATIONS:
            return lamdb0
    return lambd

@njit()
def get_lambda_dichotomy(a,b,c,d,e,f,L,left,right,TOL):
    ERROR=1000
    MAX_ITERATIONS=10000
    ITERATION_COUNTER=0
    sgnl=sign(f_lambda(a,b,c,d,e,f,L,left))
    sgnr=sign(f_lambda(a,b,c,d,e,f,L,right))
    while ITERATION_COUNTER < MAX_ITERATIONS:
        ITERATION_COUNTER+=1
        middle=0.5*(left+right)
        ff=f_lambda(a,b,c,d,e,f,L,middle)
        if abs(ff)<TOL:
            return middle
        sgnm=sign(ff)
        if sgnm==sgnr:
            right=middle
        else:
            left=middle
    print("Eigenvalue solver did not converge")
    return None

@njit()
def sign(x):
    if abs(x) < 1e-8:
        return 0.0
    elif x > 0:
        return 1.0
    else:
        return -1.0

# Returns an array with the first N eigenvalues
@njit()
def get_lambda_array(a,b,c,d,e,f,L,lambd0,N,TOL):
    lambda_array=np.zeros((N+1,),dtype=np.float64)
    MAX_ITERATIONS=1000
    ITERATION_COUNTER=0
    if a*d*L < TOL:
        STEP=0.5*math.pi/L
    else:
        STEP=math.sqrt((b*e*L-e*a+b*d)/(L*a*d))
    STEP=0.25*STEP
    lambd=lambd0   
    ERROR=abs(f_lambda(a,b,c,d,e,f,L,lambd))
    counter=0
    lambda_array[0]=0
    guess = lambda_array[0]+STEP
    sgnl=sign(f_lambda(a,b,c,d,e,f,L,guess))
    sgnr=sgnl
    while counter < N:
        while sgnl == sgnr:
            guess = guess + STEP
            sgnr=sign(f_lambda(a,b,c,d,e,f,L,guess))
        if sgnr == 0:
            guess=guess+0.5*STEP
        new_lambda=get_lambda_dichotomy(a,b,c,d,e,f,L,guess-STEP,guess,TOL)
        counter+=1
        lambda_array[counter]=new_lambda
        guess=guess+STEP
        sgnl=sign(f_lambda(a,b,c,d,e,f,L,guess))
        sgnr=sign(f_lambda(a,b,c,d,e,f,L,guess+STEP))
    return lambda_array

@njit()
def h(alpha,beta,L,x):
    return u0(x)-alpha*x-beta*(L-x)

@njit()
def get_A(alpha,beta,L,lambd,n):
    NUM_SLICES=1000
    if n == 0:
        return h(alpha,beta,L,0)
    else:
        dx=L/NUM_SLICES
        x0=0
        temp1=n*math.pi/L
        temp2=temp1/lambd
        f_old=h(alpha,beta,L,0)
        summation=0
        while x0 < L:
            x1=x0+dx
            f_new=h(alpha,beta,L,temp2*x1)*math.cos(temp1*x1)
            summation=summation+(f_old+f_new)*dx/2
            f_old=f_new
            x0=x1
        return (2/L)*summation

@njit()
def get_B(alpha,beta,L,lambd,n):
    if n == 0:
        return 0.0
    NUM_SLICES=1000
    dx=L/NUM_SLICES
    x0=0
    temp1=n*math.pi/L
    temp2=temp1/lambd
    f_old=0
    summation=0
    while x0 < L:
        x1=x0+dx
        f_new=h(alpha,beta,L,temp2*x1)*math.sin(temp1*x1)
        summation=summation+(f_old+f_new)*dx/2
        f_old=f_new
        x0=x1
    return (2/L)*summation

@njit()
def get_A_array(alpha,beta,L,lambda_array):
    A_array=np.zeros((len(lambda_array)+1,),dtype=np.float64)
    for n in range(0,len(lambda_array)):
        lambd=lambda_array[n]
        A_array[n]=get_A(alpha,beta,L,lambd,n)
    return A_array

@njit()
def get_B_array(alpha,beta,L,lambda_array):
    B_array=np.zeros((len(lambda_array)+1,),dtype=np.float64)
    for n in range(0,len(lambda_array)):
        lambd=lambda_array[n]
        B_array[n]=get_B(alpha,beta,L,lambd,n)
    return B_array

@njit()
def u(a,b,c,d,e,f,L,D,x,t,alpha,beta,lambda_array,A_array,B_array,NUM_TERMS):
    u=alpha*x+beta*(L-x)    
    for n in range(1,NUM_TERMS):
        lambd=lambda_array[n]
        A=A_array[n]
        B=B_array[n]
        increm=(A*math.cos(lambd*x)+B*math.sin(lambd*x))*math.exp(-lambd*lambd*D*t)
        u=u+increm
    return u

#@njit()
def sanity_check(a,b,c,d,e,f,L,D,t):
    assert(t >= 0)
    assert(a >= 0)
    assert(b >= 0)
    assert(c >= 0)
    assert(d >= 0)
    assert(e >= 0)
    assert(f >= 0)
    assert(D >= 0)
    assert(L >= 0)

#@njit()
def get_u_array(a,b,c,d,e,f,L,D,t):
    sanity_check(a,b,c,d,e,f,L,D,t)
    NUM_POINTS=100
    delta_x=L/NUM_POINTS
    x_array=np.zeros((NUM_POINTS+1,),dtype=np.float64)
    u_array=np.zeros((NUM_POINTS+1,),dtype=np.float64)
    x=0
    TOL=1e-4
    NUM_TERMS=1000
    alpha=get_alpha(a,b,c,d,e,f,L)
    beta=get_beta(a,b,c,d,e,f,L)
    lambda_array=get_lambda_array(a,b,c,d,e,f,L,0,NUM_TERMS,TOL)
    A_array=get_A_array(alpha,beta,L,lambda_array)
    B_array=get_B_array(alpha,beta,L,lambda_array)
    for i in range(0,NUM_POINTS+1):
        x_array[i]=x
        u_array[i]=u(a,b,c,d,e,f,L,D,x,t,alpha,beta,lambda_array,A_array,B_array,NUM_TERMS)
        x=x+delta_x
    return x_array,u_array

# The initial condition
@njit()
def u0(x):
    return 1.0

def main():
    a=0.0
    b=1.0
    c=0.0
    d=0.0
    e=1.0
    f=0.0
    L=1.0
    D=1.0
    
    times=[0.0,0.1,0.2,0.3,0.4,0.5]
    for t in times:
         x_array,u_array=get_u_array(a,b,c,d,e,f,L,D,t)
         plt.plot(x_array,u_array)
         
    plt.show()
    

main()








