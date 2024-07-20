"""
This program calculates the solution to the two-dimensional non-homogeneous Helmholtz equation

	 2                 2
	d                 d                  2
	--- (psi(x, y)) + --- (psi(x, y)) + k (x, y) psi(x, y) = f(x, y)
	  2                 2
	dy                dx

with Robin boundary conditions. The solution is bound by the domain 0 <= x <= L_x 
and 0 <= y <= L_y. The equation is discretised using an implicit finite difference
scheme on a uniform mesh. The program outputs an image of the solution.

Requires numpy, scipy and matplotlib

"""

__version__ = '0.1'

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import math

# Parameters used by the model
L_x=100
L_y=100
deltaX=0.05
deltaY=0.05

# Robin parameters for each boundary (N,S,E,W)
#
#          du
#  a u + b -- = g(x,y)
#          dn
#
# WARNING: Code does not test for unreasonable values.
#
# Format is: boundary_locat = (a,b)
boundary_west=(1,0)
boundary_east=(1,0)
boundary_north=(1,0)
boundary_south=(1,0)

# The values of psi(x,y) on each boundaries
def g_west(x,y):
	return 0.0

def g_east(x,y):
	return 0.0

def g_north(x,y):
	return 0.0

def g_south(x,y):
	return 100*math.sin(PI*x/L_x)

# the function f(x,y)
def f(x,y):
	return 0

# the function k(x,y)
def k(x,y):
	return 0

# Calculate these values just once
PI=3.1415926535
N = int(1 + L_x / (deltaX))
M = int(1 + L_y / (deltaY))
numNodes=int(N*M)
OneDeltaXSq=1.0/(deltaX*deltaX)
OneDeltaYSq=1.0/(deltaY*deltaY)
neg2deltaXsq=-2.0/(deltaX*deltaX)
neg2deltaYsq=-2.0/(deltaY*deltaY)

# returns true if the boundary condition is Dirichlet
def isDirichlet(boundary_tuple):
	tol=1e-8
	b=boundary_tuple[1]
	if abs(b) < tol:
		return True
	else:
		return False

nDirichlet=isDirichlet(boundary_north)
sDirichlet=isDirichlet(boundary_south)
eDirichlet=isDirichlet(boundary_east)
wDirichlet=isDirichlet(boundary_west)

# Returns boundary information for node n
def getBoundaryType(n):
	west=n % N == 0
	east=n % N == (N - 1)
	south=n < N
	north=n >= N*(M-1)
	if south and west:
		return "southwest"
	elif south and east:
		return "southeast"
	elif north and west:
		return "northwest"
	elif north and east:
		return "northeast"
	elif south:
		return "south"
	elif north:
		return "north"
	elif east:
		return "east"
	elif west:
		return "west"
	else:
		return "false"

# returns the x and y coordinates of node n
def getXY(n):
	x=(n % N) * deltaX
	y=int(n / M) * deltaY
	return (x,y)

# returns the coefficient matrix, A, and the rhs vector b
def getAb():
	A=scipy.sparse.lil_matrix((numNodes,numNodes), dtype=float)
	b=np.zeros(numNodes, dtype=float)
	for i in range(0,numNodes):
		boundaryType=getBoundaryType(i)
		(x,y)=getXY(i)
		kxy=k(x,y)
		mainCoeff=neg2deltaXsq+neg2deltaYsq+kxy*kxy
		xp1=OneDeltaXSq
		xm1=OneDeltaXSq
		yp1=OneDeltaYSq
		ym1=OneDeltaYSq
		if boundaryType == "false":
			A[i,i]=mainCoeff
			A[i,i-1]=xm1
			A[i,i+1]=xp1
			A[i,i-N]=ym1
			A[i,i+N]=yp1
			b[i]=f(x,y)
		elif boundaryType == "west": 
			if wDirichlet is True:
				A[i,i]=boundary_west[0]
				b[i]=g_west(x,y)
			else:
				wRobin=(boundary_west[0]*deltaX+boundary_west[1])/(boundary_west[1]*deltaX*deltaX)
				A[i,i]=mainCoeff+wRobin
				A[i,i-1]=xm1
				A[i,i+1]=xp1
				A[i,i-N]=ym1
				A[i,i+N]=yp1
				b[i]=f(x,y)+g_west(x,y)/(boundary_west[1]*deltaX)
		elif boundaryType == "east":
			if eDirichlet is True:
				A[i,i]=boundary_east[0]
				b[i]=g_east(x,y)
			else:
				eRobin=(boundary_east[1]-boundary_east[0]*deltaX)/(boundary_east[1]*deltaX*deltaX)
				A[i,i]=mainCoeff+eRobin
				A[i,i-1]=xm1
				A[i,i+1]=xp1
				A[i,i-N]=ym1
				A[i,i+N]=yp1
				b[i]=f(x,y)-g_east(x,y)/(boundary_east[1]*deltaX)
		elif boundaryType == "north":
			if nDirichlet is True:
				A[i,i]=boundary_north[0]
				b[i]=g_north(x,y)
			else:
				nRobin=(boundary_north[1]-boundary_north[0]*deltaY)/(boundary_north[1]*deltaY*deltaY)
				A[i,i]=mainCoeff+nRobin
				A[i,i-1]=xm1
				A[i,i+1]=xp1
				A[i,i-N]=ym1
				A[i,i+N]=yp1
				b[i]=f(x,y)-g_north(x,y)/(boundary_north[1]*deltaY)
		elif boundaryType == "south":
			if sDirichlet is True:
				A[i,i]=boundary_south[0]
				b[i]=g_south(x,y)
			else:
				sRobin=(boundary_south[0]*deltaY+boundary_south[1])/(boundary_south[1]*deltaY*deltaY)
				A[i,i]=mainCoeff+sRobin
				A[i,i-1]=xm1
				A[i,i+1]=xp1
				A[i,i-N]=ym1
				A[i,i+N]=yp1
				b[i]=f(x,y)+g_south(x,y)/(boundary_south[1]*deltaY)
		elif boundaryType == "southwest":
			if wDirichlet is True or sDirichlet is True:
				A[i,i]=boundary_west[0]
				b[i]=g_west(x,y)
			else:
				sRobin=(boundary_south[0]*deltaY+boundary_south[1])/(boundary_south[1]*deltaY*deltaY)
				wRobin=(boundary_west[0]*deltaX+boundary_west[1])/(boundary_west[1]*deltaX*deltaX)
				A[i,i]=mainCoeff+wRobin+sRobin
				A[i,i-1]=xm1
				A[i,i+1]=xp1
				A[i,i-N]=ym1
				A[i,i+N]=yp1
				b[i]=f(x,y)+g_west(x,y)/(boundary_west[1]*deltaX)+g_south(x,y)/(boundary_south[1]*deltaY)
		elif boundaryType == "southeast":
			if sDirichlet is True or eDirichlet is True:
				A[i,i]=boundary_south[0]
				b[i]=g_south(x,y)
			else:
				sRobin=(boundary_south[0]*deltaY+boundary_south[1])/(boundary_south[1]*deltaY*deltaY)
				eRobin=(boundary_east[1]-boundary_east[0]*deltaX)/(boundary_east[1]*deltaX*deltaX)
				A[i,i]=mainCoeff+sRobin+eRobin
				A[i,i-1]=xm1
				A[i,i+1]=xp1
				A[i,i-N]=ym1
				A[i,i+N]=yp1
				b[i]=f(x,y)+g_south(x,y)/(boundary_south[1]*deltaY)-g_east(x,y)/(boundary_east[1]*deltaX)
		elif boundaryType == "northwest":
			if nDirichlet is True or wDirichlet is True:
				A[i,i]=boundary_north[0]
				b[i]=g_north(x,y)
			else:
				nRobin=(boundary_north[1]-boundary_north[0]*deltaY)/(boundary_north[1]*deltaY*deltaY)
				wRobin=(boundary_west[0]*deltaX+boundary_west[1])/(boundary_west[1]*deltaX*deltaX)
				A[i,i]=mainCoeff+nRobin+wRobin
				A[i,i-1]=xm1
				A[i,i+1]=xp1
				A[i,i-N]=ym1
				A[i,i+N]=yp1
				b[i]=f(x,y)-g_north(x,y)/(boundary_north[1]*deltaY)+g_west(x,y)/(boundary_west[1]*deltaX)
		elif boundaryType == "northeast":
			if nDirichlet is True or eDirichlet is True:
				A[i,i]=boundary_north[0]
				b[i]=g_north(x,y)
			else:
				nRobin=(boundary_north[1]-boundary_north[0]*deltaY)/(boundary_north[1]*deltaY*deltaY)
				eRobin=(boundary_east[1]-boundary_east[0]*deltaX)/(boundary_east[1]*deltaX*deltaX)
				A[i,i]=mainCoeff+nRobin
				A[i,i-1]=xm1
				A[i,i+1]=xp1
				A[i,i-N]=ym1
				A[i,i+N]=yp1
				b[i]=f(x,y)-g_north(x,y)/(boundary_north[1]*deltaY)-g_east(x,y)/(boundary_east[1]*deltaX)

	A=scipy.sparse.csr_matrix(A)
	return (A,b)

# Saves an image
def saveImage(u):
	data=np.reshape(u,[N,M])
	filename="images/helmholtz.png"
	plt.imshow(data)
	plt.colorbar()
	plt.savefig(filename)
	plt.close()

# main
def main():
	A,b=getAb()
	psi=scipy.sparse.linalg.spsolve(A,b)
	saveImage(psi)


main()