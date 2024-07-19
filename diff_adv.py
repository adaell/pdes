"""
This program calculates the solution to the two-dimensional diffusion-advection equation

                             2
 d                          d                      d
 -- (u(x, y, t)) = D(x, y) (--- (u(x, y, t))) - b (-- (u(x, y, t)))
 dt                           2                    dy
                            dy
                             2
                            d                      d
                 + D(x, y) (--- (u(x, y, t))) - a (-- (u(x, y, t))) + q(x, y, t)
                              2                    dx
                            dx

where v = a*î + b*ĵ describes the direction of advection, D(x,y) describes the
diffusivity at (x,y) and q(x,y) is a source term. Code assumes Robin boundary 
conditions. Solution is calculated for an arbitrary initial condition u(x,y,0).
The equation is discretised using an implicit finite difference scheme on a 
uniform mesh. The program outputs images of the solution at regular time intervals. 

Requires numpy, scipy and matplotlib

"""

__version__ = '0.1'

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import math

VERBOSE=True

# Parameters used by the model
Lx=100
Ly=100
endTime=1000
deltaT=0.01
deltaX=0.5
deltaY=0.5

# x and y components of the advection vector v=a*î+b*ĵ
a=0
b=0

# save an image every so many steps
saveMod=0.1

# Robin parameters for each boundary (N,S,E,W)
#
#          du
#  a u + b -- = g
#          dn
#
# WARNING: Code does not test for unreasonable values.
#
# Format is: boundary_locat = (a,b,g)
boundary_west=(0,1,0)
boundary_east=(0,1,0)
boundary_north=(1,0,0)
boundary_south=(1,0,0)

# returns true if the boundary condition is Dirichlet
def isDirichlet(boundary_tuple):
	tol=1e-8
	b=boundary_tuple[1]
	if abs(b) < tol:
		return True
	else:
		return False

# Calculate these values just once
PI=3.1415926535
N = int(1 + Lx / (deltaX))
M = int(1 + Ly / (deltaY))
numNodes=int(N*M)
if isDirichlet(boundary_west):
	robinW=boundary_west[2]/boundary_west[0]
else:
	robinW=(boundary_west[2]*deltaX-boundary_west[1])/(boundary_west[0]*deltaX-boundary_west[1]) 
if isDirichlet(boundary_east):
	robinE=boundary_east[2]/boundary_east[0]
else:
	robinE=(boundary_east[2]*deltaX-boundary_east[1])/(boundary_east[0]*deltaX-boundary_east[1])
if isDirichlet(boundary_south):
	robinS=boundary_south[2]/boundary_south[0]
else:
	robinS=(boundary_south[0]*deltaY-boundary_south[1])/(boundary_south[2]*deltaY-boundary_south[1])
if isDirichlet(boundary_north):
	robinN=boundary_north[2]/boundary_north[0]
else:
	robinN=(boundary_north[0]*deltaY-boundary_north[1])/(boundary_north[2]*deltaY-boundary_north[1])
a2deltaX=a/(2*deltaX)
b2deltaY=b/(2*deltaY)
one2deltaX=1/(2*deltaX)
one2deltaY=1/(2*deltaY)
onedeltaT=1/(deltaT)

# Source term function
def q(x,y,t):
	return 0

# The diffusivity at (x,y)
def D(x,y):
	return 100

# The initial condition
def getU0(x,y):
	u0=np.zeros(numNodes, dtype=float)
	for i in range(0,numNodes):
		u0[i]=1.0
	return u0

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

# returns an array with values of D(x,y) at each mesh point
def getDvec():
	d_vec=np.zeros(numNodes, dtype=float)
	for i in range(0,numNodes):
		(x,y)=getXY(i)
		d_vec[i]=D(x,y)
	return d_vec

# returns the coefficient matrix, A
def getA():
	#A=np.zeros((numNodes,numNodes), dtype=float)
	A=scipy.sparse.lil_matrix((numNodes,numNodes), dtype=float)
	D_vec=getDvec()
	for i in range(0,numNodes):
		boundaryType=getBoundaryType(i)
		(x,y)=getXY(i)
		mainCoeff=onedeltaT+D_vec[i]/deltaX+D_vec[i]/deltaY
		xp1=  a2deltaX-D_vec[i]/(2*deltaX)
		xm1= -a2deltaX-D_vec[i]/(2*deltaX)
		yp1=  b2deltaY-D_vec[i]/(2*deltaY)
		ym1= -b2deltaY-D_vec[i]/(2*deltaY)
		if boundaryType == "false":
			A[i,i]=mainCoeff
			A[i-1,i]=xm1
			A[i+1,i]=xp1
			A[i-N,i]=ym1
			A[i+N,i]=yp1
		elif boundaryType == "west":
			A[i,i]=mainCoeff-xm1*robinW
			A[i+1,i]=xp1
			A[i-N,i]=ym1
			A[i+N,i]=yp1
		elif boundaryType == "east":
			A[i,i]=mainCoeff-xp1*robinE
			A[i-1,i]=xm1
			A[i-N,i]=ym1
			A[i+N,i]=yp1
		elif boundaryType == "north":
			A[i,i]=mainCoeff-yp1*robinN
			A[i-1,i]=xm1
			A[i+1,i]=xp1
			A[i-N,i]=ym1
		elif boundaryType == "south":
			A[i,i]=mainCoeff-ym1*robinS
			A[i-1,i]=xm1
			A[i+1,i]=xp1
			A[i+N,i]=yp1
		elif boundaryType == "southwest":
			A[i,i]=mainCoeff-ym1*robinS-xm1*robinW
			A[i+1,i]=xp1
			A[i+N,i]=yp1
		elif boundaryType == "southeast":
			A[i,i]=mainCoeff-ym1*robinS-xp1*robinE
			A[i-1,i]=xm1
			A[i+N,i]=yp1
		elif boundaryType == "northwest":
			A[i,i]=mainCoeff-yp1*robinN-xm1*robinW
			A[i+1,i]=xp1
			A[i-N,i]=ym1
		elif boundaryType == "northeast":
			A[i,i]=mainCoeff-yp1*robinN-xp1*robinE
			A[i-1,i]=xm1
			A[i-N,i]=ym1
	A=scipy.sparse.csr_matrix(A)
	return A

# Returns the rhs vector, b
def getB(ui,t):
	b=onedeltaT*ui
	for i in range(0,numNodes):
		(x,y)=getXY(i)
		b[i]+=q(x,y,t)
	return b

# Returns the value of u(x,y,t) at the next time step
def getU(ui,t,A):
	b=getB(ui,t)
	u=scipy.sparse.linalg.spsolve(A,b)
	return u

# Saves an image
def saveImage(u,t):
	data=np.reshape(u,[M,N])
	if t == 0:
		filename="images/diffadv.0.0.png"
	else:
		filename="images/diffadv."+str(round(t,2))+".png"
	title_text="t = " + str(t)
	plt.imshow(data)
	plt.title(title_text)
	plt.colorbar()
	plt.savefig(filename)
	plt.close()

# temporal loop
def temporalLoop():
	u0=getU0(0,0)
	A=getA()
	t=0

	saveImage(u0,0)

	while t < endTime:
		# find the next solution
		t+=deltaX
		u1=getU(u0,t,A)

		# save a picture
		if int(t % saveMod) == 0:
			saveImage(u1,t)

		# prepare for next iteration
		u0=u1

def main():
	temporalLoop()

main()