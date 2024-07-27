"""
This program calculates the solution to a two-dimensional wave equation

  2
 d                            d
 --- (u(x, y, t)) + nu(x,y) *  (-- (u(x, y, t))) = 
   2                          dt
 dt
              2                       2
          2  d                    2  d
         c  (--- (u(x, y, t))) + c  (--- (u(x, y, t))) + q(x, y, t)
               2                       2
             dy                      dx

with an arbitrary initial condition, arbitary damping function nu(x,y), source function
q(x,y,t) and Robin boundary conditions, on a uniform rectangular mesh. The governing 
equation is discretised using an implicit finite difference scheme. The program outputs 
images of the solution at regular time intervals. Code assumes that the wave is travelling
through a homogenous medium.

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
endTime=5000
deltaT=0.01
deltaX=0.5
deltaY=0.5
c=5
sigma=1

# parameters used by the source function
pulseCenterX=Lx/2
pulseCenterY=Ly/2
pulseAmplitude=1.0
pulsePeriod=2

# save an image every so many steps
saveMod=0.1

# Robin parameters for each boundary (N,S,E,W)
#          du
#  a u + b -- = g
#          dn
# Code does not test for unreasonable values.
A_west=0
B_west=1
G_west=0
A_east=0
B_east=1
G_east=0
A_north=0
B_north=1
G_north=0
A_south=0
B_south=1
G_south=0

# Calculate these values just once
PI=3.1415926535
N = int(1 + Lx / (deltaX))
M = int(1 + Ly / (deltaY))
numNodes=int(N*M)
robinW=(G_west*deltaX-B_west)/(A_west*deltaX-B_west) 
robinE=(G_east*deltaX-B_east)/(A_east*deltaX-B_east)
robinS=(A_south*deltaY-B_south)/(G_south*deltaY-B_south)
robinN=(A_north*deltaY-B_north)/(G_north*deltaY-B_north)
invDeltaTSq=1/(deltaT*deltaT)
invTwoDeltaT=1/(2*deltaT)
twodeltaT=2*deltaT
TwoInvDeltaTSq=2*invDeltaTSq
cSqDeltaXSq=(c*c)/(deltaX*deltaX)
cSqDeltaYSq=(c*c)/(deltaY*deltaY)

# Source term function
def q(x,y,t):
	xtest=abs(x-pulseCenterX) <= deltaX
	ytest=abs(y-pulseCenterY) <= deltaY
	if xtest is True and ytest is True:
		return pulseAmplitude*math.sin(pulsePeriod*2*PI*t)
	else:
		return 0

# Damping function
def nu(x,y):
	return sigma

# Returns u(x,y,0)
def getU0(x,y):
	u0=np.zeros(numNodes, dtype=float)
	for i in range(0,numNodes):
		(x,y)=getXY(i)
		u0[i]=q(x,y,0)
	return u0

# Returns u(x,y,deltaT)
def getU1(x,y):
	u1=np.zeros(numNodes, dtype=float)
	for i in range(0,numNodes):
		(x,y)=getXY(i)
		u1[i]=q(x,y,deltaT)
	return u1

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

# returns the coefficient matrix
def getA():
	#A=np.zeros((numNodes,numNodes), dtype=float)
	A=scipy.sparse.lil_matrix((numNodes,numNodes), dtype=float)
	for i in range(0,numNodes):
		boundaryType=getBoundaryType(i)
		(x,y)=getXY(i)
		mainCoeff=invDeltaTSq+(nu(x,y)/(twodeltaT))+2*cSqDeltaXSq+2*cSqDeltaYSq
		if boundaryType == "false":
			A[i,i]=mainCoeff
			A[i-1,i]=-cSqDeltaXSq
			A[i+1,i]=-cSqDeltaXSq
			A[i+N,i]=-cSqDeltaYSq
			A[i-N,i]=-cSqDeltaYSq
		elif boundaryType == "west":
			A[i,i]=mainCoeff-cSqDeltaXSq*robinW
			A[i+1,i]=-cSqDeltaXSq
			A[i+N,i]=-cSqDeltaYSq
			A[i-N,i]=-cSqDeltaYSq
		elif boundaryType == "east":
			A[i,i]=mainCoeff-cSqDeltaXSq*robinE
			A[i-1,i]=-cSqDeltaXSq
			A[i+N,i]=-cSqDeltaYSq
			A[i-N,i]=-cSqDeltaYSq
		elif boundaryType == "north":
			A[i,i]=mainCoeff-cSqDeltaYSq*robinN
			A[i-1,i]=-cSqDeltaXSq
			A[i+1,i]=-cSqDeltaXSq
			A[i-N,i]=-cSqDeltaYSq
		elif boundaryType == "south":
			A[i,i]=mainCoeff-cSqDeltaYSq*robinS
			A[i-1,i]=-cSqDeltaXSq
			A[i+1,i]=-cSqDeltaXSq
			A[i+N,i]=-cSqDeltaYSq
		elif boundaryType == "southwest":
			A[i,i]=mainCoeff-cSqDeltaYSq*robinS-cSqDeltaXSq*robinW
			A[i+1,i]=-cSqDeltaXSq
			A[i+N,i]=-cSqDeltaYSq
		elif boundaryType == "southeast":
			A[i,i]=mainCoeff-cSqDeltaYSq*robinS-cSqDeltaXSq*robinE
			A[i-1,i]=-cSqDeltaXSq
			A[i+N,i]=-cSqDeltaYSq
		elif boundaryType == "northwest":
			A[i,i]=mainCoeff-cSqDeltaYSq*robinN-cSqDeltaXSq*robinW
			A[i+1,i]=-cSqDeltaXSq
			A[i-N,i]=-cSqDeltaYSq
		elif boundaryType == "northeast":
			A[i,i]=mainCoeff-cSqDeltaYSq*robinN-cSqDeltaXSq*robinE
			A[i-1,i]=-cSqDeltaXSq
			A[i-N,i]=-cSqDeltaYSq
	A=scipy.sparse.csr_matrix(A)
	return A

# Returns the rhs vector 
def getB(ui,uim1,nuVec,t):
	b=TwoInvDeltaTSq*ui-invDeltaTSq*uim1
	for i in range(0,numNodes):
		(x,y)=getXY(i)
		b[i]+=q(x,y,t)+(nu(x,y)/(2*deltaT))*uim1[i]
	return b

# returns a vector with each nodes value for nu(x,y)
def getNuVector():
	nuVec=np.zeros(numNodes, dtype=float)
	for i in range(0,numNodes):
		(x,y)=getXY(i)
		nuVec[i]=(nu(x,y)/twodeltaT)
	return nuVec

# Returns the values at the next time step
def getU(ui,uim1,nuVec,t,A):
	b=getB(ui,uim1,nuVec,t)
	u=scipy.sparse.linalg.spsolve(A,b)
	return u

# Saves an image
def saveImage(u,t):
	data=np.reshape(u,[M,N])
	if t == 0:
		filename="images/wave.0.0.png"
	else:
		filename="images/wave."+str(round(t,2))+".png"
	title_text="t = " + str(t)
	plt.imshow(data)
	plt.title(title_text)
	plt.colorbar()
	plt.savefig(filename)
	plt.close()

# Iterates over time
@profile
def temporalLoop():
	u0=getU0(0,0)
	u1=getU1(0,0)
	A=getA()
	nuVec=getNuVector()
	t=deltaX

	saveImage(u0,0)

	while t < endTime:
		# find the next solution
		t+=deltaT
		u2=getU(u1,u0,nuVec,t,A)

		# save a picture
		if int(t % saveMod) == 0:
			saveImage(u2,t)

		# prepare for next iteration
		u0=u1
		u1=u2

def main():
	temporalLoop()

main()