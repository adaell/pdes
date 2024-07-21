"""
This program calculates the solution to the two-dimensional diffusion-advection equation

                             2
 d                          d                        d
 -- (u(x, y, t)) = D(x, y) (--- (u(x, y, t))) - v_y (-- (u(x, y, t)))
 dt                           2                      dy
                            dy

                             2
                            d                        d
                 + D(x, y) (--- (u(x, y, t))) - v_x (-- (u(x, y, t))) + q(x, y, t)
                              2                      dx
                            dx

where v = v_x*î + v_y*ĵ describes the direction of advection, D(x,y) describes the
diffusivity at (x,y) and q(x,y,t) is a source term. Code assumes Robin boundary 
conditions

							          du
							  a u + b -- = g(x,y)
							          dn

where a and b are real constants and g(x,y) is an arbitary function on the boundary. 
Solution is calculated for an arbitrary initial condition u_0(x,y) = u(x,y,0). The 
function u(x,y,t) is bounded by the domain 0 <= x <= L_x and 0 <= y <= L_y with t >= 0.

The equation is discretised using an implicit finite difference scheme on a uniform 
mesh. The program outputs images of the solution at regular time intervals. 

Requires numpy, scipy and matplotlib

"""

__version__ = '0.1'

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import math

VERBOSE=True

# Parameters used by the discretisation scheme
L_x=150
L_y=100
endTime=1000
deltaT=0.1
deltaX=0.5
deltaY=0.5

# x and y components of the advection vector v = v_x * î + v_y * ĵ
v_x=10
v_y=10

# save an image whenever t = [an integer multiple of this number]
save_interval=2.5

# Robin parameters for each boundary
#
#          du
#  a u + b -- = g(x,y)
#          dn
#
# WARNING: Code does not test for unreasonable values.
#
# Format is: boundary_locat = (a,b)
b_x_0 =(1,0)
b_x_Lx=(1,0)
b_y_0 =(1,0)
b_y_Ly=(1,0)

# g(x,y) on each boundary
def g_x_0(x,y):
	return 0.0

def g_x_Lx(x,y):
	return 0.0

def g_y_Ly(x,y):
	return 0.0

def g_y_0(x,y):
	return 0.0

# The diffusivity / thermal conductivity function D(x,y)
def D(x,y):
	return 100.0

# The source function
def q(x,y,t):
	return 0.0

# The function u(x,y,0)
def U_0(x,y):
	return 100.0

########################################################################################

# Returns an array with the initial condition
def get_u0():
	u0=np.zeros(numNodes, dtype=float)
	for i in range(0,numNodes):
		(x,y)=getXY(i)
		u0[i]=U_0(x,y)
	return u0

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
N = int(1 + L_x / (deltaX))
M = int(1 + L_y / (deltaY))
numNodes=int(N*M)
W_dirichlet=isDirichlet(b_x_0)
E_dirichlet=isDirichlet(b_x_Lx)
S_dirichlet=isDirichlet(b_y_0)
N_dirichlet=isDirichlet(b_y_Ly)

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

# returns the coefficient matrix, A
def getA():
	A=scipy.sparse.lil_matrix((numNodes,numNodes), dtype=float)
	for i in range(0,numNodes):
		boundaryType=getBoundaryType(i)
		(x,y)=getXY(i)
		Dxy=D(x,y)
		main_coeff=(1/deltaT)+2.0*Dxy/(deltaX*deltaX)+2.0*Dxy/(deltaY*deltaY)
		xp1= -Dxy/(deltaX*deltaX)+v_x/(2*deltaX)
		xm1= -Dxy/(deltaX*deltaX)-v_x/(2*deltaX)
		yp1= -Dxy/(deltaY*deltaY)+v_y/(2*deltaY)
		ym1= -Dxy/(deltaY*deltaY)-v_y/(2*deltaY)
		if boundaryType == "false":
			A[i,i]=main_coeff
			A[i,i-1]=xm1
			A[i,i+1]=xp1
			A[i,i-N]=ym1
			A[i,i+N]=yp1
		elif boundaryType == "west":
			if W_dirichlet is True:
				A[i,i]=b_x_0[0]
			else:
				robin_x_0=(b_x_0[0]*deltaX+b_x_0[1])/(b_x_0[1])
				A[i,i]=main_coeff+xm1*robin_x_0
				A[i,i+1]=xp1
				A[i,i-N]=ym1
				A[i,i+N]=yp1
		elif boundaryType == "east":
			if E_dirichlet is True:
				A[i,i]=b_x_Lx[0]
			else:
				robin_x_Lx=(b_x_Lx[1]-b_x_Lx[0]*deltaX)/(b_x_Lx[1])
				A[i,i]=main_coeff+xp1*robin_x_Lx
				A[i,i-1]=xm1
				A[i,i-N]=ym1
				A[i,i+N]=yp1
		elif boundaryType == "south":
			if S_dirichlet is True:
				A[i,i]=b_y_0[0]
			else:
				robin_y_0=(b_y_0[0]*deltaY+b_y_0[1])/(b_y_0[1])
				A[i,i]=main_coeff+ym1*robin_y_0
				A[i,i-1]=xm1
				A[i,i+1]=xp1
				A[i,i+N]=yp1
		elif boundaryType == "north":
			if N_dirichlet is True:
				A[i,i]=b_y_Ly[0]
			else:
				robin_y_Ly=(b_y_Ly[1]-b_y_Ly[0]*deltaY)/(b_y_Ly[1])
				A[i,i]=main_coeff+yp1*robin_y_Ly
				A[i,i-1]=xm1
				A[i,i+1]=xp1
				A[i,i-N]=ym1	
		elif boundaryType == "southwest":
			if S_dirichlet is True:
				A[i,i]=b_y_0[0]
			elif W_dirichlet is True:
				A[i,i]=b_x_0[0]
			else:
				robin_y_0=(b_y_0[0]*deltaY+b_y_0[1])/(b_y_0[1])
				robin_x_0=(b_x_0[0]*deltaX+b_x_0[1])/(b_x_0[1])
				A[i,i]=main_coeff+ym1*robin_y_0+xm1*robin_x_0
				A[i,i+1]=xp1
				A[i,i+N]=yp1
		elif boundaryType == "southeast":
			if S_dirichlet is True:
				A[i,i]=b_y_0[0]
			elif E_dirichlet is True:
				A[i,i]=b_x_Lx[0]
			else:
				robin_y_0=(b_y_0[0]*deltaY+b_y_0[1])/(b_y_0[1])
				robin_x_Lx=(b_x_Lx[1]-b_x_Lx[0]*deltaX)/(b_x_Lx[1])
				A[i,i]=main_coeff+ym1*robin_y_0+xp1*robin_x_Lx
				A[i,i-1]=xm1
				A[i,i+N]=yp1
		elif boundaryType == "northwest":
			if N_dirichlet is True:
				A[i,i]=b_y_Ly[0]
			elif W_dirichlet is True:
				A[i,i]=b_x_0[0]
			else:
				robin_x_0=(b_x_0[0]*deltaX+b_x_0[1])/(b_x_0[1])
				robin_y_Ly=(b_y_Ly[1]-b_y_Ly[0]*deltaY)/(b_y_Ly[1])
				A[i,i]=main_coeff+yp1*robin_y_Ly+xm1*robin_x_0
				A[i,i+1]=xp1
				A[i,i-N]=ym1	
		elif boundaryType == "northeast":
			if N_dirichlet is True:
				A[i,i]=b_y_Ly[0]
			elif E_dirichlet is True:
				A[i,i]=b_x_Lx[0]
			else:
				robin_x_Lx=(b_x_Lx[1]-b_x_Lx[0]*deltaX)/(b_x_Lx[1])
				robin_y_Ly=(b_y_Ly[1]-b_y_Ly[0]*deltaY)/(b_y_Ly[1])
				A[i,i]=main_coeff+yp1*robin_y_Ly+xp1*robin_x_Lx
				A[i,i-1]=xm1
				A[i,i-N]=ym1	
	A=scipy.sparse.csr_matrix(A)
	return A

# Returns the rhs vector, b
def getB(ui,t):
	b=(1/deltaT)*ui	
	for i in range(0,numNodes):
		boundaryType=getBoundaryType(i)
		(x,y)=getXY(i)
		Dxy=D(x,y)
		xp1= -Dxy/(deltaX*deltaX)+v_x/(2*deltaX)
		xm1= -Dxy/(deltaX*deltaX)-v_x/(2*deltaX)
		yp1= -Dxy/(deltaY*deltaY)+v_y/(2*deltaY)
		ym1= -Dxy/(deltaY*deltaY)-v_y/(2*deltaY)
		b[i]+=q(x,y,t)		
		if boundaryType == "false":
			continue
		elif boundaryType == "west":
			if W_dirichlet is True:
				b[i]=g_x_0(x,y)
			else:
				b[i]+=xm1*((g_x_0(x,y)*deltaX)/(b_x_0[1]))
		elif boundaryType == "east":
			if E_dirichlet is True:
				b[i]=g_x_Lx(x,y)
			else:
				b[i]+=-xp1*((g_x_Lx(x,y)*deltaX)/(b_x_Lx[1]))
		elif boundaryType == "north":
			if N_dirichlet is True:
				b[i]=g_y_Ly(x,y)
			else:
				b[i]+=-yp1*((g_y_Ly(x,y)*deltaY)/(b_y_Ly[1]))
		elif boundaryType == "south":
			if S_dirichlet is True:
				b[i]=g_y_0(x,y)
			else:
				b[i]+=ym1*((g_y_0(x,y)*deltaY)/(b_y_0[1]))
		elif boundaryType == "southwest":
			if S_dirichlet is True:
				b[i]=g_y_0(x,y)
			elif W_dirichlet is True:
				b[i]=g_x_0(x,y)
			else:
				b[i]+=xm1*((g_x_0(x,y)*deltaX)/(b_x_0[1]))+ym1*((g_y_0(x,y)*deltaY)/(b_y_0[1]))
		elif boundaryType == "southeast":
			if E_dirichlet is True:
				b[i]=g_x_Lx(x,y)
			elif S_dirichlet is True:
				b[i]=g_y_0(x,y)
			else:
				b[i]+=ym1*((g_y_0(x,y)*deltaY)/(b_y_0[1]))-xp1*((g_x_Lx(x,y)*deltaX)/(b_x_Lx[1]))
		elif boundaryType == "northwest":
			if N_dirichlet is True:
				b[i]=g_y_Ly(x,y)
			elif W_dirichlet is True:
				b[i]=g_x_0(x,y)
			else:
				b[i]+=-yp1*((g_y_Ly(x,y)*deltaY)/(b_y_Ly[1]))+xm1*((g_x_0(x,y)*deltaX)/(b_x_0[1]))
		elif boundaryType == "northeast":
			if N_dirichlet is True:
				b[i]=g_y_Ly(x,y)
			elif E_dirichlet is True:
				b[i]=g_x_Lx(x,y)
			else:
				b[i]+=-yp1*((g_y_Ly(x,y)*deltaY)/(b_y_Ly[1]))-xp1*((g_x_Lx(x,y)*deltaX)/(b_x_Lx[1]))
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
	plt.inferno()
	plt.colorbar()
	plt.xlabel("x")
	plt.ylabel("y")
	plt.savefig(filename)
	plt.close()

# temporal loop
def temporalLoop():
	u0=get_u0()
	A=getA()
	t=0

	saveImage(u0,0)

	next_image=save_interval
	while t < endTime:
		# find the next solution
		t+=deltaX
		u1=getU(u0,t,A)

		# save a picture
		if t >= next_image:
			saveImage(u1,t)
			next_image+=save_interval

		# prepare for next iteration
		u0=u1

def main():
	temporalLoop()

main()