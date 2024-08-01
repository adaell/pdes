'''
Calculates the solution to the diffusion-advection equation using a Markov chain 
Monte Carlo. The equation solved is

                         2
 d                      d                        d
 -- (u(x, y, t)) = D_y (--- (u(x, y, t))) - v_y (-- (u(x, y, t)))
 dt                       2                      dy
                        dy

                         2
                        d                        d
                 + D_x (--- (u(x, y, t))) - v_x (-- (u(x, y, t)))
                          2                      dx
                        dx

where D_{x,y} is the diffusivity and v = v_x*î + v_y*ĵ describes the direction of 
advection. 

The solution domain is bound by 0 <= x <= L_x and 0 <= y <= L_y. Boundary conditions
can be either homogeneous Neumann boundary conditions or nonhomogeneous/homogeneous 
Dirichlet conditions. The solution is calculated for any arbitrary initial condition
u(x,y,0).

The program outputs images of the solution at regular time intervals. 

Requires numpy and matplotlib 
'''

__version__ = '0.0.alpha'

import numpy as np
import matplotlib.pyplot as plt

from random_mc import dicecup_int, dicecup_float

# Parameters used by the model
Lx=50.0
Ly=50.0
endTime=1.0
D_x=1.0
D_y=1.0
v_x=0.0
v_y=0.0

# Number of lattice points in each direction
N=50
M=50

#The number of molecules to track
num_molecules=1000000 

# save an image whenever t = [an integer multiple of this number]
save_interval=0.5
colorbar_min=0
colorbar_max=20

# number of random numbers to generate at a time
num_random=10000

# The function u(x,y,0)
def U_0(x,y):
	return 1.0

# Robin parameters for each boundary
#
#          du
#  a u + b -- = g
#          dn
#
# WARNING: Code does not test for unreasonable values.
#
# Format is: boundary_locat = (a,b,g)
b_x_0 =(1,0,0)
b_x_Lx=(1,0,0)
b_y_0 =(1,0,0)
b_y_Ly=(1,0,0)

########################################################################################

# Calculate these values just once
numNodes=N*M
deltaX=Lx/(N+1)
deltaY=Ly/(M+1)
epsilon_x=-2.0*v_x/deltaX
epsilon_y=-2.0*v_y/deltaY
deltaT=0.5*(deltaX*deltaX/(4.0*D_x)+deltaY*deltaY/(4.0*D_y))
p_east=(1.0+epsilon_x)/4.0
p_west=(1.0-epsilon_x)/4.0
p_north=(1.0+epsilon_y)/4.0
p_south=(1.0-epsilon_y)/4.0
moleculesPerNode=num_molecules/numNodes
p_a=p_east
p_b=p_a+p_west
p_c=p_b+p_north
p_d=p_c+p_south
num_threads=1

# Returns true if the boundary condition b is homogeneous and Neumann
def isNeumann(b):
	if b[0] == 0 and b[2] == 0:
		return True
	else:
		return False

# Returns true if the boundary condition b is Dirichlet
def isDirichlet(b):
	if b[1] == 0:
		return True
	else:
		return False

# Calculate boundary information just once
x0_Neumann=isNeumann(b_x_0)
xL_Neumann=isNeumann(b_x_Lx)
y0_Neumann=isNeumann(b_y_0)
yL_Neumann=isNeumann(b_y_Ly)
x0_Dirichlet=isDirichlet(b_x_0)
xL_Dirichlet=isDirichlet(b_x_Lx)
y0_Dirichlet=isDirichlet(b_y_0)
yL_Dirichlet=isDirichlet(b_y_Ly)
c_x0=int(0)
c_xL=int(0)
c_y0=int(0)
c_yL=int(0)
if x0_Dirichlet is True:
	c_x0=int(b_x_0[2]/b_x_0[0])
if xL_Dirichlet is True:
	c_xL=int(b_x_Lx[2]/b_x_Lx[0])
if y0_Dirichlet is True:
	c_y0=int(b_y_0[2]/b_y_0[0])
if yL_Dirichlet is True:
	c_yL=int(b_y_Ly[2]/b_y_Ly[0])

# Sanity test
if (x0_Neumann == False) and (x0_Dirichlet == False):
	import sys
	print("Invalid boundary condition at x=0")
	sys.exit(1)
if (xL_Neumann == False) and (xL_Dirichlet == False):
	import sys
	print("Invalid boundary condition at x=Lx")
	sys.exit(1)
if (y0_Neumann == False) and (y0_Dirichlet == False):
	import sys
	print("Invalid boundary condition at y=0")
	sys.exit(1)
if (yL_Neumann == False) and (yL_Dirichlet == False):
	import sys
	print("Invalid boundary condition at y=Ly")
	sys.exit(1)

# initialises x_sum and y_xum for lattice L
def init_xy_sum(x_sum,y_sum,L):
	for i in range(0,N):
		x_sum[i]=0
	for j in range(0,M):
		y_sum[j]=0
	for i in range(0,N):
		for j in range(0,M):
			x_sum[i]+=L[i,j]
			y_sum[j]+=L[i,j]
	return (x_sum,y_sum)

# Calculate the scaling
def calculate_scaling():
	riemann=0.0
	for i in range(0,N):
		for j in range(0,M):
			riemann+=U_0(i*deltaX,j*deltaY)
	return num_molecules/int(riemann)

scale=calculate_scaling()

# Returns the distribution of molecules across the lattice at t=0
def get_U0(L):
	for i in range(0,N):
		for j in range(0,M):
			x=i*deltaX
			y=j*deltaY
			uxy=int(scale*U_0(x,y))
			L[i,j]=uxy
	return L

# TODO inefficient
# Returns the coordinates (x,y) of molecule n

def get_molecule_XY(n,x_sum,y_sum,L):
	summation=0
	for y in range(0,M):
		for x in range(0,N):
			summation+=L[x,y]
			if summation >= n:
				ans=(x,y)
				return ans


# Returns the current number of molecules
def get_num_molecule(L,x_sum,y_sum):
	num=np.sum(x_sum)
	return num

# returns the direction of movement corresponding to R

def getDirection(R):
	if R <= p_a:
		return "x+"
	elif R <= p_b:
		return "x-"
	elif R <= p_c:
		return "y+"
	elif R < p_d:
		return "y-"
	else:
		import sys
		print("Unknown error")
		sys.exit(1)

# returns boundary information

def getBoundaryInfo(x,y):
	x0=x==0
	xL=x==N
	y0=y==0
	yL=y==M
	if x0 and y0:
		return 'x0y0'
	elif x0 and yL:
		return 'x0yL'
	elif xL and y0:
		return 'xLy0'
	elif xL and yL:
		return 'xLyL'
	elif x0:
		return 'x0'
	elif xL:
		return 'xL'
	elif y0:
		return 'y0'
	elif yL:
		return 'yL'
	else:
		return ''

# Moves a single molecule in x- direction
def move_xm(L,x,y,x_sum,y_sum):
	L[x,y]-=1
	x_sum[x]-=1
	L[x-1,y]+=1
	x_sum[x-1]+=1
	return (L,x_sum,y_sum)

# Moves a single molecule in x+ direction
def move_xp(L,x,y,x_sum,y_sum):
	L[x,y]-=1
	x_sum[x]-=1
	L[x+1,y]+=1
	x_sum[x+1]+=1
	return (L,x_sum,y_sum)

# Moves a single molecule in y- direction
def move_ym(L,x,y,x_sum,y_sum):
	L[x,y]-=1
	y_sum[y]-=1
	L[x,y-1]+=1
	y_sum[y-1]+=1
	return (L,x_sum,y_sum)

# Moves a single molecule in y+ direction
def move_yp(L,x,y,x_sum,y_sum):
	L[x,y]-=1
	y_sum[y]-=1
	L[x,y+1]+=1
	y_sum[y+1]+=1
	return (L,x_sum,y_sum)

# wrapper function
def move(L,x,y,direction,x_sum,y_sum):
	if direction=='x+':
		(L,x_sum,y_sum)=move_xp(L,x,y,x_sum,y_sum)
		if x+1 == N-1:
			(L,x_sum,y_sum)=updateBoundary('xL',L,x_sum,y_sum)
	elif direction=='x-':
		(L,x_sum,y_sum)=move_xm(L,x,y,x_sum,y_sum)
		if x-1 == 0:
			(L,x_sum,y_sum)=updateBoundary('x0',L,x_sum,y_sum)
	elif direction=='y+':
		(L,x_sum,y_sum)=move_yp(L,x,y,x_sum,y_sum)
		if y+1 == M-1:
			(L,x_sum,y_sum)=updateBoundary('yL',L,x_sum,y_sum)
	elif direction=='y-':
		(L,x_sum,y_sum)=move_ym(L,x,y,x_sum,y_sum)
		if y-1 == 0:
			(L,x_sum,y_sum)=updateBoundary('y0',L,x_sum,y_sum)
	else:
		import sys
		print("unknown error")
		sys.exit(1)
	return (L,x_sum,y_sum)

# Updates the boundary, if the boundary conditions are Dirichlet
def updateBoundary(boundary,L,x_sum,y_sum):
	if boundary == 'x0':
		x_sum[0]=0
		for i in range(0,N):
			y_sum[i]-=L[0,i]
			L[0,i]=c_x0
			x_sum[0]+=L[0,i]
	elif boundary == 'xL':
		x_sum[N-1]=0
		for i in range(0,N):
			y_sum[i]-=L[M-1,i]
			L[M-1,i]=c_xL
			x_sum[N-1]+=L[M-1,i]
	elif boundary == 'y0':
		y_sum[0]=0
		for i in range(0,M):
			x_sum[i]-=L[i,0]
			L[i,0]=c_y0
			y_sum[0]+=L[i,0]
	elif boundary == 'yL':
		y_sum[M-1]=0
		for i in range(0,M):
			x_sum[i]-=L[i,N-1]
			L[i,N-1]=c_yL
			y_sum[M-1]+=L[i,N-1]
	else:
		import sys
		print("unknown error")
		sys.exit(1)
	return (L,x_sum,y_sum)

# Updates all boundaries
def updateAllBoundaries(L,x_sum,y_sum):
	(L,x_sum,y_sum)=updateBoundary('x0',L,x_sum,y_sum)
	(L,x_sum,y_sum)=updateBoundary('xL',L,x_sum,y_sum)
	(L,x_sum,y_sum)=updateBoundary('y0',L,x_sum,y_sum)
	(L,x_sum,y_sum)=updateBoundary('yL',L,x_sum,y_sum)
	return (L,x_sum,y_sum)

# Performs a single step on the lattice 
def make_single_step(L,molecule_to_move,R,x_sum,y_sum):
	(x,y)=get_molecule_XY(molecule_to_move,x_sum,y_sum,L)
	direction=getDirection(R)
	boundary=getBoundaryInfo(x,y)
	if boundary == 'x0y0':
		if direction == 'x-':
			if x0_Neumann is True:
				(L,x_sum,y_sum)=move(L,x,y,'x+',x_sum,y_sum)
			elif x0_Dirichlet is True:
				L=updateBoundary('x0',L,x_sum,y_sum)
		elif direction == 'y-':
			if y0_Neumann is True:
				(L,x_sum,y_sum)=move(L,x,y,'y+',x_sum,y_sum)
			elif y0_Dirichlet is True:
				L=updateBoundary('y0',L,x_sum,y_sum)
		else:
			(L,x_sum,y_sum)=move(L,x,y,direction,x_sum,y_sum)
	elif boundary == 'x0yL':
		if direction == 'x-':
			if x0_Neumann is True:
				(L,x_sum,y_sum)=move(L,x,y,'x+',x_sum,y_sum)
			elif x0_Dirichlet is True:
				L=updateBoundary('x0',L,x_sum,y_sum)
		elif direction == 'y-':
			if yL_Neumann is True:
				(L,x_sum,y_sum)=move(L,x,y,'y-',x_sum,y_sum)
			elif yL_Dirichlet is True:
				L=updateBoundary('yL',L,x_sum,y_sum)
		else:
			(L,x_sum,y_sum)=move(L,x,y,direction,x_sum,y_sum)
	elif boundary == 'xLy0':
		if direction == 'x+':
			if xL_Neumann is True:
				(L,x_sum,y_sum)=move(L,x,y,'x-',x_sum,y_sum)
			elif xL_Dirichlet is True:
				L=updateBoundary('xL',L,x_sum,y_sum)
		elif direction == 'y-':
			if y0_Neumann is True:
				(L,x_sum,y_sum)=move(L,x,y,'y+',x_sum,y_sum)
			elif y0_Dirichlet is True:
				L=updateBoundary('y0',L,x_sum,y_sum)
		else:
			(L,x_sum,y_sum)=move(L,x,y,direction,x_sum,y_sum)
	elif boundary == 'xLyL':
		if direction == 'x+':
			if xL_Neumann is True:
				(L,x_sum,y_sum)=move(L,x,y,'x-',x_sum,y_sum)
			elif xL_Dirichlet is True:
				L=updateBoundary('xL',L,x_sum,y_sum)
		elif direction == 'y+':
			if yL_Neumann is True:
				(L,x_sum,y_sum)=move(L,x,y,'y-',x_sum,y_sum)
			elif yL_Dirichlet is True:
				L=updateBoundary('yL',L,x_sum,y_sum)
		else:
			(L,x_sum,y_sum)=move(L,x,y,direction,x_sum,y_sum)
	elif boundary == 'x0':
		if direction == 'x-':
			if x0_Neumann is True:
				(L,x_sum,y_sum)=move(L,x,y,'x+',x_sum,y_sum)
			elif x0_Dirichlet is True:
				L=updateBoundary('x0',L,x_sum,y_sum)
		else:
			(L,x_sum,y_sum)=move(L,x,y,direction,x_sum,y_sum)
	elif boundary == 'xL':
		if direction == 'x+':
			if xL_Neumann is True:
				(L,x_sum,y_sum)=move(L,x,y,'x-',x_sum,y_sum)
			elif xL_Dirichlet is True:
				L=updateBoundary('xL',L,x_sum,y_sum)
		else:
			(L,x_sum,y_sum)=move(L,x,y,direction,x_sum,y_sum)
	elif boundary == 'y0':
		if direction == 'y-':
			if y0_Neumann is True:
				(L,x_sum,y_sum)=move(L,x,y,'y+')
			elif y0_Dirichlet is True:
				L=updateBoundary('y0',L,x_sum,y_sum)
		else:
			(L,x_sum,y_sum)=move(L,x,y,direction,x_sum,y_sum)
	elif boundary == 'yL':
		if direction == 'y+':
			if yL_Neumann is True:
				(L,x_sum,y_sum)=move(L,x,y,'y-',x_sum,y_sum)
			elif yL_Dirichlet is True:
				L=updateBoundary('yL',L,x_sum,y_sum)
		else:
			(L,x_sum,y_sum)=move(L,x,y,direction,x_sum,y_sum)
	elif boundary == '':
		(L,x_sum,y_sum)=move(L,x,y,direction,x_sum,y_sum)
	else:
		import sys
		print("unknown error")
		sys.exit(1)

	return (L,x_sum,y_sum)

# TODO inefficient
# Marches the simulation forward num_steps
def march(L,num_steps,R_int,R_float,x_sum,y_sum):
	for i in range(0,num_steps):
		num_molecules=get_num_molecule(L,x_sum,y_sum)
		if num_molecules == 0:
			return (L,x_sum,y_sum)		
		molecule_to_move=R_int.get_R()
		counter = 0
		while molecule_to_move > num_molecules:
			counter+=1
			molecule_to_move=R_int.get_R()
			if counter == 100:
				R_int.setNewMax(num_molecules)
		R=R_float.get_R()
		(L,x_sum,y_sum)=make_single_step(L,molecule_to_move,R,x_sum,y_sum)
	return (L,x_sum,y_sum)

# A single simulation
def run_simulation():
	# Number of steps until the next picture
	num_steps=int(save_interval/deltaT)*num_molecules
	# Number of steps in total
	total_steps=int(endTime/deltaT)*num_molecules
	# Number of loop iterations
	it=int(total_steps/num_steps)

	# Initialise variables
	L=np.zeros((N,M), dtype=int)
	x_sum=np.zeros(N, dtype=int)
	y_sum=np.zeros(M, dtype=int)
	L=get_U0(L)
	(x_sum,y_sum)=init_xy_sum(x_sum,y_sum,L)
	(L,x_sum,y_sum)=updateAllBoundaries(L,x_sum,y_sum)

	# Random numbers
	R_int=dicecup_int(num_random,0,get_num_molecule(L,x_sum,y_sum))
	R_float=dicecup_float(num_random,0,1.0)

	for i in range(0,it):
		(L,x_sum,y_sum)=march(L,num_steps,R_int,R_float,x_sum,y_sum)
		if sum(x_sum) == 0:
			break

	print(L)



def main():
	run_simulation()

main()

