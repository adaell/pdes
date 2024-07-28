"""
This program calculates the solution to the two-dimensional acoustic wave 
equation

	dv_x         1   dp 
	---- =   - ----- -- ,
	 dt         rho  dx  

	dv_y         1   dp 
	---- =   - ----- -- ,
	 dt         rho  dy  

	dp                     2           dv_y   dv_x
	-- + nu(x,y) * p  = - c  * rho * ( ---- + ---- ) + q(x,y,t) ,
	dt                                  dy     dx

or, equivalently,


  2
 d                            d
 --- (p(x, y, t)) + nu(x,y) *  (-- (p(x, y, t))) = 
   2                          dt
 dt
              2                       2
          2  d                    2  d
         c  (--- (p(x, y, t))) + c  (--- (p(x, y, t))) + Q(x, y, t) ,
               2                       2
             dy                      dx

where p=p(x,y,t) describes the air pressure at (x,y) and time t,  v = v_x*î + 
v_y*ĵ describes the velocity of the wave at (x,y) at time t, nu(x,y) is a 
damping function, q(x,y,t) is a source/sink function with temporal derivative
Q(x,y,t), rho is the density of the medium and c is the speed of sound. The equation
is solved in the domain 0 <= x <= L_x, 0 <= y <= L_y for t >= 0.

Code assumes Robin boundary conditions of the form

            du
    a u + b -- = g(x,y)
            dn

where a and b are real constants and g(x,y) is an arbitary function on the 
boundary. Perfectly matched layers can selectively be turned on/off on each 
boundary to simulate far-field conditions. Code uses an arbitrary initial 
condition p(x,y,0). The equation is discretised using a finite difference 
scheme and is solved on a uniform mesh. The temporal integration uses a 
leap-frog.

Requires numpy, matplotlib and numba

References:
Berenger, J.-P. (1994) 'A perfectly matched layer for the absorption of 
electromagnetic waves,' Journal of Computational Physics, 114(2), pp. 185–200. 
https://doi.org/10.1006/jcph.1994.1159.

"""

__version__ = '0.1'

from numba import njit, prange
import numpy as np
import matplotlib.pyplot as plt
import math

# Parameters used by the discretisation scheme
L_x=50.0
L_y=50.0
endTime=10.0
deltaT=0.01
deltaX=0.10
deltaY=0.10

# Physical parameters
rho=1.025
c=2

# save an image whenever t = [an integer multiple of this number]
save_interval=0.05
colorbar_min=-20
colorbar_max=20

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

# turn pml on/off
pml_x_0=False
pml_x_Lx=True
pml_y_0=True
pml_y_Ly=True
pml_width=20 # Width of PML in number of nodes

# g(x,y) on each boundary
@njit()
def g_x_0(x,y):
	return 0.0

@njit()
def g_x_Lx(x,y):
	return 0.0

@njit()
def g_y_0(x,y):
	return 0.0

@njit()
def g_y_Ly(x,y):
	return 0.0

# Source term function
@njit()
def q(x,y,t):
	# Pulse 1
	pulseCenterX=0.5*L_x
	pulseCenterY=0.5*L_y
	pulseAmplitude=10000.0
	pulsePeriod=2
	xtest=abs(x-pulseCenterX) <= deltaX
	ytest=abs(y-pulseCenterY) <= deltaY
	if xtest is True and ytest is True:
		return pulseAmplitude*math.sin(pulsePeriod*2*PI*t)
	return 0.0

# Damping function
@njit()
def nu(x,y):
	return 0.0

# The initial pressure
def p_0(x,y):
	return q(x,y,0)

# The initial velocity u(x,y,0)
def u_0(x,y):
	return 0.0

# The initial velocity v(x,y,0)
def v_0(x,y):
	return 0.0

#############################################################################

# Damping parameter for the perfectly matched layers
sigma_x_0=10.0
sigma_y_0=10.0
sigma_x_Lx=10.0
sigma_y_Ly=10.0

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
num_nodes=int(N*M)
x_0_dirichlet=isDirichlet(b_x_0)
x_Lx_dirichlet=isDirichlet(b_x_Lx)
y_0_dirichlet=isDirichlet(b_y_0)
y_Ly_dirichlet=isDirichlet(b_y_Ly)
oneDeltaT=(1/deltaT)
cSquared=c*c

# Returns boundary information for node n
@njit()
def getBoundaryType(n):
	x0=n % N == 0
	xLx=n % N == (N - 1)
	y0=n < N
	yLy=n >= N*(M-1)
	if y0 and x0:
		return "x_0_y_0"
	elif y0 and xLx:
		return "x_Lx_y_0"
	elif yLy and x0:
		return "x_0_y_Ly"
	elif yLy and xLx:
		return "x_Lx_y_Ly"
	elif y0:
		return "y_0"
	elif yLy:
		return "y_Ly"
	elif xLx:
		return "x_Lx"
	elif x0:
		return "x_0"
	else:
		return "false"

@njit(parallel=True)
def genBoundaryVector(boundary_vec):
	for i in prange(0,num_nodes):
		boundary_vec[i]=getBoundaryType(i)

# returns the x and y coordinates of node n
@njit()
def get_XY(n):
	x=(n % N) * deltaX
	y=int(n / M) * deltaY
	return (x,y)

@njit(parallel=True)
def genXYVector(x_vec,y_vec):
	for i in prange(0,num_nodes):
		(x,y)=get_XY(i)
		x_vec[i]=x
		y_vec[i]=y

# Returns the value of sigma at node n
@njit()
def getSigma(n):
	x = n % N
	y = int(n / M)
	xx = deltaX*x
	yy = deltaY*y
	sigmax=0
	sigmay=0
	nn=1
	if pml_x_0 is True and x <= pml_width:
		sigmax=(sigma_x_0/(pml_width*deltaX))*xx*xx-sigma_x_0
	if pml_x_Lx is True and x >= (N - pml_width):
		wd=pml_width*deltaX
		a=(-sigma_x_Lx/(2*wd*L_x-wd*wd))
		sigmax=a*xx*xx-a*(L_x-wd)*(L_x-wd)
	if pml_y_0 is True and y <= pml_width:
		sigmay=(sigma_y_0/(pml_width*deltaY))*yy*yy-sigma_y_0
	if pml_y_Ly is True and y >= (M - pml_width):
		wd=pml_width*deltaY
		a=(-sigma_y_Ly/(2*wd*L_y-wd*wd))
		sigmax=a*yy*yy-a*(L_y-wd)*(L_y-wd)
	return sigmax+sigmay

@njit(parallel=True)
def genSigmaVec(sigma_vec):
	for i in prange(0,num_nodes):
		sigma_vec[i]=getSigma(i)

# Returns an array with p(x,y,0)
def get_p0(x,y):
	p0=np.zeros(num_nodes, dtype=float)
	for i in range(0,num_nodes):
		(x,y)=get_XY(i)
		p0[i]=p_0(x,y)
	return p0

# Returns an array with u(x,y,0)
def get_u0(x,y):
	u0=np.zeros(num_nodes, dtype=float)
	for i in range(0,num_nodes):
		(x,y)=get_XY(i)
		u0[i]=u_0(x,y)
	return u0

# Returns an array with v(x,y,0)
def get_v0(x,y):
	v0=np.zeros(num_nodes, dtype=float)
	for i in range(0,num_nodes):
		(x,y)=get_XY(i)
		v0[i]=v_0(x,y)
	return v0

u_r_0=0.0
if abs(b_x_0[1]) > 1e-8:
	u_r_0=(1+deltaX*b_x_0[0]/b_x_0[1])
u_r_Lx=0.0
if abs(b_x_Lx[1]) > 1e-8:
	u_r_Lx=(1-deltaX*b_x_Lx[0]/b_x_Lx[1])
u_m=deltaT/(2.0*rho*deltaX)

# updates u
@njit(parallel=True)
def get_u(u,p,boundary_vec,x_vec,y_vec,sigma_vec):
	for i in prange(0,num_nodes):
		boundaryType=boundary_vec[i]
		x=x_vec[i]
		y=y_vec[i]
		divisor=1-deltaT*sigma_vec[i]
		pxm=0.0
		pxp=0.0
		if boundaryType == "x_0" or boundaryType == "x_0_y_0" or boundaryType == "x_0_y_Ly":
			pxp=u[i+1]
			if x_0_dirichlet is True:
				pxm=g_x_0(x,y)/b_x_0[0]
			else:
				pxm=(-deltaX*g_x_0(x,y)/b_x_0[1])+u_r_0*p[i]
		elif boundaryType == "x_Lx" or boundaryType == "x_Lx_y_0" or boundaryType == "x_Lx_y_Ly":
			pxm=u[i-1]
			if x_Lx_dirichlet is True:
				pxp=g_x_Lx(x,y)/b_x_Lx[0]
			else:
				pxp=(deltaX*g_x_Lx(x,y)/b_x_Lx[1])+u_r_Lx*p[i]
		else:
			pxm=p[i-1]
			pxp=p[i+1]
		u[i]=(u[i]-u_m*pxp+u_m*pxm)/divisor
	return u

v_r_0=0
if b_y_0[1] > 1e-8:
	v_r_y0=(1+deltaY*b_y_0[0]/b_y_0[1])
v_r_Ly=0
if b_y_Ly[1] > 1e-8:
	v_r_Ly=(1-deltaY*b_y_Ly[0]/b_y_Ly[1])
v_m=deltaT/(2*rho*deltaY)

# updates v
@njit(parallel=True)
def get_v(v,p,boundary_vec,x_vec,y_vec,sigma_vec):
	for i in prange(0,num_nodes):
		boundaryType=boundary_vec[i]
		x=x_vec[i]
		y=y_vec[i]
		divisor=1-deltaT*sigma_vec[i]
		pym=0
		pyp=0
		if boundaryType == "y_0" or boundaryType == "x_0_y_0" or boundaryType == "x_Lx_y_0":
			pyp=p[i+N]
			if y_0_dirichlet is True:
				pym=g_y_0(x,y)/b_y_0[0]
			else:
				pym=(-deltaY*g_y_0(x,y)/b_y_0[1])+v_r_0*p[i]
		elif boundaryType == "y_Ly" or boundaryType == "x_0_y_Ly" or boundaryType == "x_Lx_y_Ly":
			pym=p[i-N]
			if y_Ly_dirichlet is True:
				pyp=g_y_Ly(x,y)/b_y_Ly[0]
			else:
				pyp=(deltaY*g_y_Ly(x,y)/b_y_Ly[1])+v_r_Ly*p[i]
		else:
			pyp=p[i+N]
			pym=p[i-N]
		v[i]=(v[i]-v_m*pyp+v_m*pym)/divisor
	return v

px_r_0=0
if b_x_0[1] > 1e-8:
	px_r_0=(1.0+b_x_0[0]*deltaX/b_x_0[1])
px_r_Lx=0
if b_x_Lx[1] > 1e-8:
	px_r_Lx=(1.0-b_x_Lx[0]*deltaX/b_x_Lx[1])
px_m=rho*c*c*deltaT/(2*deltaX)

# updates p_x
@njit(parallel=True)
def get_p_x(p,u,t,boundary_vec,x_vec,y_vec,sigma_vec):
	for i in prange(0,num_nodes):
		boundaryType=boundary_vec[i]
		x=x_vec[i]
		y=y_vec[i]
		divisor=1+deltaT*nu(x,y)-deltaT*sigma_vec[i]
		uxm=0
		uxp=0
		if boundaryType == "x_0" or boundaryType == "x_0_y_0" or boundaryType == "x_0_y_Ly":
			uxp=u[i+1]
			if x_0_dirichlet is True:
				uxm=g_x_0(x,y)/b_x_0[0]
			else:
				uxm=((-g_x_0(x,y)*deltaX/b_x_0[1])+px_r_0*p_old[i])
		elif boundaryType == "x_Lx" or boundaryType == "x_Lx_y_0" or boundaryType == "x_Lx_y_Ly":
			uxm=u[i-1]
			if x_Lx_dirichlet is True:
				uxp=g_x_Lx(x,y)/b_x_Lx[0]
			else:
				uxp=((g_x_Lx(x,y)*deltaX/b_x_Lx[1])+px_r_Lx*p_old[i])
		else:
			uxm=u[i-1]
			uxp=u[i+1]
		p[i]=(p[i]-px_m*uxp+px_m*uxm+0.5*deltaT*q(x,y,t))/divisor
	return p

py_r_0=0
if b_y_0[1] > 1e-8:
	py_r_0=(1.0+b_y_0[0]*deltaY/b_y_0[1])
py_r_L=0
if b_y_Ly[1] > 1e-8:
	py_r_L=(1.0-b_y_Ly[0]*deltaY/b_y_Ly[1])
py_m=rho*c*c*deltaT/(2*deltaY)

# updates p_y
@njit(parallel=True)
def get_p_y(p,v,t,boundary_vec,x_vec,y_vec,sigma_vec):
	for i in prange(0,num_nodes):
		boundaryType=boundary_vec[i]
		x=x_vec[i]
		y=y_vec[i]
		divisor=1+deltaT*nu(x,y)-deltaT*sigma_vec[i]
		vym=0
		vyp=0
		if boundaryType == "y_0" or boundaryType == "x_0_y_0" or boundaryType == "x_Lx_y_0":
			vyp=v[i+N]
			if y_0_dirichlet is True:
				vym=g_y_0(x,y)/b_y_0[0]
			else:
				vym=((-g_y_0(x,y)*deltaY/b_y_0[1])+py_r_0*p_old[i])
		elif boundaryType == "y_Ly" or boundaryType == "x_0_y_Ly" or boundaryType == "x_Lx_y_Ly":
			vym=v[i-N]
			if y_Ly_dirichlet is True:
				vyp=g_y_Ly(x,y)/b_y_Ly[0]
			else:
				vyp=((g_y_Ly(x,y)*deltaY/b_y_Ly[1])+py_r_L*p_old[i])
		else:
			vym=v[i-N]
			vyp=v[i+N]
		p[i]=(p[i]-py_m*vyp+py_m*vym+0.5*deltaT*q(x,y,t))/divisor
	return p

# Saves an image
def saveImage(u,t):
	data=np.reshape(u,[M,N])
	if t == 0:
		filename="images/wave_2D_PML.0.0.png"
	else:
		filename="images/wave_2D_PML."+str(round(t,2))+".png"
	title_text="t = " + str(t)
	plt.imshow(data)
	plt.title(title_text)
	plt.clim(vmin=colorbar_min,vmax=colorbar_max)
	plt.colorbar()
	plt.viridis()
	plt.savefig(filename)
	plt.close()

# Iterates over time
def temporalLoop():
	px=0.5*get_p0(0,0)
	py=px
	u=get_u0(0,0)
	v=get_v0(0,0)
	saveImage(px+py,0)

	# Do these calculations only once
	boundary_vec=np.empty(num_nodes, dtype='U9')
	genBoundaryVector(boundary_vec)
	x_vec=np.zeros(num_nodes, dtype=float)
	y_vec=np.zeros(num_nodes, dtype=float)
	genXYVector(x_vec,y_vec)
	sigma_vec=np.zeros(num_nodes, dtype=float)
	genSigmaVec(sigma_vec)

	t=0
	next_image=save_interval
	while t < endTime:
		# find the next solution
		t+=deltaT
		u=get_u(u,px+py,boundary_vec,x_vec,y_vec,sigma_vec)
		v=get_v(v,px+py,boundary_vec,x_vec,y_vec,sigma_vec)
		px=get_p_x(px,u,t,boundary_vec,x_vec,y_vec,sigma_vec)
		py=get_p_y(py,v,t,boundary_vec,x_vec,y_vec,sigma_vec)

        # save a picture
		if abs(t - next_image) < 0.001:
			saveImage(px+py,t)
			#saveImage(u+v,t)
			next_image+=save_interval

def main():
	if (c >= deltaX / deltaT):
		print("Warning: Recommend that you decrease deltaX")
	if (c >= deltaY / deltaT):
		print("Warning: Recommend that you decrease deltaY")
	temporalLoop()

main()
