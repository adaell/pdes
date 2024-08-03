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
L_x=100.0
L_y=70.0
endTime=100.0
deltaT=0.01
deltaX=0.05
deltaY=0.05

# Physical parameters
rho=1.025
c=2

# save an image whenever t = [an integer multiple of this number]
save_interval=0.05
colorbar_min=-10
colorbar_max=10

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
pml_x_0=True
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
	pulseCenterX=0.25*L_x
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

VERBOSE=True

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
halfDeltaT=0.5*deltaT
cSquared=c*c

@njit()
def verbosemsg(message):
	if VERBOSE is True:
		print(message)

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

# returns the x and y coordinates of node n
@njit()
def get_XY(n):
	x=(n % N) * deltaX
	y=int(n / N) * deltaY
	return (x,y)

# Returns the value of sigma at node n
@njit()
def getSigma(n):
	x = n % N
	y = int(n / N)
	xx = deltaX*x
	yy = deltaY*y
	sigmax=0
	sigmay=0
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
		sigmay=a*yy*yy-a*(L_y-wd)*(L_y-wd)
	return sigmax+sigmay

# Calculates sigma for each node in the mesh
@njit(parallel=True)
def getSigma_v(sigma_v):
	for i in prange(0,num_nodes):
		sigma_v[i]=getSigma(i)
	return sigma_v

# Calculates nu(x,y) for each node in the mesh
@njit(parallel=True)
def getNu_v(nu_v):
	for i in prange(0,num_nodes):
		(x,y)=get_XY(i)
		nu_v[i]=nu(x,y)
	return nu_v

# Calculates the divisors for u, v, p_x, p_y 
@njit(parallel=True)
def get_divisors(div_u,div_v,div_p_x,div_p_y,sigma_v,nu_v):
	for i in prange(0, num_nodes):
		div_u[i]=1.0/(1.0-deltaT*sigma_v[i])
		div_v[i]=1.0/(1.0-deltaT*sigma_v[i])
		div_p_x[i]=1.0/(1.0+deltaT*nu_v[i]-deltaT*sigma_v[i])
		div_p_y[i]=1.0/(1.0+deltaT*nu_v[i]-deltaT*sigma_v[i])
	return (div_u,div_v,div_p_x,div_p_y)

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

# Returns five arrays of booleans with the boundary decisions in the x direction
@njit(parallel=True)
def get_x_decisions(x_b0,x_b1,x_b2,x_b3,x_b4):
	for i in prange(0,num_nodes):
		boundaryType=getBoundaryType(i)
		(x,y)=get_XY(i)
		if boundaryType == "x_0" or boundaryType == "x_0_y_0" or boundaryType == "x_0_y_Ly":
			x_b0[i]=1.0
			x_b1[i]=int(x_0_dirichlet is True)
		elif boundaryType == "x_Lx" or boundaryType == "x_Lx_y_0" or boundaryType == "x_Lx_y_Ly":
			x_b2[i]=1.0
			x_b3[i]=int(x_Lx_dirichlet is True)
		else:
			x_b4[i]=1.0
	return (x_b0,x_b1,x_b2,x_b3,x_b4)

# Returns five arrays of booleans with the boundary decisions in the y direction
@njit(parallel=True)
def get_y_decisions(y_b0,y_b1,y_b2,y_b3,y_b4):
	for i in prange(0,num_nodes):
		boundaryType=getBoundaryType(i)
		if boundaryType == "y_0" or boundaryType == "x_0_y_0" or boundaryType == "x_Lx_y_0":
			y_b0[i]=1.0
			y_b1[i]=int(y_0_dirichlet is True)
		elif boundaryType == "y_Ly" or boundaryType == "x_0_y_Ly" or boundaryType == "x_Lx_y_Ly":
			y_b2[i]=1.0
			y_b3[i]=int(y_Ly_dirichlet is True)
		else:
			y_b4[i]=1.0
	return (y_b0,y_b1,y_b2,y_b3,y_b4)

u_r_0=0.0
if abs(b_x_0[1]) > 1e-8:
	u_r_0=(1+deltaX*b_x_0[0]/b_x_0[1])
u_r_Lx=0.0
if abs(b_x_Lx[1]) > 1e-8:
	u_r_Lx=(1-deltaX*b_x_Lx[0]/b_x_Lx[1])
u_m=deltaT/(2.0*rho*deltaX)

@njit(parallel=True)
def get_u_boundaries(u_x0,u_xL,x_b0,x_b1,x_b2,x_b3,x_b4):
	for i in prange(0,num_nodes):
		boundaryType=getBoundaryType(i)
		(x,y)=get_XY(i)
		if x_b0[i]:
			if x_b1[i]:
				u_x0[i]=g_x_0(x,y)/b_x_0[0]
			else:
				u_x0[i]=(-deltaX*g_x_0(x,y)/b_x_0[1])
		elif x_b2[i]:
			if x_b3[i]:
				u_xL[i]=g_x_Lx(x,y)/b_x_Lx[0]
			else:
				u_xL[i]=(deltaX*g_x_Lx(x,y)/b_x_Lx[1])
	return (u_x0,u_xL)

# updates u
@njit(parallel=True)
def get_u(u,p,div_u,u_x0,u_xL,x_b0,x_b1,x_b2,x_b3,x_b4):
	for i in prange(0,num_nodes):
		pxm=0.0
		pxp=0.0
		if x_b4[i]:
			u[i]=(u[i]+u_m*((p[i-1])-(p[i+1])))*div_u[i]
		elif x_b0[i]:
			u[i]=(u[i]+u_m*((u_x0[i]+x_b1[i]*u_r_0*p[i])-(u[i+1])))*div_u[i]
		elif x_b2[i]:
			u[i]=(u[i]+u_m*((u[i-1])-(u_xL[i]+x_b2[i]*u_r_Lx*p[i])))*div_u[i]
		else:
			print("unknown error")
	return u

v_r_y0=0.0
if b_y_0[1] > 1e-8:
	v_r_y0=(1+deltaY*b_y_0[0]/b_y_0[1])
v_r_Ly=0.0
if b_y_Ly[1] > 1e-8:
	v_r_Ly=(1-deltaY*b_y_Ly[0]/b_y_Ly[1])
v_m=deltaT/(2*rho*deltaY)

@njit(parallel=True)
def get_v_boundaries(v_y0,v_yL,y_b0,y_b1,y_b2,y_b3,y_b4):
	for i in prange(0,num_nodes):
		(x,y)=get_XY(i)
		if y_b0[i]:
			if y_b1[i]:
				v_y0[i]=g_y_0(x,y)/b_y_0[0]
			else:
				v_y0[i]=(-deltaY*g_y_0(x,y)/b_y_0[1])
		elif y_b2[i]:
			if y_b3[i]:
				v_yL[i]=g_y_Ly(x,y)/b_y_Ly[0]
			else:
				v_yL[i]=(deltaY*g_y_Ly(x,y)/b_y_Ly[1])
	return (v_y0,v_yL)

# updates v
@njit(parallel=True)
def get_v(v,p,div_v,v_y0,v_yL,y_b0,y_b1,y_b2,y_b3,y_b4):
	for i in prange(0,num_nodes):
		pym=0.0
		pyp=0.0
		if y_b4[i] == 1.0:
			v[i]=(v[i]+v_m*((p[i-N])-(p[i+N])))*div_v[i]
		elif y_b0[i] == 1.0:
			v[i]=(v[i]+v_m*((v_y0[i]+y_b1[i]*v_r_y0*p[i])-(p[i+N])))*div_v[i]
		elif y_b2[i] == 1.0:
			v[i]=(v[i]+v_m*((p[i-N])-(v_yL[i]+y_b3[i]*v_r_Ly*p[i])))*div_v[i]
		else:
			print("Unknown error")
	return v


px_r_0=0
if b_x_0[1] > 1e-8:
	px_r_0=(1.0+b_x_0[0]*deltaX/b_x_0[1])
px_r_Lx=0
if b_x_Lx[1] > 1e-8:
	px_r_Lx=(1.0-b_x_Lx[0]*deltaX/b_x_Lx[1])
px_m=rho*c*c*deltaT/(2*deltaX)

@njit(parallel=True)
def get_p_x_boundaries(p_x_x0,p_x_xL,x_b0,x_b1,x_b2,x_b3,x_b4):
	for i in prange(0,num_nodes):
		(x,y)=get_XY(i)
		if x_b0[i]:
			if x_b1[i]:
				p_x_x0[i]=g_x_0(x,y)/b_x_0[0]
			else:
				p_x_x0[i]=((-g_x_0(x,y)*deltaX/b_x_0[1]))
		elif x_b2[i]:
			if x_b3[i]:
				p_x_xL[i]=g_x_Lx(x,y)/b_x_Lx[0]
			else:
				p_x_xL[i]=((g_x_Lx(x,y)*deltaX/b_x_Lx[1]))
	return (p_x_x0,p_x_xL)

# updates p_x
@njit(parallel=True)
def get_p_x(p,u,t,div_p_x,p_x_x0,p_x_xL,x_b0,x_b1,x_b2,x_b3,x_b4):
	for i in prange(0,num_nodes):
		boundaryType=getBoundaryType(i)
		(x,y)=get_XY(i)
		uxm=0
		uxp=0
		if x_b4[i]:
			p[i]=(p[i]+px_m*((u[i-1])-(u[i+1]))+halfDeltaT*q(x,y,t))*div_p_x[i]
		elif x_b0[i]:
			p[i]=(p[i]+px_m*((p_x_x0[i]+x_b1[i]*px_r_0*p[i])-(u[i+1]))+halfDeltaT*q(x,y,t))*div_p_x[i]
		elif x_b2[i]:
			p[i]=(p[i]+px_m*((u[i-1])-(p_x_xL[i]+x_b3[i]*px_r_Lx*p[i]))+halfDeltaT*q(x,y,t))*div_p_x[i]
		else:
			print("unknown error")
	return p

py_r_0=0
if b_y_0[1] > 1e-8:
	py_r_0=(1.0+b_y_0[0]*deltaY/b_y_0[1])
py_r_L=0
if b_y_Ly[1] > 1e-8:
	py_r_L=(1.0-b_y_Ly[0]*deltaY/b_y_Ly[1])
py_m=rho*c*c*deltaT/(2*deltaY)

@njit(parallel=True)
def get_p_y_boundaries(p_y_y0,p_y_yL,y_b0,y_b1,y_b2,y_b3,y_b4):
	for i in prange(0,num_nodes):
		(x,y)=get_XY(i)
		if y_b0[i]:
			if y_b1[i]:
				p_y_y0[i]=g_y_0(x,y)/b_y_0[0]
			else:
				p_y_y0[i]=(-g_y_0(x,y)*deltaY/b_y_0[1])
		elif y_b2[i]:
			if y_b3[i]:
				p_y_yL[i]=g_y_Ly(x,y)/b_y_Ly[0]
			else:
				p_y_yL[i]=(g_y_Ly(x,y)*deltaY/b_y_Ly[1])
	return (p_y_y0,p_y_yL)

# updates p_y
@njit(parallel=True)
def get_p_y(p,v,t,div_p_y,p_y_y0,p_y_yL,y_b0,y_b1,y_b2,y_b3,y_b4):
	for i in prange(0,num_nodes):
		boundaryType=getBoundaryType(i)
		(x,y)=get_XY(i)
		vym=0
		vyp=0
		if y_b4[i]:
			p[i]=(p[i]+py_m*((v[i-N])-(v[i+N]))+halfDeltaT*q(x,y,t))*div_p_y[i]
		if y_b0[i]:
			p[i]=(p[i]+py_m*((p_y_y0[i]+y_b1[i]*py_r_0*p[i])-(v[i+N]))+halfDeltaT*q(x,y,t))*div_p_y[i]
		elif y_b2[i]:
			p[i]=(p[i]+py_m*((v[i-N])-(p_y_yL[i]+y_b3[i]*py_r_L*p[i]))+halfDeltaT*q(x,y,t))*div_p_y[i]
		else:
			print("unknown error")
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
	py=0.5*get_p0(0,0)
	u=get_u0(0,0)
	v=get_v0(0,0)

	t=0
	next_image=save_interval
	num_images=int(endTime/save_interval)+1
	images=[None]*num_images
	timestamp=[0]*num_images
	images[0]=px+py
	image_counter=1

	# Calculate these values just once
	verbosemsg("Preliminary calculations...")

	sigma_v=np.zeros(num_nodes,dtype=float)
	sigma_v=getSigma_v(sigma_v)
	nu_v=np.zeros(num_nodes,dtype=float)
	nu_v=getNu_v(nu_v)
	div_u=np.zeros(num_nodes,dtype=float)
	div_v=np.zeros(num_nodes,dtype=float)
	div_p_x=np.zeros(num_nodes,dtype=float)
	div_p_y=np.zeros(num_nodes,dtype=float)
	(div_u,div_v,div_p_x,div_p_y)=get_divisors(div_u,div_v,div_p_x,div_p_y,sigma_v,nu_v)

	x_b0=np.zeros(num_nodes,dtype=float)
	x_b1=np.zeros(num_nodes,dtype=float)
	x_b2=np.zeros(num_nodes,dtype=float)
	x_b3=np.zeros(num_nodes,dtype=float)
	x_b4=np.zeros(num_nodes,dtype=float)
	y_b0=np.zeros(num_nodes,dtype=float)
	y_b1=np.zeros(num_nodes,dtype=float)
	y_b2=np.zeros(num_nodes,dtype=float)
	y_b3=np.zeros(num_nodes,dtype=float)
	y_b4=np.zeros(num_nodes,dtype=float)
	(x_b0,x_b1,x_b2,x_b3,x_b4)=get_x_decisions(x_b0,x_b1,x_b2,x_b3,x_b4)
	(y_b0,y_b1,y_b2,y_b3,y_b4)=get_y_decisions(y_b0,y_b1,y_b2,y_b3,y_b4)

	u_x0=np.zeros(num_nodes,dtype=float)
	u_xL=np.zeros(num_nodes,dtype=float)
	v_y0=np.zeros(num_nodes,dtype=float)
	v_yL=np.zeros(num_nodes,dtype=float)
	p_x_x0=np.zeros(num_nodes,dtype=float)
	p_x_xL=np.zeros(num_nodes,dtype=float)
	p_y_y0=np.zeros(num_nodes,dtype=float)
	p_y_yL=np.zeros(num_nodes,dtype=float)
	(u_x0,u_xL)=get_u_boundaries(u_x0,u_xL,x_b0,x_b1,x_b2,x_b3,x_b4)
	(v_y0,v_yL)=get_v_boundaries(v_y0,v_yL,y_b0,y_b1,y_b2,y_b3,y_b4)
	(p_x_x0,p_x_xL)=get_p_x_boundaries(p_x_x0,p_x_xL,x_b0,x_b1,x_b2,x_b3,x_b4)
	(p_y_y0,p_y_yL)=get_p_y_boundaries(p_y_y0,p_y_yL,y_b0,y_b1,y_b2,y_b3,y_b4)

	verbosemsg("Working out p(x,y)")
	while t < endTime:
		# find the next solution
		t+=deltaT
		pxpy=px+py
		u=get_u(u,pxpy,div_u,u_x0,u_xL,x_b0,x_b1,x_b2,x_b3,x_b4)
		v=get_v(v,pxpy,div_v,v_y0,v_yL,y_b0,y_b1,y_b2,y_b3,y_b4)
		px=get_p_x(px,u,t,div_p_x,p_x_x0,p_x_xL,x_b0,x_b1,x_b2,x_b3,x_b4)
		py=get_p_y(py,v,t,div_p_y,p_y_y0,p_y_yL,y_b0,y_b1,y_b2,y_b3,y_b4)

        # save a picture
		if abs(t - next_image) < 0.001:
			images[image_counter]=np.reshape(pxpy,[M,N])
			# images[image_counter]=u+v
			timestamp[image_counter]=t
			image_counter+=1
			next_image+=save_interval
	
	verbosemsg("Saving the images")
	for i in range(0,num_images):
		saveImage(images[i],timestamp[i])

def main():
	if (c >= deltaX / deltaT):
		print("Warning: Recommend that you decrease deltaX")
	if (c >= deltaY / deltaT):
		print("Warning: Recommend that you decrease deltaY")
	temporalLoop()

if __name__ == '__main__':
	main()
