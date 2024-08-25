#Turns the output of nbody.cpp into images

import numpy as np
import matplotlib.pyplot as plt

FILENAME='nbody.dat'

MINMAX=1
xmin=-MINMAX
xmax= MINMAX
ymin=-MINMAX
ymax= MINMAX

f = open(FILENAME,'r')
lines=f.readlines()

num_planets=int(lines[0].split('\n')[0])
positions=np.zeros((3,num_planets))
first_loop = True

for line in lines:
	# Skip line 1
	if line == lines[0]:
		continue

	tmp=line.split(":")
	if tmp[0] == 't':
		# Take down new time
		t=tmp[1].split('\n')[0]

		# Create the image from the previous timestep
		if first_loop:
			first_loop=False
			continue
		else:
			filename="images/nbody."+t+".png"
			title_text="t = " + str(t)
			for i in range(0,num_planets):
				plt.scatter(positions[0][i],positions[1][i])
			plt.title(title_text)
			axes = plt.gca()
			axes.set_xlim([xmin,xmax])
			axes.set_ylim([ymin,ymax])
			plt.savefig(filename)
			plt.close()
	else:
		n=int(tmp[0])
		x=tmp[1].split(',')
		positions[0][n]=x[0]
		positions[1][n]=x[1]
		positions[2][n]=x[2].split('\n')[0]


