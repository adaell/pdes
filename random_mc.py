# A collections of programs used by monte carlo algorithms

# A structure that generates a large number of random numbers at a time.
# A total of num_random random numbers are generated such that
# R in [min,max]

# requires numpy

import numpy as np

# Generates integers on [min,max]
class dicecup_int:
	def __init__(self,num_random,_min,_max):
		self.R_counter=0
		self.R_vec=np.random.randint(int(_min),int(_max),size=num_random)+_min*np.ones(num_random,dtype='int')+np.ones(num_random,dtype='int')
		self._min=_min
		self._max=_max
		self.num_random=num_random

	def setNewMax(self,_max):
		self.R_counter=0
		self._max=_max
		self.regenerateR()

	# regenerate the random number vector
	def regenerateR(self):
		self.R_vec=np.random.randint(int(self._min),int(self._max),size=self.num_random)+self._min*np.ones(self.num_random,dtype='int')+np.ones(self.num_random,dtype='int')

	# returns a random number from [0,1]
	def get_R(self):
		self.R_counter=self.R_counter+1
		if self.R_counter == self.num_random:
			self.R_counter=0
			self.regenerateR()
		return self.R_vec[self.R_counter]

# Generates floats on [min,max]
class dicecup_float:
	def __init__(self,num_random,_min,_max):
		self.R_counter=0
		self.R_vec=np.random.uniform(_min,_max,num_random)+_min*np.ones(num_random,dtype='float32')
		self._min=_min
		self._max=_max
		self.num_random=num_random

	# regenerate the random number vector
	def regenerateR(self):
		self.R_vec=np.random.uniform(self._min,self._max,self.num_random)+self._min*np.ones(self.num_random,dtype='float32')

	# returns a random number from [0,1]
	def get_R(self):
		self.R_counter+=1
		if self.R_counter == self.num_random:
			self.R_counter=0
			self.regenerateR()
		return self.R_vec[self.R_counter]