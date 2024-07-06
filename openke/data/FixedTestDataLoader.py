# coding:utf-8
import numpy as np
import pdb

class FixedTestDataSampler(object):

	def __init__(self, nbatches, batch_h, batch_t, batch_r, nentities):
		self.nbatches = nbatches
		self.batch_h = batch_h
		self.batch_t = batch_t
		self.batch_r = batch_r
		self.nentities = nentities
		self.batch = 0

	def __iter__(self):
		return self

	def __next__(self):
		batch_index = self.batch
		self.batch += 1
		if self.batch > self.nbatches:
			raise StopIteration()
		return [{'batch_h': np.arange(0, self.nentities, 1, dtype=int),
		  		'batch_t': np.array([self.batch_t[batch_index]]),
				'batch_r': np.array([self.batch_r[batch_index]]),
				'mode': 'head_batch'}, 
				{'batch_h': np.array([self.batch_h[batch_index]]),
		  		'batch_t': np.arange(0, self.nentities, 1, dtype=int),
				'batch_r': np.array([self.batch_r[batch_index]]),
				'mode': 'tail_batch'}]

	def __len__(self):
		return self.nbatches

class FixedTestDataLoader(object):
	def __init__(self, 
		test_file_path = "./", sampling_mode = 'link', nentities = 0):

		self.test_file_path = test_file_path
		self.sampling_mode = sampling_mode
		self.nentities = nentities
		ftest = open(self.test_file_path, "r", encoding='utf-8')
		lines = ftest.readlines()[1:]
		ftest.close()
		self.nbatches = len(lines)


		self.batch_h = np.zeros((self.nbatches,), dtype = int)
		self.batch_t = np.zeros((self.nbatches,), dtype = int)
		self.batch_r = np.zeros((self.nbatches,), dtype = int)
		self.batch_counter = 0
		counter = 0
		for i in range(self.nbatches):	# for each batch, first set positive samples, then set negative samples
			h, t, r = lines[counter].strip().split()
			h = int(h)
			t = int(t)
			r = int(r)
			self.batch_h[counter] = h
			self.batch_t[counter] = t
			self.batch_r[counter] = r
			counter += 1

	def get_ent_tot(self):
		return self.nentities

	def get_triple_tot(self):
		return self.nbatches

	def set_sampling_mode(self, sampling_mode):
		self.sampling_mode = sampling_mode

	def __iter__(self):
		if self.sampling_mode == "link":
			return FixedTestDataSampler(self.nbatches, self.batch_h, self.batch_t, self.batch_r, self.nentities)
		else:
			raise NotImplementedError
		
	def __len__(self):
		return self.nbatches
