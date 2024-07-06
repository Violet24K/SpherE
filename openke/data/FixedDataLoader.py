# coding:utf-8
import os
import ctypes
import numpy as np
import pdb

class FixedDataSampler(object):

	def __init__(self, nbatches, batch_h, batch_t, batch_r, batch_y, samples_each_batch):
		self.nbatches = nbatches
		self.batch_h = batch_h
		self.batch_t = batch_t
		self.batch_r = batch_r
		self.batch_y = batch_y
		self.samples_each_batch = samples_each_batch
		self.batch = 0

	def __iter__(self):
		return self

	def __next__(self):
		batch_index = self.batch
		self.batch += 1
		if self.batch > self.nbatches:
			raise StopIteration()
		return {'batch_h': self.batch_h[batch_index* self.samples_each_batch: (batch_index+1) * self.samples_each_batch],
		  		'batch_t': self.batch_t[batch_index* self.samples_each_batch: (batch_index+1) * self.samples_each_batch],
				'batch_r': self.batch_r[batch_index* self.samples_each_batch: (batch_index+1) * self.samples_each_batch],
				'batch_y': self.batch_y[batch_index* self.samples_each_batch: (batch_index+1) * self.samples_each_batch],
				'mode': 'normal'}

	def __len__(self):
		return self.nbatches

class FixedDataLoader(object):
	def __init__(self, 
		positive_path = "./",
		negative_path = "./",
		batch_size = None,
		nbatches = None,
		sampling_mode = "normal",
		neg_ent = 1):

		self.positive_path = positive_path
		self.negative_path = negative_path
		self.batch_size = batch_size
		self.nbatches = nbatches
		self.sampling_mode = sampling_mode
		self.neg_ent = neg_ent
		self.samples_each_batch = self.batch_size * (self.neg_ent+1)
		
		self.batch_h = np.zeros((self.batch_size * (self.neg_ent+1) * self.nbatches,), dtype = int)
		self.batch_t = np.zeros((self.batch_size * (self.neg_ent+1) * self.nbatches,), dtype = int)
		self.batch_r = np.zeros((self.batch_size * (self.neg_ent+1) * self.nbatches,), dtype = int)
		self.batch_y = np.zeros((self.batch_size * (self.neg_ent+1) * self.nbatches,), dtype = np.float32)
		self.batch_counter = 0
		positive_triple_file = open(positive_path, "r", encoding='utf-8')
		negative_triple_file = open(negative_path, "r", encoding='utf-8')
		counter = 0
		for i in range(nbatches):	# for each batch, first set positive samples, then set negative samples
			# set positive samples
			for j in range(batch_size):
				content = positive_triple_file.readline()
				h, t, r = content.strip().split()
				h = int(h)
				t = int(t)
				r = int(r)
				self.batch_h[counter] = h
				self.batch_t[counter] = t
				self.batch_r[counter] = r
				self.batch_y[counter] = 1
				counter += 1
			# set negative samples
			for j in range(batch_size*25):
				content = negative_triple_file.readline()
				h, t, r = content.strip().split()
				h = int(h)
				t = int(t)
				r = int(r)
				self.batch_h[counter] = h
				self.batch_t[counter] = t
				self.batch_r[counter] = r
				self.batch_y[counter] = -1
				counter += 1

		positive_triple_file.close()
		negative_triple_file.close()


	def __iter__(self):
		if self.sampling_mode == "normal":
			return FixedDataSampler(self.nbatches, self.batch_h, self.batch_t, self.batch_r, self.batch_y, self.samples_each_batch)
		else:
			raise NotImplementedError
		
	def __len__(self):
		return self.nbatches
