# Step1: Extract All the Positive and Negative Samples from the TrainDataLoader. The random functions in the C++ files are actually fake randoms.
import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import pdb

dataset_name = 'FB15K237'
nbatches = 100

print("Are you sure to OVERWRITE the existing files? If first run, type c then enter")
pdb.set_trace()

train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/" + dataset_name + "/", 
	nbatches = nbatches,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

f_positive = open(dataset_name + "_positive_samples.txt", 'w')
f_negative = open(dataset_name + "_negative_samples.txt", 'w')

for data in train_dataloader:
    n_samples_per_batch = len(data['batch_h'])
    n_positive_samples_per_batch = int(n_samples_per_batch/(train_dataloader.negative_ent+1))
    n_negative_samples_per_batch = int(n_samples_per_batch - n_positive_samples_per_batch)
    for i in range(n_positive_samples_per_batch):
        pos_h = data['batch_h'][i]
        pos_t = data['batch_t'][i]
        pos_r = data['batch_r'][i]
        pos_y = data['batch_y'][i]
        if pos_y != 1:
            print("Warning: This is not a positive sample!")
        f_positive.write(str(pos_h) + ' ' + str(pos_t) + ' ' + str(pos_r) + '\n')
    

    for i in range(n_positive_samples_per_batch, n_samples_per_batch):
        neg_h = data['batch_h'][i]
        neg_t = data['batch_t'][i]
        neg_r = data['batch_r'][i]
        neg_y = data['batch_y'][i]
        if neg_y != -1:
            print("Warning: This is not a negative sample!")
        f_negative.write(str(neg_h) + ' ' + str(neg_t) + ' ' + str(neg_r) + '\n')
        
f_positive.close()
f_negative.close()