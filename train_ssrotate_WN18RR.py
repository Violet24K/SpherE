import openke
from openke.config import Trainer, Tester
from openke.module.model import SSRotatE
from openke.module.loss import SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader, FixedDataLoader, FixedTestDataLoader
import openke
from openke.config import Trainer, Tester, TesterForRetrievalTest
import pdb
import torch
import os.path as osp
import argparse


def main(args):
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path = "./benchmarks/WN18RR/", 
        batch_size = 2000,
        threads = 8,
        sampling_mode = "normal", 
        bern_flag = 0, 
        filter_flag = 1, 
        neg_ent = 64,
        neg_rel = 0
    )

    # define the model
    rotate = SSRotatE(
        ent_tot = train_dataloader.get_ent_tot(),
        rel_tot = train_dataloader.get_rel_tot(),
        dim = 1024,
        margin = 6.0,
        epsilon = 2.0,
        tolerance=0.1
    )

    # define the loss function
    model = NegativeSampling(
        model = rotate, 
        loss = SigmoidLoss(adv_temperature = 2),
        batch_size = train_dataloader.get_batch_size(), 
        regul_rate = 0.0
    )

    if not args.test:
        # train the model
        trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 300, alpha = 2e-5, use_gpu = True, opt_method = "adam")
        trainer.run()
        rotate.save_checkpoint('./checkpoint/SSrotate_WN18RR.ckpt')


    dataset_name = 'WN18RR'
    # test the model
    rotate.load_checkpoint('./checkpoint/SSrotate_WN18RR.ckpt')
    # get number of entities
    entity2id_file = open("./benchmarks/" + dataset_name + "/entity2id.txt", 'r', encoding='utf-8')
    nentities = (int)(entity2id_file.readline())
    entity2id_file.close()

    # tester = TesterForRetrievalTest(model = rotate, data_loader = test_dataloader, use_gpu = True)
    specific_test_dataloader_all = FixedTestDataLoader("./benchmarks/" + dataset_name + "/test2id.txt", "link", nentities)
    specific_test_dataloader_11 = FixedTestDataLoader("./benchmarks/" + dataset_name + "/1-1.txt", "link", nentities)
    specific_test_dataloader_1n = FixedTestDataLoader("./benchmarks/" + dataset_name + "/1-n.txt", "link", nentities)
    specific_test_dataloader_n1 = FixedTestDataLoader("./benchmarks/" + dataset_name + "/n-1.txt", "link", nentities)
    specific_test_dataloader_nn = FixedTestDataLoader("./benchmarks/" + dataset_name + "/n-n.txt", "link", nentities)

    # load the dataset
    actual_head = {}    # actual_head[(r, t)] is the actual head of ?->r->t
    actual_tail = {}    # actual_tail[(h, r)] is the actual tail of h->r->?
    relation2id = open(osp.join("benchmarks", dataset_name, "relation2id.txt"), "r",encoding='utf-8')
    num_relations = int(relation2id.readline().strip())
    entity2id = open(osp.join("benchmarks", dataset_name, "entity2id.txt"), "r",encoding='utf-8')
    num_entities = int(entity2id.readline().strip())

    relation2id.close()
    entity2id.close()

    triple = open(osp.join("benchmarks", dataset_name, "train2id.txt"), "r",encoding='utf-8')
    tot = (int)(triple.readline())
    for i in range(tot):
        content = triple.readline()
        h, t, r = content.strip().split()
        h = int(h)
        t = int(t)
        r = int(r)
        if not (h, r) in actual_tail:
            actual_tail[(h, r)] = []
        if not (r, t) in actual_head:
            actual_head[(r, t)] = []
        actual_tail[(h, r)].append(t)
        actual_head[(r, t)].append(h)
    triple.close()

    valid = open(osp.join("benchmarks", dataset_name, "valid2id.txt"), "r",encoding='utf-8')
    tot = (int)(valid.readline())
    for i in range(tot):
        content = valid.readline()
        h, t, r = content.strip().split()
        h = int(h)
        t = int(t)
        r = int(r)
        if not (h, r) in actual_tail:
            actual_tail[(h, r)] = []
        if not (r, t) in actual_head:
            actual_head[(r, t)] = []
        actual_tail[(h, r)].append(t)
        actual_head[(r, t)].append(h)
    valid.close()
        
    test = open(osp.join("benchmarks", dataset_name, "test2id.txt"), "r",encoding='utf-8')
    tot = (int)(test.readline())
    for i in range(tot):
        content = test.readline()
        h, t, r = content.strip().split()
        h = int(h)
        t = int(t)
        r = int(r)
        if not (h, r) in actual_tail:
            actual_tail[(h, r)] = []
        if not (r, t) in actual_head:
            actual_head[(r, t)] = []
        actual_tail[(h, r)].append(t)
        actual_head[(r, t)].append(h)
    test.close()

    tester_all = TesterForRetrievalTest(model = rotate, data_loader = specific_test_dataloader_all, use_gpu = True, actual_head = actual_head, actual_tail = actual_tail)
    tester_11 = TesterForRetrievalTest(model = rotate, data_loader = specific_test_dataloader_11, use_gpu = True, actual_head = actual_head, actual_tail = actual_tail)
    tester_1n = TesterForRetrievalTest(model = rotate, data_loader = specific_test_dataloader_1n, use_gpu = True, actual_head = actual_head, actual_tail = actual_tail)
    tester_n1 = TesterForRetrievalTest(model = rotate, data_loader = specific_test_dataloader_n1, use_gpu = True, actual_head = actual_head, actual_tail = actual_tail)
    tester_nn = TesterForRetrievalTest(model = rotate, data_loader = specific_test_dataloader_nn, use_gpu = True, actual_head = actual_head, actual_tail = actual_tail)

    print("Test on all relationship")
    tester_all.run_link_prediction(type_constrain = False)
    print("Test on 1-1 relationship")
    tester_11.run_link_prediction(type_constrain = False)
    print("Test on 1-n relationship")
    tester_1n.run_link_prediction(type_constrain = False)
    print("Test on n-1 relationship")
    tester_n1.run_link_prediction(type_constrain = False)
    print("Test on n-n relationship")
    tester_nn.run_link_prediction(type_constrain = False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    main(args)