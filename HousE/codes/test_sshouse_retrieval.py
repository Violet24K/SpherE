import os
import os.path as osp
import sys
import argparse
import numpy as np
import pdb
from model import KGEModel
import json
import torch
import logging

from torch.utils.data import DataLoader
from dataloader import TestDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

file_path = osp.abspath(__file__)
parent = osp.dirname(osp.dirname(file_path))
folder_name = 'HousE_r_FB15k-237_0'

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def calc_f1(list1, list2):
    TP = len(set(list1) & set(list2))
    if TP == 0:
        f1 = 0
    else:
        precision = TP/len(list1)
        recall = TP/len(list2)
        f1 = (2*precision*recall)/(precision+recall)
    return f1

def test_model(model, test_triples, all_true_triples, args, nentity, nrelation, test_batch_size, cuda, cpu_num, actual_head, actual_tail):
    test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    nentity, 
                    nrelation, 
                    'head-batch'
                ), 
                batch_size=test_batch_size,
                num_workers=max(1, cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
    
    test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    nentity, 
                    nrelation, 
                    'tail-batch'
                ), 
                batch_size=test_batch_size,
                num_workers=max(1, cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
    
    test_dataset_list = [test_dataloader_head, test_dataloader_tail]       
    logs = []
    step = 0
    total_steps = sum([len(dataset) for dataset in test_dataset_list])

    # test SSHousE_r
    if hasattr(model, 'entity_radius'):
        head_retrieved = []
        head_f1s = []
        tail_retrieved = []
        tail_f1s = []
        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                    if cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()
                        filter_bias = filter_bias.cuda()

                    batch_size = positive_sample.size(0)

                    score = model.predict((positive_sample, negative_sample), mode)
                    score += filter_bias

                    #Explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(score, dim = 1, descending=True)

                    if mode == 'head-batch':    # predict head
                        positive_arg = positive_sample[:, 0]
                    elif mode == 'tail-batch':  # predict tail
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        entity_score = score[i]
                        if mode == 'head-batch':
                            rh = torch.where(entity_score<=0)[0]
                            retrieved_heads = list(rh.cpu().numpy())
                            actual_heads = actual_head[(positive_sample[i][1].item(), positive_sample[i][2].item())]
                            if positive_arg[i].item() in retrieved_heads:
                                head_retrieved.append(1)
                            else:
                                head_retrieved.append(0)
                            head_f1s.append(calc_f1(retrieved_heads, actual_heads))
                        if mode == 'tail-batch':
                            rt = torch.where(entity_score<=0)[0]
                            retrieved_tails = list(rt.cpu().numpy())
                            actual_tails = actual_tail[(positive_sample[i][0].item(), positive_sample[i][1].item())]
                            if positive_arg[i].item() in retrieved_tails:
                                tail_retrieved.append(1)
                            else:
                                tail_retrieved.append(0)
                            tail_f1s.append(calc_f1(retrieved_tails, actual_tails))
        print("Average Retrieved Rate on Head set:", np.mean(head_retrieved))
        print("Average Retrieved Rate on Tail set:", np.mean(tail_retrieved))
        print("Average F1 on head set:", np.mean(head_f1s))
        print("Average F1 on tail set:", np.mean(tail_f1s))
    # test HousE_r 
    else:        
        print('heredddddddddddddddddddddddddddddddddddddd')
        # top 1 
        head_retrieved_1 = []
        head_f1s_1 = []
        tail_retrieved_1 = []
        tail_f1s_1 = []
        # top 3
        head_retrieved_3 = []
        head_f1s_3 = []
        tail_retrieved_3 = []
        tail_f1s_3 = []
        # top 5 
        head_retrieved_5 = []
        head_f1s_5 = []
        tail_retrieved_5 = []
        tail_f1s_5 = []
        # top 10
        head_retrieved_10 = []
        head_f1s_10 = []
        tail_retrieved_10 = []
        tail_f1s_10 = []
        # top 30
        head_retrieved_30 = []
        head_f1s_30 = []
        tail_retrieved_30 = []
        tail_f1s_30 = []
        # top 50
        head_retrieved_50 = []
        head_f1s_50 = []
        tail_retrieved_50 = []
        tail_f1s_50 = []
        # top 70
        head_retrieved_70 = []
        head_f1s_70 = []
        tail_retrieved_70 = []
        tail_f1s_70 = []
        # top 100
        head_retrieved_100 = []
        head_f1s_100 = []
        tail_retrieved_100 = []
        tail_f1s_100 = []
        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                    if cuda:
                        positive_sample = positive_sample.cuda()    # torch.Size([16, 3])
                        negative_sample = negative_sample.cuda()    # torch.Size([16, 14541])
                        filter_bias = filter_bias.cuda()

                    batch_size = positive_sample.size(0)    #16

                    score = model((positive_sample, negative_sample), mode)
                    score += filter_bias    # torch.Size([16, 14541])
                    #Explicitly sort all the entities to ensure that there is no test exposure bias
                        # larger score means closer distance!
                        # positive_arg means right answer??
                    argsort = torch.argsort(score, dim = 1, descending=True)
                    if mode == 'head-batch':    # predict head
                        positive_arg = positive_sample[:, 0]
                    elif mode == 'tail-batch':  # predict tail
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        entity_score = score[i]
                        if mode == 'head-batch':
                            actual_heads = actual_head[(positive_sample[i][1].item(), positive_sample[i][2].item())]
                            # top 1 
                            rh_1 = argsort[i][:1]
                            retrieved_heads_1 = list(rh_1.cpu().numpy())
                            if positive_arg[i].item() in retrieved_heads_1:
                                head_retrieved_1.append(1)
                            else:
                                head_retrieved_1.append(0)
                            head_f1s_1.append(calc_f1(retrieved_heads_1, actual_heads))
                            # top 3
                            rh_3 = argsort[i][:3]
                            retrieved_heads_3 = list(rh_3.cpu().numpy())
                            if positive_arg[i].item() in retrieved_heads_3:
                                head_retrieved_3.append(1)
                            else:
                                head_retrieved_3.append(0)
                            head_f1s_3.append(calc_f1(retrieved_heads_3, actual_heads))
                            # top 5
                            rh_5 = argsort[i][:5]
                            retrieved_heads_5 = list(rh_5.cpu().numpy())
                            if positive_arg[i].item() in retrieved_heads_5:
                                head_retrieved_5.append(1)
                            else:
                                head_retrieved_5.append(0)
                            head_f1s_5.append(calc_f1(retrieved_heads_5, actual_heads))
                            # top 10
                            rh_10 = argsort[i][:10]
                            retrieved_heads_10 = list(rh_10.cpu().numpy())
                            if positive_arg[i].item() in retrieved_heads_10:
                                head_retrieved_10.append(1)
                            else:
                                head_retrieved_10.append(0)
                            head_f1s_10.append(calc_f1(retrieved_heads_10, actual_heads))
                            # top 30
                            rh_30 = argsort[i][:30]
                            retrieved_heads_30 = list(rh_30.cpu().numpy())
                            if positive_arg[i].item() in retrieved_heads_30:
                                head_retrieved_30.append(1)
                            else:
                                head_retrieved_30.append(0)
                            head_f1s_30.append(calc_f1(retrieved_heads_30, actual_heads))
                            # top 50
                            rh_50 = argsort[i][:50]
                            retrieved_heads_50 = list(rh_50.cpu().numpy())
                            if positive_arg[i].item() in retrieved_heads_50:
                                head_retrieved_50.append(1)
                            else:
                                head_retrieved_50.append(0)
                            head_f1s_50.append(calc_f1(retrieved_heads_50, actual_heads))
                            # top 70
                            rh_70 = argsort[i][:70]
                            retrieved_heads_70 = list(rh_70.cpu().numpy())
                            if positive_arg[i].item() in retrieved_heads_70:
                                head_retrieved_70.append(1)
                            else:
                                head_retrieved_70.append(0)
                            head_f1s_70.append(calc_f1(retrieved_heads_70, actual_heads))
                            # top 100
                            rh_100 = argsort[i][:100]
                            retrieved_heads_100 = list(rh_100.cpu().numpy())
                            if positive_arg[i].item() in retrieved_heads_100:
                                head_retrieved_100.append(1)
                            else:
                                head_retrieved_100.append(0)
                            head_f1s_100.append(calc_f1(retrieved_heads_100, actual_heads))
                        if mode == 'tail-batch':
                            actual_tails = actual_tail[(positive_sample[i][0].item(), positive_sample[i][1].item())]
                            # top 1
                            rt_1 = argsort[i][:1]
                            retrieved_tails_1 = list(rt_1.cpu().numpy())
                            if positive_arg[i].item() in retrieved_tails_1:
                                tail_retrieved_1.append(1)
                            else:
                                tail_retrieved_1.append(0)
                            tail_f1s_1.append(calc_f1(retrieved_tails_1, actual_tails))
                            # top 3
                            rt_3 = argsort[i][:3]
                            retrieved_tails_3 = list(rt_3.cpu().numpy())
                            if positive_arg[i].item() in retrieved_tails_3:
                                tail_retrieved_3.append(1)
                            else:
                                tail_retrieved_3.append(0)
                            tail_f1s_3.append(calc_f1(retrieved_tails_3, actual_tails))
                            # top 5
                            rt_5 = argsort[i][:5]
                            retrieved_tails_5 = list(rt_5.cpu().numpy())
                            if positive_arg[i].item() in retrieved_tails_5:
                                tail_retrieved_5.append(1)
                            else:
                                tail_retrieved_5.append(0)
                            tail_f1s_5.append(calc_f1(retrieved_tails_5, actual_tails))
                            # top 10
                            rt_10 = argsort[i][:10]
                            retrieved_tails_10 = list(rt_10.cpu().numpy())
                            if positive_arg[i].item() in retrieved_tails_10:
                                tail_retrieved_10.append(1)
                            else:
                                tail_retrieved_10.append(0)
                            tail_f1s_10.append(calc_f1(retrieved_tails_10, actual_tails))
                            # top 30
                            rt_30 = argsort[i][:30]
                            retrieved_tails_30 = list(rt_30.cpu().numpy())
                            if positive_arg[i].item() in retrieved_tails_30:
                                tail_retrieved_30.append(1)
                            else:
                                tail_retrieved_30.append(0)
                            tail_f1s_30.append(calc_f1(retrieved_tails_30, actual_tails))
                            # top 50
                            rt_50 = argsort[i][:50]
                            retrieved_tails_50 = list(rt_50.cpu().numpy())
                            if positive_arg[i].item() in retrieved_tails_50:
                                tail_retrieved_50.append(1)
                            else:
                                tail_retrieved_50.append(0)
                            tail_f1s_50.append(calc_f1(retrieved_tails_50, actual_tails))
                            # top 70
                            rt_70 = argsort[i][:70]
                            retrieved_tails_70 = list(rt_70.cpu().numpy())
                            if positive_arg[i].item() in retrieved_tails_70:
                                tail_retrieved_70.append(1)
                            else:
                                tail_retrieved_70.append(0)
                            tail_f1s_70.append(calc_f1(retrieved_tails_70, actual_tails))
                            # top 100
                            rt_100 = argsort[i][:100]
                            retrieved_tails_100 = list(rt_100.cpu().numpy())
                            if positive_arg[i].item() in retrieved_tails_100:
                                tail_retrieved_100.append(1)
                            else:
                                tail_retrieved_100.append(0)
                            tail_f1s_100.append(calc_f1(retrieved_tails_100, actual_tails))
        # pdb.set_trace()
        # top 1
        print("Average Retrieved Rate on Head with top 1 set:", np.mean(head_retrieved_1))
        print("Average Retrieved Rate on tailk with top 1 set:", np.mean(tail_retrieved_1))
        print("Average F1 on head with top 1 set:", np.mean(head_f1s_1))
        print("Average F1 on tail with top 1 set:", np.mean(tail_f1s_1))
        print("")
        # top 3
        print("Average Retrieved Rate on Head with top 3 set:", np.mean(head_retrieved_3))
        print("Average Retrieved Rate on tailk with top 3 set:", np.mean(tail_retrieved_3))
        print("Average F1 on head with top 3 set:", np.mean(head_f1s_3))
        print("Average F1 on tail with top 3 set:", np.mean(tail_f1s_3))
        print("")
        # top 5
        print("Average Retrieved Rate on Head with top 5 set:", np.mean(head_retrieved_5))
        print("Average Retrieved Rate on tailk with top 5 set:", np.mean(tail_retrieved_5))
        print("Average F1 on head with top 5 set:", np.mean(head_f1s_5))
        print("Average F1 on tail with top 5 set:", np.mean(tail_f1s_5))
        print("")
        # top 10
        print("Average Retrieved Rate on Head with top 10 set:", np.mean(head_retrieved_10))
        print("Average Retrieved Rate on tailk with top 10 set:", np.mean(tail_retrieved_10))
        print("Average F1 on head with top 10 set:", np.mean(head_f1s_10))
        print("Average F1 on tail with top 10 set:", np.mean(tail_f1s_10))
        print("")
        # top 30
        print("Average Retrieved Rate on Head with top 30 set:", np.mean(head_retrieved_30))
        print("Average Retrieved Rate on tailk with top 30 set:", np.mean(tail_retrieved_30))
        print("Average F1 on head with top 30 set:", np.mean(head_f1s_30))
        print("Average F1 on tail with top 30 set:", np.mean(tail_f1s_30))
        print("")
        # top 50
        print("Average Retrieved Rate on Head with top 50 set:", np.mean(head_retrieved_50))
        print("Average Retrieved Rate on tailk with top 50 set:", np.mean(tail_retrieved_50))
        print("Average F1 on head with top 50 set:", np.mean(head_f1s_50))
        print("Average F1 on tail with top 50 set:", np.mean(tail_f1s_50))
        print("")
        # top 70
        print("Average Retrieved Rate on Head with top 70 set:", np.mean(head_retrieved_70))
        print("Average Retrieved Rate on tailk with top 70 set:", np.mean(tail_retrieved_70))
        print("Average F1 on head with top 70 set:", np.mean(head_f1s_70))
        print("Average F1 on tail with top 70 set:", np.mean(tail_f1s_70))
        print("")
        # top 100          
        print("Average Retrieved Rate on Head with top 100 set:", np.mean(head_retrieved_100))
        print("Average Retrieved Rate on tailk with top 100 set:", np.mean(tail_retrieved_100))
        print("Average F1 on head with top 100 set:", np.mean(head_f1s_100))
        print("Average F1 on tail with top 100 set:", np.mean(tail_f1s_100))
        print("")

                



def main(args):
    model_path = osp.join(osp.abspath(parent), 'models', args.path)
    entity_embedding = np.load(osp.join(model_path, 'entity_embedding.npy'))
    relation_embedding = np.load(osp.join(model_path, 'relation_embedding.npy'))

    nentity = entity_embedding.shape[0]
    nrelation = relation_embedding.shape[0]

    with open(osp.join(osp.abspath(parent), 'models', args.path, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    if entity_embedding.shape[0] == 14541:  # FB15k-237
        dataset_name = 'FB15k-237'
    else:
        dataset_name = 'wn18rr'

    with open(osp.join(osp.abspath(parent), 'data', dataset_name, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(osp.join(osp.abspath(parent), 'data', dataset_name, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    train_triples = read_triple(osp.join(osp.abspath(parent), 'data', dataset_name, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(osp.join(osp.abspath(parent), 'data', dataset_name, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(osp.join(osp.abspath(parent), 'data', dataset_name, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))
    
    #All true triples
    all_true_triples = train_triples + valid_triples + test_triples


    house_dim = entity_embedding.shape[2]
    print('dim',house_dim)
    if house_dim % 2 == 0:
        house_num = house_dim
    else:
        house_num = house_dim-1


    if args.model_name == 'SSHousE_r':
        kge_model = KGEModel(
                model_name='SSHousE_r',
                nentity=nentity,
                nrelation=nrelation,
                hidden_dim=argparse_dict['hidden_dim'],
                gamma=argparse_dict['gamma'],
                house_dim=argparse_dict['house_dim'],
                house_num=house_num,
                housd_num=argparse_dict['housd_num'],
                thred=argparse_dict['thred'],
                double_entity_embedding=argparse_dict['double_entity_embedding'],
                double_relation_embedding=argparse_dict['double_relation_embedding'],
            )
    
    # used for 'HousE_r'
    else:
        kge_model = KGEModel(
                model_name='HousE_r',
                nentity=nentity,
                nrelation=nrelation,
                hidden_dim=argparse_dict['hidden_dim'],
                gamma=argparse_dict['gamma'],
                house_dim=argparse_dict['house_dim'],
                house_num=house_num,
                housd_num=argparse_dict['housd_num'],
                thred=argparse_dict['thred'],
                double_entity_embedding=argparse_dict['double_entity_embedding'],
                double_relation_embedding=argparse_dict['double_relation_embedding'],
            )
        
    kge_model.to('cuda')

    test_batch_size = argparse_dict['test_batch_size']
    cuda = argparse_dict['cuda']
    cpu_num = argparse_dict['cpu_num']
    

    actual_head = {}
    actual_tail = {}
    for triple in all_true_triples:
        if (triple[0], triple[1]) not in actual_tail:
            actual_tail[(triple[0], triple[1])] = []
        actual_tail[(triple[0], triple[1])].append(triple[2])
        if (triple[1], triple[2]) not in actual_head:
            actual_head[(triple[1], triple[2])] = []
        actual_head[(triple[1], triple[2])].append(triple[0])


    checkpoint = torch.load(osp.join(osp.abspath(parent), 'models', args.path, 'checkpoint'))

    kge_model.load_state_dict(checkpoint['model_state_dict'])
    test_model(kge_model, test_triples, all_true_triples, args, nentity, nrelation, test_batch_size, cuda, cpu_num, actual_head, actual_tail)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to the model')
    parser.add_argument('--gamma', type=float, default=5)
    parser.add_argument('--model_name', type=str, default='SSHousE_r')
    args = parser.parse_args()
    main(args)