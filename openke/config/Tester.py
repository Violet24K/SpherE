# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
from sklearn.metrics import roc_auc_score
import copy
from tqdm import tqdm
import pdb

class Tester(object):

    def __init__(self, model = None, data_loader = None, use_gpu = True):
        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        self.lib.testHead.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.testTail.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.test_link_prediction.argtypes = [ctypes.c_int64]

        self.lib.getTestLinkMRR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkMR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit10.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit3.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit1.argtypes = [ctypes.c_int64]

        self.lib.getTestLinkMRR.restype = ctypes.c_float
        self.lib.getTestLinkMR.restype = ctypes.c_float
        self.lib.getTestLinkHit10.restype = ctypes.c_float
        self.lib.getTestLinkHit3.restype = ctypes.c_float
        self.lib.getTestLinkHit1.restype = ctypes.c_float

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.model.cuda()

    def set_model(self, model):
        self.model = model

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu and self.model != None:
            self.model.cuda()

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def test_one_step(self, data):        
        return self.model.predict({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'mode': data['mode']
        })

    def run_link_prediction(self, type_constrain = False):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
        training_range = tqdm(self.data_loader)
        for index, [data_head, data_tail] in enumerate(training_range):
            score = self.test_one_step(data_head)
            self.lib.testHead(score.__array_interface__["data"][0], index, type_constrain)
            score = self.test_one_step(data_tail)
            self.lib.testTail(score.__array_interface__["data"][0], index, type_constrain)
        self.lib.test_link_prediction(type_constrain)

        mrr = self.lib.getTestLinkMRR(type_constrain)
        mr = self.lib.getTestLinkMR(type_constrain)
        hit10 = self.lib.getTestLinkHit10(type_constrain)
        hit3 = self.lib.getTestLinkHit3(type_constrain)
        hit1 = self.lib.getTestLinkHit1(type_constrain)
        print (hit10)
        return mrr, mr, hit10, hit3, hit1

    def get_best_threshlod(self, score, ans):
        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        res_mx = 0.0
        threshlod = None
        for index, [ans, score] in enumerate(res):
            if ans == 1:
                total_current += 1.0
            res_current = (2 * total_current + total_false - index - 1) / total_all
            if res_current > res_mx:
                res_mx = res_current
                threshlod = score
        return threshlod, res_mx

    def run_triple_classification(self, threshlod = None):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('classification')
        score = []
        ans = []
        training_range = tqdm(self.data_loader)
        for index, [pos_ins, neg_ins] in enumerate(training_range):
            res_pos = self.test_one_step(pos_ins)
            ans = ans + [1 for i in range(len(res_pos))]
            score.append(res_pos)

            res_neg = self.test_one_step(neg_ins)
            ans = ans + [0 for i in range(len(res_pos))]
            score.append(res_neg)

        score = np.concatenate(score, axis = -1)
        ans = np.array(ans)

        if threshlod == None:
            threshlod, _ = self.get_best_threshlod(score, ans)

        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        for index, [ans, score] in enumerate(res):
            if score > threshlod:
                acc = (2 * total_current + total_false - index) / total_all
                break
            elif ans == 1:
                total_current += 1.0

        return acc, threshlod
    












class TesterForFixedTest(object):

    def __init__(self, model = None, data_loader = None, use_gpu = True):
        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        self.lib.testHead.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.testTail.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.test_link_prediction.argtypes = [ctypes.c_int64]

        self.lib.getTestLinkMRR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkMR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit10.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit3.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit1.argtypes = [ctypes.c_int64]

        self.lib.getTestLinkMRR.restype = ctypes.c_float
        self.lib.getTestLinkMR.restype = ctypes.c_float
        self.lib.getTestLinkHit10.restype = ctypes.c_float
        self.lib.getTestLinkHit3.restype = ctypes.c_float
        self.lib.getTestLinkHit1.restype = ctypes.c_float

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.model.cuda()

    def set_model(self, model):
        self.model = model

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu and self.model != None:
            self.model.cuda()

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def test_one_step(self, data):        
        return self.model.predict({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'mode': data['mode']
        })

    def run_link_prediction(self, type_constrain = False):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
        training_range = tqdm(self.data_loader)
        
        mrr_head_list = np.zeros(self.data_loader.nbatches)
        mr_head_list = np.zeros(self.data_loader.nbatches)
        hit10_head_list = np.zeros(self.data_loader.nbatches)
        hit3_head_list = np.zeros(self.data_loader.nbatches)
        hit1_head_list = np.zeros(self.data_loader.nbatches)
        mrr_tail_list = np.zeros(self.data_loader.nbatches)
        mr_tail_list = np.zeros(self.data_loader.nbatches)
        hit10_tail_list = np.zeros(self.data_loader.nbatches)
        hit3_tail_list = np.zeros(self.data_loader.nbatches)
        hit1_tail_list = np.zeros(self.data_loader.nbatches)
        for index, [data_head, data_tail] in enumerate(training_range):
            score = self.test_one_step(data_head)
            correct_head = self.data_loader.batch_h[index]
            head_score = score[correct_head]
            rank = 1
            for i in range(self.data_loader.nentities):
                if score[i] < head_score:
                    rank += 1
            if rank == 1:
                hit1_head_list[index] = 1
            if rank <= 3:
                hit3_head_list[index] = 1
            if rank <= 10:
                hit10_head_list[index] = 1
            mrr_head_list[index] = 1/rank
            mr_head_list[index] = rank

            score = self.test_one_step(data_tail)
            correct_tail = self.data_loader.batch_t[index]
            tail_score = score[correct_tail]
            rank = 1
            for i in range(self.data_loader.nentities):
                if score[i] < tail_score:
                    rank += 1
            if rank == 1:
                hit1_tail_list[index] = 1
            if rank <= 3:
                hit3_tail_list[index] = 1
            if rank <= 10:
                hit10_tail_list[index] = 1
            mrr_tail_list[index] = 1/rank
            mr_tail_list[index] = rank
            
        print("Average MRR on Head: ", np.mean(mrr_head_list))
        print("Average MR on head: ", np.mean(mr_head_list))
        print("Average hit10 on head: ", np.mean(hit10_head_list))
        print("Average hit3 on head: ", np.mean(hit3_head_list))
        print("Average hit1 on head: ", np.mean(hit1_head_list))
    
        print("Average MRR on tail: ", np.mean(mrr_tail_list))
        print("Average MR on tail: ", np.mean(mr_tail_list))
        print("Average hit10 on tail: ", np.mean(hit10_tail_list))
        print("Average hit3 on tail: ", np.mean(hit3_tail_list))
        print("Average hit1 on tail: ", np.mean(hit1_tail_list))

        mrr = 0
        mr = 0
        hit10 = 0
        hit3 = 0
        hit1 = 0
        return mrr, mr, hit10, hit3, hit1

    def get_best_threshlod(self, score, ans):
        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        res_mx = 0.0
        threshlod = None
        for index, [ans, score] in enumerate(res):
            if ans == 1:
                total_current += 1.0
            res_current = (2 * total_current + total_false - index - 1) / total_all
            if res_current > res_mx:
                res_mx = res_current
                threshlod = score
        return threshlod, res_mx

    def run_triple_classification(self, threshlod = None):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('classification')
        score = []
        ans = []
        training_range = tqdm(self.data_loader)
        for index, [pos_ins, neg_ins] in enumerate(training_range):
            res_pos = self.test_one_step(pos_ins)
            ans = ans + [1 for i in range(len(res_pos))]
            score.append(res_pos)

            res_neg = self.test_one_step(neg_ins)
            ans = ans + [0 for i in range(len(res_pos))]
            score.append(res_neg)

        score = np.concatenate(score, axis = -1)
        ans = np.array(ans)

        if threshlod == None:
            threshlod, _ = self.get_best_threshlod(score, ans)

        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        for index, [ans, score] in enumerate(res):
            if score > threshlod:
                acc = (2 * total_current + total_false - index) / total_all
                break
            elif ans == 1:
                total_current += 1.0

        return acc, threshlod
    





class TesterForRetrievalTest(object):

    def __init__(self, model = None, data_loader = None, use_gpu = True, actual_head = None, actual_tail = None):
        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        self.lib.testHead.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.testTail.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.test_link_prediction.argtypes = [ctypes.c_int64]

        self.lib.getTestLinkMRR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkMR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit10.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit3.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit1.argtypes = [ctypes.c_int64]

        self.lib.getTestLinkMRR.restype = ctypes.c_float
        self.lib.getTestLinkMR.restype = ctypes.c_float
        self.lib.getTestLinkHit10.restype = ctypes.c_float
        self.lib.getTestLinkHit3.restype = ctypes.c_float
        self.lib.getTestLinkHit1.restype = ctypes.c_float

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu
        self.actual_head = actual_head
        self.actual_tail = actual_tail

        if self.use_gpu:
            self.model.cuda()

    def set_model(self, model):
        self.model = model

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu and self.model != None:
            self.model.cuda()

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))


    def test_one_step(self, data):        
        return self.model.predict({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'mode': data['mode']
        })
    # use real distance in test
    def test_one_step_with_radius(self, data):        
        return self.model.helper_for_test_score({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'mode': data['mode']
        })
    
    def run_link_prediction(self, type_constrain = False):
        if hasattr(self.model, 'ent_radius'):
            ent_radius = self.model.ent_radius.weight.data.cpu().numpy()
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
        training_range = tqdm(self.data_loader)
        
        # SSEmb
        if hasattr(self.model, 'ent_radius'):
            retrieve_head_list = np.zeros(self.data_loader.nbatches)
            retrieve_head_f1_list = np.zeros(self.data_loader.nbatches)
            retrieve_tail_list = np.zeros(self.data_loader.nbatches)
            retrieve_tail_f1_list = np.zeros(self.data_loader.nbatches)
            for index, [data_head, data_tail] in enumerate(training_range):
                score = self.test_one_step_with_radius(data_head)
                # test the Retrieved Rate on Head
                correct_head = self.data_loader.batch_h[index]
                head_score = score[correct_head]
                if head_score <= 0:
                    retrieve_head_list[index] = 1
                # test the F1 on Head
                retrieved_heads = np.where(score <= 0)[0]
                actual_heads = self.actual_head[(data_head['batch_r'][0], data_head['batch_t'][0])]
                TP = len(set(retrieved_heads) & set(actual_heads))
                if len(retrieved_heads) == 0:
                    retrieve_head_f1_list[index] = 0
                else:
                    precision = TP/len(retrieved_heads)
                    recall = TP/len(actual_heads)
                    if TP == 0:
                        retrieve_head_f1_list[index] = 0
                    else:
                        retrieve_head_f1_list[index] = (2*precision*recall)/(precision+recall)
                score = self.test_one_step_with_radius(data_tail)
                # test the Retrieved Rate on Tail
                correct_tail = self.data_loader.batch_t[index]
                tail_score = score[correct_tail]
                if tail_score <= 0:
                    retrieve_tail_list[index] = 1
                # test the F1 on Tail
                retrieved_tails = np.where(score <= 0)[0]
                actual_tails = self.actual_tail[(data_tail['batch_h'][0], data_tail['batch_r'][0])]
                TP = TP = len(set(retrieved_tails) & set(actual_tails))
                if len(retrieved_tails) == 0:
                    retrieve_tail_f1_list[index] = 0
                else:
                    precision = TP/len(retrieved_tails)
                    recall = TP/len(actual_tails)
                    if TP== 0:
                        retrieve_tail_f1_list[index] = 0
                    else:
                        retrieve_tail_f1_list[index] = (2*precision*recall)/(precision+recall)

            print("Average Retrieved Rate on Head: ", np.mean(retrieve_head_list))
            print("Average F1 on head: ", np.mean(retrieve_head_f1_list))
        
            print("Average Retrieved Rate on tail: ", np.mean(retrieve_tail_list))
            print("Average F1 on tail: ", np.mean(retrieve_tail_f1_list))


        else:
            retrieve_head_list1 = np.zeros(self.data_loader.nbatches)
            retrieve_head_f1_list1 = np.zeros(self.data_loader.nbatches)
            retrieve_tail_list1 = np.zeros(self.data_loader.nbatches)
            retrieve_tail_f1_list1 = np.zeros(self.data_loader.nbatches)
            retrieve_head_list3 = np.zeros(self.data_loader.nbatches)
            retrieve_head_f1_list3 = np.zeros(self.data_loader.nbatches)
            retrieve_tail_list3 = np.zeros(self.data_loader.nbatches)
            retrieve_tail_f1_list3 = np.zeros(self.data_loader.nbatches)
            retrieve_head_list10 = np.zeros(self.data_loader.nbatches)
            retrieve_head_f1_list10 = np.zeros(self.data_loader.nbatches)
            retrieve_tail_list10 = np.zeros(self.data_loader.nbatches)
            retrieve_tail_f1_list10 = np.zeros(self.data_loader.nbatches)
            # 12
            retrieve_head_list12 = np.zeros(self.data_loader.nbatches)
            retrieve_head_f1_list12 = np.zeros(self.data_loader.nbatches)
            retrieve_tail_list12 = np.zeros(self.data_loader.nbatches)
            retrieve_tail_f1_list12 = np.zeros(self.data_loader.nbatches)
            # 16
            retrieve_head_list16 = np.zeros(self.data_loader.nbatches)
            retrieve_head_f1_list16 = np.zeros(self.data_loader.nbatches)
            retrieve_tail_list16 = np.zeros(self.data_loader.nbatches)
            retrieve_tail_f1_list16 = np.zeros(self.data_loader.nbatches)
            # 20
            retrieve_head_list20 = np.zeros(self.data_loader.nbatches)
            retrieve_head_f1_list20 = np.zeros(self.data_loader.nbatches)
            retrieve_tail_list20 = np.zeros(self.data_loader.nbatches)
            retrieve_tail_f1_list20 = np.zeros(self.data_loader.nbatches)
            # 25
            retrieve_head_list25 = np.zeros(self.data_loader.nbatches)
            retrieve_head_f1_list25 = np.zeros(self.data_loader.nbatches)
            retrieve_tail_list25 = np.zeros(self.data_loader.nbatches)
            retrieve_tail_f1_list25 = np.zeros(self.data_loader.nbatches)
            # 50
            retrieve_head_list50 = np.zeros(self.data_loader.nbatches)
            retrieve_head_f1_list50 = np.zeros(self.data_loader.nbatches)
            retrieve_tail_list50 = np.zeros(self.data_loader.nbatches)
            retrieve_tail_f1_list50 = np.zeros(self.data_loader.nbatches)
            # 75
            retrieve_head_list75 = np.zeros(self.data_loader.nbatches)
            retrieve_head_f1_list75 = np.zeros(self.data_loader.nbatches)
            retrieve_tail_list75 = np.zeros(self.data_loader.nbatches)
            retrieve_tail_f1_list75 = np.zeros(self.data_loader.nbatches)

            retrieve_head_list100 = np.zeros(self.data_loader.nbatches)
            retrieve_head_f1_list100 = np.zeros(self.data_loader.nbatches)
            retrieve_tail_list100 = np.zeros(self.data_loader.nbatches)
            retrieve_tail_f1_list100 = np.zeros(self.data_loader.nbatches)
            for index, [data_head, data_tail] in enumerate(training_range):
                # heads
                score = self.test_one_step(data_head)
                correct_head = self.data_loader.batch_h[index]
                actual_heads = self.actual_head[(data_head['batch_r'][0], data_head['batch_t'][0])]
                retrieved_heads1 = np.argsort(score)[:1]
                retrieved_heads3 = np.argsort(score)[:3]
                retrieved_heads10 = np.argsort(score)[:10]
                # 12 16 20 25 75
                retrieved_heads12 = np.argsort(score)[:12]
                retrieved_heads16 = np.argsort(score)[:16]
                retrieved_heads20 = np.argsort(score)[:20]
                retrieved_heads25 = np.argsort(score)[:25]
                retrieved_heads50 = np.argsort(score)[:50]
                retrieved_heads75 = np.argsort(score)[:75]
                retrieved_heads100 = np.argsort(score)[:100]
                if correct_head in retrieved_heads1:
                    retrieve_head_list1[index] = 1
                if correct_head in retrieved_heads3:
                    retrieve_head_list3[index] = 1
                if correct_head in retrieved_heads10:
                    retrieve_head_list10[index] = 1
                # 12 16 20 25 75
                if correct_head in retrieved_heads12:
                    retrieve_head_list12[index] = 1
                if correct_head in retrieved_heads16:
                    retrieve_head_list16[index] = 1
                if correct_head in retrieved_heads20:
                    retrieve_head_list20[index] = 1
                if correct_head in retrieved_heads25:
                    retrieve_head_list25[index] = 1
                if correct_head in retrieved_heads50:
                    retrieve_head_list50[index] = 1
                if correct_head in retrieved_heads75:
                    retrieve_head_list75[index] = 1
                if correct_head in retrieved_heads100:
                    retrieve_head_list100[index] = 1
                TP = len(set(retrieved_heads1) & set(actual_heads))
                precision = TP/len(retrieved_heads1)
                recall = TP/len(actual_heads)
                if TP == 0:
                    retrieve_head_f1_list1[index] = 0
                else:
                    retrieve_head_f1_list1[index] = (2*precision*recall)/(precision+recall)
                TP = len(set(retrieved_heads3) & set(actual_heads))
                precision = TP/len(retrieved_heads3)
                recall = TP/len(actual_heads)
                if TP == 0:
                    retrieve_head_f1_list3[index] = 0
                else:
                    retrieve_head_f1_list3[index] = (2*precision*recall)/(precision+recall)
                TP = len(set(retrieved_heads10) & set(actual_heads))
                precision = TP/len(retrieved_heads10)
                recall = TP/len(actual_heads)
                if TP == 0:
                    retrieve_head_f1_list10[index] = 0
                else:
                    retrieve_head_f1_list10[index] = (2*precision*recall)/(precision+recall)
                # 12 16 20 25
                TP = len(set(retrieved_heads12) & set(actual_heads))
                precision = TP/len(retrieved_heads12)
                recall = TP/len(actual_heads)
                if TP == 0:
                    retrieve_head_f1_list12[index] = 0
                else:
                    retrieve_head_f1_list12[index] = (2*precision*recall)/(precision+recall)
                TP = len(set(retrieved_heads16) & set(actual_heads))
                precision = TP/len(retrieved_heads16)
                recall = TP/len(actual_heads)
                if TP == 0:
                    retrieve_head_f1_list16[index] = 0
                else:
                    retrieve_head_f1_list16[index] = (2*precision*recall)/(precision+recall)
                TP = len(set(retrieved_heads20) & set(actual_heads))
                precision = TP/len(retrieved_heads20)
                recall = TP/len(actual_heads)
                if TP == 0:
                    retrieve_head_f1_list20[index] = 0
                else:
                    retrieve_head_f1_list20[index] = (2*precision*recall)/(precision+recall)
                TP = len(set(retrieved_heads25) & set(actual_heads))
                precision = TP/len(retrieved_heads25)
                recall = TP/len(actual_heads)
                if TP == 0:
                    retrieve_head_f1_list25[index] = 0
                else:
                    retrieve_head_f1_list25[index] = (2*precision*recall)/(precision+recall)
                TP = len(set(retrieved_heads50) & set(actual_heads))
                precision = TP/len(retrieved_heads50)
                recall = TP/len(actual_heads)
                if TP == 0:
                    retrieve_head_f1_list50[index] = 0
                else:
                    retrieve_head_f1_list50[index] = (2*precision*recall)/(precision+recall)
                TP = len(set(retrieved_heads75) & set(actual_heads))
                precision = TP/len(retrieved_heads75)
                recall = TP/len(actual_heads)
                if TP == 0:
                    retrieve_head_f1_list75[index] = 0
                else:
                    retrieve_head_f1_list75[index] = (2*precision*recall)/(precision+recall)
                TP = len(set(retrieved_heads100) & set(actual_heads))
                precision = TP/len(retrieved_heads100)
                recall = TP/len(actual_heads)
                if TP == 0:
                    retrieve_head_f1_list100[index] = 0
                else:
                    retrieve_head_f1_list100[index] = (2*precision*recall)/(precision+recall)

                # tail
                score = self.test_one_step(data_tail)
                correct_tail = self.data_loader.batch_t[index]
                actual_tails = self.actual_tail[(data_tail['batch_h'][0], data_tail['batch_r'][0])]
                retrieved_tails1 = np.argsort(score)[:1]
                retrieved_tails3 = np.argsort(score)[:3]
                retrieved_tails10 = np.argsort(score)[:10]
                # 12 16 20 25
                retrieved_tails12 = np.argsort(score)[:12]
                retrieved_tails16 = np.argsort(score)[:16]
                retrieved_tails20 = np.argsort(score)[:20]
                retrieved_tails25 = np.argsort(score)[:25]         
                retrieved_tails50 = np.argsort(score)[:50]
                retrieved_tails75 = np.argsort(score)[:75]          
                retrieved_tails100 = np.argsort(score)[:100]
                if correct_tail in retrieved_tails1:
                    retrieve_tail_list1[index] = 1
                if correct_tail in retrieved_tails3:
                    retrieve_tail_list3[index] = 1
                if correct_tail in retrieved_tails10:
                    retrieve_tail_list10[index] = 1
                # 12 16 20 25
                if correct_tail in retrieved_tails12:
                    retrieve_tail_list12[index] = 1
                if correct_tail in retrieved_tails16:
                    retrieve_tail_list16[index] = 1
                if correct_tail in retrieved_tails20:
                    retrieve_tail_list20[index] = 1
                if correct_tail in retrieved_tails25:
                    retrieve_tail_list25[index] = 1
                if correct_tail in retrieved_tails50:
                    retrieve_tail_list50[index] = 1
                if correct_tail in retrieved_tails75:
                    retrieve_tail_list75[index] = 1
                if correct_tail in retrieved_tails100:
                    retrieve_tail_list100[index] = 1

                TP = len(set(retrieved_tails1) & set(actual_tails))
                precision = TP/len(retrieved_tails1)
                recall = TP/len(actual_tails)
                if TP == 0:
                    retrieve_tail_f1_list1[index] = 0
                else:
                    retrieve_tail_f1_list1[index] = (2*precision*recall)/(precision+recall)
                TP = len(set(retrieved_tails3) & set(actual_tails))
                precision = TP/len(retrieved_tails3)
                recall = TP/len(actual_tails)
                if TP == 0:
                    retrieve_tail_f1_list3[index] = 0
                else:
                    retrieve_tail_f1_list3[index] = (2*precision*recall)/(precision+recall)
                TP = len(set(retrieved_tails10) & set(actual_tails))
                precision = TP/len(retrieved_tails10)
                recall = TP/len(actual_tails)
                if TP == 0:
                    retrieve_tail_f1_list10[index] = 0
                else:
                    retrieve_tail_f1_list10[index] = (2*precision*recall)/(precision+recall)
                # 12 16 20 25
                TP = len(set(retrieved_tails12) & set(actual_tails))
                precision = TP/len(retrieved_tails12)
                recall = TP/len(actual_tails)
                if TP == 0:
                    retrieve_tail_f1_list12[index] = 0
                else:
                    retrieve_tail_f1_list12[index] = (2*precision*recall)/(precision+recall)
                TP = len(set(retrieved_tails16) & set(actual_tails))
                precision = TP/len(retrieved_tails16)
                recall = TP/len(actual_tails)
                if TP == 0:
                    retrieve_tail_f1_list16[index] = 0
                else:
                    retrieve_tail_f1_list16[index] = (2*precision*recall)/(precision+recall)
                TP = len(set(retrieved_tails20) & set(actual_tails))
                precision = TP/len(retrieved_tails20)
                recall = TP/len(actual_tails)
                if TP == 0:
                    retrieve_tail_f1_list20[index] = 0
                else:
                    retrieve_tail_f1_list20[index] = (2*precision*recall)/(precision+recall)
                TP = len(set(retrieved_tails25) & set(actual_tails))
                precision = TP/len(retrieved_tails25)
                recall = TP/len(actual_tails)
                if TP == 0:
                    retrieve_tail_f1_list25[index] = 0
                else:
                    retrieve_tail_f1_list25[index] = (2*precision*recall)/(precision+recall)
                TP = len(set(retrieved_tails50) & set(actual_tails))
                precision = TP/len(retrieved_tails50)
                recall = TP/len(actual_tails)
                if TP == 0:
                    retrieve_tail_f1_list50[index] = 0
                else:
                    retrieve_tail_f1_list50[index] = (2*precision*recall)/(precision+recall)
                TP = len(set(retrieved_tails75) & set(actual_tails))
                precision = TP/len(retrieved_tails75)
                recall = TP/len(actual_tails)
                if TP == 0:
                    retrieve_tail_f1_list75[index] = 0
                else:
                    retrieve_tail_f1_list75[index] = (2*precision*recall)/(precision+recall)
                TP = len(set(retrieved_tails100) & set(actual_tails))
                precision = TP/len(retrieved_tails100)
                recall = TP/len(actual_tails)
                if TP == 0:
                    retrieve_tail_f1_list100[index] = 0
                else:
                    retrieve_tail_f1_list100[index] = (2*precision*recall)/(precision+recall)

                
            print("Average Retrieved Rate on Head with top 1 set: ", np.mean(retrieve_head_list1))
            print("Average F1 on head with top 1 set: ", np.mean(retrieve_head_f1_list1))
            print("Average Retrieved Rate on tail with top 1 set: ", np.mean(retrieve_tail_list1))
            print("Average F1 on tail with top 1 set: ", np.mean(retrieve_tail_f1_list1))

            print("Average Retrieved Rate on Head with top 3 set: ", np.mean(retrieve_head_list3))
            print("Average F1 on head with top 3 set: ", np.mean(retrieve_head_f1_list3))
            print("Average Retrieved Rate on tail with top 3 set: ", np.mean(retrieve_tail_list3))
            print("Average F1 on tail with top 3 set: ", np.mean(retrieve_tail_f1_list3))

            print("Average Retrieved Rate on Head with top 10 set: ", np.mean(retrieve_head_list10))
            print("Average F1 on head with top 10 set: ", np.mean(retrieve_head_f1_list10))
            print("Average Retrieved Rate on tail with top 10 set: ", np.mean(retrieve_tail_list10))
            print("Average F1 on tail with top 10 set: ", np.mean(retrieve_tail_f1_list10))
            # 12 16 20 25
            print("Average Retrieved Rate on Head with top 12 set: ", np.mean(retrieve_head_list12))
            print("Average F1 on head with top 12 set: ", np.mean(retrieve_head_f1_list12))
            print("Average Retrieved Rate on tail with top 12 set: ", np.mean(retrieve_tail_list12))
            print("Average F1 on tail with top 12 set: ", np.mean(retrieve_tail_f1_list12))

            print("Average Retrieved Rate on Head with top 16 set: ", np.mean(retrieve_head_list16))
            print("Average F1 on head with top 16 set: ", np.mean(retrieve_head_f1_list16))
            print("Average Retrieved Rate on tail with top 16 set: ", np.mean(retrieve_tail_list16))
            print("Average F1 on tail with top 16 set: ", np.mean(retrieve_tail_f1_list16))

            print("Average Retrieved Rate on Head with top 20 set: ", np.mean(retrieve_head_list20))
            print("Average F1 on head with top 20 set: ", np.mean(retrieve_head_f1_list20))
            print("Average Retrieved Rate on tail with top 20 set: ", np.mean(retrieve_tail_list20))
            print("Average F1 on tail with top 20 set: ", np.mean(retrieve_tail_f1_list20))

            print("Average Retrieved Rate on Head with top 25 set: ", np.mean(retrieve_head_list25))
            print("Average F1 on head with top 25 set: ", np.mean(retrieve_head_f1_list25))
            print("Average Retrieved Rate on tail with top 25 set: ", np.mean(retrieve_tail_list25))
            print("Average F1 on tail with top 25 set: ", np.mean(retrieve_tail_f1_list25))

            print("Average Retrieved Rate on Head with top 50 set: ", np.mean(retrieve_head_list50))
            print("Average F1 on head with top 50 set: ", np.mean(retrieve_head_f1_list50))
            print("Average Retrieved Rate on tail with top 50 set: ", np.mean(retrieve_tail_list50))
            print("Average F1 on tail with top 50 set: ", np.mean(retrieve_tail_f1_list50))

            print("Average Retrieved Rate on Head with top 75 set: ", np.mean(retrieve_head_list75))
            print("Average F1 on head with top 75 set: ", np.mean(retrieve_head_f1_list75))
            print("Average Retrieved Rate on tail with top 75 set: ", np.mean(retrieve_tail_list75))
            print("Average F1 on tail with top 75 set: ", np.mean(retrieve_tail_f1_list75))

            print("Average Retrieved Rate on Head with top 100 set: ", np.mean(retrieve_head_list100))
            print("Average F1 on head with top 100 set: ", np.mean(retrieve_head_f1_list100))
            print("Average Retrieved Rate on tail with top 100 set: ", np.mean(retrieve_tail_list100))
            print("Average F1 on tail with top 100 set: ", np.mean(retrieve_tail_f1_list100))

        mrr = 0
        mr = 0
        hit10 = 0
        hit3 = 0
        hit1 = 0
        return mrr, mr, hit10, hit3, hit1

    def get_best_threshlod(self, score, ans):
        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        res_mx = 0.0
        threshlod = None
        for index, [ans, score] in enumerate(res):
            if ans == 1:
                total_current += 1.0
            res_current = (2 * total_current + total_false - index - 1) / total_all
            if res_current > res_mx:
                res_mx = res_current
                threshlod = score
        return threshlod, res_mx

    def run_triple_classification(self, threshlod = None):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('classification')
        score = []
        ans = []
        training_range = tqdm(self.data_loader)
        for index, [pos_ins, neg_ins] in enumerate(training_range):
            res_pos = self.test_one_step(pos_ins)
            ans = ans + [1 for i in range(len(res_pos))]
            score.append(res_pos)

            res_neg = self.test_one_step(neg_ins)
            ans = ans + [0 for i in range(len(res_pos))]
            score.append(res_neg)

        score = np.concatenate(score, axis = -1)
        ans = np.array(ans)

        if threshlod == None:
            threshlod, _ = self.get_best_threshlod(score, ans)

        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        for index, [ans, score] in enumerate(res):
            if score > threshlod:
                acc = (2 * total_current + total_false - index) / total_all
                break
            elif ans == 1:
                total_current += 1.0

        return acc, threshlod