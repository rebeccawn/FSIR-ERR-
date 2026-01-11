# coding=gbk

import os
import math
import copy
from itertools import chain
import numpy as np
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import config
from utils import *
from sklearn.metrics.pairwise import cosine_similarity  # 用于相似度计算
from scipy.stats import pearsonr
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import pandas as pd

conf = config.config()
num_classes = conf.num_classes
batchsize = conf.batchsize
datafile = '/home/wangzeyu/ENAS_new_code/local_2D_mix_5_1/'
device = conf.device_detection()
T = conf.T
lamda = 0.7655841358158099
K = 3
m = nn.Softmax(dim=1)
# ======================================================================================================================
# data loading
# ======================================================================================================================
transform=transforms.Compose([
                            ToTensor(),
                            transforms.Normalize(mean=[0.456],std=[0.224]),
                            ])
trainset=myDataset(datafile+"train/", transform=transform)
testset=myDataset(datafile+"val/", transform=transform)
# ======================================================================================================================
# model set
# ======================================================================================================================
test_load=DataLoader(testset, batch_size=1, shuffle=False)
train_load=DataLoader(trainset, batch_size=1, shuffle=False)
fin_model = []
model = torch.load('/home/wangzeyu/ENAS_new_code/modeltest/0.h5') #,map_location='cpu')
model = copy.deepcopy(model)
fin_model.append(model)
model = torch.load('/home/wangzeyu/ENAS_new_code/modeltest/1.h5') #,map_location='cpu')
model = copy.deepcopy(model)
fin_model.append(model)
model = torch.load('/home/wangzeyu/ENAS_new_code/modeltest/2.h5') #,map_location='cpu')
model = copy.deepcopy(model)
fin_model.append(model)
model = torch.load('/home/wangzeyu/ENAS_new_code/modeltest/3.h5') #,map_location='cpu')
model = copy.deepcopy(model)
fin_model.append(model)
model = torch.load('/home/wangzeyu/ENAS_new_code/modeltest/4.h5') #,map_location='cpu')
model = copy.deepcopy(model)
fin_model.append(model)
model = torch.load('/home/wangzeyu/ENAS_new_code/modeltest/5.h5') #,map_location='cpu')
model = copy.deepcopy(model)
fin_model.append(model)
model = torch.load('/home/wangzeyu/ENAS_new_code/modeltest/6.h5') #,map_location='cpu')
model = copy.deepcopy(model)
fin_model.append(model)

prob_test = np.zeros((len(fin_model), len(testset)))
pred_test = np.zeros((len(fin_model), len(testset)))

for i in range(len(fin_model)):
    model = fin_model[i]
    model = model.to(device)
    model.eval() 
    index = 0
    test_preds = torch.zeros(len(testset), 1)
    test_probs = torch.zeros(len(testset), 1)
    test_labels = torch.zeros(len(testset), 1)
    test_uncertainty = torch.zeros(len(testset), 1)
    with torch.no_grad():
        for X, y, in test_load:
            X, y = Variable(X.to(device)), Variable(y.to(device))
            pred = model(X)
            probs = m(pred)
            uncertainty = -probs[0,0] * torch.log(probs[0,0]) - (1 - probs[0,0]) * torch.log(1 - probs[0,0])
            uncertainty = uncertainty.item()
            _, pred_y = torch.max(probs, 1)
            test_labels[index:index + len(y), 0] = y
            test_probs[index:index + len(y), 0] = probs[:, 1]
            test_preds[index:index + len(y), 0] = pred_y
            test_uncertainty[index:index + len(y), 0] = uncertainty
            index += len(y)
    test_metric = compute_measures(test_labels, test_preds, test_probs)
   
    pred_test[i, :] = test_preds.t()
    prob_test[i, :] = test_probs.t()
    
    
prob_train = np.zeros((len(fin_model), len(trainset)))
pred_train = np.zeros((len(fin_model), len(trainset)))
    
for i in range(len(fin_model)):
    model = fin_model[i]
    model = model.to(device)
    model.eval() 
    index = 0
    train_preds = torch.zeros(len(trainset), 1)
    train_probs = torch.zeros(len(trainset), 1)
    train_labels = torch.zeros(len(trainset), 1)
    train_uncertainty = torch.zeros(len(trainset), 1)
    with torch.no_grad():
        for X, y, in train_load:
            X, y = Variable(X.to(device)), Variable(y.to(device))
            pred = model(X)
            probs = m(pred)
            uncertainty = -probs[0,0] * torch.log(probs[0,0]) - (1 - probs[0,0]) * torch.log(1 - probs[0,0])
            uncertainty = uncertainty.item()
            _, pred_y = torch.max(probs, 1)
            train_labels[index:index + len(y), 0] = y
            train_probs[index:index + len(y), 0] = probs[:, 1]
            train_preds[index:index + len(y), 0] = pred_y
            train_uncertainty[index:index + len(y), 0] = uncertainty
            index += len(y)
    train_metric = compute_measures(train_labels, train_preds, train_probs)
   
    pred_train[i, :] = train_preds.t()
    prob_train[i, :] = train_probs.t()
    
   
prob_test = prob_test.T
prob_train = prob_train.T
pred_test = pred_test.T
prob_train = pred_train.T

train_labels = train_labels.numpy().squeeze().astype(int)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

'''
meta_model = LogisticRegression()
meta_model.fit(prob_train, train_labels)
avg_probs = meta_model.predict_proba(prob_test)
avg_probs = avg_probs[:, 1]

svm_meta = SVC(probability=True) 
svm_meta.fit(prob_train, train_labels)
svm_probs = svm_meta.predict_proba(prob_test)
avg_probs = svm_probs[:, 1]

mlp_meta = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
mlp_meta.fit(prob_train, train_labels)
mlp_probs = mlp_meta.predict_proba(prob_test)  
avg_probs = mlp_probs[:, 1]
'''
xgb_meta = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_meta.fit(prob_train, train_labels)
xgb_probs = xgb_meta.predict_proba(prob_test)  
avg_probs = xgb_probs[:, 1]
 
#avg_probs = np.mean(p, axis=0)
avg_preds = (avg_probs > 0.5).astype(int)
Pro = np.stack([1 - avg_probs, avg_probs], axis=1)
test_metric = compute_measures(test_labels, avg_preds, avg_probs)
metrics_print(test_metric)
print()

ECE, MCE = Confidence_Calibration_Calculation(Pro, test_labels, 10)
print(ECE)
print(MCE)