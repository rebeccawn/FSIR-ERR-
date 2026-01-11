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
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import pandas as pd

conf = config.config()
num_classes = conf.num_classes
batchsize = conf.batchsize
datafile = ''
device = conf.device_detection()
T = conf.T
lamda = 0.7655841358158099

m = nn.Softmax(dim=1)
# ======================================================================================================================
# data loading
# ======================================================================================================================
transform = transforms.Compose([
    ToTensor(),
    transforms.Normalize(mean=[0.456], std=[0.224]),
])
valset = myDataset(datafile + "val/", transform=transform)
testset = myDataset(datafile + "test/", transform=transform)
# ======================================================================================================================
# model set
# ======================================================================================================================
test_load = DataLoader(testset, batch_size=1, shuffle=False)
val_load = DataLoader(valset, batch_size=1, shuffle=False)
fin_model = []
model = torch.load('/home/wangzeyu/ENAS_new_code/modeltest/resnet.h5')  # ,map_location='cpu')
model = copy.deepcopy(model)
fin_model.append(model)
model = torch.load('/home/wangzeyu/ENAS_new_code/modeltest/densenet.h5')  # ,map_location='cpu')
model = copy.deepcopy(model)
fin_model.append(model)
model = torch.load('/home/wangzeyu/ENAS_new_code/modeltest/0.h5')  # ,map_location='cpu')
model = copy.deepcopy(model)
fin_model.append(model)
model = torch.load('/home/wangzeyu/ENAS_new_code/modeltest/1.h5')  # ,map_location='cpu')
model = copy.deepcopy(model)
fin_model.append(model)
model = torch.load('/home/wangzeyu/ENAS_new_code/modeltest/2.h5')  # ,map_location='cpu')
model = copy.deepcopy(model)
fin_model.append(model)
model = torch.load('/home/wangzeyu/ENAS_new_code/modeltest/3.h5')  # ,map_location='cpu')
model = copy.deepcopy(model)
fin_model.append(model)
model = torch.load('/home/wangzeyu/ENAS_new_code/modeltest/4.h5')  # ,map_location='cpu')
model = copy.deepcopy(model)
fin_model.append(model)
model = torch.load('/home/wangzeyu/ENAS_new_code/modeltest/5.h5')  # ,map_location='cpu')
model = copy.deepcopy(model)
fin_model.append(model)
model = torch.load('/home/wangzeyu/ENAS_new_code/modeltest/6.h5')  # ,map_location='cpu')
model = copy.deepcopy(model)
fin_model.append(model)
# ======================================================================================================================
# feature extraction
# ======================================================================================================================

val_features = []
test_features = []
feature_dims = []
fc_layers = []

for model in fin_model[2:len(fin_model)]:
    model = model.to(device)
    model.eval()
    index = 0

    with torch.no_grad():
        sample_images, _ = next(iter(val_load))
        sample_images = sample_images.to(device)
        sample_feature = model(sample_images, extract_features=True)
        feature_dim = sample_feature.size(1)
    feature_dims.append(feature_dim)

    features = torch.zeros(len(valset), feature_dim).to(device)

    for images, labels in val_load:
        images = images.to(device)
        with torch.no_grad():
            feature = model(images, extract_features=True)
            features[index:index + 1, :] = feature
            index += 1

    val_features.append(features.cpu().numpy())

feature_dims = np.array(feature_dims)

for model in fin_model[2:len(fin_model)]:
    model = model.to(device)
    model.eval()
    index = 0

    with torch.no_grad():
        sample_images, _ = next(iter(test_load))
        sample_images = sample_images.to(device)
        sample_feature = model(sample_images, extract_features=True)
        feature_dim = sample_feature.size(1)
    features = torch.zeros(len(testset), feature_dim).to(device)

    for images, labels in test_load:
        images = images.to(device)
        with torch.no_grad():
            feature = model(images, extract_features=True)
            features[index:index + 1, :] = feature
            index += 1

    test_features.append(features.cpu().numpy())

all_features = [
    np.vstack([val_feat, test_feat])
    for val_feat, test_feat in zip(val_features, test_features)
]

# ======================================================================================================================
# test single model
# ======================================================================================================================
pareto = np.zeros((len(fin_model), 2))
AUC = np.zeros((len(fin_model), 1))
p = np.zeros((len(fin_model), len(testset)))
u = np.zeros((len(fin_model), len(testset)))
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
            uncertainty = -probs[0, 0] * torch.log(probs[0, 0]) - (1 - probs[0, 0]) * torch.log(1 - probs[0, 0])
            uncertainty = uncertainty.item()
            _, pred_y = torch.max(probs, 1)
            test_labels[index:index + len(y), 0] = y
            test_probs[index:index + len(y), 0] = probs[:, 1]
            test_preds[index:index + len(y), 0] = pred_y
            test_uncertainty[index:index + len(y), 0] = uncertainty
            index += len(y)
    test_metric = compute_measures(test_labels, test_preds, test_probs)
    metrics_print(test_metric)
    print()
    pareto[i][0] = test_metric['sen']
    pareto[i][1] = test_metric['spe']
    AUC[i] = test_metric['auc']
    p[i, :] = test_probs.t()
    u[i, :] = test_uncertainty.t()
# ======================================================================================================================
# ER-rule
# ======================================================================================================================
fin_model.pop(0)
fin_model.pop(0)
pareto = np.delete(pareto, [0, 1], axis=0)
AUC = np.delete(AUC, [0, 1], axis=0)
p = np.delete(p, [0, 1], axis=0)
u = np.delete(u, [0, 1], axis=0)
ww = cal_weight(pareto, AUC, lamda)
print(ww)

for K in range(1, 10):  # 寻找合适的K
    test_preds = torch.zeros(len(testset), 1)
    test_probs = torch.zeros(len(testset), 1)
    test_uncertainty = torch.zeros(len(testset), 1)
    index = 0
    for i in range(len(p[0])):
        sum1 = 1 + u[:, i]
        ppp = np.zeros((len(fin_model), 2))
        ppp[:, 0] = (1 - p[:, i]) / sum1
        ppp[:, 1] = p[:, i] / sum1
        r = np.zeros((len(fin_model), 1))
        for j in range(len(p)):
            similarities = [mahalanobis(test_features[j][i, :], test_features[j][t, :], inv_cov_matrix) for t in
                            range(len(test_features[j]))]
            similarities = np.array(similarities)
            most_similar_indices = similarities.argsort()[:K]
            most_similar_indices = most_similar_indices.tolist()
            valset_sub = torch.utils.data.Subset(valset, most_similar_indices)
            val_sub_load = DataLoader(valset_sub, batch_size=1, shuffle=False)
            model = fin_model[j]
            model = model.to(device)
            model.eval()  #
            index_1 = 0
            val_preds = torch.zeros(len(valset_sub), 1)
            val_labels = torch.zeros(len(valset_sub), 1)
            val_probs = torch.zeros(len(valset_sub), 1)
            val_uncertainty = torch.zeros(len(valset), 1)
            with torch.no_grad():
                for X, y, in val_sub_load:
                    X, y = Variable(X.to(device)), Variable(y.to(device))
                    pred = model(X)
                    probs = m(pred)
                    uncertainty = -probs[0, 0] * torch.log(probs[0, 0]) - (1 - probs[0, 0]) * torch.log(1 - probs[0, 0])
                    uncertainty = uncertainty.item()
                    _, pred_y = torch.max(probs, 1)
                    val_labels[index_1:index_1 + 1, 0] = y
                    val_preds[index_1:index_1 + 1, 0] = pred_y
                    val_probs[index:index + 1, 0] = probs[:, 1]
                    val_uncertainty[index:index + 1, 0] = uncertainty
                    index_1 += 1
            num = 0
            for k in range(K):
                if val_labels[k] == val_preds[k]:
                    num += 1
            NV = num / K
            r[j] = NV
        probs = Analytic_ER_rule_1(r, ww, ppp)
        probs = list(chain.from_iterable(probs))
        probs = torch.tensor([float(np.squeeze(p)) for p in probs])
        _, pred_y = torch.max(probs, 0)
        test_preds[index:index + len(y), 0] = pred_y
        test_probs[index:index + len(y), 0] = probs[1]
        # test_uncertainty[index:index + len(y), 0] = uncertainty

        index += len(y)
    test_metric = compute_measures(test_labels, test_preds, test_probs)
    print()
    metrics_print(test_metric)

M = 10
ECE, MCE = Confidence_Calibration_Calculation(test_preds, test_labels, M)
print('ECE=', ECE)
print('MCE=', MCE)

# ======================================================================================================================
# ERE:uncertainty_sort + compute_measures
# ======================================================================================================================
test_uncertainty = test_uncertainty / test_uncertainty.max()
uncertainty_sort, indices = torch.sort(torch.flatten(test_uncertainty), descending=True,
                                       dim=-1)  # ??test_uncertainty????????
print(uncertainty_sort[0])
length = len(test_uncertainty)

num = 0
for i in range(length):
    if test_uncertainty[i] <= uncertainty_sort[int(length / 4)]:
        num += 1
index = 0
test_preds2 = torch.zeros(num, 1)
test_probs2 = torch.zeros(num, 1)
test_u2 = torch.zeros(num, 1)
test_labels2 = torch.zeros(num, 1)
for i in range(length):
    if test_uncertainty[i] <= uncertainty_sort[int(length / 4)]:
        test_preds2[index:index + 1, 0] = test_preds[i]
        test_probs2[index:index + 1, 0] = test_probs[i]
        test_u2[index:index + 1, 0] = test_uncertainty[i]
        test_labels2[index:index + 1, 0] = test_labels[i]
        index += 1
test_metric = compute_measures(test_labels2, test_preds2, test_probs2)
metrics_print(test_metric)
print(uncertainty_sort[int(length / 4)])
print('\t')

num = 0
for i in range(length):
    if test_uncertainty[i] <= uncertainty_sort[int(length / 2)]:
        num += 1
index = 0
test_preds2 = torch.zeros(num, 1)
test_probs2 = torch.zeros(num, 1)
test_u2 = torch.zeros(num, 1)
test_labels2 = torch.zeros(num, 1)
for i in range(length):
    if test_uncertainty[i] <= uncertainty_sort[int(length / 2)]:
        test_preds2[index:index + 1, 0] = test_preds[i]
        test_probs2[index:index + 1, 0] = test_probs[i]
        test_u2[index:index + 1, 0] = test_uncertainty[i]
        test_labels2[index:index + 1, 0] = test_labels[i]
        index += 1
test_metric = compute_measures(test_labels2, test_preds2, test_probs2)
metrics_print(test_metric)
print(uncertainty_sort[int(length / 2)])
print('\t')

num = 0
for i in range(length):
    if test_uncertainty[i] <= uncertainty_sort[int(length * 3 / 4)]:
        num += 1
index = 0
test_preds2 = torch.zeros(num, 1)
test_probs2 = torch.zeros(num, 1)
test_u2 = torch.zeros(num, 1)
test_labels2 = torch.zeros(num, 1)
for i in range(length):
    if test_uncertainty[i] <= uncertainty_sort[int(length * 3 / 4)]:
        test_preds2[index:index + 1, 0] = test_preds[i]
        test_probs2[index:index + 1, 0] = test_probs[i]
        test_u2[index:index + 1, 0] = test_uncertainty[i]
        test_labels2[index:index + 1, 0] = test_labels[i]
        index += 1
test_metric = compute_measures(test_labels2, test_preds2, test_probs2)
metrics_print(test_metric)
print(uncertainty_sort[int(length * 3 / 4)])
print()



