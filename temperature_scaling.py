import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS
import config
from utils import *

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

def ensemble_logits(model_list, data, device):
    model_outputs = []
    for model in model_list:
        model.eval()
        with torch.no_grad():
            logits = model(data.to(device))
            model_outputs.append(logits)

    return torch.mean(torch.stack(model_outputs), dim=0)

def tune_temperature(model_list, val_loader, device):
    logits_list = []
    labels_list = []

    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        logits = ensemble_logits(model_list, data, device)
        logits_list.append(logits)
        labels_list.append(target)

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    temp_scaler = TemperatureScaler().to(device)
    optimizer = LBFGS([temp_scaler.temperature], lr=0.01, max_iter=50)

    def eval():
        optimizer.zero_grad()
        scaled_logits = temp_scaler(logits)
        loss = F.cross_entropy(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(eval)
    print(f"Optimal temperature: {temp_scaler.temperature.item():.4f}")
    return temp_scaler
    
def predict_with_temperature(model_list, temp_scaler, test_loader, device):
    all_probs = []
    all_preds = []
    all_targets = []

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        logits = ensemble_logits(model_list, data, device)
        scaled_logits = temp_scaler(logits)
        probs = F.softmax(scaled_logits, dim=1)
        preds = probs.argmax(dim=1)

        all_probs.append(probs)
        all_preds.append(preds)
        all_targets.append(target)

    probs = torch.cat(all_probs)
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    return preds, probs, targets

conf = config.config()
num_classes = conf.num_classes
batchsize = conf.batchsize
datafile = '/home/wangzeyu/ENAS_new_code/local_2D_mix_5_1/'
device = conf.device_detection()
# ======================================================================================================================
# data loading
# ======================================================================================================================
transform=transforms.Compose([
                            ToTensor(),
                            transforms.Normalize(mean=[0.456],std=[0.224]),
                            ])
trainset=myDataset(datafile+"val/", transform=transform)
testset=myDataset(datafile+"test/", transform=transform)
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


    
temp_scaler = tune_temperature(fin_model, train_load, device)

preds, probs, targets = predict_with_temperature(fin_model, temp_scaler, test_load, device)

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

def compute_metrics(preds, probs, targets, pos_label=1):
    if probs.ndim == 2 and probs.shape[1] == 2:
        probs = probs[:, 1]
    
    # Accuracy
    acc = accuracy_score(targets, preds)

    # AUC
    auc = roc_auc_score(targets, probs)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(targets, preds)
    tn, fp, fn, tp = cm.ravel()

    # Sensitivity (Recall / True Positive Rate)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "Accuracy": acc,
        "AUC": auc,
        "Sensitivity": sensitivity,
        "Specificity": specificity
    }
    
metrics = compute_metrics(
    preds.detach().cpu().numpy(),
    probs.detach().cpu().numpy(),
    targets.detach().cpu().numpy()
)

for k, v in metrics.items():
    print(f"{k}: {v:.4f}")




