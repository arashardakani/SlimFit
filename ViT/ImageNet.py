
import json
from pathlib import Path
from tqdm import tqdm
#from PIL import Image
from datasets import load_metric
from transformers import ViTFeatureExtractor
from datasets import Features, ClassLabel, Array3D, Image
from transformers import default_data_collator, set_seed
from ILSViT import ViTForImageClassification
#from transformers import ViTForImageClassification, set_seed
from transformers import TrainingArguments, Trainer
from transformers import AdamW

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from datasets import load_dataset
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import random
import re
import gc
import torchvision
import torchvision.transforms as transforms
from loaddatasets import build_transform
seed = 43

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


set_seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128
num_train_epochs = 3


metric = load_metric("accuracy")
def compute_metrics(eval_pred, labels):

    #predictions, labels = eval_pred
    out = metric.compute(predictions=eval_pred, references=labels)
    return out

class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))




train_path = 'Path_to_train_folder'
val_path = 'Path_to_val_folder'


transform_train = build_transform(True)
transform_valid = build_transform(False)
train_imagenet_data = torchvision.datasets.ImageFolder(train_path, transform=transform_train)
val_imagenet_data = torchvision.datasets.ImageFolder(val_path, transform=transform_valid)

 




num_train_optimization_steps = int(
        len(train_imagenet_data) / batch_size) * num_train_epochs

train_dataloader = DataLoader(train_imagenet_data,
                                shuffle=True,
                                pin_memory = True,
                                num_workers=8,
                                batch_size=batch_size)


eval_dataloader = DataLoader(val_imagenet_data,
                                shuffle=False,
                                pin_memory = True,
                                num_workers=8,
                                batch_size=batch_size)

metric = load_metric("accuracy")



model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-384', num_labels=1000).to(device)

model.to(device)
model.train()

t = 0
for name, p in model.named_parameters():
    if (p.dim() > 1 or re.search('layernorm_before.weight',name) or re.search('layernorm_after.weight',name)): 
        t += 1
        
print(t)


class ParameterDiffer(object):
    def __init__(self, network):
        network_params = []
        no_params = 0
        for name, p in network.named_parameters():
            if (p.dim() > 1 or re.search('layernorm_before.weight',name) or re.search('layernorm_after.weight',name)):
                network_params.append(p.data.clone())
                no_params += 1
        self.no_params = no_params
        self.network_params = network_params

    def get_difference(self, network):
        total_diff = torch.zeros(self.no_params)
        i = 0
        for name, p in network.named_parameters():
            if (p.dim() > 1 or re.search('layernorm_before.weight',name) or re.search('layernorm_after.weight',name)):
                p_np = p.data.clone()
                diff = torch.norm((self.network_params[i] - p_np)/self.network_params[i]/self.network_params[i][:].flatten().shape[0], p = 1) 
                total_diff[i] = diff
                self.network_params[i] = p_np#.clone()
                i += 1
        return total_diff


diff = ParameterDiffer(model)
lr = 5e-5
num_train_optimization_steps = len(train_dataloader) * num_train_epochs

optim = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = WarmupLinearSchedule(optim, warmup_steps=int(0. * num_train_optimization_steps), t_total=num_train_optimization_steps + 1)
scheduler.step()

diff_his = []
diff_vec = torch.rand(t) * 1e20 
pitch = int(0.95 * t)
print(pitch, lr, seed)
for epoch in range(num_train_epochs):
    
    model.train()
    for i, batch in enumerate(tqdm(train_dataloader)):
        cnt = 0
        model.train()
        sorted_diff = diff_vec.sort(descending=False)[1]
        for name, p in model.named_parameters():
            if (p.dim() > 1 or re.search('layernorm_before.weight',name) or re.search('layernorm_after.weight',name)):# and not (re.search('classifier.weight',name) or re.search('embeddings',name)):
                if cnt in sorted_diff[0:pitch]:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
                cnt += 1

        optim.zero_grad()
        input = {
                    "pixel_values": batch[0].to(device),
                    "labels": batch[1].to(device)
                }
        outputs = model(**input)
        loss = outputs.loss
        loss.backward()
        optim.step()
        a = diff.get_difference(model)
        diff_vec[sorted_diff[pitch:]] = a[sorted_diff[pitch:]]
        scheduler.step()   

        if ((i % 16000)==0) and not (i == 0):
            with torch.no_grad():
                model.eval()
                eval_loss = 0
                nb_eval_steps = 0
                preds = []
                labels = []
                for j, batch1 in enumerate(tqdm(eval_dataloader)):
                    input = {
                            "pixel_values": batch1[0].to(device),
                            "labels": batch1[1].to(device)
                        }
                    output_eval = model(**input)
                    nb_eval_steps += 1
                    if len(preds) == 0:
                        preds.append(output_eval.logits.detach().cpu().numpy())
                        labels.append(input["labels"].detach().cpu().numpy())
                    else:
                        preds[0] = np.append(preds[0],
                                            output_eval.logits.detach().cpu().numpy(),
                                            axis=0)
                        labels[0] = np.append(labels[0],
                                            input["labels"].detach().cpu().numpy(),
                                            axis=0)
                preds = preds[0]
                labels = labels[0]
                preds = np.argmax(preds, axis=1)
                #result = {}
                result = compute_metrics(preds, labels)
                print(result)
         
    with torch.no_grad():
        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []
        labels = []
        for j, batch1 in enumerate(tqdm(eval_dataloader)):
            input = {
                    "pixel_values": batch1[0].to(device),
                    "labels": batch1[1].to(device)
                }
            output_eval = model(**input)
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(output_eval.logits.detach().cpu().numpy())
                labels.append(input["labels"].detach().cpu().numpy())
            else:
                preds[0] = np.append(preds[0],
                                    output_eval.logits.detach().cpu().numpy(),
                                    axis=0)
                labels[0] = np.append(labels[0],
                                    input["labels"].detach().cpu().numpy(),
                                    axis=0)
        preds = preds[0]
        labels = labels[0]
        preds = np.argmax(preds, axis=1)
        #result = {}
        result = compute_metrics(preds, labels)
        print(result)
