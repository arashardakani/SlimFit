
import json
from pathlib import Path
from tqdm import tqdm
#from PIL import Image
from datasets import load_metric
from transformers import ViTFeatureExtractor
from datasets import Features, ClassLabel, Array3D, Image
from transformers import default_data_collator, set_seed
from ILSViT import ViTForImageClassification
#from transformers import ViTForImageClassification
from transformers import TrainingArguments, Trainer
#from transformers import AdamW
from AdamW import *
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from datasets import load_dataset
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import random
import re
import gc
seed = 43

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

set_seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32
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



data = load_dataset('cifar10')

train = data['train']
train_ds = train
test_ds = data['test']


feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k", seed = 43)


def preprocess_images(examples):

    images = examples['img']
    images = [np.array(image, dtype=np.uint8) for image in images]
    images = [np.moveaxis(image, source=-1, destination=0) for image in images]
    inputs = feature_extractor(images=images)
    examples['pixel_values'] = inputs['pixel_values']

    return examples


features = Features({
    'label': ClassLabel(names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']),
    'img': Image(decode=True, id=None),
    'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
})



preprocessed_train_ds = train_ds.map(preprocess_images, batched=True, features=features)
preprocessed_test_ds = test_ds.map(preprocess_images, batched=True, features=features)

preprocessed_train_ds = preprocessed_train_ds.remove_columns('img')
preprocessed_val_ds = preprocessed_test_ds.remove_columns('img')



data_collator = default_data_collator

num_train_optimization_steps = int(
        len(preprocessed_train_ds) / batch_size) * num_train_epochs
train_data = preprocessed_train_ds
train_dataloader = DataLoader(train_data,
                                shuffle=True,
                                collate_fn=data_collator,
                                pin_memory = True,
                                num_workers=8,
                                batch_size=batch_size)


eval_dataloader = DataLoader(preprocessed_val_ds,
                                shuffle=False,
                                collate_fn=data_collator,
                                pin_memory = True,
                                num_workers=8,
                                batch_size=batch_size)

metric = load_metric("accuracy")



model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=10).to(device)
model.to(device)
model.train()

model.classifier.weight.data.normal_(mean=0.0, std=model.config.initializer_range)



t = 0
for name, p in model.named_parameters():
    if (p.dim() > 1 or re.search('layernorm_before.weight',name) or re.search('layernorm_after.weight',name)) and not re.search('classifier.weight',name):
        t += 1
print(t)


class ParameterDiffer(object):
    def __init__(self, network):
        network_params = []
        no_params = 0
        for name, p in network.named_parameters():
            if (p.dim() > 1 or re.search('layernorm_before.weight',name) or re.search('layernorm_after.weight',name)) and not re.search('classifier.weight',name):
                network_params.append(p.data.clone())
                no_params += 1
        self.no_params = no_params
        self.network_params = network_params

    def get_difference(self, network):
        total_diff = torch.zeros(self.no_params)
        i = 0
        for name, p in network.named_parameters():
            if (p.dim() > 1 or re.search('layernorm_before.weight',name) or re.search('layernorm_after.weight',name)) and not re.search('classifier.weight',name):
                p_np = p.data.clone()
                diff = torch.norm((self.network_params[i] - p_np)/self.network_params[i]/self.network_params[i][:].flatten().shape[0], p = 1) 
                total_diff[i] = diff
                self.network_params[i] = p_np#.clone()
                i += 1
        return total_diff


diff = ParameterDiffer(model)
lr = 6.5e-5
num_train_optimization_steps = len(train_dataloader) * num_train_epochs

optim = AdamW(model.parameters(), lr=lr)
scheduler = WarmupLinearSchedule(optim, warmup_steps=int(0. * num_train_optimization_steps), t_total= num_train_optimization_steps + 1)
scheduler.step()

diff_vec = torch.rand(t) * 10000000 
pitch = int(0.90 * t)
print(pitch, lr, seed)
for epoch in range(num_train_epochs):
    
    model.train()
    for i, batch in enumerate(tqdm(train_dataloader)):
        cnt = 0
        sorted_diff = diff_vec.sort(descending=False)[1]
        for name, p in model.named_parameters():
            if (p.dim() > 1 or re.search('layernorm_before.weight',name) or re.search('layernorm_after.weight',name)) and not re.search('classifier.weight',name):
                if cnt in sorted_diff[0:pitch]:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
                cnt += 1

        optim.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()} 
        outputs = model(**batch)

        
        loss = outputs.loss
        loss.backward()
        optim.step()
        a = diff.get_difference(model)
        diff_vec[sorted_diff[pitch:]] = a[sorted_diff[pitch:]]
        scheduler.step()     
    with torch.no_grad():
        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []
        labels = []
        for j, batch1 in enumerate(tqdm(eval_dataloader)):
            batch1 = {k: v.to(device) for k, v in batch1.items()} 
            output_eval = model(**batch1)
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(output_eval.logits.detach().cpu().numpy())
                labels.append(batch1["labels"].detach().cpu().numpy())
            else:
                preds[0] = np.append(preds[0],
                                    output_eval.logits.detach().cpu().numpy(),
                                    axis=0)
                labels[0] = np.append(labels[0],
                                    batch1["labels"].detach().cpu().numpy(),
                                    axis=0)
        preds = preds[0]
        labels = labels[0]
        preds = np.argmax(preds, axis=1)
        #result = {}
        result = compute_metrics(preds, labels)
        print(result)


