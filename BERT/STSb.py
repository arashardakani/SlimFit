import json
from pathlib import Path
from tqdm import tqdm
import torch
import gc
import torch
from transformers import AutoModel, AutoTokenizer, BertTokenizer
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig, set_seed, get_scheduler
import numpy as np
from torch.utils.data import DataLoader
from transformers import PretrainedConfig, default_data_collator, DataCollatorWithPadding
from ILSBERT import *
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import random
import re
from AdamW import *

seed = 43
PYTORCH_NO_CUDA_MEMORY_CACHING=1
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




GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

#useful in preprocessing, this sets what each task does. 
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}



#setting some model parameters
task = "stsb"
sentence1_key, sentence2_key = task_to_keys[task]
model_checkpoint = "bert-base-cased"
num_train_epochs = 3
batch_size = 32
pad_to_max_length = True
max_length = 128



set_seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


raw_datasets = load_dataset("glue", task)

is_regression = task == "stsb"
if not is_regression:
    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)
else:
    num_labels = 1

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

config = AutoConfig.from_pretrained(model_checkpoint, num_labels=num_labels, finetuning_task=task)

model = BertForSequenceClassification.from_pretrained(model_checkpoint, config = config)

    
model.to(device)

        
# Preprocessing the datasets

sentence1_key, sentence2_key = task_to_keys[task]

# Some models have set the order of the labels to use, so let's make sure we do use it.
label_to_id = None
if (
    model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
    and task is not None
    and not is_regression
):
    # Some have all caps in their config, some don't.
    label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
    if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
        label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
    
elif task is None:
    label_to_id = {v: i for i, v in enumerate(label_list)}

if label_to_id is not None:
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}
elif task is not None and not is_regression:
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = {id: label for label, id in config.label2id.items()}

padding = "max_length" if pad_to_max_length else False

def preprocess_function(examples):
    # Tokenize the texts
    texts = (
        (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*texts, padding=padding, max_length=max_length, truncation=True)

    if "label" in examples:
        if label_to_id is not None:
            # Map labels to IDs (not necessary for GLUE tasks)
            result["labels"] = [label_to_id[l] for l in examples["label"]]
        else:
            # In all cases, rename the column to labels because the model will expect that.
            result["labels"] = examples["label"]
    return result

#with accelerator.main_process_first():
processed_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["validation_matched" if task == "mnli" else "validation"]

train_max_length = 0
dev_max_length = 0
for item in train_dataset:
    if len(item['input_ids']) > train_max_length:
        train_max_length = len(item['input_ids'])
for item in eval_dataset:
    if len(item['input_ids']) > dev_max_length:
        dev_max_length = len(item['input_ids'])

# DataLoaders creation:
if pad_to_max_length:
    data_collator = default_data_collator
else:
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(None))

train_loader = DataLoader(
    train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size, num_workers=8, pin_memory = True)
val_loader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size, num_workers=8, pin_memory = True)




class ParameterDiffer(object):
    def __init__(self, network):
        network_params = []
        no_params = 0
        for name, p in network.named_parameters():
            if (p.dim() > 1 or re.search('LayerNorm.weight',name)) and not re.search('classifier.weight',name):# and re.search('bert.encoder',name):
                network_params.append(p.data.clone())
                no_params += 1
        self.no_params = no_params
        self.network_params = network_params

    def get_difference(self, network):
        total_diff = torch.zeros(self.no_params)
        i = 0
        for name, p in network.named_parameters():
            if (p.dim() > 1 or re.search('LayerNorm.weight',name)) and not re.search('classifier.weight',name):
                p_np = p.data.clone()
                diff = torch.norm((self.network_params[i] - p_np)/self.network_params[i]/self.network_params[i][:].flatten().shape[0], p = 1)
                total_diff[i] = diff 
                self.network_params[i] = p_np
                i += 1
        return total_diff


diff = ParameterDiffer(model)

num_train_optimization_steps = len(train_loader) * num_train_epochs
lr = 8e-5 # 16e-5 
print("LR:", lr, "Seed:", seed)

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optim = AdamW(optimizer_grouped_parameters, lr=lr, weight_decay = 0.01) #torch.optim.
scheduler = WarmupLinearSchedule(optim, warmup_steps=int(0), t_total=int(1 * num_train_optimization_steps + 1))
lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optim,
        num_warmup_steps=0,
        num_training_steps=num_train_optimization_steps,
    )


t = 0
for name, p in model.named_parameters():
    if (p.dim() > 1 or re.search('LayerNorm.weight',name)) and not re.search('classifier.weight',name):
        t += 1
print(t)

diff_his = []
diff_vec = torch.rand(t) * 10000000000 #1e-8
pitch = int(0.85 * t)
print("pitch:",pitch)
metric = load_metric('glue', task)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for epoch in range(num_train_epochs):
    
    model.train()
    for i, batch in enumerate(tqdm(train_loader)):
        cnt = 0
        sorted_diff = diff_vec.sort(descending=False)[1]
        for name, p in model.named_parameters():
            if (p.dim() > 1 or re.search('LayerNorm.weight',name)) and not re.search('classifier.weight',name):
                if cnt in sorted_diff[0:pitch]:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
                cnt += 1
        
        
        batch = {k: v.to(device) for k, v in batch.items()} 

        outputs = model(**batch)
        
        loss = outputs.loss#[0]#.sum()
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        a = diff.get_difference(model)
        diff_vec[sorted_diff[pitch:]] = a[sorted_diff[pitch:]]
        lr_scheduler.step()      
            
        if i % 2000 ==1999:
            with torch.no_grad():
                model.eval()
                for j, batch1 in enumerate(tqdm(val_loader)):
                    batch1 = {k: v.to(device) for k, v in batch1.items()} 
                    outputs = model(**batch1)
                    metric.add_batch(predictions=outputs.logits.squeeze(), references=batch1["labels"] )
                print(metric.compute())
    with torch.no_grad():
        model.eval()
        for j, batch1 in enumerate(tqdm(val_loader)):
            batch1 = {k: v.to(device) for k, v in batch1.items()} 
            outputs = model(**batch1)
            metric.add_batch(predictions=outputs.logits.squeeze(), references=batch1["labels"] )
        print(metric.compute())
end.record()
print(start.elapsed_time(end))



