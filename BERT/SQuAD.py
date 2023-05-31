from datasets import ClassLabel, Sequence
import random
import pandas as pd
from transformers import AutoTokenizer
import transformers
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

from transformers import default_data_collator, get_scheduler

import time, copy, math
from sys import argv
import numpy as np
from pathlib import Path
import re

import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AdamW, DistilBertForQuestionAnswering, set_seed
from datasets import load_dataset, load_metric
from transformers import TrainingArguments, Trainer, DistilBertConfig

from ILSBERT import BertForQuestionAnswering


import collections
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)


seed = 23 #43
print(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
set_seed(seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()

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


class ParameterDiffer(object):
    def __init__(self, network):
        network_params = []
        no_params = 0
        for name, p in network.named_parameters():
            if (p.dim() > 1 or re.search('LayerNorm.weight',name)):# and re.search('bert.encoder',name):
                network_params.append(p.data.clone())
                no_params += 1
        self.no_params = no_params
        self.network_params = network_params

    def get_difference(self, network):
        total_diff = torch.zeros(self.no_params)
        i = 0
        for name, p in network.named_parameters():
            if (p.dim() > 1 or re.search('LayerNorm.weight',name)):
                p_np = p.data.clone()
                
                diff = torch.norm((self.network_params[i] - p_np)/self.network_params[i]/self.network_params[i][:].flatten().shape[0], p = 1)
                total_diff[i] = diff 
                self.network_params[i] = p_np
                i += 1
        return total_diff


class SquadTrainer:
    def __init__(self, model, num_epochs = 3, batch_size = 128, learning_rate = 18e-5):
        self.squad_v2 = True
        model_checkpoint = "bert-base-uncased"

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.max_length = 384 # The maximum length of a feature (question and context)
        self.doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.

        self.datasets = load_dataset("squad_v2" if self.squad_v2 else "squad")

        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        self.pad_on_right = self.tokenizer.padding_side == "right"

        assert isinstance(self.tokenizer, transformers.PreTrainedTokenizerFast)

        tokenized_datasets = self.datasets.map(self.prepare_train_features, batched=True, remove_columns=self.datasets["train"].column_names)

        self.validation_features = self.datasets["validation"].map(
            self.prepare_validation_features,
            batched=True,
            remove_columns=self.datasets["validation"].column_names
        )
        self.eval_loader = DataLoader(tokenized_datasets["validation"], batch_size=batch_size, shuffle=False, collate_fn  = default_data_collator)
        self.model = model.to(self.device)
        self.train_loader = DataLoader(tokenized_datasets["train"], batch_size=batch_size, shuffle=True, collate_fn  = default_data_collator)
        self.batch_size = batch_size
        print(learning_rate)
        self.optim = AdamW(self.model.parameters(), lr=learning_rate, weight_decay = 0.01)
        self.num_epochs = num_epochs
        num_train_optimization_steps = int(len(tokenized_datasets["train"]) / batch_size) * num_epochs
        self.scheduler = get_scheduler(name='linear', optimizer=self.optim, num_warmup_steps=0.1, num_training_steps=num_train_optimization_steps)
        self.diff = ParameterDiffer(self.model)

        





    def train(self):
        t = 0
        for name, p in self.model.named_parameters():
            if (p.dim() > 1 or re.search('LayerNorm.weight',name)):
                t += 1
        print(t)
        diff_vec = torch.rand(t) * 10000000000 #1e-8
        pitch = int(0.8 * t)
        print("pitch:",pitch)
        for epoch in range(self.num_epochs):
            
            self.model.train()
            for i, batch in enumerate(tqdm(self.train_loader)):
                cnt = 0
                sorted_diff = diff_vec.sort(descending=False)[1]
                for name, p in self.model.named_parameters():
                    if (p.dim() > 1 or re.search('LayerNorm.weight',name)):
                        if cnt in sorted_diff[0:pitch]:
                            p.requires_grad = False
                        else:
                            p.requires_grad = True
                        cnt += 1
                self.optim.zero_grad()
                batch = {k: v.to(self.device) for k, v in batch.items()} 
                
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                self.optim.step()
                a = self.diff.get_difference(self.model)
                diff_vec[sorted_diff[pitch:]] = a[sorted_diff[pitch:]]
                self.scheduler.step()
                

            self.model.eval()
            self.evaluate()

    def prepare_train_features(self, examples):
        # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples["question" if self.pad_on_right else "context"],
            examples["context" if self.pad_on_right else "question"],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if self.pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if self.pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    def prepare_validation_features(self, examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples["question" if self.pad_on_right else "context"],
            examples["context" if self.pad_on_right else "question"],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # We keep the example_id that gave us this feature and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if self.pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    def postprocess_qa_predictions(self, examples, features, raw_predictions, n_best_size = 20, max_answer_length = 30):
        all_start_logits, all_end_logits = raw_predictions
        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        # The dictionaries we have to fill.
        predictions = collections.OrderedDict()

        # Logging.
        print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

        # Let's loop over all the examples!
        for example_index, example in enumerate(tqdm(examples)):
            # Those are the indices of the features associated to the current example.
            feature_indices = features_per_example[example_index]

            min_null_score = None # Only used if squad_v2 is True.
            valid_answers = []
            
            context = example["context"]
            # Looping through all the features associated to the current example.
            for feature_index in feature_indices:
                # We grab the predictions of the model for this feature.
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]
                # This is what will allow us to map some the positions in our logits to span of texts in the original
                # context.
                offset_mapping = features[feature_index]["offset_mapping"]

                # Update minimum null prediction.
                cls_index = features[feature_index]["input_ids"].index(self.tokenizer.cls_token_id)
                feature_null_score = start_logits[cls_index] + end_logits[cls_index]
                if min_null_score is None or min_null_score < feature_null_score:
                    min_null_score = feature_null_score

                # Go through all possibilities for the `n_best_size` greater start and end logits.
                start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
                end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                        # to part of the input_ids that are not in the context.
                        if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                        ):
                            continue
                        # Don't consider answers with a length that is either < 0 or > max_answer_length.
                        if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                            continue

                        start_char = offset_mapping[start_index][0]
                        end_char = offset_mapping[end_index][1]
                        valid_answers.append(
                            {
                                "score": start_logits[start_index] + end_logits[end_index],
                                "text": context[start_char: end_char]
                            }
                        )
            
            if len(valid_answers) > 0:
                best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
            else:
                # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
                # failure.
                best_answer = {"text": "", "score": 0.0}
            
            # Let's pick our final answer: the best one or the null answer (only for squad_v2)
            if not self.squad_v2:
                predictions[example["id"]] = best_answer["text"]
            else:
                answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
                predictions[example["id"]] = answer

        return predictions

    def evaluate(self):
        validation_features = self.datasets["validation"].map(
            self.prepare_validation_features,
            batched=True,
            remove_columns=self.datasets["validation"].column_names
        )
        start_logits = []
        end_logits = []
        for j, batch in enumerate(tqdm(self.eval_loader)):
            self.model.eval()
            #batch = tuple(t.to(self.device) for t in batch)
            batch = {k: v.to(self.device) for k, v in batch.items()} 
            with torch.no_grad():
                inputs = {
                    "input_ids": batch['input_ids'],
                    "attention_mask": batch['attention_mask'],
                    "token_type_ids": batch['token_type_ids'],
                }


                outputs = self.model(**inputs)

                start_logits.append(outputs.start_logits)
                end_logits.append(outputs.end_logits)

        start_logits = torch.cat(start_logits, dim = 0).cpu().numpy()
        end_logits = torch.cat(end_logits, dim = 0).cpu().numpy()
        raw_predictions = start_logits, end_logits #self.trainer.predict(validation_features)

        validation_features.set_format(type=validation_features.format["type"], columns=list(validation_features.features.keys()))

        max_answer_length = 30

        for batch in self.eval_loader:
            break
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            output = self.model(**batch)

        start_logits = output.start_logits[0].cpu().numpy()
        end_logits = output.end_logits[0].cpu().numpy()
        offset_mapping = validation_features[0]["offset_mapping"]
        # The first feature comes from the first example. For the more general case, we will need to be match the example_id to
        # an example index
        context = self.datasets["validation"][0]["context"]

        n_best_size = 20
        # Gather the indices the best start/end logits:
        start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
        end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
        valid_answers = []
        for start_index in start_indexes:
            for end_index in end_indexes:
                # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                # to part of the input_ids that are not in the context.
                if (
                    start_index >= len(offset_mapping)
                    or end_index >= len(offset_mapping)
                    or offset_mapping[start_index] is None
                    or offset_mapping[end_index] is None
                ):
                    continue
                # Don't consider answers with a length that is either < 0 or > max_answer_length.
                if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                    continue
                if start_index <= end_index: # We need to refine that test to check the answer is inside the context
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )

        valid_answers = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[:n_best_size]

        examples = self.datasets["validation"]

        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(validation_features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        final_predictions = self.postprocess_qa_predictions(self.datasets["validation"], validation_features, raw_predictions)

        metric = load_metric("squad_v2" if self.squad_v2 else "squad")

        if self.squad_v2:
            formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in final_predictions.items()]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in self.datasets["validation"]]
        print(metric.compute(predictions=formatted_predictions, references=references))


model_checkpoint = "bert-base-uncased"
model = BertForQuestionAnswering.from_pretrained(model_checkpoint)
trainer = SquadTrainer(model)
trainer.train()
































































