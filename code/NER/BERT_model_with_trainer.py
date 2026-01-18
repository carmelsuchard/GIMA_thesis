from prepare_dataset_for_BERT import prepare_dataset, make_collator
import pandas as pd
import numpy as np
from tqdm import trange
import torch

from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import trange
from transformers import BertForTokenClassification, TrainingArguments, logging, AutoModelForSequenceClassification, Trainer
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

import numpy as np
import os
import pickle

from transformers import get_scheduler
import torch
import evaluate
from BERT_settings import checkpoint
from code.NER.BERT_model_helper_functions import compute_metrics, count_entities


###################################################################################################
## This script:
# 1. Prepares files in a CONLL-2000 format with a BIO annotaion scheme
# 2. Trains a BERT Named Entity Recognition Model
# 3. Saves the one with the best accuracy
# 4. Generates a figure with the model performance (loss curve)
####################################################################################################

tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True) #TEMP

###### Prepare and tokenize dataset ######
tokenized_datasets, label2id, id2label = prepare_dataset()
data_collator = make_collator()

###### Setting parameters ######

print("Train labels:", count_entities(tokenized_datasets["train"]))
print("Val labels:", count_entities(tokenized_datasets["test"]))

### Hardware settings ###
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = BertForTokenClassification.from_pretrained(
    checkpoint,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
    output_attentions=False,
    output_hidden_states=False,
)

# optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

training_args = TrainingArguments(
    output_dir="./models/ner_bert_model",
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
)

# def compute_metrics_for_trainer(p, tokenized_inputs=None):
#     """
#     Compute token classification metrics using seqeval.
    
#     Args:
#         p: tuple(predictions, labels) from Trainer.
#         tokenized_inputs: optional, the tokenized batch used (to show original tokens).
    
#     Returns:
#         dict of precision, recall, f1
#     """
#     predictions, labels = p
#     preds = predictions.argmax(-1)
    
#     # Convert label IDs to label names, ignoring -100
#     true_labels = [[id2label[l] for l in label_seq if l != -100] 
#                    for label_seq in labels]
#     true_preds = [[id2label[pred] for (pred, l) in zip(pred_seq, label_seq) if l != -100]
#                   for pred_seq, label_seq in zip(preds, labels)]
    
#     # DEBUGGING: show tokens, true/pred labels, and correctness for first few sequences
#     print("="*80)
#     for i, (seq_true, seq_pred) in enumerate(zip(true_labels, true_preds)):
#         if i > 0:  # only show first sequence for clarity
#             break
#         # If tokenized_inputs provided, recover original tokens
#         if tokenized_inputs is not None:
#             tokens = tokenized_inputs[i]["tokens"] if "tokens" in tokenized_inputs[i] else tokenized_inputs[i]["input_ids"]
#         else:
#             tokens = ["<token?>"] * len(seq_true)
        
#         print("First sequence tokens (truncated to 30):", tokens[:30])
#         print("True labels (truncated to 30):", seq_true[:30])
#         print("Pred labels (truncated to 30):", seq_pred[:30])
#         correctness = ["✓" if t==p else "✗" for t, p in zip(seq_true[:30], seq_pred[:30])]
#         print("Correctness (truncated to 30):", correctness[:30])
        
#         # Optional: show token, true, pred, correct in a mini table
#         print("\nToken | True | Pred | Correct")
#         print("-"*35)
#         for tok, t, p, corr in zip(tokens[:30], seq_true[:30], seq_pred[:30], correctness[:30]):
#             print(f"{tok:10} | {t:6} | {p:6} | {corr}")
    
#     print("="*80)
    
#     # Print full classification report
#     report = classification_report(true_labels, true_preds, digits=4)
#     print("Seqeval Classification Report:\n", report)
    
#     # Compute metrics
#     return {
#         "precision": precision_score(true_labels, true_preds),
#         "recall": recall_score(true_labels, true_preds),
#         "f1": f1_score(true_labels, true_preds),
#     }


best_f1_so_far = 0.0
best_epoch_info = None

def compute_metrics_for_trainer(p, tokenized_inputs=None, epoch=None):
    """
    Compute token classification metrics using seqeval, with debug prints and best F1 tracking.

    Args:
        p: tuple(predictions, labels) from Trainer.
        tokenized_inputs: optional, the tokenized batch used (to show original tokens).
        epoch: optional, the current epoch number (int).

    Returns:
        dict of precision, recall, f1
    """
    global best_f1_so_far, best_epoch_info

    predictions, labels = p
    preds = predictions.argmax(-1)
    
    # Convert label IDs to label names, ignoring -100
    true_labels = [[id2label[l] for l in label_seq if l != -100] 
                   for label_seq in labels]
    true_preds = [[id2label[pred] for (pred, l) in zip(pred_seq, label_seq) if l != -100]
                  for pred_seq, label_seq in zip(preds, labels)]
    
    # DEBUG: show first sequence
    print("="*80)
    for i, (seq_true, seq_pred) in enumerate(zip(true_labels, true_preds)):
        if i > 0:  # only first sequence
            break
        tokens = tokenized_inputs[i]["tokens"] if tokenized_inputs is not None and "tokens" in tokenized_inputs[i] else ["<token?>"]*len(seq_true)
        print(f"Epoch {epoch} - First sequence tokens (truncated to 30):", tokens[:30])
        print("True labels (truncated to 30):", seq_true[:30])
        print("Pred labels (truncated to 30):", seq_pred[:30])
        correctness = ["✓" if t==p else "✗" for t, p in zip(seq_true[:30], seq_pred[:30])]
        print("Correctness (truncated to 30):", correctness[:30])
        
        # Mini table
        print("\nToken | True | Pred | Correct")
        print("-"*35)
        for tok, t, p, corr in zip(tokens[:30], seq_true[:30], seq_pred[:30], correctness[:30]):
            print(f"{tok:10} | {t:6} | {p:6} | {corr}")
    
    print("="*80)
    
    # Seqeval metrics
    precision = precision_score(true_labels, true_preds)
    recall = recall_score(true_labels, true_preds)
    f1 = f1_score(true_labels, true_preds)
    
    print(f"Epoch {epoch} metrics -- Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Track best F1
    if f1 > best_f1_so_far:
        best_f1_so_far = f1
        best_epoch_info = {
            "epoch": epoch,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        print(f"*** New best F1 at epoch {epoch}: {f1:.4f} ***")
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_for_trainer
)

trainer.train()

def print_best_epoch():
    global best_epoch_info
    if best_epoch_info:
        print("="*80)
        print(f"Best epoch: {best_epoch_info['epoch']}")
        print(f"Precision: {best_epoch_info['precision']:.4f}")
        print(f"Recall:    {best_epoch_info['recall']:.4f}")
        print(f"F1:        {best_epoch_info['f1']:.4f}")
        print("="*80)


print_best_epoch()


# scheduler = get_scheduler(
#     "linear",
#     optimizer=optimizer,
#     num_warmup_steps=int(0.1 * num_training_steps),
#     num_training_steps=num_training_steps,
# )


    # avg_eval_loss = eval_loss / len(validation_dataloader) # Calculates average validation loss across all batches
    # validation_loss_values.append(avg_eval_loss)  # Saves this epoch's validation loss average to tracking list
    
    # token_accuracy = correct_tokens / total_tokens
    # token_accuracies.append(token_accuracy)
    # print(f"Validation loss: {avg_eval_loss:.4f} | Token accuracy: {token_accuracy:.4f}")

    # # Compute F1 metrics
    # precision, recall, f1, report = compute_metrics(all_predictions, all_labels, id2label)
    # f1_values.append(f1)
    # print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    # print("Per-class report:\n", report)

    # if f1 > best_f1:
    #     best_f1 = f1
    #     best_epoch = epoch
    #     best_model_state = model.state_dict()

# # Restore best model
# model.load_state_dict(best_model_state)
# print(f"Best model restored from epoch {best_epoch+1} with F1 = {best_f1:.4f}")

# # Save full model and tokenizer
# save_path = "best_ner_model"
# os.makedirs(save_path, exist_ok=True)
# model.save_pretrained(save_path)
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# tokenizer.save_pretrained(save_path)













# FIGURE OUT THESE PARAMETERS FOR optimizer and scheduler
# FULL_FINETUNING = True
# if FULL_FINETUNING:
#     param_optimizer = list(model.named_parameters())
#     no_decay = ['bias', 'gamma', 'beta']
#     optimizer_grouped_parameters = [
#         {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
#          'weight_decay_rate': 0.01},
#         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
#          'weight_decay_rate': 0.0}
#     ]
# else:
#     param_optimizer = list(model.classifier.named_parameters())
#     optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]



# n_gpu = torch.cuda.device_count()
# torch.cuda.get_device_name(0)
# print(torch.cuda.get_device_name(0))
# ^ Test these on the school computer

### Model parameters ###
# Batch size would have already been defined
# epsilon = 1e-8      #Adam’s epsilon for numerical stability.
# weight_decay = 0 # form of regularization to lower the chance of overfitting, default is 0
# nr_articles_labeled = 5 DONT KNOW WHAT THESE 3 DO YET
