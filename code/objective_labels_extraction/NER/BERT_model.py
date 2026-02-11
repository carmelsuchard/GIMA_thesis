from prepare_dataset_for_BERT import prepare_dataset, make_collator
import pandas as pd
import numpy as np
from tqdm import trange
import torch
from collections import defaultdict

from transformers import BertForTokenClassification, BertTokenizer, logging
from seqeval.metrics import accuracy_score
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import trange
from transformers import BertForTokenClassification, BertTokenizer, logging, AutoModelForSequenceClassification, DataCollatorWithPadding
import numpy as np
import os
import pickle

from transformers import get_scheduler
import torch
from BERT_settings import checkpoint, epochs_count, LABELS, batch_size, training_datasets_path
from BERT_model_helper_functions import compute_metrics, count_entities, compute_ner_metrics
from plot_loss import plot_learning_curve

from time import time


###################################################################################################
## This script:
# 1. Prepares files in a CONLL-2000 format with a BIO annotaion scheme
# 2. Trains a BERT Named Entity Recognition Model
# 3. Saves the one with the best accuracy
# 4. Generates a figure with the model performance (loss curve)
####################################################################################################
start_time = time()

tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)

###### Prepare and tokenize dataset ######
tokenized_datasets, label2id, id2label = prepare_dataset()
data_collator = make_collator()

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator
)
validation_dataloader = DataLoader(
    tokenized_datasets["test"], batch_size=batch_size, collate_fn=data_collator
)

###### Setting parameters ######

print("Train labels:", count_entities(tokenized_datasets["train"]))
print("Val labels:", count_entities(tokenized_datasets["test"]))

### Hardware settings ###
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Model is on device: {device}")

model = BertForTokenClassification.from_pretrained(
    checkpoint,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
    output_attentions=False,
    output_hidden_states=False,
)

model.to(device)

learning_rate = 4e-5
num_training_steps = epochs_count * len(train_dataloader)
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps,
)

best_f1 = -1
best_epoch = 0
best_model_state = None

training_loss_values, validation_loss_values, token_accuracies = [], [], []
unique_labels = list(set([label.split("-")[1] for label in LABELS if label != "O"]))
overall_metrics_df = pd.DataFrame(columns=unique_labels)

starting_training_time = time()
print("Tokenization time: %.2f seconds" % (starting_training_time - start_time))
###### Training loop ######
for epoch in trange(epochs_count, desc="Epoch"):
    start_epoch_time = time()
    # --- Training ---
    print("\n Going through training epoch ", epoch+1)
    model.train() # Set model to training mode
    total_loss = 0
    total_sequences = 0

    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss
        # print(f"Training loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        scheduler.step()


        size_of_this_batch = batch["input_ids"].size(0)
        total_loss += loss.item() * size_of_this_batch
        total_sequences += size_of_this_batch

    avg_train_loss = total_loss / total_sequences
    training_loss_values.append(avg_train_loss)
    print(f"Epoch {epoch+1} | Training loss: {avg_train_loss:.4f}")

    # --- Validation ---
    model.eval() # Set model to evaluation mode
    print("\n Going through validation epoch ", epoch+1)
    eval_loss = 0
    total_sequences = 0
    correct_tokens = 0
    total_tokens = 0

    all_predictions, all_labels = [], []

    with torch.no_grad():
        for batch in validation_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            size_of_this_batch = batch["input_ids"].size(0)
            eval_loss += loss.item() * size_of_this_batch # Add this batch's loss to the total
            total_sequences += size_of_this_batch

            predictions = torch.argmax(logits, dim=-1) # Takes the logits (raw scores) and picks the highest one, and get the prediction
            labels = batch["labels"] # Get the true labels from the batch

            # Calculate how many of none-padded tokens were correctly predicted
            mask = labels != -100
            correct_tokens += (predictions[mask] == labels[mask]).sum().item()
            total_tokens += mask.sum().item()

            # Convert the original ids, predictions and labels to numpy arrays for metric computation
            input_ids = batch["input_ids"].cpu().numpy()
            preds = predictions.cpu().numpy()
            labs = labels.cpu().numpy()

            for sentence_ids, pred_seq, label_seq in zip(input_ids, preds, labs): # Iterate over the individual sequences in the batch, getting their id, predictions and labels
                seq_preds = []
                seq_labels = []
                
                for p, l in zip(pred_seq, label_seq):
                    if l != -100:
                        seq_preds.append(p)
                        seq_labels.append(l)
                
                # Collect all predictions and labels for metric computation
                all_predictions.append(seq_preds)
                all_labels.append(seq_labels)
                
                # Only print on last epoch
            # if epoch == epochs - 1:
                # print("This is epoch ", epoch+1, " validation example:")
                tokens = tokenizer.convert_ids_to_tokens(sentence_ids) # Converts token IDs back to actual words
                clean_tokens = [tok for tok, l in zip(tokens, label_seq) if l != -100]
                clean_preds = [id2label[p] for p in seq_preds]
                clean_labels = [id2label[l] for l in seq_labels]

                print("TOKENS: ", clean_tokens)
                print("REFERENCE LABELS: ", clean_labels)
                print("PREDICTED LABELS: ", clean_preds)
                print("-" * 60)

    avg_eval_loss = eval_loss / total_sequences # Calculates average validation loss across all batches
    validation_loss_values.append(avg_eval_loss)  # Saves this epoch's validation loss average to tracking list
    
    token_accuracy = correct_tokens / total_tokens
    token_accuracies.append(token_accuracy)
    print(f"Validation loss: {avg_eval_loss:.4f} | Token accuracy: {token_accuracy:.4f}")

 # Compute F1 metrics
    metrics_table = compute_metrics(all_predictions, all_labels, id2label)
    overall_metrics_df = pd.concat([overall_metrics_df, metrics_table])
    
    
    f1 = metrics_table["overall"].values[0]
    if f1 >= best_f1:
        best_f1 = f1
        best_epoch = epoch
        best_model_state = model.state_dict()

    end_epoch_time = time()
    print("Epoch time: %.2f seconds" % (end_epoch_time - start_epoch_time))
    print("================= END OF EPOCH ==============\n")

# print("Token accuracies:", token_accuracies)
overall_metrics_df["training_loss"] = training_loss_values
overall_metrics_df["validation_loss"] = validation_loss_values

output_model = ("objective_labels_extraction/NER/Models/model_" + os.path.basename(training_datasets_path) + "_epoch_" + str(best_epoch+1) +'_f1_' + str(best_f1) + '.pt')
torch.save(best_model_state, output_model)

# # Visualize the training loss
print("This is the final metrics df, going into the plot:", overall_metrics_df)
learning_curve = plot_learning_curve(overall_metrics_df, metric_name="F1")
learning_curve.savefig('objective_labels_extraction/NER/Models/learning_curve_' + os.path.basename(training_datasets_path) + "_epoch_" + str(best_epoch+1) +'_f1_' + str(best_f1) + '.png', bbox_inches='tight')
learning_curve.show()
learning_curve.close()

# Restore best model
model.load_state_dict(best_model_state)
print(f"Best model restored from epoch {best_epoch+1} with F1 = {best_f1:.4f}")











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
