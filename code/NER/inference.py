import os
from datasets import Dataset, DatasetDict
import re
from itertools import product
from transformers import logging, AutoTokenizer, DataCollatorWithPadding, DataCollatorForTokenClassification
from transformers import BertForTokenClassification
from BERT_settings import checkpoint, training_datasets_path, LABELS, epochs_count, archive_theses_path
import torch
from BERT_model_helper_functions import make_label_dicts


tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

label2id, id2label = make_label_dicts()
model = BertForTokenClassification.from_pretrained(
    checkpoint,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
    output_attentions=False,
    output_hidden_states=False,
)
model.load_state_dict(torch.load("C:/Users/5298954/Documents/Github_Repos/GIMA_thesis/code/NER/models/model_cleaned_epoch_2_precision_0.3125.pt"))
model.to(device)
model.eval()


def read_text_to_list(file_path):
    print("Reading file...")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        # Split the sentences both along new lines and perriods into lists. Then within the lists, split into words.
        sentences = re.split(r'[\.]', text)
        sentences = [sen.replace('\n', ' ').strip() for sen in sentences if sen.strip()]
        sentences = [sen.split(' ') for sen in sentences if sen]
    
    return sentences

def prepare_sentence_for_inference(sentence_words, tokenizer, device):
    print("Tokenizing...")
    # Tokenize the sentence (sentence_words is already split into words)
    tokenized = tokenizer(
        sentence_words, 
        truncation=True, 
        is_split_into_words=True, 
        max_length=512,
        padding=True,
        return_tensors="pt"
    )
    tokenized = {k: v.to(device) for k, v in tokenized.items()}

    return tokenized


def align_predictions_to_words(tokenized_inputs, predictions, id2label):
    word_predictions = []
    
    for i, pred_seq in enumerate(predictions):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        sentence_preds = []
        
        previous_word_idx = None
        for word_idx, pred in zip(word_ids, pred_seq):
            if word_idx is None:
                # This is a special token ([CLS], [SEP], padding), skip it
                continue
            elif word_idx != previous_word_idx:
                # This is the first token of a word, record the prediction
                sentence_preds.append(id2label[pred])
            # else: this is a subword token of a word we already have a prediction for, skip it
            previous_word_idx = word_idx
        
        word_predictions.append(sentence_preds)
    
    return word_predictions


def get_predictions_for_batch(tokenized_inputs, model):
    with torch.no_grad():
        outputs = model(**tokenized_inputs)
        logits = outputs.logits
    
    predictions = torch.argmax(logits, dim=-1)
    
    return predictions.cpu().numpy()

def get_labels_for_thesis(file_path):
    sentences = read_text_to_list(file_path)
    tokenized_thesis = prepare_sentence_for_inference(sentences, tokenizer, device)
    predictions = get_predictions_for_batch(tokenized_thesis, model)
    word_predictions = align_predictions_to_words(tokenized_thesis, predictions, id2label)
    
    for sent_words, sent_preds in zip(sentences, word_predictions):
        for word, pred in zip(sent_words, sent_preds):
            print(f"{word}: {pred}")
        print("-" * 40)


if __name__ == "__main__":
    
    archive_theses_path = "C:/Users/5298954/Documents/Github_Repos/GIMA_thesis/code/full_archive/original/English/1982_Overtoom_Paul_Regional_economic_development_in_northwest_Mexico_UU.txt"
    get_labels_for_thesis(archive_theses_path)
    