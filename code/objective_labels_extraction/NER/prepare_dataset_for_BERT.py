import os
from datasets import Dataset, DatasetDict
from transformers import logging, AutoTokenizer, DataCollatorWithPadding, DataCollatorForTokenClassification
from BERT_settings import checkpoint, training_datasets_path, validation_datasets_path, LABELS, epochs_count
from BERT_model_helper_functions import count_entities, make_label_dicts
import sys
# Import test split
# from sklearn.model_selection import train_test_split

# TAGS_TO_REMOVE = ["summary", "publisher", "dataSource", "method", "question", "goal", "null"]
# replacement_dict = {identifier + bad_tag: "O" for identifier, bad_tag in product(["B-", "I-"], TAGS_TO_REMOVE)}

tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)

def read_conll_file(file_path):
    training_examples = []
    current_tokens = []
    current_labels = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            if not line:
                if current_tokens:
                    training_examples.append({
                        "tokens": current_tokens,
                        "ner_tags": current_labels
                    })
                    current_tokens = []
                    current_labels = []
                continue
            
            try:
                token, label = line.split()
            except Exception as e:
                print(f"{file_path} failed on this line: {line}")
                sys.exit()
                
            current_tokens.append(token)
            current_labels.append(label)
    
    # Save the last section if file doesn't end with blank line
    if current_tokens:
        training_examples.append({
            "tokens": current_tokens,
            "ner_tags": current_labels
        })

    return training_examples

def compile_theses(dir_path):
    print("Compiling .conll files from directory:", dir_path, "...")
    conlls = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f.endswith(".conll")]
    all_examples = []
    for conll in conlls:
        examples = read_conll_file(conll)
        all_examples.extend(examples)
    return all_examples

def convert_to_dataset(thesis_data, label_map):
    return Dataset.from_dict({
        "tokens": [thesis["tokens"] for thesis in thesis_data],
        "ner_tags": [[label_map[l] for l in thesis["ner_tags"]] for thesis in thesis_data],
    })

def tokenize_and_align_labels(examples):
    print("Tokenizing and aligning labels...")
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, max_length=512
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def prepare_dataset():
    print("Preparing dataset...")
    combined_list = compile_theses(training_datasets_path)
    label2id, id2label = make_label_dicts()
    
    raw_dataset = convert_to_dataset(combined_list, label2id)
    
    # Split dataset
    dataset_dict = raw_dataset.train_test_split(test_size=0.3, seed=6)
    
    # Tokenization
    tokenized_datasets = dataset_dict.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=["tokens", "ner_tags"]
    )
    
    return tokenized_datasets, label2id, id2label

def make_collator():
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    return data_collator

if __name__ == "__main__":
    tokenized_datasets, label2id, id2label = prepare_dataset()
    print(id2label)