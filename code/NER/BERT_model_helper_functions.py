from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from nervaluate.evaluator import Evaluator
from collections import Counter
import pandas as pd
from collections import namedtuple
from BERT_settings import LABELS

def compute_metrics(predictions, labels, id2label):
    print("\n Computing metrics...")    
    true_predictions = []
    true_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        seq_preds = []
        seq_labels = []
        for p, l in zip(pred_seq, label_seq):
            if l != -100:  # Ignore padding
                seq_preds.append(id2label[p])
                seq_labels.append(id2label[l])
        true_predictions.append(seq_preds)
        true_labels.append(seq_labels)

    print("Here is what's going into the calculation of the metrics:")
    print(f"Here are the predictions fyi ({len(true_predictions)}): {true_predictions} \n And here are the labels ({len(true_labels)}): {true_labels}\n")

    precision = precision_score(true_labels, true_predictions)
    recall = recall_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)
    report = classification_report(true_labels, true_predictions, digits=4)
    return precision, recall, f1, report

# def compute_ner_metrics(true_labels, true_predictions):
def compute_ner_metrics(epoch_predictions, epoch_labels, id2label):
    print("\n Computing metrics...")
    doc_predictions = {int(k): v for k, v in epoch_predictions.items()}
    doc_labels = {int(k): v for k, v in epoch_labels.items()}
    
    print(f"This is what the inputs are: doc_predictions: {doc_predictions}, doc_labels:  {doc_labels} ")
    for doc_id in doc_predictions:
        print("Working on doc id:", doc_id)
        
    #     full_labels = []
    #     full_predictions = []
    #     for pred_seq, label_seq in zip(doc_predictions[doc_id], doc_labels[doc_id]):
    #         print(f"Doc ID: {doc_id} \n Predictions: {pred_seq} \n Labels: {label_seq} \n")
    #         seq_preds = []
    #         seq_labels = []
    #         for p, l in zip(pred_seq, label_seq):
    #             if id2label[l] != -100:  # Ignore padding
    #                 seq_preds.append(id2label[p])
    #                 seq_labels.append(id2label[l])
    #         full_predictions.append(seq_preds)
    #         full_labels.append(seq_labels)
        
        
    #     print(f"\n This is the true labels going into the evaluator: {full_labels[0:20]}")
    #     print(f"\n This is the true predictions going into the evaluator: {full_predictions[0:20]}")
        
    #     print("Length of full labels:", len(full_labels), "Length of full predictions:", len(full_predictions))
    #     return


    # evaluator = Evaluator(good_labels, good_predictions, tags=['title', 'spatial', 'author', 'issued', 'inGroup', 'subject', 'AUTHOR', 'TITLE', 'ISSUED'], loader="list")
    # results = evaluator.evaluate()

    # print(evaluator.summary_report())

        
    # metrics_dict = {key: value['ent_type'].precision for key, value in results["entities"].items()}
    # metrics_dict["overall"] = results["overall"]["ent_type"].precision
    # metric_per_label = pd.DataFrame([metrics_dict])
    
    # print(metric_per_label)

    # return metric_per_label
    

def count_entities(dataset):
    counter = Counter()
    for ex in dataset:
        for l in ex["labels"]:
            if l != -100:
                counter[l] += 1
    return counter

def make_label_dicts():
    label2id  = {label: i for i, label in enumerate(LABELS)}
    id2label = {id: label for label, id in label2id.items()}

    return label2id, id2label

if __name__ == "__main__":
    predictions = [['O'], ['B-title', 'O', 'B-spatial', 'I-title', 'I-title', 'B-author', 'I-title', 'O', 'O', 'I-title', 'I-title', 'O', 'I-title', 'I-title', 'I-title', 'I-issued', 'I-issued', 'I-title', 'O', 'I-title']]
    labels = [['O'], ['B-title', 'I-title', 'B-spatial', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-issued', 'I-issued', 'I-issued', 'O', 'O', 'O', 'I-title']]
    compute_ner_metrics(predictions, labels)