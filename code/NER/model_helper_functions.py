from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from nervaluate.evaluator import Evaluator
from collections import Counter
import pandas as pd
from collections import namedtuple

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
    f1 = f1_score(true_laplot_learning_curvebels, true_predictions)
    report = classification_report(true_labels, true_predictions, digits=4)
    return precision, recall, f1, report

# def compute_ner_metrics(true_labels, true_predictions):
def compute_ner_metrics(predictions, labels, id2label):
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
        
    evaluator = Evaluator(true_labels, true_predictions, tags=['title', 'spatial', 'author', 'issued', 'inGroup', 'subject'], loader="list")
    results = evaluator.evaluate()
        
    metrics_dict = {key: value['ent_type'].precision for key, value in results["entities"].items()}
    metrics_dict["overall"] = results["overall"]["ent_type"].precision
    metric_per_label = pd.DataFrame([metrics_dict])
    
    print(metric_per_label)

    return metric_per_label
    

def count_entities(dataset):
    counter = Counter()
    for ex in dataset:
        for l in ex["labels"]:
            if l != -100:
                counter[l] += 1
    return counter

if __name__ == "__main__":
    predictions = [['O'], ['B-title', 'O', 'B-spatial', 'I-title', 'I-title', 'B-author', 'I-title', 'O', 'O', 'I-title', 'I-title', 'O', 'I-title', 'I-title', 'I-title', 'I-issued', 'I-issued', 'I-title', 'O', 'I-title']]
    labels = [['O'], ['B-title', 'I-title', 'B-spatial', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-issued', 'I-issued', 'I-issued', 'O', 'O', 'O', 'I-title']]
    compute_ner_metrics(predictions, labels)