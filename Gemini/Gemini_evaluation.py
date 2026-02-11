from collections import namedtuple
from thefuzz import fuzz
import re
import pandas as pd
import ast
from boxplot_metrics import boxplot_metrics 

# This file is used to evaluate NER performance on thesis documents using the Gemini model. It takes a different approach compared to standard NER inference.


# I think that the only thing I need to do is to go through each of the captured entities and see if the model predicted it. Line-by-line doesn't matter so much.
# However, I do want to know what this is called bcause it is not a standard batch by batch thing like in regular NER

# TODO: Go ghtough the entities captured from the thesis and see if they are in the model predictions.
# TODO: Do the metrics calculation based on that.
# TODO: Come up with fuzziness
# TODO: incorporate the differnet tags
# Return a result for each tag, and then a weighted one combined
# TODO: Figure out how much fuziness is okay
# TODO: A special comparison for the titles? <- Wait, I think I already wrote some of these
# TODO: These are based only on string simliarity, not semantic similarity
# TODO: Implement the fuzziness into the metrics calculation



# Get a dict of lists of refernece entities, and a list of predictions
# Write function that can be applied to each one
# Do some counting




def normalize_string(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def string_similarity(str1, str2):
    str1 = normalize_string(str1)
    str2 = normalize_string(str2)
    
    dist = fuzz.ratio(str1, str2)
    # print("Calculating string similarity between:", str1, "AND", str2, "GIVES DISTANCE OF:", dist)

    return dist

def determine_entity_captured(entity_to_check, entity_list_to_compare_against, label):
    #  I want to know if the reference entity is similar enough to any of the predicted entities
    captured_bool = any(string_similarity(entity_to_check, ent) > 90 for ent in entity_list_to_compare_against)
    # print(f"Reference entity: '{entity_to_check}' in predicted_entities_in_label '{entity_list_to_compare_against}' captured: {captured_bool}")
    return captured_bool


def compute_metrics(reference_entities_dict, predicted_entities_dict):
    metrics_dict = {}    
    rates_dict = {}
    
    references_count = len([ent.lower() for sublist in reference_entities_dict.values() for ent in sublist])

    rates_tuple = namedtuple(typename="rates_tuple", field_names=["count", "weight", "TP", "FN", "FP"])
            
    for label in reference_entities_dict:
        # if label != "spatial":
        #     continue
        
        reference_entities_in_label = list(set([ent.lower() for ent in reference_entities_dict[label]]))
        predicted_entities_in_label = list(set([ent.lower() for ent in predicted_entities_dict[label]]))
        
        captured_bool_list = [determine_entity_captured(ref_ent, predicted_entities_in_label, label) for ref_ent in reference_entities_in_label]
        predicted_in_reference = [determine_entity_captured(pred_ent, reference_entities_in_label, label) for pred_ent in predicted_entities_in_label]

        results = rates_tuple(
            count = len(reference_entities_in_label),
            weight = len(reference_entities_in_label) / references_count,
            TP = sum(captured_bool_list), # How many of the reference entities were captured
            FN = len(reference_entities_in_label) - sum(captured_bool_list) , # How many of the reference entities were NOT captured
            FP = len(predicted_entities_in_label) - sum(predicted_in_reference) # How many extra entities were predicted that were NOT in the reference (subtraction because we want those not captured)
        )
        
        rates_dict [label] = results
    
    for label in rates_dict:
        counts = rates_dict[label].count
        TP = rates_dict[label].TP
        FN = rates_dict[label].FN
        FP = rates_dict[label].FP
        
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        jaccard = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

        metrics_dict[label] = {
            "recall": round(recall, 2),
            "precision": round(precision, 2),
            "f1": round(f1, 2),
            "jaccard": round(jaccard, 2)
        }
    
    metrics_dict["weighted_overall"] = {}
    metrics_dict["weighted_overall"]["recall"] = round(sum(rates_dict[label].weight * metrics_dict[label]["recall"] for label in metrics_dict if label != "weighted_overall"), 2)
    metrics_dict["weighted_overall"]["precision"] = round(sum(rates_dict[label].weight * metrics_dict[label]["precision"] for label in metrics_dict if label != "weighted_overall"), 2)
    metrics_dict["weighted_overall"]["f1"] = round(sum(rates_dict[label].weight * metrics_dict[label]["f1"] for label in metrics_dict if label != "weighted_overall"), 2)
    metrics_dict["weighted_overall"]["jaccard"] = round(sum(rates_dict[label].weight * metrics_dict[label]["jaccard"] for label in metrics_dict if label != "weighted_overall"), 2)


    return metrics_dict
    # return metrics_dict
                
if __name__ == "__main__":
    # predictions = pd.read_csv(r"C:\Users\carme\OneDrive\Documents\Git_Repos\GIMA_thesis\code\NER\inference_results\multimodal_results.csv")
    # references = pd.read_csv(r"C:\Users\carme\OneDrive\Documents\Git_Repos\GIMA_thesis\code\data_cleaning\annotations_summary.csv")
    
    # combined_recall_dict = {"title": [], "author": [], "issued": [], "spatial": [], "inGroup": [], "subject": [], "weighted_overall": []}
    # combined_precision_dict = combined_recall_dict.copy()
    # combined_f1_dict = combined_recall_dict.copy()
    # combined_jaccard_dict = combined_recall_dict.copy()
    
    # for row in predictions.iterrows():
    #     # print("Processing row:", row)
    #     file_name = row[1]["Thesis_File"].replace("('", "").replace("',)", "")
    #     print(file_name)
    #     pred = {
    #         "title": row[1]["title"],
    #         "author": row[1]["author"],
    #         "issued": row[1]["issued"],
    #         "spatial": row[1]["spatial"],
    #         "inGroup": row[1]["inGroup"],
    #         "subject": row[1]["subject"]
    #     }
    #     pred = {k: list(ast.literal_eval(v)) for k, v in pred.items()}
        
    #     reference_row = references[references["Thesis_File"] == file_name]
    #     ref = {
    #         "title": reference_row["title"].values[0],
    #         "author": reference_row["author"].values[0],
    #         "issued": reference_row["issued"].values[0],
    #         "spatial": reference_row["spatial"].values[0],
    #         "inGroup": reference_row["inGroup"].values[0],
    #         "subject": reference_row["subject"].values[0]
    #     }
    #     ref = {k: [] if pd.isna(v) else list(ast.literal_eval(v))for k, v in ref.items()}
        
    
    #     metrics_dict = compute_metrics(ref, pred)
    #     # print("Metrics dicts:", metrics_dict)
        
    #     for label in metrics_dict:
    #         # print(metrics_dict[label])
    #         combined_recall_dict[label] = combined_recall_dict[label] + [metrics_dict[label]["recall"]]
    #         combined_precision_dict[label] = combined_precision_dict[label] + [metrics_dict[label]["precision"]]
    #         combined_f1_dict[label] = combined_f1_dict[label] + [metrics_dict[label]["f1"]]
    #         combined_jaccard_dict[label] = combined_jaccard_dict[label] + [metrics_dict[label]["jaccard"]]
        
    # print("Combined recall dict:", combined_recall_dict)
    # print("Combined precision dict:", combined_precision_dict)
    # print("Combined f1 dict:", combined_f1_dict)
    # print("Combined jaccard dict:", combined_jaccard_dict)
    
    # boxplot_metrics(combined_recall_dict, "Multimodal Gemini NER Recall",
    #                 r"C:\Users\carme\OneDrive\Documents\Git_Repos\GIMA_thesis\code\NER\Figures\multimodal_recall_boxplot.png")
    
    # boxplot_metrics(combined_precision_dict, "Multimodal Gemini NER Precision",
    #                 r"C:\Users\carme\OneDrive\Documents\Git_Repos\GIMA_thesis\code\NER\Figures\multimodal_precision_boxplot.png")
    
    # boxplot_metrics(combined_f1_dict, "Multimodal Gemini NER F1 Score",
    #                 r"C:\Users\carme\OneDrive\Documents\Git_Repos\GIMA_thesis\code\NER\Figures\multimodal_f1_boxplot.png")
        
    # boxplot_metrics(combined_jaccard_dict, "Multimodal Gemini NER Jaccard Index",
    #                 r"C:\Users\carme\OneDrive\Documents\Git_Repos\GIMA_thesis\code\NER\Figures\multimodal_jaccard_boxplot.png")
        
    #     data = {
    #     'title': [0.9, 0.85, 0.88, 0.92, 0.87],
    #     'author': [0.8, 0.75, 0.78, 0.82, 0.77],
    #     'issued': [0.95, 0.9, 0.93, 0.96, 0.91],
    #     'spatial': [0.7, 0.65, 0.68, 0.72, 0.67],
    #     'inGroup': [0.85, 0.8, 0.83, 0.87, 0.82],
    #     'subject': [0.9, 0.88, 0.89, 0.91, 0.87]
    # }    
        
    # result = string_similarity('een onderzoek de koopgerichtheid van huishoudens in nieuwegein', 'een onderzoek naar de koopgerichtheid van huishoudens in nieuwegein')
    # print(result)
    
    
#     ref = {
#     "title": ['een onderzoek naar de koopgerichtheid van huishoudens in nieuwegein', 'GROEIKERN OOK KOOPKERN?', 'een onderzoek naar de koopgerichtheid van huishoudens in Nieuwegein'],
#     "author": ['Henk Baten'],
#     "issued": ['1982', 'juni 1982'],
#     "spatial": ['Nieuw....egein', 'Nieuwegein', 'Nieuwegein', 'Nieuwegein', 'Nieuwegein', 'de negen woonwijken van Nieuwegein'],
#     "inGroup": ['Nieuwegeinse huishoudens', 'Nieuwegeinse huishoudens', 'echtparen', 'Nieuwegeinse huishoudens', 'Nieuwegeinse huishoudens', 'NIEUWEGEINSE HUISHOUDENS'],
#     "subject": ['ruimtelijk koopgedrag', 'ruimtelijk koopgedrag', 'winkelvoor - zieningen', 'ruimtelijk koopgedrag', 'ruimtelijk koopgedrag', 'ruimtelijk ( koop - ) gedrag', 'ruimtelijk koopgedrag', 'ruimtelijk koopgedrag', 'koopgerichtheid', 'prefe - renties tructuur', 'koopgerichtheid', 'koopgerichtheid .', 'koopgerichtheid .', 'preferentie - structuur', 'koopgerichtheid', 'ruimtelijk koopgedrag', 'distributieve voorzieningen', 'preferentiestructuur', 'preferen - tiestructuur', 'koopgerichtheid', 'winkelvoorzieningen', 'winkelvoorzieningen', 'winkelvoorzieningen', 'winkelvoorzieningen', 'winkelvoorzieningen', 'winkelvoorzieningen', 'distributieve voorzieningen']
# }

# pred = {
#     "title": ['een onderzoek naar de koopgerichtheid van huishoudens in nieuwegein', 'GROEIKERN OOK KOOPKERN ?'],
#     "author": ['Henk Baten'],
#     "issued": ['1982', 'juni 1982'],
#     "spatial": ['nieuwegein', 'Jutphaas', 'Vreeswijk', 'Utrecht', 'IJsselstein', 'Vianen', 'Nieuwegein-Noord', 'Nieuwegein-Zuid', 'Jutphaas-Wijkersloot', 'Batau-Noord', 'Batau-Zuid', 'Zuilenstein', 'Doorslag', 'Fokkesteeg', 'Hoog-Zandveld', 'Lekboulevard', 'Kanaleneiland', 'Utrecht-Centrum', 'Utrecht-Overvecht'],
#     "inGroup": ['huishoudens', 'migranten', 'woonforensen', 'voormalig Utrechtenaren', 'echtparen', 'alleenstaanden'],
#     "subject": ['koopgerichtheid', 'ruimtelijk koopgedrag', 'winkelvoorzieningen', 'koopkrachtbinding', 'distributieve structuur', 'duurzame goederen', 'dagelijks benodigde goederen', 'ruimtelijk beleid', 'ruimtelijke planning']
    
# }

