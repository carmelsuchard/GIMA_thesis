from collections import namedtuple
from rapidfuzz import fuzz, distance

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

pred = {
    "title": ['een onderzoek naar de koopgerichtheid van huishoudens in nieuwegein', 'GROEIKERN OOK KOOPKERN ?'],
    "author": ['Henk Baten'],
    "issued": ['1982', 'juni 1982'],
    "spatial": ['nieuwegein', 'Jutphaas', 'Vreeswijk', 'Utrecht', 'IJsselstein', 'Vianen', 'Nieuwegein-Noord', 'Nieuwegein-Zuid', 'Jutphaas-Wijkersloot', 'Batau-Noord', 'Batau-Zuid', 'Zuilenstein', 'Doorslag', 'Fokkesteeg', 'Hoog-Zandveld', 'Lekboulevard', 'Kanaleneiland', 'Utrecht-Centrum', 'Utrecht-Overvecht'],
    "inGroup": ['huishoudens', 'migranten', 'woonforensen', 'voormalig Utrechtenaren', 'echtparen', 'alleenstaanden'],
    "subject": ['koopgerichtheid', 'ruimtelijk koopgedrag', 'winkelvoorzieningen', 'koopkrachtbinding', 'distributieve structuur', 'duurzame goederen', 'dagelijks benodigde goederen', 'ruimtelijk beleid', 'ruimtelijke planning']
    
}
ref = {
    "title": ['een onderzoek naar de koopgerichtheid van huishoudens in nieuwegein', 'GROEIKERN OOK KOOPKERN?', 'een onderzoek naar de koopgerichtheid van huishoudens in Nieuwegein'],
    "author": ['Henk Baten'],
    "issued": ['1982', 'juni 1982'],
    "spatial": ['Nieuwegein', 'Nieuwegein', 'Nieuwegein', 'Nieuwegein', 'Nieuwegein', 'de negen woonwijken van Nieuwegein'],
    "inGroup": ['Nieuwegeinse huishoudens', 'Nieuwegeinse huishoudens', 'echtparen', 'Nieuwegeinse huishoudens', 'Nieuwegeinse huishoudens', 'NIEUWEGEINSE HUISHOUDENS'],
    "subject": ['ruimtelijk koopgedrag', 'ruimtelijk koopgedrag', 'winkelvoor - zieningen', 'ruimtelijk koopgedrag', 'ruimtelijk koopgedrag', 'ruimtelijk ( koop - ) gedrag', 'ruimtelijk koopgedrag', 'ruimtelijk koopgedrag', 'koopgerichtheid', 'prefe - renties tructuur', 'koopgerichtheid', 'koopgerichtheid .', 'koopgerichtheid .', 'preferentie - structuur', 'koopgerichtheid', 'ruimtelijk koopgedrag', 'distributieve voorzieningen', 'preferentiestructuur', 'preferen - tiestructuur', 'koopgerichtheid', 'winkelvoorzieningen', 'winkelvoorzieningen', 'winkelvoorzieningen', 'winkelvoorzieningen', 'winkelvoorzieningen', 'winkelvoorzieningen', 'distributieve voorzieningen']
}

def string_similarity(str1, str2):
    dist = fuzz.ratio(str1, str2)
    # print("Calculating string similarity between:", str1, "AND", str2, "GIVES DISTANCE OF:", dist)

    return dist

def determine_entity_captured(ref_ent, predicted_entities_in_label, label):
    #  I want to know if the reference entity is similar enough to any of the predicted entities
    captured_bool = any(string_similarity(ref_ent, pred_ent) > 90 for pred_ent in predicted_entities_in_label)
    print(f"Reference entity: '{ref_ent}' in predicted_entities_in_label '{predicted_entities_in_label}' captured: {captured_bool}")
    return captured_bool


def get_recall(reference_entities_dict, predicted_entities_dict):
    recall_dict = {}    
    rates_dict = {}
    
    references_count = len([ent.lower() for sublist in reference_entities_dict.values() for ent in sublist])

    rates_tuple = namedtuple(typename="rates_tuple", field_names=["count", "weight", "TP", "FN"])
            
    for label in reference_entities_dict:
        # if label != "spatial":
        #     continue
        
        predicted_entities_in_label = [ent.lower() for ent in predicted_entities_dict[label]]
        reference_entities_in_label = [ent.lower() for ent in reference_entities_dict[label]]
        
        captured_bool_list = [determine_entity_captured(ref_ent, predicted_entities_in_label, label) for ref_ent in reference_entities_in_label]
        print(captured_bool_list)
        results = rates_tuple(
            count = len(reference_entities_in_label),
            weight = len(reference_entities_in_label) / references_count,
            TP = sum(captured_bool_list),
            FN = len(reference_entities_in_label) - sum(captured_bool_list)
        )
        
        rates_dict [label] = results
    
    for label in rates_dict:
        counts = rates_dict[label].count
        TP = rates_dict[label].TP
        FN = rates_dict[label].FN
        
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        recall_dict[label] = recall
    
    recall_dict["weighted_overall"] = sum(rates_dict[label].weight * recall_dict[label] for label in recall_dict)
    
    print(recall_dict)
    return recall_dict
                


### Concepts
# Accuracy is a metric that measures how often a machine learning model correctly predicts the outcome.
# You can calculate accuracy by dividing the number of correct predictions by the total number of predictions. 


if __name__ == "__main__":
    get_recall(ref, pred)
    # result = string_similarity('een onderzoek de koopgerichtheid van huishoudens in nieuwegein', 'een onderzoek naar de koopgerichtheid van huishoudens in nieuwegein')
    # print(result)