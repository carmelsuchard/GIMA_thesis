# 1. Read the annotations of a thesis
# 2. For spatial annotations, put them through one process
# 3. For other annotations, put them through a separate process 
# 3.25 - Convert to Lemas?
#. Semantic normalization!
# 3.30 Word Sense Disambiguation (WSD). -> Pick using word similarily either lesk or another one
# 3.5 - Concept linking
#4. For the diambiguation process, first try to find something similar in wordnet/babel net
import json
import requests
from annotations_reader import conll_annotations_reader
import spacy

base_url = "https://babelnet.io/v9/"
key = "6fe98d9f-cb06-4434-81c8-add6f9903089"

def get_noun_lemmas(text):
    nlp = spacy.load("nl_core_news_sm")
    doc = nlp(text)
    token_details = []
    for idx, token in enumerate(doc):
        # token_details.append((idx, token.text, token.lemma_, token.pos_, token.tag_, token.dep_))
        token_details.append((idx, token.lemma_, token.text, token.pos_))
        # print(token_details)

    noun_tokens = [(lemma, text, pos) for idx, lemma, text, pos in token_details if pos == "NOUN"]
    prop_noun_tokens = [(lemma, text, pos) for idx, lemma, text, pos in token_details if pos == "PROPN"]
    return noun_tokens, prop_noun_tokens

def find_existence_in_babelnet(lemma, return_language="EN"):
    service_url = f'{base_url}/getSynsetIds'
    params = {
        'lemma' : lemma,
        'searchLang' : return_language,
        'key'  : key
    }
    header = {"Accept-Encoding" : "gzip"}

    response = requests.get(service_url, params=params, headers=header)    
    response.raise_for_status()
    
    synsets = response.json()
    return synsets if synsets else None


if __name__ == "__main__":
    
    # conll_path = r"C:\Users\carme\OneDrive\Documents\Graduate_Programs\Thesis\code\annotated_conll_files\cleaned\1971_Abcouwer_NF_Ontwikkelingen_in_de_industriele_structuur_van_zuid-west-Nederland_1945-1970.conll"
    conll_path = r"C:\Users\carme\OneDrive\Documents\Graduate_Programs\Thesis\code\annotated_conll_files\cleaned\1974_Harts_Jan_Migratie.conll"
    
    reader = conll_annotations_reader(conll_path)
    tags_dict = reader.get_tags_dict()
    
    for value in tags_dict["subject"]:
        noun_tokens, prop_noun_tokens = get_noun_lemmas(value)
        print(noun_tokens, prop_noun_tokens)
        
    #     for noun in noun_tokens:
    #         lemma, text, pos = noun
    #         print(f"Searching for {lemma}")
    #         data = find_existence_in_babelnet(lemma, return_language="NL")
    #         if data:
    #             print(f"found: {lemma}\n")
    #             print(data)
                
    #         else:
    #             print(f"Nothing found for: {lemma}\n")
    #             # How many definitions are there?
    #             # Add to RDF
                

#### Results example
# ('Middengebied', 'Middengebied', 'NOUN')
# ('Middengebied', 'Middengebied', 'NOUN')
# ('Middengebied', 'Middengebied', 'NOUN')
# ('suburbaniseringsproces', 'suburbaniseringsproces', 'NOUN')
# ('verband', 'verband', 'NOUN')
# ('werkgelegenheid', 'werkgelegenheid', 'NOUN')
# ('woningbouw', 'woningbouw', 'NOUN')
# ('afstand', 'afstand', 'NOUN')
# ('stedenring', 'stedenring', 'NOUN')
# ('proces', 'processen', 'NOUN')
# ('werkgelegenheid', 'werkgelegenheids', 'NOUN')
# ('bouwprocesontwikkeling', 'bouwprocesontwikkeling', 'NOUN')

#####


        
        
        # print("Original tag: ", value)
        # print("noun_tokens: ", noun_tokens, "\n\n")
    
    
    # get_noun_lemmas("industriële structuren")
    
    # for tag, values in tags_dict[5].items():
    #     print(tag)
    #     for value in values:
    #         print(value)
    
    
    # if tags_dict:
    #     for tag, annotations in tags_dict.items():
    #         print(f"{tag}: {annotations}")
    
    # lemma = "Apjhkjple"
    
    # find_existence_in_babelnet(lemma, return_language="EN")
    



# To do at home: find an example that finds 2 examples in babelnet
# See if the closest distacne algorithm finds the correct one, per the definition.
