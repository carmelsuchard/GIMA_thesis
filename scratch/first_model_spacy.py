import spacy

# texts = [
#     "This thesis studies the population of Utrecht",
#     "This thesis studies aging in Southern Amsterdam",
# ]

nlp = spacy.load("en_core_web_sm")
# ner_labels = nlp.get_pipe('ner')

# # print(ner_labels.labels)

# categories = ["ORG", "PERSON", "LOC", "GPE"]
# docs = [nlp(text) for text in texts]

# for doc in docs:
#     entities = []
#     for ent in doc.ents:
#         if ent.label_ in categories:
#             entities.append((ent.text, ent.label_))
#     print(entities)

import random
from spacy.util import minibatch
from spacy.training.example import Example

train_data = [
    ("This thesis studies the population of Utrecht", {"entities":[(39, 46, "SPATIAL")]}),
    ("This thesis studies aging in Southern Amsterdam", {"entities":[(38, 47, "SPATIAL")]})
]

if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")
    
    
for _, annotations in train_data:
    for ent in annotations["entities"]:
        if ent[2] not in ner.labels:
            ner.add_label(ent[2])

other_pipes = [pipe for pipe in nlp.pipe_names if pipe !="ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()
    
    epochs = 50
    for epoch in range(epochs):
        random.shuffle(train_data)
        losses = {}
        batches = minibatch(train_data, size =2)
        for batch in batches:
            examples = []
            for text, annotations in batch:
                doc = nlp.makedoc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)
                
            nlp.update(examples, drop=0.5, losses=losses)
        print(f'Epoch {epoch+1}, losses = {losses}')
                
nlp.to_disk("first_ner_model")
trained_nlp = spacy.load("first_ner_model")