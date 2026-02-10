from datasets import load_dataset
import torch

from transformers import AutoTokenizer # Tokenizer
from transformers import AutoModelForSequenceClassification, AutoModel, BertModel
#  Automodel: An object that returns the correct architecture based on the checkpoint

# So I will need to know the chekcpoint belonging to the model that I use, which I can see on its model card

checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


raw_inputs = ["This thesis studies Munich but doesn't hate it",
              "This one is about Utrecht"]

#BERT was only pretrained with sequences up to 512 tokens, so I will need to chop it up to sentences like Tabea did
# WordPiece is the subword-based tokenizer used in BERT


# raw_datasets = load_dataset("conll2003")
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt") # This returns my pyTorch tensors to feed into the model
print(inputs)

# model = AutoModel.from_pretrained(checkpoint)
model = BertModel.from_pretrained("bert-base-cased")

outputs = model(**inputs)
# print(outputs.logits)

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
model.config.id2label

# Batching is the act of sending multiple sentences through the model, all at once. Batching allows the model to work when you feed it multiple sentences. 

# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# sequence = "What's the use of wondering if he's good or if he's bad?"

# tokens = tokenizer.tokenize(sequence)
# ids = tokenizer.convert_tokens_to_ids(tokens)
# batched_ids = [ids, ids]
# print('Batched ids:', batched_ids)

# input_ids = torch.tensor(batched_ids)
# print("Input IDs:", input_ids)

# output = model(input_ids)
# print("Logits:", output.logits)
# predictions = torch.nn.functional.softmax(output.logits, dim=-1)
# print(predictions)
# model.config.id2label

####### FINE_TUNING ########
