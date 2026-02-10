import os
from datasets import Dataset, DatasetDict
import re
from itertools import product
from transformers import logging, AutoTokenizer, DataCollatorWithPadding, DataCollatorForTokenClassification
from BERT_settings import checkpoint, training_datasets_path, LABELS, epochs_count
from model_helper_functions import count_entities
import sys
from torch.utils.data import DataLoader





if __name__ == "__main__":
    pass