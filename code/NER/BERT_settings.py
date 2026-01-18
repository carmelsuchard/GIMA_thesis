import pandas as pd
import os

### Model settings ###
checkpoint = "GroNLP/bert-base-dutch-cased"
epochs_count = 2
batch_size = 8
# checkpoint = "bert-base-multilingual-cased"

### Training settings ###
# LABELS = [
#         "O",
#         "B-TITLE",
#         "I-TITLE",
#         "B-AUTHOR",
#         "I-AUTHOR",
#     ]

LABELS = [
        "O",
        "B-title",
        "I-title",
        "B-author",
        "I-author",
        "B-issued",
        "I-issued",
        "B-subject",
        "I-subject",
        "B-spatial",
        "I-spatial",
        "B-inGroup",
        "I-inGroup"
    ]

archive_theses_path = r"./code/full_archive/Dutch"
training_datasets_path = r"./code/annotated_conll_files/testing2"

if __name__ == "__main__":
    # print(LABELS)
    # unique_labels = list(set([label.split("-")[1] for label in LABELS if label != "O"]))
    # metrics_df = pd.DataFrame(columns=unique_labels)
    # print(metrics_df)
    pass
