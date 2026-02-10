from itertools import product
from pathlib import Path
import os
import pandas as pd
import numpy as np
import re

TAGS = ["title", "author", "issued", "inGroup", "spatial", "subject"]
TAGS_TO_REMOVE = ["summary", "publisher", "dataSource", "method", "question", "goal", "null"]

def clean_unused_tags(tgt_file_path, dst_file_path=""):
    if os.path.exists(dst_file_path):
        print(os.path.basename(dst_file_path), " already cleaned \n")
        return

    with open(tgt_file_path, mode="r", encoding="utf-8") as f:
        text = f.read()
        
        replacement_dict = {identifier + bad_tag: "O" for identifier, bad_tag in product(["B-", "I-"], TAGS_TO_REMOVE)}
        
        for old, new in replacement_dict.items():
            text = text.replace(old, new)
    
    with open(dst_file_path, mode = "w", encoding="utf-8") as f:
        f.write(text)

def find_replace_in_file(file_path, replacement_list, new):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    for old in replacement_list:
        content = content.replace(old, new)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def conll200_to_tags_dict(file_path):
    if not os.path.exists(file_path):
        print(f"Path to file does not exist: {file_path}")
        return
    
    tags_dict = {}
    with open(file_path, encoding="utf-8") as f:
        counter = 0
        new_annotation = False
        previous_tag = ""
        for line in f:
            line = line.strip()
            if line.endswith(' O') or line == "":
                new_annotation = True
                continue
            for tag in TAGS:
                if line.endswith(tag):
                    found_tag = tag
                    break
            if not found_tag: # There is a label but we're not using it, skip
                continue
            if found_tag != previous_tag:
                new_annotation = True
            else:
                new_annotation = False
            
            type = line.split(" ")[1][0]
            if type == "B":
                new_annotation = True
            
            previous_tag = found_tag
            clean_word = line.split(" ")[0]
            if found_tag in tags_dict:
                if new_annotation:
                    tags_dict[found_tag].append(clean_word)
                else:
                    tags_dict[found_tag][-1] = tags_dict[found_tag][-1] + " " + clean_word
            else:
                tags_dict[found_tag] = [clean_word]
            counter += 1
            new_annotation = False
    return tags_dict

def propegate_labels(file_path, tags_dict):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Path does not exist: {file_path}")

    # Read the full text once
    with open(file_path, encoding="utf-8") as f:
        text = f.read()

    print(f"Starting label propagation on {file_path}...")

    for label, sequences in tags_dict.items():
        # print(f"Propagating label: {label}")
        for value in set(sequences):
            # print(f"  Processing sequence: '{value}'")

            # Build a pattern that only matches sequences where all tokens are currently O
            tokens = value.split()
            pattern_lines = [re.escape(t) + r" O" for t in tokens]  # only match O labels
            pattern = "\n".join(pattern_lines)

            # Compile regex; allow multi-line matching
            regex = re.compile(pattern)

            # Build replacement text with correct BIO labels
            replacement_lines = [f"{tokens[0]} B-{label}"] + [f"{t} I-{label}" for t in tokens[1:]]
            replacement_text = "\n".join(replacement_lines)

            # Apply replacement
            text, num_subs = regex.subn(replacement_text, text)
            # print(f"    Replaced {num_subs} occurrences")

    # Write the modified text back to the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

    # print("Label propagation completed.")


if __name__ == "__main__":
    target_dir_path = r"C:\Users\carme\OneDrive\Documents\Graduate_Programs\Thesis\code\data\original_text"
    destination_dir_path = r"C:\Users\carme\OneDrive\Documents\Graduate_Programs\Thesis\code\data\cleaned"
    
    conlls = [f for f in os.listdir(target_dir_path) if os.path.isfile(os.path.join(target_dir_path, f)) and f.endswith(".conll")]

    all_thesis_df = pd.DataFrame()

    for conll in conlls:
        clean_unused_tags(Path(target_dir_path)/conll, Path(destination_dir_path)/conll)
        
        bad_characters = ["•", "»", "■", "◦", "«"]
        find_replace_in_file(Path(destination_dir_path)/conll, bad_characters, ".")
        
        tags_dict = conll200_to_tags_dict(Path(destination_dir_path)/conll)
        propegate_labels(Path(destination_dir_path)/conll, tags_dict)

        concat_tags = {key:tuple(value) for key, value in tags_dict.items()}
        df = pd.DataFrame([concat_tags])
        df["Thesis_File"] = conll
        
        all_thesis_df = pd.concat([all_thesis_df, df], ignore_index=True)
        
    all_thesis_df.to_csv(r"C:\Users\carme\OneDrive\Documents\Graduate_Programs\Thesis\code\annotations_summary.csv", index=False)