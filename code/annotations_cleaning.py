from itertools import product
from pathlib import Path
import os
import pandas as pd
import numpy as np

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

def find_replace_in_file(file_path, old, new):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    content = content.replace(old, new)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def conll200_to_tags_dict(file_path):
    tags_dict = {}
    with open(file_path, encoding="utf-8") as f:
        counter = 0
        new_annotation = False
        previous_tag = ""
        for line in f:
            # if counter > 50:
            #     break
            line = line.strip()
            if line.endswith(' O') or line == "":
                new_annotation = True
                continue
            for tag in TAGS:
                if line.endswith(tag):
                    found_tag = tag
                    break
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

if __name__ == "__main__":
    target_dir_path = r"C:\Users\carme\OneDrive\Documents\Graduate_Programs\Thesis\code\annotated_conll_files\original"
    destination_dir_path = r"C:\Users\carme\OneDrive\Documents\Graduate_Programs\Thesis\code\annotated_conll_files\cleaned"
    
    conlls = [f for f in os.listdir(target_dir_path) if os.path.isfile(os.path.join(target_dir_path, f)) and f.endswith(".conll")]

    # for conll in conlls:
    #     clean_unused_tags(Path(target_dir_path)/conll, Path(destination_dir_path)/conll)

    # print(conlls)
    
    all_thesis_df = pd.DataFrame()
    for conll in conlls:
        
        tags_dict = conll200_to_tags_dict(os.path.join(destination_dir_path, conll))    

        concat_tags = {key:tuple(value) for key, value in tags_dict.items()}
        df = pd.DataFrame([concat_tags])
        df["Thesis_File"] = conll
        
        all_thesis_df = pd.concat([all_thesis_df, df], ignore_index=True)
        
    all_thesis_df.to_csv(r"C:\Users\carme\OneDrive\Documents\Graduate_Programs\Thesis\code\annotations_summary.csv", index=False)