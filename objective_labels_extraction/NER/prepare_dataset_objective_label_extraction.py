# Read a file
# Remove everything that's not one of our labels that we want
# Cut it off some amount after the last one.

# Save it back to the place.
import os
from itertools import product
from pathlib import Path

TAGS_TO_REMOVE = ["summary", "publisher", "dataSource", "method", "question", "goal", "null", "spatial", "inGroup", "subject"]
TAGS_TO_EXTRACT = ["title", "author", "issued"]


def clean_unused_tags(src_file_path, dst_file_path=""):
    # if os.path.exists(dst_file_path):
    #     print(os.path.basename(dst_file_path), " already cleaned \n")
    #     return

    with open(src_file_path, mode="r", encoding="utf-8") as f:
        text = f.read()
        
        replacement_dict = {identifier + bad_tag: "O" for identifier, bad_tag in product(["B-", "I-"], TAGS_TO_REMOVE)}
        
        for old, new in replacement_dict.items():
            text = text.replace(old, new)

        print(len(text))    
    
    # with open(dst_file_path, mode = "w", encoding="utf-8") as f:
    #     f.write(text)
    # Find the last index of a row that ends with I-title, I-author, or I-issued
    # Print the first 20 lines of the text
    print("First 20 lines of cleaned text:")
    print("\n".join(text.splitlines()[:20]))

    last_index = [text.rfind("I-" + tag) for tag in TAGS_TO_EXTRACT]
    print("Last index of relevant tag:", last_index)


if __name__ == "__main__":
    file_name = "1971_Abcouwer_NF_Ontwikkelingen_in_de_industriele_structuur_van_zuid-west-Nederland_1945-1970.conll"

    src_file_path = Path("C:/Users/5298954/Documents/Github_Repos/GIMA_thesis/code/data/original_text") / file_name
    dst_file_path = Path("C:/Users/5298954/Documents/Github_Repos/GIMA_thesis/code/data/objective_label_NER") / file_name

    clean_unused_tags(src_file_path, dst_file_path)
