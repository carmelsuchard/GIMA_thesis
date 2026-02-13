# Read a file
# Remove everything that's not one of our labels that we want
# Get the first 1000 lines that are longer than 1

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

    text_lines = text.splitlines()
    lines_counter = 0
    last_line_index = 0
    for i, line in enumerate(text_lines):
        if len(line) > 3:
            lines_counter += 1
        if lines_counter >= 1000:
            last_line_index = i
            break

    text_first_1000_rows = "\n".join(text_lines[0:last_line_index])

    with open(dst_file_path, mode = "w", encoding="utf-8") as f:
        f.write(text_first_1000_rows)

if __name__ == "__main__":
    for file_name in os.listdir("code/data/original_text"):
        if file_name.endswith(".conll"):
            src_file_path = Path("code/data/original_text") / file_name
            dst_file_path = Path("code/data/objective_label_NER") / file_name

            clean_unused_tags(src_file_path, dst_file_path)
    