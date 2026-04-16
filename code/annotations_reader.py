import os

class conll_annotations_reader():
    TAGS = ("title", "author", "issued", "inGroup", "spatial", "subject")

    def __init__(self, conll_path):
        self.conll_path = conll_path
    
    def conll200_to_tags_dict(self):
        """Reads the conll file and returns a dictionary with the tags as keys and the corresponding annotations as values."""
        if not os.path.exists(self.conll_path):
            print(f"Path to file does not exist: {self.conll_path}")
            return
        tags_dict = {}
        with open(self.conll_path, encoding="utf-8") as f:
            new_annotation = False
            previous_tag = ""
            for line in f:
                line = line.strip()
                if line.endswith(' O') or line == "":
                    new_annotation = True
                    continue
                found_tag = ""
                for tag in conll_annotations_reader.TAGS:
                    if line.endswith(tag):
                        found_tag = tag
                        break
                if not found_tag: # There is a tag but we're not using it, skip
                    continue
                if found_tag != previous_tag:
                    new_annotation = True
                else:
                    new_annotation = False
                
                previous_tag = found_tag

                type = line.split(" ")[1][0]
                if type == "B":
                    new_annotation = True
                clean_word = line.split(" ")[0]
                
                if found_tag in tags_dict:
                    if new_annotation:
                        tags_dict[found_tag].append(clean_word)
                    else:
                        tags_dict[found_tag][-1] = tags_dict[found_tag][-1] + " " + clean_word
                else:
                    tags_dict[found_tag] = [clean_word]
                new_annotation = False
        
        tags_dict = {k:set(v) for k,v in tags_dict.items()}
        
        return tags_dict


if __name__ == "__main__":

    conlls = [r"C:\Users\carme\Downloads\1973_Harts_Jan_Migratie_UU.conll",
              r"C:\Users\carme\Downloads\1973_Harts_Jan_Migratie_UU (1).conll"
              ]

    for conll in conlls:
        reader = conll_annotations_reader(conll)
        tags_dict = reader.conll200_to_tags_dict()
        print(f"Tags dict for {conll}: {tags_dict['title']}")