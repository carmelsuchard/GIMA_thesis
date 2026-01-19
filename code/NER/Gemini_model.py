import os
from google import genai
from Gemini_api import API_key
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import json

os.environ['GEMINI_API_KEY'] = API_key

class Label(BaseModel):
    spatial: List[str]
    author: List[str]
    title: List[str]
    issued: List[str]
    subject: List[str]
    inGroup: List[str]

client = genai.Client()

def build_prompt(training_dir_path, inference_file_path, training_examples):
    
    with open(inference_file_path, "r", encoding="utf-8") as f:
        inference_text = f.read()

    if training_examples: # Few-shot version of the prompt
        conlls = [os.path.join(training_dir_path, f) for f in os.listdir(training_dir_path) if os.path.isfile(os.path.join(training_dir_path, f)) and f.endswith(".conll")]
        conll_training_paths = conlls[:1]

        full_text = []
        for indx, conll in enumerate(conll_training_paths):
            with open(conll, "r", encoding="utf-8") as f:
                full_text.append(f"Example {indx+1} \n")
                full_text.append(f.read())
        
        combined_training_text = "\n".join(full_text)

        prompt = f"""
        Named Entity Recognition task in Dutch.

        Below are {len(conll_training_paths)} examples showing the text of human-geography theses. Each word in the thesis is labeled as a title (Title of the thesis),
        author(Author of the thesis), issued (Time of publication), spatial (The spatial extent, study area or spatial coverage of the entire thesis, given as placenames or place descriptions),
        subject (Concept that is the main subject of the thesis), inGroup (the group of persons studied), or as O (empty label). Non-empty labeled can start with a B- to indicate the beginning of an entity span,
        or as I- to indicate the inside of an entity span. After reading the examples, extratct the tags from the new thesis text, and return it as a JSON, where each key is a label and each value is a list of entities that have that label.
        Do not extract anything that is not in the text.

        Examples:

        {combined_training_text}

        Please annotated the following student thesis:

        {inference_text}

        """

    else:
        prompt = f"""
        Named Entity Recognition task in Dutch.

        Label each word in this human-geography thesis as a title (Title of the thesis), author(Author of the thesis), issued (Time of publication), spatial (The spatial extent, study area or spatial coverage of the entire thesis, given as placenames or place descriptions),
        subject (Concept that is the main subject of the thesis), inGroup (the group of persons studied), or as O (empty label). Return it as a JSON, where each key is a label and each value is a list of entities that have that label.
        Do not extract anything that is not in the text.
        
        {inference_text}
        """

    # Save the prompt to text if needed to see it
    # with open(r"C:\Users\5298954\Documents\Github_Repos\GIMA_thesis\code\NER\prompt_preview.txt", "w", encoding="utf-8") as f:
    #     f.write(prompt)

    return prompt


def predict_labels(training_dir_path, inference_file_path, type):
    if type == "few-shot":
        prompt = build_prompt(training_dir_path, inference_file_path, True)
        results_csv = r"code\NER\inference_results\few_shot_results.csv"

    elif type == "zero-shot":
        prompt = build_prompt(training_dir_path, inference_file_path, False)
        results_csv = r"code\NER\inference_results\zero_shot_results.csv"
    else:
        print("Specify if predict zero-shot or few-shot")

    # total_tokens = client.models.count_tokens(
    # model="gemini-3.0-flash", contents=prompt
    # )
    # print("total_tokens: ", total_tokens)

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": Label.model_json_schema(),
        }
    )

    labels = Label.model_validate_json(response.text)

    response_dict = json.loads(response.text)
    response_dict["Thesis_File"] = [os.path.basename(inference_file_path)]
    
    results_df = pd.read_csv(results_csv)
    new_row = pd.DataFrame([{key:tuple(value) for key, value in response_dict.items()}])

    # results_df = results_df[results_df['Thesis_File'] != new_row["Thesis_File"]]

    results_df = pd.concat([results_df, new_row], ignore_index=True)
    results_df.to_csv(results_csv, index=False)

    print(results_df)


if __name__ == "__main__":

    training_dir_path = r"C:\Users\5298954\Documents\Github_Repos\GIMA_thesis\code\annotated_conll_files\cleaned"
    # inference_file_path = r"C:\Users\5298954\Documents\Github_Repos\GIMA_thesis\code\annotated_conll_files\Gemini_data\Validation\1994_Slabbertje_Martin_Het_PPP-project.conll"
    inference_file_path = r"C:\Users\5298954\Documents\Github_Repos\GIMA_thesis\code\annotated_conll_files\Gemini_data\Validation\2007_Jong_de_Stefan_Een_onderzoek_naar_e-commerce_succes_in_de_binnenstad.conll"

    predict_labels(training_dir_path, inference_file_path, "zero-shot")




# For env: pip install google-genai



# model_info = client.models.get(model="gemini-3.0-flash")
# print(f"{model_info.input_token_limit=}")
# print(f"{model_info.output_token_limit=}")