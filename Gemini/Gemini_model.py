import os
from google import genai
from Gemini.Gemini_api import API_key
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import json
import random
import time
import pathlib
import httpx

os.environ['GEMINI_API_KEY'] = API_key

class Label(BaseModel):
    spatial: List[str]
    author: List[str]
    title: List[str]
    issued: List[str]
    subject: List[str]
    inGroup: List[str]

client = genai.Client()

def build_prompt(training_dir_path, inference_file_path, type):
    
    with open(inference_file_path, "r", encoding="utf-8") as f:
        inference_text = f.read()

    if type == "few-shot": # Few-shot version of the prompt
        conlls = [os.path.join(training_dir_path, f) for f in os.listdir(training_dir_path) if os.path.isfile(os.path.join(training_dir_path, f)) and f.endswith(".conll")]
        # Take two random conll files as training examples
        conll_training_paths = random.sample(conlls, 2)

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
        or as I- to indicate the inside of an entity span. After reading the examples, extract the tags from the new thesis text, and return it as a JSON, where each key is a label and each value is a list of entities that have that label.
        Only extract entities that are in the text. Follow the annotation rules.
        
        Annotation rules:

        • Don’t tag exhaustively, but only the first 5 mentions (e.g. a particular place, or a particular method) of any given concept in those sections that should be searched 
        • Restrict search to particular sections: Don’t use TOC, no prefaces. Focus on the main sections (introduction/method/conclusion). Avoid using sections literature review, background, results, or TOC or literature list
        • If several different concepts are mentioned in a sentence, annotate them separately
        • inGroup: should be specific for the research design
        • Subject: Leave out subjects unless there is explicitly a conceptual model (key-concepts)
        • Spatial: the largest extent of the research area related to a goal/question. Any spatial level that links to a different research goal can appear separately. In case there is no placename available for this, encode the information on most specific level that is there (“plein in Amersfoort”).
        
        Examples:

        {combined_training_text}

        Please annotate the following student thesis:

        {inference_text}

        """

    elif type == "zero-shot":  # Zero-shot version of the prompt
        prompt = f"""
        Named Entity Recognition task in Dutch.

        Label each word in this human-geography thesis as a title (Title of the thesis), author(Author of the thesis), issued (Time of publication), spatial (The spatial extent, study area or spatial coverage of the entire thesis, given as placenames or place descriptions),
        subject (Concept that is the main subject of the thesis), inGroup (the group of persons studied), or as O (empty label). Return it as a JSON, where each key is a label and each value is a list of entities that have that label.
        Only extract entities that are in the text. Follow the annotation rules.
        
        Annotation rules:

        • Don’t tag exhaustively, but only the first 5 mentions (e.g. a particular place, or a particular method) of any given concept in those sections that should be searched 
        • Restrict search to particular sections: Don’t use TOC, no prefaces. Focus on the main sections (introduction/method/conclusion). Avoid using sections literature review, background, results, or TOC or literature list
        • If several different concepts are mentioned in a sentence, annotate them separately
        • inGroup: should be specific for the research design
        • Subject: Leave out subjects unless there is explicitly a conceptual model (key-concepts)
        • Spatial: the largest extent of the research area related to a goal/question. Any spatial level that links to a different research goal can appear separately. In case there is no placename available for this, encode the information on most specific level that is there (“plein in Amersfoort”).
        
        Please annotate the following student thesis:
        {inference_text}
        """
    elif type == "multimodal":  # Zero-shot version of the prompt
        print("Building multi-modal prompt...")
        prompt = f"""
        Named Entity Recognition task in Dutch.

        Label each word in this human-geography thesis as a title (Title of the thesis), author(Author of the thesis), issued (Time of publication), spatial (The spatial extent, study area or spatial coverage of the entire thesis, given as placenames or place descriptions),
        subject (Concept that is the main subject of the thesis), inGroup (the group of persons studied), or as O (empty label). Return it as a JSON, where each key is a label and each value is a list of entities that have that label.
        Only extract entities that are in the text. Follow the annotation rules. The thesis is provided as a PDF file attachment and scanned to text, which may have OCR errors.
        
        Annotation rules:

        • Don’t tag exhaustively, but only the first 5 mentions (e.g. a particular place, or a particular method) of any given concept in those sections that should be searched 
        • Restrict search to particular sections: Don’t use TOC, no prefaces. Focus on the main sections (introduction/method/conclusion). Avoid using sections literature review, background, results, or TOC or literature list
        • If several different concepts are mentioned in a sentence, annotate them separately
        • inGroup: should be specific for the research design
        • Subject: Leave out subjects unless there is explicitly a conceptual model (key-concepts)
        • Spatial: the largest extent of the research area related to a goal/question. Any spatial level that links to a different research goal can appear separately. In case there is no placename available for this, encode the information on most specific level that is there (“plein in Amersfoort”).
        
        Please annotate the following student thesis:
        {inference_text}
        """


    # Save the prompt to text if needed to see it
    # with open(r"C:\Users\5298954\Documents\Github_Repos\GIMA_thesis\code\NER\prompt_preview.txt", "w", encoding="utf-8") as f:
    #     f.write(prompt)

    return prompt


def predict_labels(training_dir_path, inference_file_path, type):
    if type == "few-shot":
        prompt = build_prompt(training_dir_path, inference_file_path, "few-shot")
        results_csv = "code/NER/inference_results/few_shot_results.csv"

    elif type == "zero-shot":
        prompt = build_prompt(training_dir_path, inference_file_path, "zero-shot")
        results_csv = "code/NER/inference_results/zero_shot_results.csv"
    
    elif type == "multimodal":
        prompt = build_prompt(training_dir_path, inference_file_path, "multimodal")
        results_csv = "code/NER/inference_results/multimodal_results.csv"
    else:
        print("Specify if predict zero-shot, few-shot, or multi-modal")

    # total_tokens = client.models.count_tokens(
    # model="gemini-3-flash", contents=prompt
    # )
    # print("total_tokens: ", total_tokens)
    
    if type == "multimodal":
        client = genai.Client()
        file_path = pathlib.Path("C:/Users/carme/Downloads/1994_Slabbertje_Martin_Het_PPP-project_UU_searchable.pdf - SURFdrive.pdf")
        sample_file = client.files.upload(
            file=file_path,
        )

        file_info = client.files.get(name=sample_file.name)
        print(file_info.model_dump_json(indent=4))

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[prompt, sample_file],
            config={
                "response_mime_type": "application/json",
                "response_json_schema": Label.model_json_schema(),
            }
        )
    
    else:
        client = genai.Client()
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
    print("Let's sleep for 60 seconds...")
    time.sleep(60)  # Add a delay to avoid hitting rate limits or overwhelming the API


if __name__ == "__main__":
    training_dir_path = "C:/Users/carme/OneDrive/Documents/Git_Repos/GIMA_thesis/code/annotated_conll_files/Gemini_data/Training"

    # Zero-shot loop
    # for validation_thesis in os.listdir(r"C:\Users\5298954\Documents\Github_Repos\GIMA_thesis\code\annotated_conll_files\Gemini_data\Validation"):
    #     predict_labels(training_dir_path, os.path.join(r"C:\Users\5298954\Documents\Github_Repos\GIMA_thesis\code\annotated_conll_files\Gemini_data\Validation", validation_thesis), "zero-shot")

    # # Few-shot loop
    # for validation_thesis in os.listdir(r"C:\Users\5298954\Documents\Github_Repos\GIMA_thesis\code\annotated_conll_files\Gemini_data\Validation"):
    #     predict_labels(training_dir_path, os.path.join(r"C:\Users\5298954\Documents\Github_Repos\GIMA_thesis\code\annotated_conll_files\Gemini_data\Validation", validation_thesis), "few-shot")

    # Single file prediction
    inference_file_path = r"C:/Users/carme/OneDrive/Documents/Git_Repos/GIMA_thesis/code/annotated_conll_files/Gemini_data/Validation/1994_Slabbertje_Martin_Het_PPP-project.conll"
    predict_labels(training_dir_path, inference_file_path, "multimodal")







# model_info = client.models.get(model="gemini-3.0-flash")
# print(f"{model_info.input_token_limit=}")
# print(f"{model_info.output_token_limit=}")