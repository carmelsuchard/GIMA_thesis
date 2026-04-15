from collections import namedtuple
from thefuzz import fuzz
import re
import pandas as pd
import ast

GROUND_TRUTH_CSV = r"C:\Users\carme\OneDrive\Documents\Git_Repos\GIMA_thesis\code\data_cleaning\annotations_summary_manually_edited.csv"
PREDICTIONS_CSV = r"C:\Users\carme\OneDrive\Documents\Git_Repos\GIMA_thesis\code\subjective_labels_extraction\keyword_extraction\keyword_extraction.csv"

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = SentenceTransformer(
    "NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers"
)

def embed_phrases(phrases):
    """Convert list of phrases into embeddings."""
    if not phrases:
        return np.array([])
    return EMBEDDING_MODEL.encode(phrases)


def build_similarity_matrix(ref_embeddings, pred_embeddings):
    """Compute cosine similarity matrix (pred x ref)."""
    if len(ref_embeddings) == 0 or len(pred_embeddings) == 0:
        return np.array([])
    return cosine_similarity(pred_embeddings, ref_embeddings)


def soft_precision(sim_matrix, threshold):
    """Prediction-focused: rows = predictions."""
    if sim_matrix.size == 0:
        return 0.0

    max_per_pred = sim_matrix.max(axis=1)  # best match per prediction
    matches = (max_per_pred >= threshold).sum()

    return matches / len(max_per_pred)


def soft_recall(sim_matrix, threshold):
    """Ground-truth-focused: columns = reference."""
    if sim_matrix.size == 0:
        return 0.0

    max_per_ref = sim_matrix.max(axis=0)  # best match per ground truth
    matches = (max_per_ref >= threshold).sum()

    return matches / len(max_per_ref)


def compute_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def jaccard_similarity(ref, pred):
    """Exact match baseline."""
    ref_set = set(ref)
    pred_set = set(pred)

    if not ref_set and not pred_set:
        return 1.0

    intersection = len(ref_set & pred_set)
    union = len(ref_set | pred_set)

    if union == 0:
        return 0.0

    return intersection / union


def to_phrase_list(value):
    """Convert CSV cell to list[str]."""
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    text = str(value).strip()
    if not text:
        return []

    # Case 1: list-like string, e.g. "['a', 'b']"
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except (ValueError, SyntaxError):
            pass

    # Case 2: comma-separated string, e.g. "a, b"
    return [part.strip() for part in text.split(",") if part.strip()]


def compute_subject_metrics(subject_reference, subject_predicted, threshold=0.6):
    """
    Compute soft precision, recall, F1 using embedding similarity,
    plus exact-match Jaccard similarity.
    """

    # Step 1: embed
    subject_reference = to_phrase_list(subject_reference)
    subject_predicted = to_phrase_list(subject_predicted)

    ref_emb = embed_phrases(subject_reference)
    pred_emb = embed_phrases(subject_predicted)

    # Step 2: similarity matrix
    sim_matrix = build_similarity_matrix(ref_emb, pred_emb)

    # Step 3: metrics
    precision = soft_precision(sim_matrix, threshold)
    recall = soft_recall(sim_matrix, threshold)
    f1 = compute_f1(precision, recall)
    jaccard = jaccard_similarity(subject_reference, subject_predicted)

    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "jaccard": jaccard,
    }

def main():
    subject_col = "subject"
    # Read input files
    df_ref = pd.read_csv(GROUND_TRUTH_CSV, usecols=["thesisID", subject_col], encoding="latin-1")
    df_pred = pd.read_csv(PREDICTIONS_CSV, usecols=["thesisID", "subject_predicted"], encoding="latin-1")

    # Match rows using thesisID
    df_merged = pd.merge(df_pred, df_ref, on="thesisID", how="inner")

    # Build output rows
    rows = []
    for _, row in df_merged.iterrows():
        metrics = compute_subject_metrics(
            subject_reference=row["subject"],
            subject_predicted=row["subject_predicted"],
            threshold=0.6
        )
        rows.append(
            {
                "thesisID": row["thesisID"],
                "recall": metrics["recall"],
                "precision": metrics["precision"],
                "f1": metrics["f1"],
                "jaccard": metrics["jaccard"],
            }
        )

    # Write output
    df_out = pd.DataFrame(rows, columns=["thesisID", "recall", "precision", "f1", "jaccard"])
    
    OUTPUT_CSV = r"C:\Users\carme\OneDrive\Documents\Git_Repos\GIMA_thesis\code\subjective_labels_extraction\keyword_extraction\results\keyword_extraction_results" + f"_{subject_col}_{threshold}" + ".csv"

    df_out.to_csv(OUTPUT_CSV, index=False, encoding="latin-1")


if __name__ == "__main__":
    main()