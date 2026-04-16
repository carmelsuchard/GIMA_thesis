import pandas as pd
subject_col = "subject"

GROUND_TRUTH_CSV = r"C:\Users\5298954\Documents\Github_Repos\GIMA_thesis\code\data_cleaning\annotations_summary_manually_edited.csv"
PREDICTIONS_CSV = r"C:\Users\5298954\Documents\Github_Repos\GIMA_thesis\code\subjective_labels_extraction\keyword_extraction\keyword_extraction.csv"

df_ref = pd.read_csv(GROUND_TRUTH_CSV, usecols=["thesisID", subject_col], encoding="latin-1", sep=";")
# df_pred = pd.read_csv(PREDICTIONS_CSV, usecols=["thesisID", "subject_predicted"], encoding="latin-1")

print(df_ref)