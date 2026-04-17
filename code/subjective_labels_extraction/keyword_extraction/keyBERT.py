from pathlib import Path
import os
import time

import pandas as pd
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer

DUTCH_STOPWORDS = [
  # Common Dutch stopwords (not exhaustive)
  "aan", "aangaande", "aangezien", "achte", "achter", "achterna", "af",
  "afgelopen", "al", "aldaar", "aldus", "alhoewel", "alias", "alle",
  "allebei", "alleen", "alles", "als", "alsnog", "altijd", "altoos",
  "ander", "andere", "anders", "anderszins", "beetje", "behalve",
  "behoudens", "beide", "beiden", "ben", "beneden", "bent", "bepaald",
  "betreffende", "bij", "bijna", "bijv", "binnen", "binnenin",
  "blijkbaar", "blijken", "boven", "bovenal", "bovendien", "bovengenoemd",
  "bovenstaand", "bovenvermeld", "buiten", "bv", "daar", "daardoor",
  "daarheen", "daarin", "daarna", "daarnet", "daarom", "daarop",
  "daaruit", "daarvanlangs", "dan", "dat", "de", "deden", "deed", "der",
  "derde", "derhalve", "dertig", "deze", "dhr", "die", "dikwijls", "dit",
  "doch", "doe", "doen", "doet", "door", "doorgaand", "drie", "duizend",
  "dus", "echter", "een", "eens", "eer", "eerdat", "eerder", "eerlang",
  "eerst", "eerste", "eigen", "eigenlijk", "elk", "elke", "en", "enig",
  "enige", "enigszins", "enkel", "er", "erdoor", "erg", "ergens", "etc",
  "etcetera", "even", "eveneens", "evenwel", "gauw", "ge", "gedurende",
  "geen", "gehad", "gekund", "geleden", "gelijk", "gemoeten", "gemogen",
  "genoeg", "geweest", "gewoon", "gewoonweg", "haar", "haarzelf", "had",
  "hadden", "hare", "heb", "hebben", "hebt", "hedden", "heeft", "heel",
  "hem", "hemzelf", "hen", "het", "hetzelfde", "hier", "hierbeneden",
  "hierboven", "hierin", "hierna", "hierom", "hij", "hijzelf", "hoe",
  "hoewel", "honderd", "hun", "hunne", "ieder", "iedere", "iedereen",
  "iemand", "iets", "ik", "ikzelf", "in", "inderdaad", "inmiddels",
  "intussen", "inzake", "is", "ja", "je", "jezelf", "jij", "jijzelf",
  "jou", "jouw", "jouwe", "juist", "jullie", "kan", "klaar", "kon",
  "konden", "krachtens", "kun", "kunnen", "kunt", "laatst", "later",
  "liever", "lijken", "lijkt", "maak", "maakt", "maakte", "maakten",
  "maar", "mag", "maken", "me", "meer", "meest", "meestal", "men",
  "met", "mevr", "mezelf", "mij", "mijn", "mijnent", "mijner", "mijzelf",
  "minder", "miss", "misschien", "missen", "mits", "mocht", "mochten",
  "moest", "moesten", "moet", "moeten", "mogen", "mr", "mrs", "mw", "na",
  "naar", "nadat", "nam", "namelijk", "nee", "neem", "negen", "nemen",
  "nergens", "net", "niemand", "niet", "niets", "niks", "noch", "nochtans",
  "nog", "nogal", "nooit", "nu", "nv", "of", "ofschoon", "om", "omdat",
  "omhoog", "omlaag", "omstreeks", "omtrent", "omver", "ondanks", "onder",
  "ondertussen", "ongeveer", "ons", "onszelf", "onze", "onzeker", "ooit",
  "ook", "op", "opnieuw", "opzij", "over", "overal", "overeind", "overige",
  "overigens", "paar", "pas", "per", "precies", "recent", "redelijk",
  "reeds", "rond", "rondom", "samen", "sedert", "sinds", "sindsdien",
  "slechts", "sommige", "spoedig", "steeds", "tamelijk", "te", "tegen",
  "tegenover", "tenzij", "terwijl", "thans", "tien", "tiende", "tijdens",
  "tja", "toch", "toe", "toen", "toenmaals", "toenmalig", "tot", "totdat",
  "tussen", "twee", "tweede", "u", "uit", "uitgezonderd", "uw", "vaak",
  "vaakwat", "van", "vanaf", "vandaan", "vanuit", "vanwege", "veel",
  "veeleer", "veertig", "verder", "verscheidene", "verschillende",
  "vervolgens", "via", "vier", "vierde", "vijf", "vijfde", "vijftig",
  "vol", "volgend", "volgens", "voor", "vooraf", "vooral", "vooralsnog",
  "voorbij", "voordat", "voordezen", "voordien", "voorheen", "voorop",
  "voorts", "vooruit", "vrij", "vroeg", "waar", "waarom", "waarschijnlijk",
  "wanneer", "want", "waren", "was", "wat", "we", "wederom", "weer", "weg",
  "wegens", "weinig", "wel", "weldra", "welk", "welke", "werd", "werden",
  "werder", "wezen", "whatever", "wie", "wiens", "wier", "wij", "wijzelf",
  "wil", "wilden", "willen", "word", "worden", "wordt", "zal", "ze", "zei",
  "zeker", "zelf", "zelfde", "zelfs", "zes", "zeven", "zich", "zichzelf",
  "zij", "zijn", "zijne", "zijzelf", "zo", "zoals", "zodat", "zodra",
  "zonder", "zou", "zouden", "zowat", "zulk", "zulke", "zullen", "zult",
]

STRUCTURAL_STOPWORDS = [
    # Document structure
    "hoofdstuk", "paragraaf", "sectie", "alinea", "bijlage", "appendix",
    "inleiding", "conclusie", "samenvatting", "inhoudsopgave", "voorwoord",
    "nawoord", "literatuurlijst", "bibliografie", "woordenlijst", "glossarium",
    "hoofdstukken", "paragrafen", "secties", "alinea's", "bijlagen", "appendices",
    "inleidingen", "conclusies", "samenvattingen", "inhoudsopgaven", "voorwoorden",
    "nawoorden", "literatuurlijsten", "bibliografieën", "woordenlijsten", "glossaria",
    
    # Common heading words
    "voorgeschiedenis", "achtergrond", "aanleiding", "doelstelling", "doelstellingen",
    "werkwijze", "methode", "methodologie", "resultaten", "bevindingen",
    "aanbevelingen", "discussie", "toelichting", "verantwoording", "hoofdlijnen",

    # Reference/numbering language
    "zie", "figuur", "tabel", "grafiek", "diagram", "pagina", "bladzijde",
    "noot", "voetnoot", "bron", "bronnen", "referentie", "verwijzing",
    "bovenstaand", "onderstaand", "bijgaand", "hierboven", "hieronder",
    "voorgaand", "volgende", "respectievelijk",

    # Filler/transition words
    "aldus", "immers", "echter", "tevens", "eveneens", "voorts", "derhalve",
    "derden", "namelijk", "waarbij", "waardoor", "waarmee", "waarvan",
    "hierbij", "hierdoor", "hiermee", "hiervan", "hierop", "hierin",

    # Vague/generic nouns
    "aspect", "aspecten", "onderdeel", "onderdelen", "geheel", "gehelen",
    "gebied", "gebieden", "kader", "kaders", "niveau", "niveaus",
    "mate", "wijze", "manier", "vorm", "vormen", "soort", "soorten",
]


ROOT = Path(__file__).resolve().parents[3]
TEXTS_DIR = ROOT / "code" / "data" / "unlabeled_texts"
IDS_PATH = ROOT / "code" / "data_cleaning" / "thesis_IDs.csv"
OUTPUT_PATH = ROOT / "code" / "subjective_labels_extraction" / "keyword_extraction" / "keyword_extraction.csv"
FAILED_OUTPUT_PATH = ROOT / "code" / "subjective_labels_extraction" / "keyword_extraction" / "keyword_extraction_failed.csv"

# MODEL = KeyBERT(model="robbert-2022-dutch-sentence-transformers")
MODEL = KeyBERT(model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# MODEL = KeyBERT()

def extract_keywords(text: str, top_n: int = 3) -> list[str]:
    print("Extracting keywords from text...")
    start_time = time.perf_counter()
    
    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words=DUTCH_STOPWORDS+STRUCTURAL_STOPWORDS)
    keywords = MODEL.extract_keywords(
        text,
        top_n=top_n,
        vectorizer=vectorizer,
        keyphrase_ngram_range=(1, 2),
        stop_words=DUTCH_STOPWORDS+STRUCTURAL_STOPWORDS,
    )

    elapsed_seconds = time.perf_counter() - start_time
    print(f"Keyword extraction completed in {elapsed_seconds:.2f} seconds: {len(keywords)} keywords found.")
    print("Extracted keywords and saliency scores:")
    for keyword, score in keywords:
        print(f"- {keyword}: {score:.4f}")
    return [keyword for keyword, _score in keywords]


def load_results(output_path: Path) -> pd.DataFrame:
    if output_path.exists():
        print(f"Loading existing results from {output_path}")
        return pd.read_csv(output_path)
    print("No existing results found. Creating a new results table.")
    return pd.DataFrame(columns=["id", "subject_predicted"])


def main() -> None:
    print("Starting keyword extraction script...")
    results_df = load_results(OUTPUT_PATH)
    failed_rows = []

    print(f"Reading thesis IDs from {IDS_PATH}")
    file_ids = pd.read_csv(IDS_PATH)
    file_ids_dict = dict(zip(file_ids["file_name"], file_ids["thesisID"]))

    new_rows = []

    print(f"Scanning text files in {TEXTS_DIR}")
    for file_path in TEXTS_DIR.iterdir():
        if not file_path.is_file():
            continue

        file_name = file_path.stem
        thesis_id = file_ids_dict.get(file_name)

        if thesis_id is None or pd.isna(thesis_id):
            message = "No thesis ID found"
            print(f"Skipping file without thesis ID: {file_path.name}")
            failed_rows.append({
                "file_name": file_path.name,
                "thesis_id": None,
                "reason": message,
            })
            continue

        file_id = int(thesis_id)

        print(f"Processing file: {file_path.name} -> thesis ID {file_id}")
        try:
            with file_path.open("r", encoding="utf-8") as f:
                text = f.read()

            keywords = extract_keywords(text)
            new_rows.append({"thesisID": file_id, "subject_predicted": keywords})
            print(f"Finished file: {file_path.name}")

        except Exception as exc:
            print(f"Failed file: {file_path.name} ({exc})")
            failed_rows.append({
                "file_name": file_path.name,
                "thesis_id": file_id,
                "reason": str(exc),
            })

    if new_rows:
        print(f"Appending {len(new_rows)} new rows to the results table.")
        results_df = pd.concat([results_df, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        print("No new rows were generated.")

    print(f"Saving results to {OUTPUT_PATH}")
    results_df.to_csv(OUTPUT_PATH, index=False)

    if failed_rows:
        failed_df = pd.DataFrame(failed_rows)
        print(f"Saving {len(failed_rows)} failed files to {FAILED_OUTPUT_PATH}")
        failed_df.to_csv(FAILED_OUTPUT_PATH, index=False)

        print("Files that could not be completed:")
        for row in failed_rows:
            print(f"- {row['file_name']}: {row['reason']}")
    else:
        print("No failed files.")

    print("Keyword extraction script completed.")

if __name__ == "__main__":
    main()