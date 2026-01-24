# AI Coding Agent Instructions for GIMA Thesis Project

## Project Overview
This is a **Dutch human geography thesis archive NER and Knowledge Graph project**. The system extracts structured metadata from historical geography theses (1970s-1990s) using Named Entity Recognition, then constructs an RDF knowledge graph. The NER pipeline supports both BERT (fine-tuned on Dutch theses) and Gemini (LLM-based) approaches.

## Architecture Overview

### 1. Data Pipeline & Directory Structure
```
annotated_conll_files/
├── original/          # Raw CONLL files (BIO-tagged)
├── cleaned/           # After removing unwanted tags (used for BERT training)
├── cleaned_I_annotations/
├── BERT_data/         # Pre-split train/val sets
├── Gemini_data/
└── cleaned_small/     # Subset for testing
```

**Key Data Format**: CONLL-2000 with BIO annotation scheme  
```
token LABEL
word  B-author
name  I-author

(blank line = sentence boundary)
```

**Entity Tags** (defined in [code/annotations_reader.py](code/annotations_reader.py#L2)):
- `title`, `author`, `issued`, `spatial`, `subject`, `inGroup`, `O` (null)

### 2. NER Model Approaches

**BERT Pipeline** (`code/NER/BERT_*.py`):
- Checkpoint: `GroNLP/bert-base-dutch-cased` (config: [BERT_settings.py](code/NER/BERT_settings.py))
- Flow: `prepare_dataset_for_BERT.py` → tokenize+align labels → train [BERT_model.py](code/NER/BERT_model.py) → save checkpoint to `models/`
- Dataset prep: Reads from `training_datasets_path`, compiles all CONLL files, splits 80/20
- Metrics: Uses `seqeval.metrics` for token-level accuracy (see [BERT_model_helper_functions.py](code/NER/BERT_model_helper_functions.py#L1))
- Settings stored in [BERT_settings.py](code/NER/BERT_settings.py) - DO NOT hardcode paths

**Gemini Pipeline** (`code/NER/Gemini_*.py`):
- Uses Google GenAI API with few-shot prompting
- Pydantic schema: defines expected JSON output structure (`spatial`, `author`, `title`, `issued`, `subject`, `inGroup`)
- Few-shot examples loaded from training CONLL files (configurable count)
- API key stored in [Gemini_api.py](code/NER/Gemini_api.py) (injected via `os.environ`)
- Returns structured JSON instead of CONLL format

### 3. Knowledge Graph Construction
Pipeline: [code/build_knowledge_graph.py](code/build_knowledge_graph.py)
- Reads NER tags via [annotations_reader.py](code/annotations_reader.py#L1): extracts entity lists from CONLL
- Maps entities to RDF triples using Dublin Core (`dcterms:*`), GeoSPARQL
- **Geographic disambiguation**:
  - [find_geography.py](code/find_geography.py#L1): queries DBPedia + Geonames SPARQL endpoints
  - Stores `owl:sameAs` links to DBPedia URIs
- Ontology defined in [SGPLThesis_ontology.ttl](SGPLThesis_ontology.ttl): defines ThesisSpatial, ThesisSubject classes

### 4. Data Cleaning
[annotations_cleaning.py](code/annotations_cleaning.py): Removes unwanted tags (summary, publisher, dataSource, etc.) by replacing with `O` labels in CONLL files.

## Critical Conventions & Patterns

### Paths & Configuration
- **Relative paths from project root** in settings files (e.g., `./code/annotated_conll_files/cleaned`)
- Settings not hardcoded in scripts → centralized in `BERT_settings.py` and `Gemini_settings.py`
- API credentials in `config.py` (loaded via imports, never passed as args)

### Tag Parsing Logic
- **BIO scheme enforcement**: `B-` marks entity start, `I-` marks continuation
- Tag merging: consecutive `I-` tags without new `B-` = single entity (see [annotations_reader.py](code/annotations_reader.py#L32-L42))
- Example: `["word", "I-author", "name", "I-author"]` → single author "word name"

### Dataset Handling
- CONLL files must have blank lines separating sentences (mandatory for parsing)
- Tokenizer respects subword boundaries: `-100` labels for subword tokens (see [prepare_dataset_for_BERT.py](code/NER/prepare_dataset_for_BERT.py#L75))
- Train/test splits computed per-dataset (no global shuffle to preserve thesis boundaries)

### Gemini Integration Pattern
- Few-shot prompt construction: load N training examples, embed in system prompt
- JSON output validation via Pydantic (enforces exact field names)
- Batch inference vs. single-file: [prepare_archive_for_inference.py](code/NER/prepare_archive_for_inference.py) handles bulk processing

## Typical Workflows

### Training a BERT Model
1. Place cleaned CONLL files in `./code/annotated_conll_files/cleaned/`
2. Update `LABELS` and `training_datasets_path` in [BERT_settings.py](code/NER/BERT_settings.py)
3. Run [BERT_model.py](code/NER/BERT_model.py) → trains, saves best checkpoint
4. Outputs: model in `code/NER/models/`, loss curve in `Figures/`

### Running Inference (Gemini)
1. Set API key in [Gemini_api.py](code/NER/Gemini_api.py)
2. Prepare inference CONLL via [prepare_archive_for_inference.py](code/NER/prepare_archive_for_inference.py)
3. Run [inference.py](code/NER/inference.py) → outputs JSON entities for each thesis

### Building Knowledge Graph
1. Ensure NER-tagged CONLL files exist
2. Run [build_knowledge_graph.py](code/NER/build_knowledge_graph.py)
3. Outputs: RDF triples (in-memory graph, can serialize to Turtle)
4. Geographic entities linked to DBPedia/Geonames via SPARQL queries

## Common Gotchas & Debugging

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `FileNotFoundError` in BERT pipeline | Path hardcoded or relative from wrong dir | Check settings in [BERT_settings.py](code/NER/BERT_settings.py), use `.` prefix |
| Gemini API errors | Invalid credentials in [Gemini_api.py](code/NER/Gemini_api.py) | Verify `API_key` variable, check quota |
| CONLL parsing fails | Missing blank lines between sentences | Validate CONLL format: each sentence needs `\n\n` separator |
| Low NER accuracy | Label-token misalignment after tokenization | Debug [tokenize_and_align_labels](code/NER/prepare_dataset_for_BERT.py#L68): check `word_ids` mapping |
| Geographic linking fails | SPARQL endpoint timeouts | Add retry logic, cache DBPedia results to avoid re-querying |

## Key Files Reference
- **Data I/O**: [annotations_reader.py](code/annotations_reader.py), [annotations_cleaning.py](code/annotations_cleaning.py)
- **BERT Training**: [BERT_model.py](code/NER/BERT_model.py), [prepare_dataset_for_BERT.py](code/NER/prepare_dataset_for_BERT.py), [BERT_settings.py](code/NER/BERT_settings.py)
- **Gemini**: [Gemini_model.py](code/NER/Gemini_model.py), [Gemini_settings.py](code/NER/Gemini_settings.py)
- **KG**: [build_knowledge_graph.py](code/build_knowledge_graph.py), [find_geography.py](code/find_geography.py)
- **Ontology**: [SGPLThesis_ontology.ttl](SGPLThesis_ontology.ttl)
