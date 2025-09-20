# EHR Concept Mining (Bio_ClinicalBERT + MLP)

This repo trains an MLP classifier on Bio_ClinicalBERT embeddings of clinical phrases (obtained via concatenation or anchoring) and then predicts labels for *new* phrases mined from EHRs. It’s a cleaned, CLI-based version of a Colab workflow.

## Features
- Bio_ClinicalBERT embeddings for short clinical phrases
- 5-fold CV + grid search over a compact MLP config
- Saves a reusable `.joblib` model
- Predicts labels for new phrases and writes an Excel file

## Install
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
