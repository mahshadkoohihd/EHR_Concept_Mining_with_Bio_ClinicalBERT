#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train an MLP classifier on embedded clinical phrases (Bio_ClinicalBERT),
then embed new phrases and predict labels.

Inputs
- Training Excel: columns ["Phrase", "Lable"]  (spelling kept for compatibility)
- New phrases Excel: column ["Phrase"]

Outputs
- outputs/mlp_clinical_phrases.joblib         (trained model)
- outputs/label_encoder.joblib                 (LabelEncoder for y)
- outputs/new_phrases_predictions.xlsx         (Phrase, PredLabel)
- outputs/metrics.json                         (CV accuracy, confusion matrix, params)
- optional: embedded_train.pt / embedded_new.pt (if --save_embeddings)

"""
import argparse
import json
import os
import sys
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from transformers import AutoModel, AutoTokenizer

# ---------------------------
# Utils
# ---------------------------
def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)

def set_seeds(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def fail(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)

# ---------------------------
# Embedding utilities
# ---------------------------
def load_bioclinicalbert(device: str = "cpu") -> Tuple[AutoTokenizer, AutoModel]:
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model.to(device)
    model.eval()
    return tokenizer, model

@torch.no_grad()
def embed_phrases_batch(
    phrases: List[str],
    tokenizer,
    model,
    device: str = "cpu",
    max_length: int = 16,
    batch_size: int = 64,
) -> torch.Tensor:
    """Mean-pool last_hidden_state; batched for speed/memory."""
    all_vecs = []
    n = len(phrases)
    for i in range(0, n, batch_size):
        batch = [str(p) for p in phrases[i : i + batch_size]]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc).last_hidden_state  # [B, T, H]
        vecs = out.mean(dim=1).detach().cpu()  # [B, H]
        all_vecs.append(vecs)
    return torch.cat(all_vecs, dim=0)

# ---------------------------
# Training
# ---------------------------
def build_mlp_pipeline(random_state: int = 42) -> Pipeline:
    # Standardize -> MLP (helps MLP convergence a lot)
    mlp = MLPClassifier(random_state=random_state)
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", mlp),
    ])
    return pipe

def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    use_stratified_cv: bool = True,
):
    param_grid = {
        "clf__hidden_layer_sizes": [(100,)],
        "clf__activation": ["relu"],
        "clf__solver": ["adam"],
        "clf__alpha": [1.0],
        "clf__max_iter": [300],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state) if use_stratified_cv \
         else KFold(n_splits=5, shuffle=True, random_state=random_state)

    grid = GridSearchCV(build_mlp_pipeline(random_state), param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    grid.fit(X, y)

    best = grid.best_estimator_
    accs = cross_val_score(best, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    y_pred_cv = cross_val_predict(best, X, y, cv=cv, n_jobs=-1)
    cm = confusion_matrix(y, y_pred_cv)
    return best, grid.best_params_, float(np.mean(accs)), cm.tolist()

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_xlsx", required=True, help="Excel with columns ['Phrase','Lable']")
    ap.add_argument("--new_xlsx", required=True, help="Excel with column ['Phrase']")
    ap.add_argument("--save_dir", default="outputs", help="Directory to save model/embeddings/predictions")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Torch device preference")
    ap.add_argument("--max_length", type=int, default=16, help="Tokenizer max_length (16–32 is typical for short phrases)")
    ap.add_argument("--batch_size", type=int, default=64, help="Embedding batch size")
    ap.add_argument("--save_embeddings", action="store_true", help="Save .pt embeddings for reproducibility")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = ap.parse_args()

    set_seeds(args.seed)
    ensure_dir(args.save_dir)

    # ---------- Device ----------
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available; falling back to CPU.")
            device = "cpu"
        else:
            device = args.device
    print(f"[INFO] Using device: {device}")

    # ---------- Load data ----------
    if not os.path.exists(args.train_xlsx):
        fail(f"Training file not found: {args.train_xlsx}")
    if not os.path.exists(args.new_xlsx):
        fail(f"New phrases file not found: {args.new_xlsx}")

    train_df = pd.read_excel(args.train_xlsx)
    new_df = pd.read_excel(args.new_xlsx)

    for col in ["Phrase", "Lable"]:
        if col not in train_df.columns:
            fail(f"Training file must contain column '{col}'. Found: {list(train_df.columns)}")
    if "Phrase" not in new_df.columns:
        fail(f"New phrases file must contain column 'Phrase'. Found: {list(new_df.columns)}")

    train_df = train_df[["Phrase", "Lable"]].dropna()
    new_df = new_df[["Phrase"]].dropna()

    if train_df.empty:
        fail("Training dataframe is empty after dropping NA.")
    if new_df.empty:
        fail("New phrases dataframe is empty after dropping NA.")

    phrases = train_df["Phrase"].astype(str).tolist()
    labels_raw = train_df["Lable"].astype(str).tolist()
    new_phrases = new_df["Phrase"].astype(str).tolist()

    # Encode labels and persist encoder
    le = LabelEncoder()
    y = le.fit_transform(labels_raw)
    joblib.dump(le, os.path.join(args.save_dir, "label_encoder.joblib"))

    # ---------- Embedding ----------
    tokenizer, model = load_bioclinicalbert(device=device)
    print(f"[INFO] Embedding {len(phrases)} training phrases...")
    X_train = embed_phrases_batch(
        phrases, tokenizer, model, device=device, max_length=args.max_length, batch_size=args.batch_size
    )
    print(f"[INFO] Embedding {len(new_phrases)} new phrases...")
    X_new = embed_phrases_batch(
        new_phrases, tokenizer, model, device=device, max_length=args.max_length, batch_size=args.batch_size
    )

    if args.save_embeddings:
        torch.save(X_train, os.path.join(args.save_dir, "embedded_train.pt"))
        torch.save(X_new, os.path.join(args.save_dir, "embedded_new.pt"))

    # Convert to numpy for sklearn
    X_train_np = X_train.numpy()
    X_new_np = X_new.numpy()

    # ---------- Train ----------
    print("[INFO] Training MLP with 5-fold Stratified CV and grid search...")
    clf, best_params, mean_acc, cm = train_mlp(X_train_np, y, random_state=args.seed, use_stratified_cv=True)
    model_path = os.path.join(args.save_dir, "mlp_clinical_phrases.joblib")
    joblib.dump(clf, model_path)

    metrics = {
        "cv_mean_accuracy": round(mean_acc, 6),
        "best_params": best_params,
        "confusion_matrix": cm,
        "n_train": int(len(phrases)),
        "n_new": int(len(new_phrases)),
        "max_length": int(args.max_length),
        "batch_size": int(args.batch_size),
        "device": device,
        "seed": int(args.seed),
    }
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Saved model: {model_path}")
    print(f"[INFO] CV mean accuracy: {mean_acc:.4f}")
    print(f"[INFO] Metrics written to metrics.json")

    # ---------- Predict ----------
    print("[INFO] Predicting labels for new phrases...")
    preds_enc = clf.predict(X_new_np)
    preds = le.inverse_transform(preds_enc)

    out = pd.DataFrame({"Phrase": new_phrases, "PredLabel": preds})
    out_path = os.path.join(args.save_dir, "new_phrases_predictions.xlsx")
    out.to_excel(out_path, index=False)
    print(f"[INFO] Saved predictions to: {out_path}")

if __name__ == "__main__":
    main()
