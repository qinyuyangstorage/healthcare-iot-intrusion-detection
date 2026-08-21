from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from iot_ids.features import PaperFeaturePipeline
from iot_ids.model import LightweightIDS
from iot_ids.synthetic import make_synthetic_iomt


def train_model(frame: pd.DataFrame, epochs: int = 100, seed: int = 42) -> dict[str, object]:
    torch.manual_seed(seed)
    train, test = train_test_split(frame, test_size=0.30, random_state=seed, stratify=frame["label"])
    pipeline = PaperFeaturePipeline()
    x_train, y_train = pipeline.fit_transform(train)
    x_test, y_test = pipeline.transform(test)

    model = LightweightIDS()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=64,
        shuffle=True,
    )
    model.train()
    final_loss = 0.0
    for _ in range(epochs):
        for features, labels in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(features), labels)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())

    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        predictions = (torch.sigmoid(model(torch.from_numpy(x_test))) >= 0.5).numpy().astype(int)
    elapsed_ms = (time.perf_counter() - start) * 1000
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "dataset": "synthetic non-clinical IoMT traffic",
        "seed": seed,
        "train_rows": len(train),
        "test_rows": len(test),
        "epochs": epochs,
        "retained_features_before_pca": len(pipeline.retained_columns_),
        "pca_explained_variance_ratio": pipeline.pca_.explained_variance_ratio_.tolist(),
        "model_parameters": parameter_count,
        "final_batch_loss": final_loss,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "false_positive_rate": fp / max(fp + tn, 1),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "total_inference_ms": elapsed_ms,
        "mean_inference_ms_per_record": elapsed_ms / len(test),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce the PCA-guided lightweight IoMT IDS pipeline")
    parser.add_argument("--input", type=Path, help="Optional CSV with a binary 'label' column")
    parser.add_argument("--synthetic-rows", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/metrics.json"))
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.input) if args.input else make_synthetic_iomt(args.synthetic_rows, args.seed)
    metrics = train_model(frame, epochs=args.epochs, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
