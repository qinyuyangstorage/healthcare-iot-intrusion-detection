from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class PaperFeaturePipeline:
    correlation_threshold: float = 0.75
    label_column: str = "label"
    dropped_columns: tuple[str, ...] = ("class", "ip.proto")
    categorical_columns_: list[str] = field(default_factory=list)
    retained_columns_: list[str] = field(default_factory=list)
    frequency_maps_: dict[str, dict[object, float]] = field(default_factory=dict)
    scaler_: StandardScaler | None = None
    pca_: PCA | None = None

    def _split(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        if self.label_column not in frame:
            raise ValueError(f"Missing label column: {self.label_column}")
        y = frame[self.label_column].astype(int).to_numpy()
        x = frame.drop(columns=[self.label_column, *self.dropped_columns], errors="ignore").copy()
        return x, y

    def fit_transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x, y = self._split(frame)
        self.categorical_columns_ = x.select_dtypes(include=["object", "category"]).columns.tolist()
        for column in self.categorical_columns_:
            frequencies = x[column].value_counts(normalize=True, dropna=False).to_dict()
            self.frequency_maps_[column] = frequencies
            x[column] = x[column].map(frequencies).fillna(0.0)
        x = x.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        correlation = x.corr().abs()
        upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
        removed = {column for column in upper.columns if (upper[column] > self.correlation_threshold).any()}
        self.retained_columns_ = [column for column in x.columns if column not in removed]
        if len(self.retained_columns_) < 2:
            raise ValueError("Correlation filtering retained fewer than two usable features")

        values = x[self.retained_columns_].to_numpy(dtype=np.float64)
        self.scaler_ = StandardScaler()
        scaled = self.scaler_.fit_transform(values)
        self.pca_ = PCA(n_components=2, random_state=42)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            components = self.pca_.fit_transform(scaled)
        if not np.isfinite(components).all():
            raise ValueError("PCA produced non-finite training components")
        return augment_components(components), y

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.scaler_ is None or self.pca_ is None:
            raise RuntimeError("Pipeline must be fitted before transform")
        x, y = self._split(frame)
        for column in self.categorical_columns_:
            mapping = self.frequency_maps_[column]
            x[column] = x[column].map(mapping).fillna(0.0)
        x = x.reindex(columns=self.retained_columns_, fill_value=0.0)
        x = x.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        scaled = self.scaler_.transform(x.to_numpy(dtype=np.float64))
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            components = self.pca_.transform(scaled)
        if not np.isfinite(components).all():
            raise ValueError("PCA produced non-finite test components")
        return augment_components(components), y


def augment_components(components: np.ndarray) -> np.ndarray:
    if components.ndim != 2 or components.shape[1] != 2:
        raise ValueError("Expected PCA components with shape (n, 2)")
    x1, x2 = components[:, 0], components[:, 1]
    return np.column_stack([x1, x2, x1**2, x2**2, np.sin(x1), np.sin(x2)]).astype(np.float32)
