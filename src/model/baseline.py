"""
Baseline Classifier Model.
Standard isolated-step Logistic Regression classifier evaluating static flow/packet features
without temporal dynamics modeling. Used for comparative benchmarking against the World Model.
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


class LogisticRegressionBaseline:
    """Logistic Regression baseline operating on isolated time-window feature vectors."""

    def __init__(self, c_param: float = 1.0, max_iter: int = 1000, random_state: int = 42):
        self.c_param = c_param
        self.max_iter = max_iter
        self.random_state = random_state
        self.infil_model = LogisticRegression(
            C=c_param,
            max_iter=max_iter,
            random_state=random_state,
            class_weight="balanced"
        )
        self.stage_model = LogisticRegression(
            C=c_param,
            max_iter=max_iter,
            random_state=random_state,
            class_weight="balanced"
        )
        self.is_fitted = False

    def fit(
        self,
        x_features: np.ndarray,
        y_infil: np.ndarray,
        y_stages: Optional[np.ndarray] = None
    ) -> "LogisticRegressionBaseline":
        """
        Fit baseline on isolated feature vectors S_t.
        If 3D array [N, seq_len, D] is passed, extracts the current step S_t (last element of sequence).
        """
        if x_features.ndim == 3:
            x_step = x_features[:, -1, :]
        else:
            x_step = x_features

        # If y_infil is 2D [N, K], take next step t+1 target
        if y_infil.ndim == 2:
            y_infil_step = y_infil[:, 0]
        else:
            y_infil_step = y_infil

        self.infil_model.fit(x_step, y_infil_step)

        if y_stages is not None:
            if y_stages.ndim == 2:
                y_stages_step = y_stages[:, 0]
            else:
                y_stages_step = y_stages
            self.stage_model.fit(x_step, y_stages_step)

        self.is_fitted = True
        return self

    def predict_infil_proba(self, x_features: np.ndarray) -> np.ndarray:
        """Predict infiltration probability for the immediate next step."""
        if x_features.ndim == 3:
            x_step = x_features[:, -1, :]
        else:
            x_step = x_features

        if not self.is_fitted:
            raise RuntimeError("Baseline model is not fitted yet.")

        # Return probability of positive class (infiltration)
        if len(self.infil_model.classes_) == 1:
            return np.zeros(len(x_step))
        return self.infil_model.predict_proba(x_step)[:, 1]

    def predict_stage(self, x_features: np.ndarray) -> np.ndarray:
        """Predict MITRE stage category."""
        if x_features.ndim == 3:
            x_step = x_features[:, -1, :]
        else:
            x_step = x_features

        if not self.is_fitted:
            raise RuntimeError("Baseline model is not fitted yet.")

        return self.stage_model.predict(x_step)

    def evaluate(
        self,
        x_test: np.ndarray,
        y_infil_test: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Compute F1, Precision, Recall, and False Positive Rate (FPR)."""
        probas = self.predict_infil_proba(x_test)
        preds = (probas >= threshold).astype(int)

        if y_infil_test.ndim == 2:
            targets = y_infil_test[:, 0].astype(int)
        else:
            targets = y_infil_test.astype(int)

        # Metrics
        f1 = float(f1_score(targets, preds, zero_division=0))
        prec = float(precision_score(targets, preds, zero_division=0))
        rec = float(recall_score(targets, preds, zero_division=0))

        # Confusion Matrix for False Positive Rate: FPR = FP / (FP + TN)
        cm = confusion_matrix(targets, preds, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            fpr = float(fp / max(fp + tn, 1))
        else:
            fpr = 0.0

        return {
            "f1_score": f1,
            "precision": prec,
            "recall": rec,
            "false_positive_rate": fpr
        }

    def save(self, output_path: str = "model/weights/baseline_final.pkl") -> None:
        """Save baseline model to pickle file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, input_path: str = "model/weights/baseline_final.pkl") -> "LogisticRegressionBaseline":
        """Load baseline model from pickle file."""
        with open(input_path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    np.random.seed(42)
    x = np.random.randn(200, 10, 28)
    y_inf = np.random.randint(0, 2, size=(200, 5))
    y_sg = np.random.randint(0, 6, size=(200, 5))

    baseline = LogisticRegressionBaseline()
    baseline.fit(x[:150], y_inf[:150], y_sg[:150])
    metrics = baseline.evaluate(x[150:], y_inf[150:])
    print("Baseline Test Metrics:", metrics)
