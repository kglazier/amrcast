"""XGBoost-based MIC prediction model (Tier 1 baseline)."""

import json
import logging
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class MICPredictor:
    """XGBoost regression model for log2(MIC) prediction.

    One model per antibiotic. Predicts log2(MIC) and rounds to nearest
    valid doubling dilution.
    """

    VALID_LOG2_MICS = np.arange(-4, 11, dtype=float)  # 0.0625 to 1024 ug/mL

    def __init__(self, antibiotic: str):
        self.antibiotic = antibiotic
        self.model: xgb.XGBRegressor | None = None
        self.feature_names: list[str] = []

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        val_fraction: float = 0.15,
    ) -> dict:
        """Train XGBoost regression on log2(MIC) values.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: log2(MIC) target values.
            feature_names: Optional feature names for interpretability.
            val_fraction: Fraction of data to hold out for early stopping.

        Returns:
            Dict with training metrics.
        """
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_fraction, random_state=42
        )

        self.model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            early_stopping_rounds=20,
            eval_metric="mae",
        )

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Evaluate on validation set
        y_pred = self.predict_raw(X_val)
        metrics = self._compute_metrics(y_val, y_pred)
        metrics["n_train"] = len(X_train)
        metrics["n_val"] = len(X_val)
        metrics["best_iteration"] = self.model.best_iteration

        logger.info(
            f"[{self.antibiotic}] Trained: MAE={metrics['mae']:.2f}, "
            f"EA={metrics['essential_agreement']:.1%}, "
            f"n_train={metrics['n_train']}, n_val={metrics['n_val']}"
        )

        return metrics

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Predict log2(MIC) values (continuous)."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict log2(MIC) values rounded to nearest valid dilution."""
        raw = self.predict_raw(X)
        return self._round_to_dilution(raw)

    def predict_mic(self, X: np.ndarray) -> np.ndarray:
        """Predict MIC values in ug/mL."""
        log2_mic = self.predict(X)
        return np.power(2.0, log2_mic)

    def _round_to_dilution(self, log2_values: np.ndarray) -> np.ndarray:
        """Round continuous log2(MIC) to nearest valid doubling dilution."""
        rounded = np.round(log2_values)
        return np.clip(rounded, self.VALID_LOG2_MICS.min(), self.VALID_LOG2_MICS.max())

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute clinical microbiology evaluation metrics."""
        y_pred_rounded = self._round_to_dilution(y_pred)

        mae = float(np.mean(np.abs(y_true - y_pred)))
        diff = np.abs(y_true - y_pred_rounded)

        # Essential Agreement: within +/- 1 doubling dilution
        ea = float(np.mean(diff <= 1.0))

        # Exact match
        exact = float(np.mean(diff == 0.0))

        return {
            "mae": mae,
            "essential_agreement": ea,
            "exact_match": exact,
        }

    def save(self, model_dir: Path) -> None:
        """Save model and metadata."""
        if self.model is None:
            raise RuntimeError("No model to save.")

        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"xgb_{self.antibiotic}.json"
        self.model.get_booster().save_model(str(model_path))

        meta = {
            "antibiotic": self.antibiotic,
            "feature_names": self.feature_names,
            "n_features": len(self.feature_names),
        }
        meta_path = model_dir / f"xgb_{self.antibiotic}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved model to {model_path}")

    def load(self, model_dir: Path) -> None:
        """Load model and metadata."""
        model_path = model_dir / f"xgb_{self.antibiotic}.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        booster = xgb.Booster()
        booster.load_model(str(model_path))
        self.model = xgb.XGBRegressor()
        self.model._Booster = booster

        meta_path = model_dir / f"xgb_{self.antibiotic}_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.feature_names = meta.get("feature_names", [])

        logger.info(f"Loaded model from {model_path}")
