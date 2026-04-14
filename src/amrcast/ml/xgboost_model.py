"""XGBoost-based MIC prediction model (Tier 1 baseline)."""

import json
import logging
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold, KFold, train_test_split

logger = logging.getLogger(__name__)


class MICPredictor:
    """XGBoost regression model for log2(MIC) prediction.

    One model per antibiotic. Predicts log2(MIC) and rounds to nearest
    valid doubling dilution.
    """

    VALID_LOG2_MICS = np.arange(-7, 11, dtype=float)  # 0.008 to 1024 ug/mL

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

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        n_folds: int = 5,
        random_state: int = 42,
        groups: np.ndarray | None = None,
        y_lower: np.ndarray | None = None,
        y_upper: np.ndarray | None = None,
    ) -> dict:
        """Run k-fold cross-validation and return aggregated metrics.

        Also trains a final model on all data (stored as self.model).

        Args:
            groups: Optional group labels for each sample. When provided,
                uses GroupKFold so all samples in the same group stay in
                the same fold. Use this for phylogenetic/clonal grouping
                to prevent data leakage from related genomes.
            y_lower: Lower bound of MIC interval (log2). -inf for left-censored.
                When provided along with y_upper, uses XGBoost AFT objective
                for proper censored interval regression.
            y_upper: Upper bound of MIC interval (log2). +inf for right-censored.

        Returns:
            Dict with per-fold and aggregated (mean ± std) metrics.
        """
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        use_aft = y_lower is not None and y_upper is not None

        if groups is not None:
            n_unique_groups = len(set(groups))
            actual_folds = min(n_folds, n_unique_groups)
            kf = GroupKFold(n_splits=actual_folds)
            split_iter = kf.split(X, y, groups)
        else:
            actual_folds = n_folds
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            split_iter = kf.split(X)

        fold_metrics = []
        for fold_i, (train_idx, val_idx) in enumerate(split_iter):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Hold out 15% of training fold for early stopping
            X_tr, X_es, y_tr, y_es = train_test_split(
                X_train, y_train, test_size=0.15, random_state=random_state
            )

            if use_aft:
                lo_tr, lo_es = y_lower[train_idx], y_lower[val_idx]
                hi_tr, hi_es = y_upper[train_idx], y_upper[val_idx]
                # Split interval bounds matching the early-stop split
                lo_tr_tr, lo_tr_es, hi_tr_tr, hi_tr_es = train_test_split(
                    lo_tr, hi_tr, test_size=0.15, random_state=random_state
                )
                model = self._train_interval_fold(
                    X_tr, lo_tr_tr, hi_tr_tr,
                    X_val=X_es, y_val=y_es,
                    random_state=random_state,
                )
            else:
                model = xgb.XGBRegressor(
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=3,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=random_state,
                    early_stopping_rounds=20,
                    eval_metric="mae",
                )
                model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)

            if use_aft:
                y_pred = model.predict(xgb.DMatrix(X_val))
            else:
                y_pred = model.predict(X_val)

            metrics = self._compute_metrics(y_val, y_pred)
            metrics["n_train"] = len(X_train)
            metrics["n_val"] = len(X_val)
            fold_metrics.append(metrics)

            logger.info(
                f"[{self.antibiotic}] Fold {fold_i+1}/{n_folds}: "
                f"MAE={metrics['mae']:.2f}, EA={metrics['essential_agreement']:.1%}"
            )

        # Aggregate
        agg = {}
        for key in ["mae", "essential_agreement", "exact_match"]:
            values = [m[key] for m in fold_metrics]
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))

        agg["n_samples"] = len(y)
        agg["n_folds"] = actual_folds
        agg["grouped"] = groups is not None
        agg["aft"] = use_aft
        agg["fold_metrics"] = fold_metrics

        logger.info(
            f"[{self.antibiotic}] CV result: "
            f"MAE={agg['mae_mean']:.2f}±{agg['mae_std']:.2f}, "
            f"EA={agg['essential_agreement_mean']:.1%}±{agg['essential_agreement_std']:.1%}"
        )

        # Train final model on all data for deployment
        if use_aft:
            X_tr, X_es, y_tr, y_es = train_test_split(
                X, y, test_size=0.15, random_state=random_state
            )
            lo_tr, lo_es, hi_tr, hi_es = train_test_split(
                y_lower, y_upper, test_size=0.15, random_state=random_state
            )
            self.model = self._train_interval_fold(
                X_tr, lo_tr, hi_tr, X_val=X_es, y_val=y_es,
                random_state=random_state,
            )
        else:
            X_tr, X_es, y_tr, y_es = train_test_split(
                X, y, test_size=0.15, random_state=random_state
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
                random_state=random_state,
                early_stopping_rounds=20,
                eval_metric="mae",
            )
            self.model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)

        return agg

    def _make_interval_objective(self, y_lower: np.ndarray, y_upper: np.ndarray):
        """Create a custom XGBoost objective for interval-censored regression.

        Loss is 0 when prediction falls inside [lower, upper].
        Outside the interval, loss is squared distance to nearest bound.

        This properly handles censored MIC data:
          - Exact (==):   [4.0, 4.0]   — standard squared loss
          - Left (<=/< ): [-inf, -6.0] — no penalty for predicting lower
          - Right (>=/>): [5.0, +inf]   — no penalty for predicting higher
        """
        lo = y_lower.copy()
        hi = y_upper.copy()

        def interval_obj(predt: np.ndarray, dtrain: xgb.DMatrix) -> tuple:
            n = len(predt)
            grad = np.zeros(n)
            hess = np.ones(n)  # constant hessian for stability

            # Below lower bound: gradient pushes prediction up
            below = predt < lo
            grad[below] = predt[below] - lo[below]

            # Above upper bound: gradient pushes prediction down
            above = predt > hi
            grad[above] = predt[above] - hi[above]

            # Inside interval: zero gradient (no loss)
            return grad, hess

        return interval_obj

    def _train_interval_fold(
        self,
        X: np.ndarray,
        y_lower: np.ndarray,
        y_upper: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        random_state: int = 42,
    ) -> xgb.Booster:
        """Train one model with interval-censored custom objective.

        Returns a Booster trained with the interval loss.
        """
        dtrain = xgb.DMatrix(X)
        # Use midpoint of interval as label (needed for eval metric)
        y_mid = np.where(
            np.isfinite(y_lower) & np.isfinite(y_upper),
            (y_lower + y_upper) / 2,
            np.where(np.isfinite(y_lower), y_lower, y_upper),
        )
        dtrain.set_label(y_mid)

        obj = self._make_interval_objective(y_lower, y_upper)

        params = {
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "seed": random_state,
        }

        evals = []
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dval, "val")]

        model = xgb.train(
            params, dtrain, num_boost_round=500, obj=obj,
            evals=evals, verbose_eval=False,
            early_stopping_rounds=20 if evals else None,
        )
        return model

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Predict log2(MIC) values (continuous)."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        if isinstance(self.model, xgb.Booster):
            # AFT predictions are in shifted positive space — shift back
            return self.model.predict(xgb.DMatrix(X)) - self.AFT_SHIFT
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
        if isinstance(self.model, xgb.Booster):
            self.model.save_model(str(model_path))
        else:
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
