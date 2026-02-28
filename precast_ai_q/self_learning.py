from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


class SelfLearningLayer:
    FEATURE_RANGES = {
        "cement": (250.0, 600.0),
        "water_cement_ratio": (0.25, 0.80),
        "admixture": (0.0, 4.0),
        "temperature": (10.0, 90.0),
        "humidity": (20.0, 100.0),
        "time_hours": (2.0, 48.0),
    }

    def __init__(
        self,
        state_path: str = "data/self_learning_state.json",
        feedback_path: str = "data/production_feedback.csv",
        learning_rate: float = 0.08,
    ):
        self.state_path = self._resolve_path(state_path)
        self.feedback_path = self._resolve_path(feedback_path)
        self.learning_rate = learning_rate
        self.state = self._load_state()

    def _resolve_path(self, relative_or_absolute: str) -> Path:
        candidate = Path(relative_or_absolute)
        if candidate.is_absolute():
            candidate.parent.mkdir(parents=True, exist_ok=True)
            return candidate

        module_dir = Path(__file__).resolve().parent
        preferred = module_dir / relative_or_absolute
        preferred.parent.mkdir(parents=True, exist_ok=True)
        return preferred

    def _default_state(self) -> dict:
        return {
            "global_bias": 0.0,
            "curing_bias": {},
            "feature_weights": {k: 0.0 for k in self.FEATURE_RANGES.keys()},
            "updates_count": 0,
            "mae_before": 0.0,
            "mae_after": 0.0,
        }

    def _load_state(self) -> dict:
        if not self.state_path.exists():
            return self._default_state()

        with open(self.state_path, "r", encoding="utf-8") as file:
            loaded = json.load(file)

        default = self._default_state()
        default.update(loaded)

        for feature in self.FEATURE_RANGES.keys():
            default["feature_weights"].setdefault(feature, 0.0)

        default.setdefault("curing_bias", {})
        return default

    def _save_state(self):
        with open(self.state_path, "w", encoding="utf-8") as file:
            json.dump(self.state, file, indent=2)

    def _normalize(self, feature_name: str, value: float) -> float:
        lower, upper = self.FEATURE_RANGES[feature_name]
        if upper <= lower:
            return 0.0
        normalized = (value - lower) / (upper - lower)
        return float(max(0.0, min(1.0, normalized)))

    def _feature_vector(self, cement, water_cement_ratio, admixture, temperature, humidity, time_hours) -> dict:
        return {
            "cement": self._normalize("cement", cement),
            "water_cement_ratio": self._normalize("water_cement_ratio", water_cement_ratio),
            "admixture": self._normalize("admixture", admixture),
            "temperature": self._normalize("temperature", temperature),
            "humidity": self._normalize("humidity", humidity),
            "time_hours": self._normalize("time_hours", time_hours),
        }

    def adjustment(self, cement, water_cement_ratio, admixture, temperature, humidity, time_hours, curing_type) -> float:
        feature_vector = self._feature_vector(
            cement,
            water_cement_ratio,
            admixture,
            temperature,
            humidity,
            time_hours,
        )

        weighted_component = sum(
            self.state["feature_weights"][feature] * value for feature, value in feature_vector.items()
        )
        curing_component = self.state["curing_bias"].get(curing_type, 0.0)

        return float(self.state["global_bias"] + weighted_component + curing_component)

    def predict(self, base_prediction, cement, water_cement_ratio, admixture, temperature, humidity, time_hours, curing_type):
        corrected = base_prediction + self.adjustment(
            cement,
            water_cement_ratio,
            admixture,
            temperature,
            humidity,
            time_hours,
            curing_type,
        )
        return float(max(0.0, min(100.0, corrected)))

    def update(
        self,
        cement,
        water_cement_ratio,
        admixture,
        temperature,
        humidity,
        time_hours,
        curing_type,
        predicted_strength,
        actual_strength,
    ) -> dict:
        error_before = float(actual_strength - predicted_strength)

        feature_vector = self._feature_vector(
            cement,
            water_cement_ratio,
            admixture,
            temperature,
            humidity,
            time_hours,
        )

        self.state["global_bias"] += self.learning_rate * error_before * 0.35

        previous_curing_bias = self.state["curing_bias"].get(curing_type, 0.0)
        self.state["curing_bias"][curing_type] = previous_curing_bias + self.learning_rate * error_before * 0.45

        for feature, value in feature_vector.items():
            self.state["feature_weights"][feature] += self.learning_rate * error_before * value * 0.08

        predicted_after = self.predict(
            predicted_strength,
            cement,
            water_cement_ratio,
            admixture,
            temperature,
            humidity,
            time_hours,
            curing_type,
        )
        error_after = float(actual_strength - predicted_after)

        previous_count = int(self.state["updates_count"])
        new_count = previous_count + 1
        self.state["updates_count"] = new_count

        self.state["mae_before"] = (
            (self.state["mae_before"] * previous_count) + abs(error_before)
        ) / new_count
        self.state["mae_after"] = (
            (self.state["mae_after"] * previous_count) + abs(error_after)
        ) / new_count

        self._save_state()
        self._append_feedback(
            cement,
            water_cement_ratio,
            admixture,
            temperature,
            humidity,
            time_hours,
            curing_type,
            predicted_strength,
            actual_strength,
            predicted_after,
            error_before,
            error_after,
        )

        return {
            "predicted_before": float(predicted_strength),
            "predicted_after": float(predicted_after),
            "actual": float(actual_strength),
            "error_before": float(error_before),
            "error_after": float(error_after),
            "updates_count": int(self.state["updates_count"]),
        }

    def _append_feedback(
        self,
        cement,
        water_cement_ratio,
        admixture,
        temperature,
        humidity,
        time_hours,
        curing_type,
        predicted_strength,
        actual_strength,
        predicted_after,
        error_before,
        error_after,
    ):
        row = pd.DataFrame(
            [
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "cement": cement,
                    "water_cement_ratio": water_cement_ratio,
                    "admixture": admixture,
                    "temperature": temperature,
                    "humidity": humidity,
                    "time_hours": time_hours,
                    "curing_type": curing_type,
                    "predicted_before": predicted_strength,
                    "actual_strength": actual_strength,
                    "predicted_after": predicted_after,
                    "error_before": error_before,
                    "error_after": error_after,
                }
            ]
        )

        if self.feedback_path.exists():
            row.to_csv(self.feedback_path, mode="a", index=False, header=False)
        else:
            row.to_csv(self.feedback_path, mode="w", index=False, header=True)

    def get_summary(self) -> dict:
        return {
            "updates_count": int(self.state["updates_count"]),
            "mae_before": float(self.state["mae_before"]),
            "mae_after": float(self.state["mae_after"]),
            "global_bias": float(self.state["global_bias"]),
        }
