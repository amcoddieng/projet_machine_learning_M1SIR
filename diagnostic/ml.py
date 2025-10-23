"""Prédiction de diagnostic tensionnel via modèle entraîné ou règles de secours."""
from dataclasses import dataclass
from typing import Literal, List
from pathlib import Path

Label = Literal['Normal', 'Élevée', 'Hypertension Stade 1', 'Hypertension Stade 2', 'Crise hypertensive']


@dataclass
class PredictionResult:
    """Conteneur du résultat de prédiction (étiquette et score)."""
    label: Label
    risk_score: float

# Chargement du modèle entraîné (si présent)
MODEL = None
LABELS: List[str] = ['Normal', 'Élevée', 'Hypertension Stade 1', 'Hypertension Stade 2', 'Crise hypertensive']
MODEL_PATH = Path(__file__).resolve().parent / 'model' / 'bp_model.joblib'

try:
    if MODEL_PATH.exists():
        import joblib
        # Charger le bundle joblib (modèle + labels)
        bundle = joblib.load(MODEL_PATH)
        MODEL = bundle.get('model')
        if isinstance(bundle, dict) and 'labels' in bundle and isinstance(bundle['labels'], list):
            LABELS = bundle['labels']  # type: ignore
except Exception:
    MODEL = None


def _predict_rule_based(systolic: int, diastolic: int) -> PredictionResult:
    """Règles simples basées sur les catégories de TA."""
    if systolic >= 180 or diastolic >= 120:
        return PredictionResult('Crise hypertensive', 0.99)
    if systolic >= 140 or diastolic >= 90:
        return PredictionResult('Hypertension Stade 2', 0.85)
    if 130 <= systolic <= 139 or 80 <= diastolic <= 89:
        return PredictionResult('Hypertension Stade 1', 0.65)
    if 120 <= systolic <= 129 and diastolic < 80:
        return PredictionResult('Élevée', 0.4)
    return PredictionResult('Normal', 0.1)


def predict_bp(systolic: int, diastolic: int) -> PredictionResult:
    """Prédit l'étiquette et un score de confiance à partir de (SYS,DIA)."""
    if MODEL is not None:
        try:
            import numpy as np
            X = np.array([[float(systolic), float(diastolic)]], dtype=float)
            if hasattr(MODEL, 'predict_proba'):
                proba = MODEL.predict_proba(X)[0]
                idx = int(proba.argmax())
                label = LABELS[idx] if idx < len(LABELS) else 'Hypertension Stade 1'
                score = float(proba[idx])
            else:
                idx = int(MODEL.predict(X)[0])
                label = LABELS[idx] if idx < len(LABELS) else 'Hypertension Stade 1'
                score = 0.5
            return PredictionResult(label, score)
        except Exception:
            return _predict_rule_based(systolic, diastolic)

    return _predict_rule_based(systolic, diastolic)
