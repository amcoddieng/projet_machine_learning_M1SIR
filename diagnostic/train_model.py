"""Script d'entraînement pour un modèle de classification tensionnelle."""
import os
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Ordre des étiquettes cibles
LABELS = ['Normal', 'Élevée', 'Hypertension Stade 1', 'Hypertension Stade 2', 'Crise hypertensive']


def rule_label(sys: int, dia: int) -> int:
    """Mappe (SYS,DIA) vers un indice de classe selon des règles simples."""
    if sys >= 180 or dia >= 120:
        return 4
    if sys >= 140 or dia >= 90:
        return 3
    if 130 <= sys <= 139 or 80 <= dia <= 89:
        return 2
    if 120 <= sys <= 129 and dia < 80:
        return 1
    return 0


def synthesize(n_per_class: int = 800):
    """Génère un jeu synthétique équilibré par classes autour de plages de TA."""
    rng = np.random.default_rng(42)
    X = []
    y = []

    for _ in range(n_per_class):
        sys = rng.integers(95, 120)
        dia = rng.integers(60, 80)
        X.append([sys, dia])
        y.append(rule_label(sys, dia))

    for _ in range(n_per_class):
        sys = rng.integers(120, 130)
        dia = rng.integers(55, 79)
        X.append([sys, dia])
        y.append(rule_label(sys, dia))

    for _ in range(n_per_class):
        sys = rng.integers(130, 140)
        dia = rng.integers(80, 90)
        X.append([sys, dia])
        y.append(rule_label(sys, dia))

    for _ in range(n_per_class):
        sys = rng.integers(140, 180)
        dia = rng.integers(90, 120)
        X.append([sys, dia])
        y.append(rule_label(sys, dia))

    for _ in range(n_per_class):
        sys = rng.integers(180, 220)
        dia = rng.integers(110, 130)
        X.append([sys, dia])
        y.append(rule_label(sys, dia))

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)
    return X, y


def main():
    """Entraîne un pipeline, évalue et sauvegarde le modèle au format joblib."""
    X, y = synthesize()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)

    # Pipeline: normalisation + régression logistique multinomiale
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(multi_class="multinomial", max_iter=1000, random_state=7)),
    ])

    pipe.fit(Xtr, ytr)

    ypred = pipe.predict(Xte)
    print(classification_report(yte, ypred, target_names=LABELS))

    # Sauvegarde du modèle et des labels
    model_dir = Path(__file__).resolve().parent / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / "bp_model.joblib"

    joblib.dump({
        "model": pipe,
        "labels": LABELS,
    }, out_path)

    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
