from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.units import cm
from pathlib import Path

TITLE = "Rapport de Projet — Plateforme Django de Diagnostic de Tension (ML)"


def build_paragraph(text, style_name="BodyText", styles=None):
    styles = styles or getSampleStyleSheet()
    return Paragraph(text, styles[style_name])


def main(output_path: Path = None):
    if output_path is None:
        output_path = Path.cwd() / "rapport_projet_diagnostic_tension.pdf"

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"]))
    styles.add(ParagraphStyle(name="H3", parent=styles["Heading3"]))

    doc = SimpleDocTemplate(str(output_path), pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    story = []

    # Title
    story.append(Paragraph(TITLE, styles["TitleCenter"]))
    story.append(Spacer(1, 0.5*cm))

    # 1. Contexte et Objectifs
    story.append(Paragraph("1. Contexte et Objectifs", styles["H2"]))
    story.append(build_paragraph("<b>Problématique</b> : Aider au screening rapide de l’hypertension via des valeurs de tension artérielle (TA)."))
    story.append(build_paragraph("<b>Objectif</b> : Développer une mini-plateforme Web (Django) permettant de saisir <i>systolic</i> et <i>diastolic</i> et d’obtenir un diagnostic indicatif, d’abord par règles, puis via un modèle ML entraîné."))
    story.append(Spacer(1, 0.3*cm))

    # 2. Données
    story.append(Paragraph("2. Données", styles["H2"]))
    story.append(build_paragraph("<b>CSV interne</b> : <code>diagnostic/data/bp_dataset.csv</code>"))
    story.append(build_paragraph("<b>Colonnes</b> : <code>systolic, diastolic, label</code>"))
    story.append(build_paragraph("<b>Classes</b> : Normal, Élevée, Hypertension Stade 1, Hypertension Stade 2, Crise hypertensive."))
    story.append(build_paragraph("<b>Remarque</b> : Dataset d’exemple (synthétique) conforme aux catégories usuelles de TA."))
    story.append(Spacer(1, 0.3*cm))

    # 3. Méthode et Architecture
    story.append(Paragraph("3. Méthode et Architecture", styles["H2"]))
    story.append(build_paragraph("<b>Stack</b> : Django 5, scikit-learn (Logistic Regression), joblib."))
    story.append(build_paragraph("<b>Approche</b> : Règles de base (fallback) dans <code>diagnostic/ml.py</code>; modèle ML entraîné et chargé si présent."))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph("Fichiers clés", styles["H3"]))
    items = [
        "diagnostic/ml.py → predict_bp(), chargement modèle et fallback",
        "diagnostic/train_model.py → pipeline d’entraînement et sauvegarde",
        "diagnostic/forms.py → BloodPressureForm",
        "diagnostic/views.py → home()",
        "diagnostic/templates/diagnostic/home.html → interface utilisateur",
        "diagnostic/data/bp_dataset.csv → dataset d’exemple",
        "projet_m1Sir/settings.py → INSTALLED_APPS inclut diagnostic",
        "projet_m1Sir/urls.py → include('diagnostic.urls') à la racine",
    ]
    story.append(ListFlowable([ListItem(Paragraph(i, styles["BodyText"])) for i in items], bulletType='bullet'))
    story.append(Spacer(1, 0.3*cm))

    # 4. Implémentation ML
    story.append(Paragraph("4. Implémentation ML", styles["H2"]))
    story.append(build_paragraph("<b>Entrées</b> : [[systolic, diastolic]]."))
    story.append(build_paragraph("<b>Labels</b> : ['Normal','Élevée','Hypertension Stade 1','Hypertension Stade 2','Crise hypertensive']."))
    story.append(build_paragraph("<b>Pipeline</b> : StandardScaler + LogisticRegression(multi_class='multinomial')."))
    story.append(build_paragraph("<b>Persistance</b> : <code>diagnostic/model/bp_model.joblib</code> via joblib."))
    story.append(Spacer(1, 0.3*cm))

    # 5. Résultats et Évaluation
    story.append(Paragraph("5. Résultats et Évaluation", styles["H2"]))
    story.append(build_paragraph("Après entraînement, un rapport de classification est affiché (précision, rappel, f1). Sur données synthétiques, performance élevée attendue."))
    story.append(Spacer(1, 0.3*cm))

    # 6. Démonstration
    story.append(Paragraph("6. Démonstration", styles["H2"]))
    demo_items = [
        "Installer les dépendances : pip install -r requirements.txt",
        "Entraîner le modèle : python diagnostic/train_model.py",
        "Lancer le serveur : python manage.py migrate && python manage.py runserver",
        "Tester : http://127.0.0.1:8000/ puis saisir systolic/diastolic",
    ]
    story.append(ListFlowable([ListItem(Paragraph(i, styles["BodyText"])) for i in demo_items], bulletType='1'))
    story.append(Spacer(1, 0.3*cm))

    # 7. Avertissements et Éthique
    story.append(Paragraph("7. Avertissements et Éthique", styles["H2"]))
    story.append(build_paragraph("Outil indicatif, non destiné au diagnostic médical. Pas de stockage patient implémenté. Les performances sur données réelles requièrent un dataset cliniquement représentatif."))
    story.append(Spacer(1, 0.3*cm))

    # 8. Limites et Améliorations
    story.append(Paragraph("8. Limites et Améliorations", styles["H2"]))
    lim_items = [
        "Variables limitées (uniquement TA). Ajouter d’autres facteurs cliniques.",
        "Dataset de démo synthétique. Intégrer des données réelles anonymisées.",
        "Améliorer la robustesse (détection anomalies, unités, valeurs manquantes).",
        "Élargir les modèles (RF, XGBoost), calibration des probabilités, tuning.",
    ]
    story.append(ListFlowable([ListItem(Paragraph(i, styles["BodyText"])) for i in lim_items], bulletType='bullet'))

    doc.build(story)
    print(f"PDF généré : {output_path}")


if __name__ == "__main__":
    main()
