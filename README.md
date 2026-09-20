# SDG Text Classifier — NLP + Machine Learning

A multiclass text classifier that maps free-form Spanish text to one of the 17 UN Sustainable Development Goals (SDGs), deployed as an interactive web app.

**Live demo:** [[clasificador-ods-blhkm5ky27fcijjctfnymk.streamlit.app](https://clasificador-ods-blhkm5ky27fcijjctfnymk.streamlit.app)](https://clasificador-ods-uniandes.streamlit.app/)

## Overview

Public institutions and organizations like UNFPA need to relate large volumes of participatory-planning text to the SDG framework — a task that traditionally requires manual expert annotation. This project builds an automated pipeline that classifies text into SDGs using classical NLP and machine learning, trained on the [OSDG Community Dataset](https://osdg.ai/).

The project covers the full lifecycle: text preprocessing, unsupervised topic exploration, supervised classification, model evaluation, and deployment as a usable application.

## Pipeline

```
Raw text
   │
   ▼
Preprocessing (tokenization, stopword removal, Spanish stemming)
   │
   ▼
TF-IDF vectorization (Bag-of-Words + weighting)
   │
   ▼
Dimensionality reduction (TruncatedSVD / LSA)
   │
   ▼
Logistic Regression classifier
   │
   ▼
Predicted SDG + confidence score
```

All steps are wrapped in a single `scikit-learn` `Pipeline`, so the exact same transformations used in training are applied at inference time — no train/serve skew.

## Methodology highlights

- **Topic modeling (LSA):** explored 10–20 SVD components, selecting the configuration that **maximizes topic coherence** (`c_v`, computed with `gensim`) rather than just explained variance — the two metrics disagreed, and coherence was prioritized since the goal was topic interpretability, not variance retention.
- **Hyperparameter tuning:** `GridSearchCV` over SVD components, regularization strength, and class weighting, with stratified cross-validation.
- **Train/test/validation split:** an explicit three-way split — training + CV, a held-out test set for aggregate metrics, and a separate held-out validation set for qualitative inspection of individual predictions — so no single split is reused across different evaluation purposes.
- **Error analysis:** confusion matrix findings were cross-validated against the topic-modeling results — SDGs that share vocabulary in the LSA topics (e.g. poverty, decent work, inequality) are the same ones the classifier confuses most, which is documented and interpreted rather than left as a bare number.

## Results

| Metric | Cross-validation | Test set |
|---|---|---|
| F1 macro | 0.8098 | 0.8071 |
| Accuracy | — | 0.8416 |
| F1 weighted | — | 0.8417 |

Performance is not uniform across classes: SDGs with distinctive vocabulary (16, 4, 5) reach F1 > 0.89, while SDGs sharing socioeconomic vocabulary (8, 9, 10) are harder to separate (F1 between 0.53 and 0.66) — a limitation expected from a bag-of-words representation, discussed in detail in the analysis notebook.

## Tech stack

- **Data processing / ML:** pandas, NumPy, scikit-learn, gensim
- **NLP:** NLTK (Spanish stopwords, stemming)
- **Visualization:** matplotlib, WordCloud
- **Deployment:** Streamlit, Streamlit Community Cloud

## Project structure

```
.
├── app.py                 # Streamlit app: loads the trained pipeline and serves predictions
├── modelo_ods.pkl          # Trained pipeline (TF-IDF + SVD + Logistic Regression)
├── requirements.txt        # Python dependencies
└── notebook/                # Full analysis: EDA, LSA, model selection, evaluation
```

## Running locally

```bash
git clone https://github.com/andres-ricoq/clasificador-ods.git
cd clasificador-ods
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

[OSDG Community Dataset (OSDG-CD)](https://osdg.ai/), 2023 version — 40,067 texts labeled by SDG relevance through community-driven annotation, machine-translated to Spanish and augmented for this project.

## Author

Andres — Data Science / Machine Learning
