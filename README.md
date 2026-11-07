# ❤️ Heart Disease Prediction using Machine Learning

A complete, end-to-end machine-learning application that predicts the **likelihood of heart
disease** from routine clinical parameters. It trains several ML models on the public Cleveland
heart-disease dataset, evaluates them honestly on a hold-out test set, and serves predictions
through a **FastAPI** backend with a premium **Streamlit** dashboard — including **SHAP**
explainability so every result can be understood.

> ⚠️ **Important**: this is a *screening aid*, **not** a medical diagnosis. The model
> never replaces a qualified healthcare professional.

---

## ✨ Highlights

- Real ML pipeline: **Logistic Regression**, **Random Forest**, **XGBoost** + a **soft-voting
  ensemble**.
- Metrics computed on a real test set: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
- **SHAP** explainability (global feature importance + per-patient explanations).
- FastAPI REST API with Pydantic validation (`/health`, `/model-info`, `/predict`, `/explain`).
- Streamlit dashboard (Home, Prediction, Results, Model Performance, Explainable AI, About).
- Joblib-serialised preprocessing + prediction pipeline (no training/serving skew).

---

## 🗂️ Project structure

```
heart-disease-prediction/
├── app.py                     # Streamlit frontend
├── backend/
│   ├── main.py                # FastAPI app
│   ├── prediction.py          # Prediction engine
│   └── schemas.py             # Pydantic request/response models
├── models/
│   ├── ensemble_pipeline.joblib   # saved ensemble pipeline
│   ├── individual_models.joblib   # saved individual models
│   ├── shap_explainer.joblib      # SHAP explainer
│   ├── metrics.json               # real evaluation metrics
│   ├── feature_importances.json
│   └── plots/                     # ROC + confusion matrix PNGs
├── data/
│   └── heart.csv              # Cleveland heart-disease dataset
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluation.py
│   └── explainability.py
├── utils/
│   └── recommendations.py     # preventive suggestion engine
├── scripts/
│   └── fetch_data.py          # dataset downloader
├── notebooks/
│   └── model_training.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ 1. Install dependencies

**Recommended:** create a virtual environment (Python 3.11+).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

Install the packages:

```bash
pip install -r requirements.txt
```

If the dataset is not already present in `data/heart.csv`, download it:

```bash
python scripts/fetch_data.py
```

---

## 🏋️ 2. Train the models

```bash
python -m src.train
```

This runs the full workflow — loads & cleans data, splits stratified train/test, trains the three
individual models plus the ensemble, evaluates them on the test set, and saves:

- `models/ensemble_pipeline.joblib` — the serving pipeline
- `models/individual_models.joblib`
- `models/shap_explainer.joblib`
- `models/metrics.json` — real accuracy / precision / recall / F1 / ROC-AUC / confusion matrices
- `models/plots/roc_curves.png`, `confusion_matrices.png`

Training is also reproduced step-by-step in `notebooks/model_training.ipynb`.

> 🔒 **Reproducibility**: `RANDOM_STATE = 42`, stratified split, and a single shared
> preprocessor are used to prevent leakage and keep results reproducible.

---

## ⚙️ 3. Start the FastAPI backend

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Check it is alive:

- Health: `http://127.0.0.1:8000/health`
- Model info: `http://127.0.0.1:8000/model-info`
- Interactive docs: `http://127.0.0.1:8000/docs`

---

## 🖥️ 4. Start the Streamlit app

In a second terminal:

```bash
python -m streamlit run app.py
```

Open the printed URL (default `http://localhost:8501`). The frontend talks to the backend at
`http://127.0.0.1:8000` — keep both running.

---

## 🧪 5. Test the prediction functionality

You can exercise the API directly (example with `curl`):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"age\":63,\"sex\":1,\"cp\":3,\"trestbps\":145,\"chol\":233,\"fbs\":1,\"restecg\":0,\"thalach\":150,\"exang\":0,\"oldpeak\":2.3,\"slope\":0,\"ca\":0,\"thal\":1}"
```

Or through the dashboard: **Prediction → enter parameters → Run Prediction → Results** (with a
**Show SHAP Explanation** button).

---

## 📦 6. Upload to GitHub

```bash
git init
git add .
git commit -m "Heart Disease Prediction using Machine Learning"
git branch -M main
git remote add origin https://github.com/<your-username>/heart-disease-prediction.git
git push -u origin main
```

> The trained model files are committed by design so the app runs out-of-the-box.
> `models/plots/` and `__pycache__/` are git-ignored.

---

## 🎤 7. Explaining the project (interview cheat-sheet)

- **Problem**: cardiovascular disease remains a leading cause of death; early, accessible risk
  screening is limited.
- **Data**: public Cleveland heart-disease dataset — 303 patients, 14 attributes (age, sex, chest
  pain type, blood pressure, cholesterol, ECG, max heart rate, exercise angina, ST depression,
  slope, vessels, thalassemia).
- **Preprocessing**: drop duplicates, one-hot encode categoricals, standard-scale numerics, with a
  single `ColumnTransformer` reused for training and serving (**no data leakage**).
- **Models**: Logistic Regression (linear baseline), Random Forest & XGBoost (non-linear), combined
  into a **soft-voting ensemble**.
- **Evaluation**: stratified 80/20 split; accuracy, precision, recall, F1, ROC-AUC, confusion
  matrix — all computed on the real test set.
- **Explainability**: SHAP values show which features push a prediction up/down.
- **Architecture**: Streamlit frontend → FastAPI REST API → joblib model pipeline; Pydantic
  validates all inputs.
- **Ethics**: clear disclaimers; the tool is a screening aid, never a diagnosis.

---

## 📊 Actual results (test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|----------|
| Logistic Regression | 0.820 | 0.844 | 0.818 | 0.831 | 0.887 |
| Random Forest | 0.754 | 0.750 | 0.818 | 0.783 | 0.874 |
| XGBoost | 0.705 | 0.703 | 0.788 | 0.743 | 0.839 |
| Ensemble (Soft Voting) | 0.754 | 0.750 | 0.818 | 0.783 | 0.866 |

> These values are generated from your actual training run and stored in `models/metrics.json`.
> They are **not** fabricated and will reflect the specific train/test split.

---

## 🙏 Acknowledgements

Built as a BE mini-project by **Abhilash B N** (1BI24AI400) and **Nihar D** (1BI24AI403),
Department of AI & ML, Bangalore Institute of Technology, under the guidance of Dr. Kempanna M.
