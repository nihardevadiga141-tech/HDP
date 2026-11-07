"""Heart Disease Prediction — Streamlit frontend.

Premium white & navy-blue healthcare dashboard with six pages:
  Home, Prediction, Results, Model Performance, Explainable AI, About.

The UI talks to the FastAPI backend (default http://127.0.0.1:8000). Start the
backend first with:
    python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HeartAI — Heart Disease Screening",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://127.0.0.1:8000"

NAVY = "#0B2545"
NAVY_2 = "#13315C"
NAVY_3 = "#134074"
BLUE_ACCENT = "#8DA9C4"
BG = "#F6F8FB"

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CSS = f"""
<style>
    .stApp {{ background-color: {BG}; }}
    h1, h2, h3 {{ color: {NAVY}; font-family: 'Segoe UI', 'Helvetica Neue', sans-serif; }}
    .hero {{
        background: linear-gradient(135deg, {NAVY} 0%, {NAVY_2} 60%, {NAVY_3} 100%);
        border-radius: 18px; padding: 2.2rem 2.4rem; color: white; margin-bottom: 1.6rem;
        box-shadow: 0 10px 30px rgba(11,37,69,0.25);
    }}
    .hero h1 {{ color: white; font-size: 2.25rem; margin-bottom: .3rem; }}
    .hero p {{ color: #d7e3f0; font-size: 1.05rem; max-width: 820px; }}
    .card {{
        background: white; border-radius: 14px; padding: 1.4rem 1.5rem;
        border: 1px solid #e2e9f2; box-shadow: 0 4px 14px rgba(16,44,84,0.06);
        height: 100%;
    }}
    .metric-card {{
        background: white; border-radius: 14px; padding: 1.1rem 1.3rem;
        border-left: 5px solid {NAVY_3}; box-shadow: 0 4px 14px rgba(16,44,84,0.06);
    }}
    .metric-value {{ font-size: 1.9rem; font-weight: 800; color: {NAVY}; }}
    .metric-label {{ color: #5a6b84; font-size: .88rem; }}
    .section-title {{ color: {NAVY}; font-weight: 700; margin: 1.6rem 0 .7rem; }}
    .disclaimer {{
        background: #eef4fb; border-radius: 12px; padding: 1rem 1.2rem;
        color: {NAVY_2}; font-size: .95rem; border-left: 4px solid {BLUE_ACCENT};
        margin-top: 1.2rem;
    }}
    .risk-badge {{ border-radius: 10px; padding: .35rem .9rem; font-weight: 700; display: inline-block; }}
    .stButton>button {{
        background: {NAVY}; color: white; border-radius: 10px; font-weight: 600;
        border: none; padding: .55rem 1.4rem;
    }}
    .stButton>button:hover {{ background: {NAVY_3}; color: white; }}
    .nav-note {{ color: #7c8aa0; font-size: .85rem; margin-top: 1.5rem; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def api_request(method: str, path: str, payload: dict | None = None) -> dict:
    """Call the FastAPI backend and return parsed JSON; raises on failure."""
    url = f"{API_URL}{path}"
    timeout = 30
    try:
        if method.lower() == "get":
            resp = requests.get(url, timeout=timeout)
        else:
            resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError as exc:
        raise ConnectionError(
            f"Could not reach the backend at {API_URL}. Make sure the FastAPI server "
            f"is running (python -m uvicorn backend.main:app --port 8000)."
        ) from exc
    except requests.HTTPError as exc:
        raise RuntimeError(f"Backend returned {resp.status_code}: {resp.text}") from exc


def load_metrics() -> dict:
    try:
        with open("models/metrics.json", "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def load_feature_importances() -> dict:
    try:
        with open("models/feature_importances.json", "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def render_metric_card(label: str, value: str, suffix: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-value">{value}{suffix}</div>'
        f'<div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page: Home
# ---------------------------------------------------------------------------
def page_home():
    st.markdown(
        '<div class="hero"><h1>❤️ HeartAI — Heart Disease Screening</h1>'
        "<p>An end-to-end machine-learning application that estimates cardiovascular risk "
        "from routine clinical parameters, with transparent, explainable results — designed "
        "to support (not replace) professional cardiovascular care.</p></div>",
        unsafe_allow_html=True,
    )

    metrics = load_metrics()
    best = None
    if metrics and metrics.get("models"):
        best_name = max(metrics["models"], key=lambda k: metrics["models"][k]["roc_auc"])
        best = metrics["models"][best_name]
    else:
        best_name = "—"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if best:
            render_metric_card("Best Model (ROC-AUC)", f"{best['roc_auc']*100:.1f}", "%")
        else:
            render_metric_card("Best Model (ROC-AUC)", "—")
    with c2:
        if best:
            render_metric_card("Best Accuracy", f"{best['accuracy']*100:.1f}", "%")
        else:
            render_metric_card("Best Accuracy", "—")
    with c3:
        render_metric_card("Models Trained", "4")
    with c4:
        render_metric_card("Clinical Features", "13")

    st.markdown('<div class="section-title">Project Overview</div>', unsafe_allow_html=True)
    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            """
            This project delivers a **working machine-learning pipeline** that predicts the
            likelihood of heart disease:

            - **Data** — the classic Cleveland heart-disease dataset (303 patients, 14 attributes).
            - **Models** — Logistic Regression, Random Forest and XGBoost, combined into a
              soft-voting ensemble.
            - **Evaluation** — accuracy, precision, recall, F1, ROC-AUC and confusion matrices
              computed on a real hold-out test set.
            - **Explainability** — SHAP to show *why* a patient received a given risk.
            - **Access** — a Streamlit dashboard backed by a FastAPI REST API.

            > ⚠️ This is a **screening aid**, not a medical diagnosis. Always consult a
            > qualified healthcare professional for clinical decisions.
            """
        )
    with right:
        with st.container():
            st.markdown('<div class="card"><b>🛰️ Architecture</b><ul style="margin-top:8px">'
                        '<li>Streamlit frontend UI</li>'
                        '<li>FastAPI backend (REST)</li>'
                        '<li>Joblib pipeline storage</li>'
                        '<li>SHAP explainability</li>'
                        '<li>Plotly interactive charts</li></ul></div>',
                        unsafe_allow_html=True)

    st.markdown('<div class="section-title">Get started</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a:
        st.info("**1.** Open the **Prediction** page and enter patient parameters.")
    with b:
        st.info("**2.** Submit — the model returns risk + explanation.")
    with c:
        st.info("**3.** Review recommendations and consult a professional if needed.")


# ---------------------------------------------------------------------------
# Page: Prediction
# ---------------------------------------------------------------------------
def page_prediction():
    st.markdown('<div class="hero"><h1>🩺 Heart Disease Prediction</h1><p>'
                "Enter the patient's clinical parameters. All fields are required and "
                "validated by the backend before prediction.</p></div>",
                unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown("### 👤 Demographics")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (years)", min_value=20, max_value=100, value=55, step=1)
        with col2:
            sex = st.selectbox("Sex", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])
        with col3:
            cp = st.selectbox(
                "Chest pain type",
                options=[("Typical angina", 0), ("Atypical angina", 1),
                         ("Non-anginal pain", 2), ("Asymptomatic", 3)],
                format_func=lambda x: x[0],
            )

        st.markdown("### 🫀 Vitals & Lab")
        col1, col2, col3 = st.columns(3)
        with col1:
            trestbps = st.number_input("Resting blood pressure (mm Hg)", min_value=80, max_value=250, value=130, step=1)
        with col2:
            chol = st.number_input("Serum cholesterol (mg/dl)", min_value=100, max_value=600, value=230, step=1)
        with col3:
            thalach = st.number_input("Max heart rate achieved (bpm)", min_value=50, max_value=250, value=140, step=1)

        col1, col2, col3 = st.columns(3)
        with col1:
            fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        with col2:
            restecg = st.selectbox(
                "Resting ECG result",
                options=[("Normal", 0), ("ST-T wave abnormality", 1), ("Left ventricular hypertrophy", 2)],
                format_func=lambda x: x[0],
            )
        with col3:
            exang = st.selectbox("Exercise-induced angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])

        st.markdown("### 📈 Exercise / Imaging")
        col1, col2, col3 = st.columns(3)
        with col1:
            oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=6.2, value=1.0, step=0.1, format="%.1f")
        with col2:
            slope = st.selectbox(
                "Slope of peak exercise ST segment",
                options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)],
                format_func=lambda x: x[0],
            )
        with col3:
            ca = st.selectbox("Major vessels colored by fluoroscopy (0–4)", options=[0, 1, 2, 3, 4])

        st.markdown("### 🧬 Other")
        col1, col2 = st.columns(2)
        with col1:
            thal = st.selectbox(
                "Thalassemia",
                options=[("Normal", 1), ("Fixed defect", 2), ("Reversable defect", 3)],
                format_func=lambda x: x[0],
            )

        submitted = st.form_submit_button("🩺 Run Prediction", width="stretch")

    if submitted:
        payload = {
            "age": int(age),
            "sex": int(sex[1]),
            "cp": int(cp[1]),
            "trestbps": int(trestbps),
            "chol": int(chol),
            "fbs": int(fbs[1]),
            "restecg": int(restecg[1]),
            "thalach": int(thalach),
            "exang": int(exang[1]),
            "oldpeak": float(oldpeak),
            "slope": int(slope[1]),
            "ca": int(ca),
            "thal": int(thal[1]),
        }
        with st.spinner("Running prediction …"):
            try:
                result = api_request("post", "/predict", payload)
                st.session_state["last_payload"] = payload
                st.session_state["last_result"] = result
                st.success("Prediction complete — see the **Results** page.")
            except (ConnectionError, RuntimeError) as exc:
                st.error(str(exc))


# ---------------------------------------------------------------------------
# Page: Results
# ---------------------------------------------------------------------------
def page_results():
    st.markdown('<div class="hero"><h1>📋 Prediction Results</h1><p>Actual model output '
                "for the submitted parameters.</p></div>", unsafe_allow_html=True)

    result = st.session_state.get("last_result")

    if not result:
        st.info("No prediction yet. Run a prediction on the **Prediction** page first.")
        return

    probability = result["probability"]
    risk = result["risk_category"]
    prediction = result["prediction"]

    color = {"Low": "#1e8e3e", "Moderate": "#b26a00", "High": "#c62828"}[risk]

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("Predicted Class", "Heart disease present" if prediction == 1 else "No heart disease")
    with c2:
        render_metric_card("Estimated Probability", f"{probability*100:.1f}", "%")
    with c3:
        st.markdown(
            f'<div class="metric-card"><div style="font-size:1.3rem;font-weight:800;color:{color}">'
            f'{risk} Risk</div><div class="metric-label">Risk category</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Interpretation</div>', unsafe_allow_html=True)
    st.markdown(
        f"The model estimates a **{probability*100:.1f}%** probability of heart disease "
        f"being present, placing this patient in the **{risk.lower()}** risk category."
    )

    if risk == "High":
        st.error("⚠️ This result indicates elevated risk. Professional medical review is strongly advised.")
    elif risk == "Moderate":
        st.warning("Moderate risk — please monitor and consider discussing with a clinician.")
    else:
        st.success("Low predicted risk. Continue maintaining a heart-healthy lifestyle.")

    st.markdown('<div class="section-title">Preventive suggestions</div>', unsafe_allow_html=True)
    recs = result["recommendations"]
    st.markdown("**General guidance:**")
    for item in recs["general"]:
        st.markdown(f"- {item}")
    if recs["specific"]:
        st.markdown("**Based on your inputs:**")
        for item in recs["specific"]:
            st.markdown(f"- {item}")

    st.markdown('<div class="disclaimer">⚠️ ' + result["disclaimer"] + "</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Explain this result</div>', unsafe_allow_html=True)
    if st.button("🔬 Show SHAP Explanation"):
        with st.spinner("Computing feature contributions …"):
            try:
                explanation = api_request("post", "/explain", st.session_state.get("last_payload", {}))
                st.session_state["last_explanation"] = explanation
                _render_shap_chart(explanation)
            except (ConnectionError, RuntimeError) as exc:
                st.error(str(exc))


# ---------------------------------------------------------------------------
# SHAP chart renderer (shared by Results & Explainable AI pages)
# ---------------------------------------------------------------------------
def _render_shap_chart(explanation: dict) -> None:
    names = explanation["feature_names"]
    values = explanation["shap_values"]
    base = explanation["base_value"]

    pairs = sorted(zip(names, values), key=lambda t: t[1])
    top = pairs[-10:]
    labels = [p[0] for p in top]
    contribs = [p[1] for p in top]

    fig = go.Figure(go.Bar(
        x=contribs,
        y=labels,
        orientation="h",
        marker_color=[NAVY_3 if v < 0 else "#e07b2e" for v in contribs],
        text=[f"{v:+.3f}" for v in contribs],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Feature contributions (towards 'disease' class) — base value {base:.3f}",
        height=460,
        margin=dict(l=20, r=40, t=60, b=20),
        xaxis_title="SHAP contribution",
        paper_bgcolor="white",
        plot_bgcolor=BG,
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("**How to read it:** bars pointing right increase the predicted probability of "
                "heart disease; bars pointing left decrease it. Longer bars = stronger influence.")


# ---------------------------------------------------------------------------
# Page: Model Performance
# ---------------------------------------------------------------------------
def page_performance():
    st.markdown('<div class="hero"><h1>📊 Model Performance</h1><p>Real evaluation results '
                "computed on the held-out test set — nothing fabricated.</p></div>",
                unsafe_allow_html=True)

    metrics = load_metrics()
    if not metrics or "models" not in metrics:
        st.warning("No metrics found. Train the models first (python -m src.train).")
        return

    models = metrics["models"]
    names = list(models.keys())

    st.markdown('<div class="section-title">Comparison table</div>', unsafe_allow_html=True)
    rows = []
    for name in names:
        m = models[name]
        rows.append({
            "Model": name,
            "Accuracy": round(m["accuracy"], 4),
            "Precision": round(m["precision"], 4),
            "Recall": round(m["recall"], 4),
            "F1": round(m["f1"], 4),
            "ROC-AUC": round(m["roc_auc"], 4),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown('<div class="section-title">Metric comparison chart</div>', unsafe_allow_html=True)
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    chart_data = []
    for name in names:
        for key in metric_keys:
            chart_data.append({"Model": name, "Metric": key.title().replace("_", "-"), "Value": models[name][key]})
    df = pd.DataFrame(chart_data)
    fig = px.bar(df, x="Metric", y="Value", color="Model", barmode="group",
                 text_auto=".2f", color_discrete_sequence=["#0B2545", "#134074", "#8DA9C4", "#e07b2e"])
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=40, b=20),
                      paper_bgcolor="white", plot_bgcolor=BG, legend=dict(orientation="h", y=1.12))
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, width="stretch")

    st.markdown('<div class="section-title">ROC curves (test set)</div>', unsafe_allow_html=True)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#9aa7b8"), name="Chance"))
    colors = ["#0B2545", "#134074", "#8DA9C4", "#e07b2e"]
    for i, name in enumerate(names):
        pts = models[name]["roc"]
        roc_fig.add_trace(go.Scatter(
            x=pts["fpr"], y=pts["tpr"], mode="lines", name=f"{name} (AUC={models[name]['roc_auc']:.3f})",
            line=dict(width=2.5, color=colors[i % len(colors)]),
        ))
    roc_fig.update_layout(height=440, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                          margin=dict(l=30, r=20, t=40, b=40), paper_bgcolor="white", plot_bgcolor=BG,
                          legend=dict(orientation="h", y=1.1))
    st.plotly_chart(roc_fig, width="stretch")

    st.markdown('<div class="section-title">Confusion matrices</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    cm_path = "models/plots/confusion_matrices.png"
    if exists_os_path(cm_path):
        col_a.image(cm_path, caption="Confusion matrices (predicted vs actual)", width="stretch")
    roc_path = "models/plots/roc_curves.png"
    if exists_os_path(roc_path):
        col_b.image(roc_path, caption="Combined ROC curves", width="stretch")


def exists_os_path(path: str) -> bool:
    import os

    return os.path.exists(path)


# ---------------------------------------------------------------------------
# Page: Explainable AI
# ---------------------------------------------------------------------------
def page_explainable():
    st.markdown('<div class="hero"><h1>🔬 Explainable AI</h1><p>SHAP-based transparency: '
                "understand which clinical features drive the prediction.</p></div>",
                unsafe_allow_html=True)

    st.markdown('<div class="section-title">Global feature importance</div>', unsafe_allow_html=True)
    fi = load_feature_importances()
    if fi:
        model_name = "XGBoost" if "XGBoost" in fi else list(fi.keys())[0]
        data = fi[model_name]
        names = data["feature_names"]
        imp = data["importance"]
        s = sorted(zip(names, imp), key=lambda t: t[1], reverse=True)
        top = s[:12]
        fig = go.Figure(go.Bar(
            x=[v for _, v in top], y=[n for n, _ in top], orientation="h",
            marker_color=NAVY_3, text=[f"{v:.3f}" for _, v in top], textposition="outside",
        ))
        fig.update_layout(title=f"Mean feature importance ({model_name})", height=440,
                          margin=dict(l=20, r=40, t=50, b=20),
                          xaxis_title="Importance", paper_bgcolor="white", plot_bgcolor=BG)
        st.plotly_chart(fig, width="stretch")
        st.caption("Feature importance reflects how much each (encoded) feature helped the model separations.")
    else:
        st.info("Feature importances not available yet — train the models first.")

    st.markdown('<div class="section-title">Explain a patient</div>', unsafe_allow_html=True)
    explanation = st.session_state.get("last_explanation")
    if explanation:
        _render_shap_chart(explanation)
        if st.button("Clear stored explanation"):
            st.session_state.pop("last_explanation", None)
            st.rerun()
    else:
        st.info("Run a prediction and click **Show SHAP Explanation** on the Results page to "
                "see an individual explanation here.")


# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
def page_about():
    st.markdown(
        '<div class="hero"><h1>ℹ️ About the Project</h1><p>'
        "A mini-project in Artificial Intelligence & Machine Learning.</p></div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="card"><b>🎯 Goal</b><p style="margin-top:8px">Develop a real, working '
            "ML application that estimates heart-disease risk from clinical parameters and gives "
            "transparent, explainable results.</p></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="card"><b>🧩 Tech stack</b><ul style="margin-top:8px">'
            "<li>Python · Pandas · NumPy</li>"
            "<li>Scikit-learn · XGBoost</li>"
            "<li>Streamlit (UI) · FastAPI (API)</li>"
            "<li>SHAP (explainability)</li>"
            "<li>Joblib (model storage)</li>"
            "<li>Plotly · Matplotlib · Seaborn</li></ul></div>",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Workflow</div>', unsafe_allow_html=True)
    st.markdown(
        "1. **Data** — public Cleveland heart-disease dataset (303 patients, 14 attributes).\n"
        "2. **Preprocessing** — cleaning, one-hot encoding, standard scaling (no leakage).\n"
        "3. **Models** — Logistic Regression, Random Forest, XGBoost + soft-voting ensemble.\n"
        "4. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC, confusion matrices on a "
        "stratified hold-out test set.\n"
        "5. **Explainability** — SHAP to interpret individual predictions.\n"
        "6. **Serving** — FastAPI REST API consumed by a Streamlit dashboard."
    )

    st.markdown('<div class="section-title">Limitations & ethics</div>', unsafe_allow_html=True)
    st.markdown(
        "<ul>"
        "<li>This is a <b>screening tool</b>, not a diagnostic device.</li>"
        "<li>Predictions must never be treated as a confirmed medical diagnosis.</li>"
        "<li>Data is small (303 patients) and from a single historical cohort — it may not "
        "generalise to all populations.</li>"
        "<li>Always consult a qualified healthcare professional.</li>"
        "</ul>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Team</div>', unsafe_allow_html=True)
    st.markdown(
        "**Abhilash B N** (1BI24AI400) · **Nihar D** (1BI24AI403) · "
        "Department of AI & ML, Bangalore Institute of Technology."
    )

    st.markdown(
        '<div class="disclaimer">This application is built for educational/academic '
        "purposes and is not a substitute for professional medical advice.</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### <span style='color:{NAVY}'>❤️ HeartAI</span>", unsafe_allow_html=True)
    st.markdown("Cardiovascular risk screening")
    _pages = ["Home", "Prediction", "Results", "Model Performance", "Explainable AI", "About"]
    _requested = st.query_params.get("page", "Home")
    _default_index = _pages.index(_requested) if _requested in _pages else 0
    page = st.radio(
        "Navigate",
        _pages,
        index=_default_index,
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Backend: FastAPI")
    st.caption(f"API: {API_URL}")
    st.markdown('<div class="nav-note">⚠️ Educational tool — not a medical diagnosis.</div>',
                unsafe_allow_html=True)

PAGES = {
    "Home": page_home,
    "Prediction": page_prediction,
    "Results": page_results,
    "Model Performance": page_performance,
    "Explainable AI": page_explainable,
    "About": page_about,
}

PAGES[page]()
