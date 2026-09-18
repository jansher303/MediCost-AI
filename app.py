"""MediCost-AI: medical insurance cost estimator.

Run with:  streamlit run app.py
Needs:     streamlit, pandas, and whatever library trained model.pkl (e.g. scikit-learn)
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).with_name("model.pkl")

# "Northeast" is the baseline (all region columns = 0), matching one-hot
# encoding with drop_first=True during training.
REGIONS = ["Northeast", "Northwest", "Southeast", "Southwest"]

st.set_page_config(page_title="MediCost-AI", page_icon="🏥", layout="centered")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;800&display=swap');

    html, body, .stApp { font-family: 'Manrope', sans-serif; }

    .block-container { max-width: 980px; padding-top: 2rem; padding-bottom: 3rem; }
    #MainMenu, footer { visibility: hidden; }

    .app-header { text-align: center; padding: 0.5rem 0 1.5rem; }
    .app-icon {
        width: 88px; height: 88px; margin: 0 auto 0.9rem;
        display: flex; align-items: center; justify-content: center;
        background: #e8f4ff; border-radius: 50%;
    }
    .app-title { font-size: 2.6rem; font-weight: 800; color: #123c66; line-height: 1.1; }
    .app-subtitle { font-size: 1.1rem; color: #64748b; margin-top: 0.4rem; }

    div[data-testid="stWidgetLabel"] p { font-weight: 600; color: #263b53; }

    div[data-testid="stFormSubmitButton"] > button {
        width: 100%; min-height: 3.4rem; border-radius: 14px;
        font-size: 1.1rem; font-weight: 700;
    }

    .result-box {
        background: #ffffff; border: 2px solid #d8ebfa; border-radius: 20px;
        padding: 1.8rem 1.5rem; text-align: center; margin-top: 1.5rem;
        box-shadow: 0 8px 25px rgba(25, 80, 120, 0.08);
    }
    .result-label { font-size: 1rem; font-weight: 600; color: #64748b; }
    .result-value { font-size: 2.8rem; font-weight: 800; color: #1769aa; margin: 0.3rem 0; }
    .result-note { font-size: 0.95rem; color: #64748b; }

    .creator {
        text-align: center; margin-top: 3rem; padding-top: 1.2rem;
        border-top: 1px solid #dce8f2; color: #64748b; font-size: 0.95rem;
    }
    .creator b { color: #1769aa; font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading model...")
def load_model(path: Path):
    """Load the trained model once and reuse it across reruns and sessions."""
    with path.open("rb") as f:
        return pickle.load(f)


def build_features(model, *, age, sex, bmi, children, smoker, region) -> pd.DataFrame:
    """Turn form values into the exact feature frame the model was trained on."""
    row = {
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "bmi": bmi,
        "children": children,
        "smoker": 1 if smoker == "Yes" else 0,
        "region_northwest": int(region == "Northwest"),
        "region_southeast": int(region == "Southeast"),
        "region_southwest": int(region == "Southwest"),
    }
    frame = pd.DataFrame([row])

    # Match the training column order when the model records it (scikit-learn does).
    expected = getattr(model, "feature_names_in_", None)
    if expected is not None:
        missing = set(expected) - set(frame.columns)
        if missing:
            raise ValueError(f"Model expects columns the app does not provide: {sorted(missing)}")
        frame = frame[list(expected)]
    return frame


def predict_cost(model, **inputs) -> float:
    """Predict the cost, never returning a negative number."""
    return max(float(model.predict(build_features(model, **inputs))[0]), 0.0)


def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Healthy weight"
    if bmi < 30:
        return "Overweight"
    return "Obesity range"


# ---------------------------------------------------------------------------
# Load model (fail with a clear message, not a traceback)
# ---------------------------------------------------------------------------

if not MODEL_PATH.exists():
    st.error(f"Model file not found: `{MODEL_PATH.name}`. Place it in the same folder as `app.py`.")
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as exc:  # noqa: BLE001 - show any load problem to the developer
    st.error(f"Could not load the model: {exc}")
    st.stop()

st.session_state.setdefault("history", [])

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="app-header">
        <div class="app-icon">
            <svg width="58" height="58" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                <circle cx="32" cy="32" r="29" fill="#1769aa"/>
                <rect x="27" y="13" width="10" height="38" rx="3" fill="white"/>
                <rect x="13" y="27" width="38" height="10" rx="3" fill="white"/>
            </svg>
        </div>
        <div class="app-title">MediCost-AI</div>
        <div class="app-subtitle">Estimate a medical insurance cost in seconds</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Optional BMI helper (outside the form so it updates instantly)
# ---------------------------------------------------------------------------

with st.expander("Don't know your BMI? Calculate it"):
    h_col, w_col, r_col = st.columns(3)
    height_cm = h_col.number_input("Height (cm)", 100.0, 250.0, 170.0, step=1.0)
    weight_kg = w_col.number_input("Weight (kg)", 20.0, 300.0, 70.0, step=0.5)
    calc_bmi = weight_kg / (height_cm / 100) ** 2
    r_col.metric("Your BMI", f"{calc_bmi:.1f}", bmi_category(calc_bmi), delta_color="off")

# ---------------------------------------------------------------------------
# Input form (one submit = one rerun)
# ---------------------------------------------------------------------------

with st.form("prediction_form"):
    st.subheader("Personal information")

    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    sex = c2.selectbox("Gender", ["Female", "Male"])
    bmi = c3.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1, format="%.1f")

    c4, c5, c6 = st.columns(3)
    children = c4.number_input("Number of children", min_value=0, max_value=10, value=0, step=1)
    smoker = c5.selectbox("Smoker", ["No", "Yes"])
    region = c6.selectbox("Region", REGIONS)

    submitted = st.form_submit_button("Estimate cost", type="primary")

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

if submitted:
    inputs = dict(age=age, sex=sex, bmi=bmi, children=children, smoker=smoker, region=region)
    try:
        cost = predict_cost(model, **inputs)
        flipped = "No" if smoker == "Yes" else "Yes"
        alt_cost = predict_cost(model, **{**inputs, "smoker": flipped})
    except Exception as exc:  # noqa: BLE001
        st.error(f"Prediction failed: {exc}")
        st.stop()

    st.session_state["result"] = {"inputs": inputs, "cost": cost, "alt_cost": alt_cost}
    st.session_state["history"].append(
        {
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Age": age,
            "Gender": sex,
            "BMI": bmi,
            "Children": children,
            "Smoker": smoker,
            "Region": region,
            "Estimated cost ($)": round(cost, 2),
        }
    )

result = st.session_state.get("result")

if result:
    cost, alt_cost, used = result["cost"], result["alt_cost"], result["inputs"]

    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-label">Estimated medical insurance cost</div>
            <div class="result-value">${cost:,.2f}</div>
            <div class="result-note">Based on the details you entered above.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    m1, m2 = st.columns(2)
    m1.metric("BMI category", bmi_category(used["bmi"]))

    diff = alt_cost - cost
    if used["smoker"] == "Yes":
        m2.metric("If you didn't smoke", f"${alt_cost:,.2f}", f"{diff:+,.2f}", delta_color="inverse")
    else:
        m2.metric("If you smoked", f"${alt_cost:,.2f}", f"{diff:+,.2f}", delta_color="inverse")

    st.caption(
        "This is a statistical estimate from a machine learning model, not an insurance quote. "
        "Real premiums depend on the insurer, plan, and factors this model doesn't see."
    )

# ---------------------------------------------------------------------------
# Session history
# ---------------------------------------------------------------------------

if st.session_state["history"]:
    with st.expander(f"Estimates this session ({len(st.session_state['history'])})"):
        history_df = pd.DataFrame(st.session_state["history"])
        st.dataframe(history_df, hide_index=True)

        left, right = st.columns(2)
        left.download_button(
            "Download as CSV",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="medicost_estimates.csv",
            mime="text/csv",
        )
        if right.button("Clear history"):
            st.session_state["history"] = []
            st.session_state.pop("result", None)
            st.rerun()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="creator">Models trained by<br><b>Jan Sher</b></div>',
    unsafe_allow_html=True,
)