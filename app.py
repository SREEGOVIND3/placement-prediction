import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="Placement Predictor", page_icon="🎓", layout="centered")

# Simple UI Styling
st.markdown(
    """
    <style>
        .main-title {
            font-size: 40px;
            font-weight: 800;
            text-align: center;
        }
        .sub-title {
            text-align: center;
            color: gray;
            margin-top: -10px;
        }
        .card {
            padding: 20px;
            border-radius: 16px;
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
        }
        .result-card {
            padding: 18px;
            border-radius: 16px;
            background-color: #ffffff;
            border: 1px solid #e9ecef;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">🎓 Placement Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Logistic Regression | Real-time Placement Prediction</div>', unsafe_allow_html=True)
st.write("")

# Sidebar info
st.sidebar.title("📌 Placement Predictor")
st.sidebar.write("Fill the details and click **Predict**")
st.sidebar.markdown("---")
st.sidebar.info("✅ Model: Logistic Regression\n\n✅ Accuracy: 81%")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("placement.csv")

df = load_data()

# -----------------------------
# Remove useless columns
# -----------------------------
drop_cols = ["StudentID", "student_id", "ID", "RollNo", "Roll Number", "Name"]
for c in drop_cols:
    if c in df.columns:
        df = df.drop(c, axis=1)

# -----------------------------
# Target Conversion
# -----------------------------
if df["PlacementStatus"].dtype == "object":
    df["PlacementStatus"] = df["PlacementStatus"].astype("category").cat.codes

# -----------------------------
# Only Important Features
# -----------------------------
important_features = [
    "CGPA",
    "Internships",
    "Projects",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "Workshops/Certifications",
    "PlacementTraining",
    "ExtracurricularActivities",
]
important_features = [f for f in important_features if f in df.columns]

# -----------------------------
# Convert Yes/No columns in dataset (if text)
# -----------------------------
yes_no_cols = ["PlacementTraining", "ExtracurricularActivities"]
for col in yes_no_cols:
    if col in df.columns and df[col].dtype == "object":
        df[col] = df[col].str.strip().str.lower().map({
            "yes": 1, "no": 0,
            "y": 1, "n": 0,
            "true": 1, "false": 0
        })

for col in yes_no_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# -----------------------------
# Train Model
# -----------------------------
X = df[important_features]
y = df["PlacementStatus"]

X_encoded = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# -----------------------------
# UI Input Card
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📝 Enter Student Details")

c1, c2 = st.columns(2)
user_data = {}

# ✅ CGPA (0–10)
if "CGPA" in important_features:
    user_data["CGPA"] = c1.number_input(
        "CGPA (0 - 10)",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.1
    )

# ✅ Internships (no restriction)
if "Internships" in important_features:
    user_data["Internships"] = c2.number_input("Internships", value=0)

# ✅ Projects (no restriction)
if "Projects" in important_features:
    user_data["Projects"] = c1.number_input("Projects", value=0)

# ✅ AptitudeTestScore (0–100)
if "AptitudeTestScore" in important_features:
    user_data["AptitudeTestScore"] = c2.number_input(
        "Aptitude Test Score (0 - 100)",
        min_value=0,
        max_value=100,
        value=60,
        step=1
    )

# ✅ SoftSkillsRating (0–10)
if "SoftSkillsRating" in important_features:
    user_data["SoftSkillsRating"] = c1.number_input(
        "Soft Skills Rating (0 - 10)",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.1
    )

# ✅ Workshops/Certifications (no restriction)
if "Workshops/Certifications" in important_features:
    user_data["Workshops/Certifications"] = c2.number_input("Workshops/Certifications", value=0)

# ✅ PlacementTraining (Yes/No)
if "PlacementTraining" in important_features:
    training_choice = c1.selectbox("Placement Training", ["No", "Yes"])
    user_data["PlacementTraining"] = 1 if training_choice == "Yes" else 0

# ✅ ExtracurricularActivities (Yes/No)
if "ExtracurricularActivities" in important_features:
    extra_choice = c2.selectbox("Extracurricular Activities", ["No", "Yes"])
    user_data["ExtracurricularActivities"] = 1 if extra_choice == "Yes" else 0

st.markdown("</div>", unsafe_allow_html=True)

st.write("")
predict_btn = st.button("🚀 Predict Placement", use_container_width=True)

# -----------------------------
# Prediction
# -----------------------------
if predict_btn:
    input_df = pd.DataFrame([user_data])

    # Apply same encoding
    input_encoded = pd.get_dummies(input_df, drop_first=True)

    # Add missing columns
    for col in X_encoded.columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Match column order
    input_encoded = input_encoded[X_encoded.columns]

    # Scale + Predict
    input_scaled = scaler.transform(input_encoded)

    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    # Results
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("✅ Prediction Result")

    if pred == 1:
        st.success("🎉 Student is likely to GET PLACED ✅")
    else:
        st.error("❌ Student is likely to NOT get placed")

    st.write("### 📊 Confidence Score")
    st.progress(float(proba[1]))
    st.write(f"Placed Probability: **{proba[1]*100:.2f}%**")
    st.write(f"Not Placed Probability: **{proba[0]*100:.2f}%**")
    st.markdown("</div>", unsafe_allow_html=True)

st.caption("Built by **Sree Govind** | Logistic Regression + Streamlit 🚀")
