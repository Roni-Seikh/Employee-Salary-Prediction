import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score

# Load model and encoders
model = joblib.load("best_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# Set page config
st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns **>50K or ≤50K** based on employee details.")

# --- Sidebar Inputs ---
st.sidebar.header("🧾 Input Employee Details")

age = st.sidebar.slider("📅 Age", 18, 65, 30)
education = st.sidebar.selectbox("🎓 Education Level", [
    "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"
])
occupation = st.sidebar.selectbox("💼 Job Role", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv", "Armed-Forces"
])
hours_per_week = st.sidebar.slider("⏱️ Hours per week", 1, 80, 40)
experience = st.sidebar.slider("🧠 Years of Experience", 0, 40, 5)

input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
})

# --- Preprocessing ---
for col in ['education', 'occupation']:
    if input_df[col][0] not in label_encoders[col].classes_:
        st.warning(f"⚠️ Unknown {col} value: {input_df[col][0]}. Defaulting to most frequent.")
        input_df[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])
    else:
        input_df[col] = label_encoders[col].transform(input_df[col])

# --- Single Prediction ---
st.markdown("### 🔍 Prediction on Single Entry")
st.dataframe(input_df)

if st.button("🔮 Predict Salary Class"):
    prediction = model.predict(input_df)
    predicted_label = target_encoder.inverse_transform(prediction)[0]
    st.success(f"✅ **Prediction: {predicted_label}**")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)
        conf = round(np.max(proba) * 100, 2)
        st.info(f"🧪 Confidence: {conf}%")

# --- Batch Prediction ---
st.markdown("---")
st.markdown("### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    try:
        batch_data = pd.read_csv(uploaded_file)
        original_cols = batch_data.columns.tolist()

        for col in ['education', 'occupation']:
            batch_data[col] = batch_data[col].apply(lambda x: x if x in label_encoders[col].classes_ else label_encoders[col].classes_[0])
            batch_data[col] = label_encoders[col].transform(batch_data[col])

        predictions = model.predict(batch_data)
        batch_data['PredictedClass'] = target_encoder.inverse_transform(predictions)

        st.success("✅ Batch predictions complete!")
        st.dataframe(batch_data.head())

        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Predictions", data=csv, file_name="batch_predictions.csv", mime='text/csv')

    except Exception as e:
        st.error(f"❌ Error: {e}")

# --- Data Visualization ---
with st.expander("📊 Show Input Distribution Charts"):
    st.markdown("Visualize the distribution of employee features")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot([age], bins=10, kde=True, ax=axs[0], color="skyblue")
    axs[0].set_title("Age")
    sns.histplot([hours_per_week], bins=10, kde=True, ax=axs[1], color="lightgreen")
    axs[1].set_title("Hours per Week")
    sns.histplot([experience], bins=10, kde=True, ax=axs[2], color="salmon")
    axs[2].set_title("Experience")
    st.pyplot(fig)

# --- Download Template ---
with st.expander("📁 Download Input Template"):
    example_df = pd.DataFrame({
        'age': [30],
        'education': ['Bachelors'],
        'occupation': ['Sales'],
        'hours-per-week': [40],
        'experience': [5]
    })
    st.dataframe(example_df)
    template_csv = example_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV Template", template_csv, "input_template.csv", mime='text/csv')

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit · Developed by **Roni Seikh** · ")
