# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Dummy dataset
data = {
    'age': [25, 45, 35, 50, 23],
    'education': ['Bachelors', 'Masters', 'PhD', 'HS-grad', 'Assoc'],
    'occupation': ['Sales', 'Exec-managerial', 'Tech-support', 'Craft-repair', 'Adm-clerical'],
    'hours-per-week': [40, 50, 60, 30, 20],
    'experience': [2, 20, 10, 25, 1],
    'income_class': ['<=50K', '>50K', '>50K', '<=50K', '<=50K']
}

df = pd.DataFrame(data)

# Encode categorical columns
label_encoders = {}
for col in ['education', 'occupation']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode target
target_le = LabelEncoder()
df['income_class'] = target_le.fit_transform(df['income_class'])

# Features and target
X = df.drop('income_class', axis=1)
y = df['income_class']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model and encoders
joblib.dump(model, "best_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(target_le, "target_encoder.pkl")

print("✅ Model and encoders saved.")
