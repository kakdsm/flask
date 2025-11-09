# create_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Define the class IN THIS FILE
class JobRecommenderSystem:
    def __init__(self, model):
        self.model = model
    
    def get_top_5_jobs(self, skills_input):
        skills_array = np.array(skills_input).reshape(1, -1)
        probabilities = self.model.predict_proba(skills_array)[0]
        
        job_probs = pd.DataFrame({
            'Job': self.model.classes_,
            'Probability': probabilities
        }).sort_values('Probability', ascending=False).head(5)
        
        return [(row['Job'], float(row['Probability'])) for _, row in job_probs.iterrows()]
    
    def get_5_soft_traits(self, skills_input):
        return {
            'Critical Thinking': int(skills_input[10]),
            'Problem Solving': int(skills_input[11]), 
            'Communication': int(skills_input[12]),
            'Teamwork': int(skills_input[13]),
            'Adaptability': int(skills_input[14])
        }

# Load and train model
file_path = "jft_dataset.xlsx"
df = pd.read_excel(file_path)
df = df.drop_duplicates()
if df.isnull().sum().any():
    df = df.dropna()

print(f"Dataset shape: {df.shape}")

X = df.iloc[:, 0:15]
y = df.iloc[:, 15]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("🤖 Training Random Forest Model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Model Accuracy: {accuracy*100:.1f}%")

# Create and save system
recommender_system = JobRecommenderSystem(model)
joblib.dump(recommender_system, "job_recommender_system.pkl")
print("💾 Complete system saved as job_recommender_system.pkl")

# Quick test
print("\n🎯 QUICK TEST:")
sample_skills = [4, 3, 2, 3, 4, 4, 3, 2, 3, 3, 4, 4, 3, 4, 4]
print("Top jobs:", recommender_system.get_top_5_jobs(sample_skills))
print("Soft traits:", recommender_system.get_5_soft_traits(sample_skills))