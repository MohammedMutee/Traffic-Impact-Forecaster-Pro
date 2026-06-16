
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import joblib
from prepare_data import prepare_data
import pickle

class TrafficImpactModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=8,
            objective='multi:softprob',
            num_class=4,
            random_state=42,
            early_stopping_rounds=10
        )
        self.feature_names = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.feature_names = X_train.columns.tolist()
        if X_val is not None:
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.model.fit(X_train, y_train)
        print("Training complete.")

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        print("\n=== Model Performance ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High', 'Extreme']))
        
        # Feature Importance
        print("\n=== Top Contributing Factors (Feature Importance) ===")
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for i in range(min(5, len(self.feature_names))):
            print(f"{i+1}. {self.feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

    def get_action_plan(self, impact_score, features_row):
        impact_levels = {0: "Low", 1: "Medium", 2: "High", 3: "Extreme"}
        level = impact_levels.get(impact_score, "Unknown")
        
        recommendations = {
            0: {
                "Manpower": "1 Traffic Officer (Monitoring)",
                "Barricades": "0 required",
                "Diversion": "None",
                "Priority": "P4 (Routine)"
            },
            1: {
                "Manpower": "2 Officers (On-site)",
                "Barricades": "2-4 (Local containment)",
                "Diversion": "Local advisory (Twitter/Astram)",
                "Priority": "P3 (Standard Response)"
            },
            2: {
                "Manpower": "4-6 Officers + 1 Inspector",
                "Barricades": "10+ (Lane management)",
                "Diversion": "Active Local Diversion; Update Google Maps/Waze",
                "Priority": "P2 (High Priority)"
            },
            3: {
                "Manpower": "12+ Officers + ACP Oversight",
                "Barricades": "30+ (Complete perimeter)",
                "Diversion": "Systemic Grid Diversion Plan; Emergency Alerts",
                "Priority": "P1 (Emergency/Extreme)"
            }
        }
        
        plan = recommendations.get(impact_score, recommendations[0])
        print(f"\n[FORECAST] Impact Level: {level}")
        print(f"--- RESOURCE DEPLOYMENT PLAN ---")
        for k, v in plan.items():
            print(f"- {k}: {v}")
        print(f"--------------------------------")

if __name__ == "__main__":
    file_path = 'Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv'
    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data(file_path)
    
    # Split train for validation
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    print("Initializing Model...")
    impact_model = TrafficImpactModel()
    impact_model.train(X_tr, y_tr, X_val, y_val)
    
    impact_model.evaluate(X_test, y_test)
    
    # Showcase logic
    print("\n=== Real-World Action Plan Sample ===")
    sample_preds = impact_model.model.predict(X_test.iloc[:3])
    for i in range(3):
        impact_model.get_action_plan(sample_preds[i], X_test.iloc[i])
