"""
Test script for the Earthquake Prediction API
Run this to verify the application is working correctly
"""

import requests
import json
import time

def test_api():
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Earthquake Prediction API")
    print("=" * 50)
    
    # Test 1: Check if server is running
    print("1. Testing server connection...")
    try:
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print("✅ Server is running!")
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        print("💡 Make sure to run 'python app_simple.py' first")
        return False
    
    # Test 2: Test countries API
    print("\n2. Testing countries API...")
    try:
        response = requests.get(f"{base_url}/api/countries")
        countries = response.json()
        print(f"✅ Got {len(countries)} countries")
        print(f"   Sample: {countries[:3]}")
    except Exception as e:
        print(f"❌ Countries API failed: {e}")
    
    # Test 3: Test continents API
    print("\n3. Testing continents API...")
    try:
        response = requests.get(f"{base_url}/api/continents")
        continents = response.json()
        print(f"✅ Got {len(continents)} continents")
        print(f"   All: {continents}")
    except Exception as e:
        print(f"❌ Continents API failed: {e}")
    
    # Test 4: Test prediction API with sample data
    print("\n4. Testing prediction API...")
    
    test_cases = [
        {
            "name": "Small earthquake",
            "data": {
                "magnitude": 4.5,
                "depth": 15,
                "latitude": 35.0,
                "longitude": 139.0,
                "alert": "none",
                "country": "Japan",
                "continent": "Asia",
                "magType": "ml"
            }
        },
        {
            "name": "Large shallow earthquake",
            "data": {
                "magnitude": 7.8,
                "depth": 8,
                "latitude": -33.0,
                "longitude": -72.0,
                "alert": "red",
                "country": "Chile",
                "continent": "South America",
                "magType": "mw"
            }
        },
        {
            "name": "Deep earthquake",
            "data": {
                "magnitude": 6.2,
                "depth": 150,
                "latitude": 40.0,
                "longitude": 25.0,
                "alert": "yellow",
                "country": "Greece",
                "continent": "Europe",
                "magType": "mb"
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n   Test 4.{i}: {test_case['name']}")
        try:
            response = requests.post(
                f"{base_url}/predict",
                json=test_case['data'],
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result['success']:
                    predictions = result['predictions']
                    
                    # High impact prediction
                    high_impact = predictions['high_impact']
                    print(f"      High Impact: {high_impact['probability']:.2%} ({high_impact['risk_level']})")
                    
                    # Tsunami prediction
                    tsunami = predictions['tsunami']
                    print(f"      Tsunami Risk: {tsunami['probability']:.2%} ({tsunami['risk_level']})")
                    
                    print("      ✅ Prediction successful!")
                else:
                    print(f"      ❌ Prediction failed: {result.get('error', 'Unknown error')}")
            else:
                print(f"      ❌ HTTP error: {response.status_code}")
                
        except Exception as e:
            print(f"      ❌ Request failed: {e}")
    
    # Test 5: Test invalid input handling
    print("\n5. Testing error handling...")
    try:
        response = requests.post(
            f"{base_url}/predict",
            json={"invalid": "data"},
            headers={'Content-Type': 'application/json'}
        )
        
        result = response.json()
        if not result.get('success', True):
            print("✅ Error handling works correctly")
        else:
            print("⚠️ Error handling might need improvement")
            
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 API testing complete!")
    print("💡 Open http://localhost:5000 in your browser to see the UI")
    
    return True

def test_ui_elements():
    """Test if the UI loads correctly"""
    print("\n🎨 Testing UI Elements")
    print("-" * 30)
    
    try:
        response = requests.get("http://localhost:5000")
        html_content = response.text
        
        # Check for key UI elements
        ui_elements = [
            "Earthquake Impact Predictor",
            "magnitude",
            "depth",
            "latitude",
            "longitude",
            "Predict Impact",
            "style.css",
            "script.js"
        ]
        
        missing_elements = []
        for element in ui_elements:
            if element not in html_content:
                missing_elements.append(element)
        
        if not missing_elements:
            print("✅ All UI elements found!")
        else:
            print(f"⚠️ Missing UI elements: {missing_elements}")
            
    except Exception as e:
        print(f"❌ UI test failed: {e}")

if __name__ == "__main__":
    print("🌍 Earthquake Prediction System Test Suite")
    print("Make sure the server is running first!")
    print()
    
    # Wait a moment for user to start server
    print("Starting tests in 3 seconds...")
    time.sleep(3)
    
    # Run tests
    if test_api():
        test_ui_elements()
    
    print("\n🏁 Testing finished!")
    
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
import joblib


DATA_PATH = "/content/earthquake_1995-2023.csv"
df = pd.read_csv(DATA_PATH)


for col in ["alert", "continent", "country", "location", "magType", "net"]:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")


df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce", dayfirst=True)


df["year"] = df["date_time"].dt.year
df["month"] = df["date_time"].dt.month
df["day"] = df["date_time"].dt.day
df["hour"] = df["date_time"].dt.hour
df["day_of_week"] = df["date_time"].dt.dayofweek


for col in ["magnitude", "depth", "latitude", "longitude", "gap", "dmin", "cdi", "mmi", "sig", "tsunami"]:
    if col not in df.columns:
        df[col] = np.nan


df["log_depth"] = np.log1p(df["depth"].fillna(0))
df["energy_release"] = 10 ** (1.5 * df["magnitude"].fillna(df["magnitude"].median()) + 4.8)
df["is_shallow"] = (df["depth"] < 70).astype(int).fillna(0)
df["abs_latitude"] = df["latitude"].abs()
df["abs_longitude"] = df["longitude"].abs()

df["cdi"] = df["cdi"].fillna(df["cdi"].median() if df["cdi"].notna().any() else 0)
df["mmi"] = df["mmi"].fillna(df["mmi"].median() if df["mmi"].notna().any() else 0)
df["sig"] = df["sig"].fillna(df["sig"].median() if df["sig"].notna().any() else 0)
df["tsunami"] = df["tsunami"].fillna(0)

df["impact_score"] = df["cdi"] * 0.3 + df["mmi"] * 0.3 + df["sig"] * 0.4 + df["tsunami"] * 2


impact_thresh = df["impact_score"].quantile(0.75)
df["high_impact"] = (df["impact_score"] >= impact_thresh).astype(int)

df["tsunami"] = df["tsunami"].astype(int)


numeric_features = [
    "magnitude",
    "depth",
    "log_depth",
    "energy_release",
    "gap",
    "dmin",
    "abs_latitude",
    "abs_longitude",
    "year",
    "month",
    "hour",
    "day_of_week",
    "cdi",
    "mmi",
    "sig",
]

numeric_features = [c for c in numeric_features if c in df.columns]

categorical_features = [c for c in ["alert", "magType", "net", "continent", "country"] if c in df.columns]

df_model = df.copy().reset_index(drop=True)
df_model = df_model.dropna(subset=numeric_features, how="all")

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",
)


X_processed = preprocessor.fit_transform(df_model[numeric_features + categorical_features])


def train_and_evaluate(X_proc, y, model_name="model", do_gridsearch=False):
    """
    Trains a RandomForest pipeline and reports performance.
    Saves best model to disk as joblib (model_name + '.joblib').
    Expects preprocessed X_proc.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.25, stratify=y, random_state=42
    )


    clf = Pipeline(
        steps=[
            ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1)),
        ]
    )

    if do_gridsearch:
        param_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_split": [2, 5],
        }
        gs = GridSearchCV(
            clf, param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=1
        )
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
    else:
        clf.fit(X_train, y_train)
        best = clf


    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1] if hasattr(best, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    print(f"--- {model_name} results ---")
    print(f"Accuracy: {acc:.4f}")
    if roc is not None:
        print(f"ROC-AUC: {roc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))


    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{model_name}.joblib"
    joblib.dump(best, model_path)
    print(f"Saved model to: {model_path}")


    return best, X_test, y_test


y_high_impact = df_model["high_impact"]
y_tsunami = df_model["tsunami"]

print("Training HIGH-IMPACT classifier (top 25% impact)...")
best_high, X_test_high_impact, y_test_high_impact = train_and_evaluate(
    X_processed, y_high_impact, model_name="rf_high_impact", do_gridsearch=False
)

print("\nTraining TSUNAMI classifier...")
best_tsun, X_test_tsunami, y_test_tsunami = train_and_evaluate(
    X_processed, y_tsunami, model_name="rf_tsunami", do_gridsearch=False
)


def print_feature_importances(pipeline, preprocessor, feature_names, top_n=20):
    """
    Attempt to extract importances from the RandomForest inside the pipeline.
    Uses the preprocessor to get expanded feature names for OneHotEncoder.
    """
    clf = pipeline.named_steps["classifier"]

    try:
        feature_names_out = preprocessor.get_feature_names_out()
    except:
        print("Could not get feature names from preprocessor. Is it fitted?")
        return

    importances = clf.feature_importances_
    feat_imp = sorted(zip(feature_names_out, importances), key=lambda x: x[1], reverse=True)[:top_n]
    print("\nTop feature importances:")
    for name, imp in feat_imp:
        print(f"{name}: {imp:.4f}")


try:
    print("Feature Importances for High Impact Model:")
    print_feature_importances(best_high, preprocessor, numeric_features + categorical_features)

    print("\nFeature Importances for Tsunami Model:")
    print_feature_importances(best_tsun, preprocessor, numeric_features + categorical_features)

except Exception as e:
    print("Could not extract feature importances:", e)


print("Done.")
