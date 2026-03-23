#!/usr/bin/env python
"""
Quick test script to verify Flask app and all components are working
"""
import os
import sys
import json

print("\n" + "="*70)
print("WATER POTABILITY DASHBOARD - VERIFICATION SCRIPT")
print("="*70)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Test 1: Check required files
print("\n[1/5] Checking required files...")
required_files = [
    ("data/kaggle_water_quality.csv", "Dataset"),
    ("models/best_model.pkl", "Trained Model"),
    ("models/scaler.pkl", "Scaler"),
    ("models/imputer.pkl", "Imputer"),
    ("models/comparison.json", "Model Comparison"),
    ("models/confusion_matrix.json", "Confusion Matrix"),
    ("models/feature_importance.json", "Feature Importance"),
    ("templates/index.html", "HTML Template"),
]

all_exist = True
for filepath, name in required_files:
    full_path = os.path.join(BASE_DIR, filepath)
    exists = os.path.exists(full_path)
    status = "✓" if exists else "✗"
    print(f"   {status} {name:.<40} {filepath}")
    if not exists:
        all_exist = False

if all_exist:
    print("\n   ✓ All required files exist!")
else:
    print("\n   ⚠ Some files are missing. Please train the model first:")
    print("   $ python train.py")
    sys.exit(1)

# Test 2: Check Python dependencies
print("\n[2/5] Checking Python dependencies...")
required_packages = [
    ("flask", "Flask"),
    ("pandas", "Pandas"),
    ("numpy", "NumPy"),
    ("sklearn", "Scikit-Learn"),
    ("joblib", "Joblib"),
]

for package, name in required_packages:
    try:
        __import__(package)
        print(f"   ✓ {name:.<40} installed")
    except ImportError:
        print(f"   ✗ {name:.<40} NOT installed")
        print(f"\n   Install with: pip install {package}")
        sys.exit(1)

print("\n   ✓ All dependencies are installed!")

# Test 3: Check JSON file formats
print("\n[3/5] Validating JSON files...")
json_files = [
    "models/comparison.json",
    "models/confusion_matrix.json",
    "models/feature_importance.json",
]

for json_file in json_files:
    full_path = os.path.join(BASE_DIR, json_file)
    try:
        with open(full_path, 'r') as f:
            data = json.load(f)
        print(f"   ✓ {json_file:.<45} valid")
    except json.JSONDecodeError as e:
        print(f"   ✗ {json_file:.<45} INVALID: {e}")
        sys.exit(1)

print("\n   ✓ All JSON files are valid!")

# Test 4: Check data integrity
print("\n[4/5] Checking data integrity...")
try:
    import pandas as pd
    
    df = pd.read_csv(os.path.join(BASE_DIR, "data/kaggle_water_quality.csv"))
    n_rows, n_cols = df.shape
    print(f"   ✓ Dataset loaded: {n_rows} rows × {n_cols} columns")
    
    required_cols = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
                     "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity", "Potability"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"   ✗ Missing columns: {missing_cols}")
        sys.exit(1)
    
    print(f"   ✓ All required columns present")
    
    pot_count = (df["Potability"] == 1).sum()
    npot_count = (df["Potability"] == 0).sum()
    print(f"   ✓ Class distribution: {pot_count} potable, {npot_count} not potable")
    
except Exception as e:
    print(f"   ✗ Data check failed: {e}")
    sys.exit(1)

# Test 5: Test model loading
print("\n[5/5] Testing model loading...")
try:
    import joblib
    
    model = joblib.load(os.path.join(BASE_DIR, "models/best_model.pkl"))
    print(f"   ✓ Model loaded: {type(model).__name__}")
    
    scaler = joblib.load(os.path.join(BASE_DIR, "models/scaler.pkl"))
    print(f"   ✓ Scaler loaded: {type(scaler).__name__}")
    
    imputer = joblib.load(os.path.join(BASE_DIR, "models/imputer.pkl"))
    print(f"   ✓ Imputer loaded: {type(imputer).__name__}")
    
except Exception as e:
    print(f"   ✗ Model loading failed: {e}")
    sys.exit(1)

# Success!
print("\n" + "="*70)
print("✓ ALL CHECKS PASSED!")
print("="*70)
print("\nYour Flask dashboard is ready to run:")
print("   $ python app.py")
print("\nThen open your browser to: http://127.0.0.1:5000")
print("\nThe dashboard will show:")
print("   • Dataset statistics")
print("   • Model performance metrics")
print("   • Confusion matrix & charts")
print("   • Feature importance analysis")
print("   • IoT live predictor")
print("   • Water quality standards reference")
print("\n" + "="*70 + "\n")

