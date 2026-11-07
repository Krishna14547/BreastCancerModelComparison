# train_models.py
# Trains the 5 classifiers on the breast cancer data and saves everything the app needs.

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from evaluation import score_model

# Random seed: controls the train/test split and the models that use randomness.
# CHANGE THIS to your own number so your results are unique to you.
SEED = 2026

# ---- 1. Load the data we saved in Step 2 ----
df = pd.read_csv("data/data.csv")
X = df.drop(columns="target")   # the 30 measurements
y = df["target"]                # 1 = malignant, 0 = benign
print("Data:", df.shape, "| classes:", y.value_counts().to_dict())

# ---- 2. Split into train (80%) and test (20%) ----
# stratify=y keeps the benign/malignant ratio the same in both parts,
# otherwise a small test set could end up with too few malignant cases.
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
print("Train rows:", len(X_tr), "| Test rows:", len(X_te))

# ---- 3. Save the test set as test_data.csv ----
# This is the file we upload into the Streamlit app later.
# It has the features AND the target column, as the assignment requires.
test_out = X_te.copy()
test_out["target"] = y_te
test_out.to_csv("test_data.csv", index=False)

# ---- 4. Define the 5 models ----
# (display name, file name, the classifier itself)
# Feel free to tweak these settings. Different choices give different results.
model_list = [
    ("Logistic Regression", "logistic_regression", LogisticRegression(max_iter=2000)),
    ("Decision Tree",       "decision_tree",       DecisionTreeClassifier(max_depth=5, random_state=SEED)),
    ("KNN",                 "knn",                 KNeighborsClassifier(n_neighbors=7)),
    ("Naive Bayes",         "naive_bayes",         GaussianNB()),
    # probability=True is needed so SVC can output probabilities for the AUC score
    ("Support Vector Machine", "svm",              SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED)),
]

# ---- 5. Train, evaluate and save each model ----
results = []
for display_name, file_name, clf in model_list:

    # A Pipeline = scaler + classifier packed together.
    # Scaling matters for KNN, SVM and Logistic Regression because the features
    # are on very different ranges (area is in the hundreds, smoothness is below 1).
    # Saving the whole pipeline means the app doesn't have to scale anything itself.
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    # Learn from the TRAINING data only
    pipe.fit(X_tr, y_tr)

    # Save the trained pipeline so the app can load it without retraining
    joblib.dump(pipe, f"model/{file_name}.joblib")

    # Score on the TEST data, which the model has never seen
    scores = score_model(pipe, X_te, y_te)
    results.append({"ML Model Name": display_name, **scores})
    print("Trained and saved:", display_name)

# ---- 6. Build the comparison table and save it ----
results_df = pd.DataFrame(results).round(4)
results_df.to_csv("model/metrics.csv", index=False)

print("\n=== Comparison table (test set) ===")
print(results_df.to_string(index=False))
