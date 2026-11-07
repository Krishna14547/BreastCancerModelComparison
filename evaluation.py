# evaluation.py
# One place for the metric calculations so training and the app never disagree.

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)


def score_model(model, X, y):
    """Return the 6 assignment metrics for a fitted model on data (X, y)."""

    # Hard class predictions (0 or 1). Used for the metrics that need a decision.
    pred = model.predict(X)

    # Probability of class 1 (malignant). AUC needs a score, not a hard 0/1,
    # because it measures how well the model ranks malignant above benign.
    # [:, 1] picks the "class 1" column of the probability table.
    prob = model.predict_proba(X)[:, 1]

    return {
        "Accuracy": accuracy_score(y, pred),     # share of all predictions that were right
        "AUC": roc_auc_score(y, prob),           # ranking quality, 0.5 = guessing, 1.0 = perfect
        "Precision": precision_score(y, pred),   # of those flagged malignant, how many truly were
        "Recall": recall_score(y, pred),         # of the truly malignant, how many we caught
        "F1": f1_score(y, pred),                 # balance between precision and recall
        "MCC": matthews_corrcoef(y, pred),       # one balanced score that uses all 4 confusion-matrix cells
    }
