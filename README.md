# Breast Cancer Classification: Five-Model Comparison

**Live Streamlit app:** https://breastcancermodelcomparison-bitspilani-mudhukrishna.streamlit.app/

**GitHub repository:** https://github.com/Krishna14547/BreastCancerModelComparison

## a. Problem statement

Predict whether a breast mass is **malignant or benign** from 30 numeric measurements of cell nuclei. This is a binary classification problem, and the positive class (1) is malignant. Five classifiers are trained on the same data and compared on the same held-out test set using Accuracy, AUC, Precision, Recall, F1 and MCC. The trained models are served through an interactive Streamlit app.

## b. Dataset description

- **Source:** Breast Cancer Wisconsin (Diagnostic) dataset from the UCI Machine Learning Repository, loaded through the copy bundled with scikit-learn (`load_breast_cancer`).
- **Instances:** 569 (357 benign, 212 malignant).
- **Features:** 30 numeric features. They are the mean, standard error and worst value of 10 measurements taken from digitised images of fine-needle aspirates of breast masses: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension.
- **Target:** `target`, where 1 = malignant and 0 = benign. scikit-learn encodes this the other way round, so the labels were flipped to make malignant the positive class.
- **Data quality:** no missing values.
- **Preprocessing:** features are standardised with `StandardScaler` inside each model pipeline, so scaling is learned from training data only.
- **Split:** 80% train (455 rows) and 20% test (114 rows), stratified on the target, random seed 2026.

## c. GitHub repository

https://github.com/Krishna14547/BreastCancerModelComparison

- `app.py`: Streamlit app
- `train_models.py`: trains the five models and saves them
- `evaluation.py`: metric calculations shared by training and the app
- `requirements.txt`: pinned dependencies
- `test_data.csv`: held-out test data (features plus `target`)
- `data/data.csv`: full dataset
- `model/`: saved pipelines (`.joblib`) for all five classifiers and `metrics.csv`

To run locally: install the packages in `requirements.txt`, then run `streamlit run app.py` from the project folder.

## d. Models used

Each model is a pipeline of `StandardScaler` followed by the classifier. Settings: Logistic Regression (max_iter=2000), Decision Tree (max_depth=5), KNN (k=7), Gaussian Naive Bayes (defaults), SVM (RBF kernel, C=1.0, probability enabled).

### Comparison table (test set, 114 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9737 | 0.9987 | 0.9756 | 0.9524 | 0.9639 | 0.9433 |
| Decision Tree | 0.9561 | 0.9590 | 0.9111 | 0.9762 | 0.9425 | 0.9085 |
| KNN | 0.9737 | 0.9974 | 0.9535 | 0.9762 | 0.9647 | 0.9439 |
| Naive Bayes | 0.9561 | 0.9964 | 0.9302 | 0.9524 | 0.9412 | 0.9064 |
| Support Vector Machine | 0.9825 | 1.0000 | 0.9545 | 1.0000 | 0.9767 | 0.9633 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | It reaches 0.9737 accuracy and an AUC of 0.9987. It has the highest precision (0.9756) but a lower recall (0.9524), so it misses a few malignant cases while rarely raising false alarms. |
| Decision Tree | It is the weakest on AUC (0.9590) and accuracy (0.9561). Its precision is the lowest (0.9111), which means more benign cases are flagged as malignant, even though recall is high (0.9762). |
| KNN | It matches Logistic Regression on accuracy (0.9737) and has an MCC of 0.9439, the second best. Its recall (0.9762) is higher than Logistic Regression's, with an AUC of 0.9974. |
| Naive Bayes | It shares the lowest accuracy (0.9561) and has the lowest MCC (0.9064), but its AUC is high (0.9964). Precision (0.9302) and recall (0.9524) are balanced, so it is a reasonable but not leading option. |
| Support Vector Machine | It scores best on every headline metric: accuracy 0.9825, AUC 1.0000, F1 0.9767 and MCC 0.9633. It also has perfect recall (1.0000), so it misses no malignant cases. |
| **Overall, winner for this dataset?** | Support Vector Machine (RBF). It has the highest accuracy, AUC, F1 and MCC. Its recall of 1.0000 matters most here, because a missed malignant case is the costliest error. Its precision (0.9545) is only slightly below Logistic Regression's. The test set has only 114 rows, so the gaps between the top models are small and may not hold on other splits. |
