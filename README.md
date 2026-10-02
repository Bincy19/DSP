# Loan Status Prediction – Data Science Programming Exam

Jupyter notebook analysing `exam_2024.csv` (loan data): exploration, PCA + KMeans clustering, Random Forest tuning, and a learning curve built from an incremental teaching schedule.

## Contents
1. **Data loading** – read CSV, one-hot encode, shuffle.
2. **Feature ranges** – histograms of the 5 numeric features with the largest range.
3. **Clustering** – standardise, reduce to 2D with PCA, KMeans (k=2), coloured by `loan_status`.
4. **Hyperparameter tuning** – `GridSearchCV` (5-fold) on a `RandomForestClassifier` (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`).
5. **Teaching schedule** – `create_teaching_schedule()` grows the training set (500 initial rows, +100 per batch).
6. **Learning curve** – trains the best model on each schedule step and plots accuracy vs. training size.

## Setup
```bash
pip install -r requirements.txt
```
Place `exam_2024.csv` in the same folder as the notebook.

## Run
```bash
jupyter notebook DATA_SCIENCE_PROGRAMMING_EXAM__2_.ipynb
```

## Notes
- The target column is named `" loan_status"` (leading space) in the CSV; the code relies on this.
- Dataset is not included.
