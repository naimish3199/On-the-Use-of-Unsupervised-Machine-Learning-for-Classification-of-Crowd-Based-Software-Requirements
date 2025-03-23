import itertools
import numpy as np
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from utils import import_data

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_class_combinations(labels):
    """
    Generate all possible class combinations for classification.

    Args:
        labels (list): List of unique class labels.

    Returns:
        list: List of class combinations.
    """
    unique_classes = sorted(set(labels))
    class_combinations = []
    for i in range(2, len(unique_classes) + 1):  # Start from binary classification
        class_combinations.extend(itertools.combinations(unique_classes, i))
    return class_combinations

def run_logistic_regression(X_filtered, y_filtered, param_grid):
    """
    Perform Logistic Regression with Stratified K-Fold Cross-Validation and hyperparameter tuning.

    Args:
        X_filtered (list): TF-IDF transformed feature matrix.
        y_filtered (list): Filtered labels.
        param_grid (dict): Hyperparameter grid for Logistic Regression.

    Returns:
        dict: Results containing precision, recall, F1-score, and best parameters.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    precision_list, recall_list, f1_list = [], [], []
    best_f1 = 0
    best_model = None
    best_params = None

    for train_idx, test_idx in skf.split(X_filtered, y_filtered):
        X_train, X_test = X_filtered[train_idx], X_filtered[test_idx]
        y_train, y_test = y_filtered[train_idx], y_filtered[test_idx]

        # Hyperparameter tuning using GridSearchCV
        clf = GridSearchCV(
            LogisticRegression(max_iter=1000, random_state=42),
            param_grid,
            cv=3,
            scoring="f1_macro",
            n_jobs=-1
        )
        clf.fit(X_train, y_train)

        # Get the best model and parameters
        best_clf = clf.best_estimator_
        best_params = clf.best_params_

        # Predictions
        y_pred = best_clf.predict(X_test)

        # Compute metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        # Track the best performing model
        if f1 > best_f1:
            best_f1 = f1
            best_model = best_clf

    # Average scores across folds
    avg_precision = round(np.mean(precision_list), 3)
    avg_recall = round(np.mean(recall_list), 3)
    avg_f1 = round(np.mean(f1_list), 3)

    return {
        "Precision": avg_precision,
        "Recall": avg_recall,
        "F1 Score": avg_f1,
        "Best Model": best_model,
        "Best Params": best_params
    }

def main():
    """
    Main function to run Logistic Regression classification for all class combinations.
    """
    # Load and preprocess the requirements data
    _, _, texts, _, _, _, labels, _ = import_data()

    # Define all possible class combinations
    class_combinations = get_class_combinations(labels)

    # Define hyperparameter grid for Logistic Regression
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],  # Regularization strength
        "solver": ["liblinear", "lbfgs"]
    }

    # Define TF-IDF vectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))

    # Initialize results
    results = []
    best_results = []

    # Class mapping for readability
    mapping = {0: "Health", 1: "Energy", 2: "Entertainment", 3: "Safety", 4: "Other"}

    # Process each class combination
    for classes in class_combinations:
        logging.info(f"Processing classification for classes: {[mapping[i] for i in classes]}")

        # Filter dataset for selected classes
        mask = [label in classes for label in labels]
        X_filtered = [texts[i] for i in range(len(texts)) if mask[i]]
        y_filtered = [labels[i] for i in range(len(labels)) if mask[i]]

        # Transform labels to indices (0, 1, 2, ...)
        label_mapping = {cls: idx for idx, cls in enumerate(classes)}
        y_filtered = [label_mapping[label] for label in y_filtered]

        # TF-IDF transformation
        X_tfidf = vectorizer.fit_transform(X_filtered)
        y_filtered = np.array(y_filtered)

        # Run Logistic Regression
        result = run_logistic_regression(X_tfidf, y_filtered, param_grid)

        # Log results
        logging.info(f"Best Parameters: {result['Best Params']}")
        logging.info(f"Macro Precision: {result['Precision']:.3f}, Recall: {result['Recall']:.3f}, F1 Score: {result['F1 Score']:.3f}")
        print('\n')

if __name__ == "__main__":
    main()

