import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from data_preparation import load_titanic, load_diabetes


# ============================================================
#  DECISION TREE EXPERIMENT
# ============================================================
def run_experiment(X_train, X_test, y_train, y_test, dataset_name, feature_names):
    """
    Train a Decision Tree using GridSearchCV to find best parameters.
    Evaluate with K-Fold cross validation and test accuracy.
    """
    print(f"\n{'='*60}")
    print(f"  DATASET: {dataset_name}")
    print(f"{'='*60}")

    # ── Step 1: Find best parameters with GridSearchCV ───────
    param_grid = {
        'criterion'        : ['gini', 'entropy'],
        'max_depth'        : [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf' : [1, 2, 4],
    }

    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv      = 5,
        scoring = 'accuracy',
        n_jobs  = -1
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print(f"\n  Best parameters (GridSearchCV):")
    for k, v in best_params.items():
        print(f"    {k:22s}: {v}")

    # ── Step 2: Train best model ──────────────────────────────
    best_model = DecisionTreeClassifier(**best_params, random_state=42)
    best_model.fit(X_train, y_train)

    # ── Step 3: Evaluate ──────────────────────────────────────
    train_acc = accuracy_score(y_train, best_model.predict(X_train))
    test_acc  = accuracy_score(y_test,  best_model.predict(X_test))

    print(f"\n  Training Accuracy : {train_acc*100:.2f}%")
    print(f"  Test Accuracy     : {test_acc*100:.2f}%")

    # ── Step 4: K-Fold Cross Validation (k=10) ───────────────
    X_all = np.concatenate([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    cv_scores = cross_val_score(best_model, X_all, y_all, cv=10, scoring='accuracy')
    print(f"\n  K-Fold (k=10) Cross Validation:")
    print(f"    Mean Accuracy : {cv_scores.mean()*100:.2f}%")
    print(f"    Std           : {cv_scores.std()*100:.2f}%")

    # ── Step 5: Classification report ────────────────────────
    print(f"\n  Classification Report:")
    print(classification_report(y_test, best_model.predict(X_test)))

    return best_model, train_acc, test_acc, cv_scores


# ============================================================
#  PLOT: Confusion matrix
# ============================================================
def plot_confusion_matrix(model, X_test, y_test, dataset_name):
    """Plot confusion matrix for the trained model."""
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.title(f'Confusion Matrix — {dataset_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'dt_confusion_{dataset_name.lower().replace(" ", "_")}.png', dpi=150)
    plt.show()
    print(f"  📊 Confusion matrix saved.")


# ============================================================
#  PLOT: Decision tree structure
# ============================================================
def plot_decision_tree(model, feature_names, dataset_name):
    """Plot the decision tree structure (limited to depth 3 for readability)."""
    plt.figure(figsize=(20, 8))
    plot_tree(
        model,
        feature_names = feature_names,
        class_names   = ['Class 0', 'Class 1'],
        filled        = True,
        max_depth     = 3,
        fontsize      = 9
    )
    plt.title(f'Decision Tree — {dataset_name} (max depth shown: 3)')
    plt.tight_layout()
    plt.savefig(f'dt_tree_{dataset_name.lower().replace(" ", "_")}.png', dpi=150)
    plt.show()
    print(f"  📊 Decision tree plot saved.")


# ============================================================
#  MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  FASE 3 — Decision Tree")
    print("=" * 60)

    # ── Load datasets ─────────────────────────────────────────
    X_train_t, X_test_t, y_train_t, y_test_t = load_titanic('datasets/Titanic-Dataset.csv')
    X_train_d, X_test_d, y_train_d, y_test_d = load_diabetes('datasets/diabetes.csv')

    titanic_features  = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    diabetes_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                         'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    # ── Run experiments ───────────────────────────────────────
    model_t, train_t, test_t, cv_t = run_experiment(
        X_train_t, X_test_t, y_train_t, y_test_t,
        "Titanic", titanic_features
    )

    model_d, train_d, test_d, cv_d = run_experiment(
        X_train_d, X_test_d, y_train_d, y_test_d,
        "Diabetes", diabetes_features
    )

    # ── Plots ─────────────────────────────────────────────────
    plot_confusion_matrix(model_t, X_test_t, y_test_t, "Titanic")
    plot_confusion_matrix(model_d, X_test_d, y_test_d, "Diabetes")

    plot_decision_tree(model_t, titanic_features,  "Titanic")
    plot_decision_tree(model_d, diabetes_features, "Diabetes")

    # ── Final summary ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY — Decision Tree")
    print(f"{'='*60}")
    print(f"  {'Dataset':12} | {'Train Acc':>10} | {'Test Acc':>10} | {'CV Mean':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Titanic':12} | {train_t*100:>9.2f}% | {test_t*100:>9.2f}% | {cv_t.mean()*100:>9.2f}%")
    print(f"  {'Diabetes':12} | {train_d*100:>9.2f}% | {test_d*100:>9.2f}% | {cv_d.mean()*100:>9.2f}%")