import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.inspection import permutation_importance

# Reuse pipeline building utilities
from patient_priority_ml import load_data, preprocess_data, feature_engineering


def aggregate_importances(feature_names, importances, categorical_cols=None):
    """
    Aggregate transformed feature importances back to original columns.

    - Numeric features come through as 'num__<col>'
    - Categorical one-hots come through as 'cat__<col>_<category>'
    """
    agg = {}
    for name, imp in zip(feature_names, importances):
        if name.startswith('num__'):
            orig = name.split('num__', 1)[1]
        elif name.startswith('cat__'):
            # Map one-hot back to its original categorical column robustly
            tail = name.split('cat__', 1)[1]
            orig = None
            if categorical_cols:
                # Prefer longest match to avoid truncating names with underscores
                for col in sorted(categorical_cols, key=len, reverse=True):
                    if tail.startswith(f"{col}_"):
                        orig = col
                        break
            if orig is None:
                # Fallback: best-effort split
                orig = tail.split('_', 1)[0]
        else:
            # Fallback: use the raw name
            orig = name

        agg[orig] = agg.get(orig, 0.0) + float(imp)

    return agg


def main():
    # Load data and prepare splits consistently with the training pipeline
    df = load_data('patient_priority_extended.csv')
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test, preprocessor, label_encoder = feature_engineering(df)

    # Load best saved model (Pipeline with preprocessor + estimator)
    model_path = Path('best_model.pkl')
    if not model_path.exists():
        raise FileNotFoundError("best_model.pkl not found. Please run training to save the model.")

    pipeline = joblib.load(model_path)

    # Compute permutation importance on the held-out test set
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=1,
    )

    # Derive transformed feature names from the fitted preprocessor
    try:
        preproc = pipeline.named_steps['preprocessor']
        feature_names = preproc.get_feature_names_out()
        # Extract original categorical column list from fitted ColumnTransformer
        categorical_cols = None
        for name, transformer, cols in preproc.transformers_:
            if name == 'cat':
                categorical_cols = list(cols)
                break
    except Exception:
        # Fallbacks
        preproc = None
        categorical_cols = None
        feature_names = [f"f_{i}" for i in range(len(result.importances_mean))]

    # Aggregate one-hot importances to original feature level
    agg = aggregate_importances(feature_names, result.importances_mean, categorical_cols)

    # Create sorted ranking
    ranking = sorted(agg.items(), key=lambda x: x[1], reverse=True)

    # Save to CSV and print top items
    out_df = pd.DataFrame(ranking, columns=['feature', 'importance'])
    out_df.to_csv('feature_importance.csv', index=False)

    # Pretty print
    print("\nImportant features affecting triage (permutation importance, aggregated):")
    for i, (feat, score) in enumerate(ranking, start=1):
        print(f"{i:2d}. {feat}: {score:.6f}")

    # Also save a plot for quick reference
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(10, 6))
        top_k = min(15, len(ranking))
        sns.barplot(x=[v for _, v in ranking[:top_k]], y=[k for k, _ in ranking[:top_k]])
        plt.xlabel('Aggregated Importance')
        plt.ylabel('Feature')
        plt.title('Top Features Affecting Triage')
        plt.tight_layout()
        # Ensure plots directory exists
        Path('plots').mkdir(exist_ok=True)
        plt.savefig('plots/feature_importance.png')
        plt.close()
    except Exception as e:
        print(f"Could not save plot: {e}")


if __name__ == '__main__':
    main()
