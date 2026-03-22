"""
Feature Selection for Gene Expression Cancer Classification

This script performs:
1. Loading preprocessed data
2. Feature selection using SelectKBest with various k values
3. Comparison of before vs after feature reduction
4. Saving selected features for model training

Why Feature Selection Matters:
- Gene expression data is high-dimensional (~22,000 genes)
- Small sample size (286 samples) → high risk of overfitting
- Many genes are likely irrelevant to cancer relapse prediction
- Reducing features improves model interpretability and performance
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

# Import preprocessing pipeline
from preprocess import preprocess_pipeline


def select_k_best_features(X_train: pd.DataFrame, 
                           y_train: pd.Series, 
                           k: int = 100,
                           score_func=f_classif) -> tuple:
    """
    Select top k features using SelectKBest.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training feature matrix
    y_train : pd.Series
        Training labels
    k : int
        Number of features to select
    score_func : callable
        Scoring function (f_classif for ANOVA F-test, mutual_info_classif for MI)
        
    Returns:
    --------
    selector : SelectKBest
        Fitted selector object
    selected_features : list
        Names of selected features
    feature_scores : pd.DataFrame
        All features with their scores
    """
    print(f"\nSelecting top {k} features using {score_func.__name__}...")
    
    selector = SelectKBest(score_func=score_func, k=k)
    selector.fit(X_train, y_train)
    
    # Get feature scores
    feature_scores = pd.DataFrame({
        'feature': X_train.columns,
        'score': selector.scores_,
        'pvalue': selector.pvalues_ if hasattr(selector, 'pvalues_') and selector.pvalues_ is not None else np.nan
    }).sort_values('score', ascending=False)
    
    # Get selected feature names
    selected_mask = selector.get_support()
    selected_features = X_train.columns[selected_mask].tolist()
    
    print(f"  Selected {len(selected_features)} features")
    print(f"  Top 10 features by score:")
    for i, row in feature_scores.head(10).iterrows():
        pval_str = f", p={row['pvalue']:.2e}" if not np.isnan(row['pvalue']) else ""
        print(f"    {row['feature']}: {row['score']:.2f}{pval_str}")
    
    return selector, selected_features, feature_scores


def transform_with_selector(selector: SelectKBest, 
                           X: pd.DataFrame, 
                           selected_features: list) -> pd.DataFrame:
    """Transform data using fitted selector, preserving feature names."""
    X_selected = pd.DataFrame(
        selector.transform(X),
        columns=selected_features,
        index=X.index
    )
    return X_selected


def evaluate_feature_selection(X_train: pd.DataFrame, 
                               y_train: pd.Series,
                               k_values: list = [50, 100, 200, 500]) -> pd.DataFrame:
    """
    Evaluate different k values using cross-validation.
    
    Compares model performance with different numbers of selected features.
    """
    print("\n" + "=" * 60)
    print("EVALUATING DIFFERENT K VALUES")
    print("=" * 60)
    
    results = []
    
    for k in k_values:
        print(f"\nTesting k={k}...")
        
        # Select features
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X_train, y_train)
        
        # Quick evaluation with Logistic Regression
        model = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(model, X_selected, y_train, cv=5, scoring='accuracy')
        
        results.append({
            'k': k,
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'features_shape': X_selected.shape
        })
        
        print(f"  Accuracy: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "-" * 40)
    print("SUMMARY")
    print("-" * 40)
    print(results_df.to_string(index=False))
    
    return results_df


def compare_before_after(X_train_full: pd.DataFrame,
                        X_train_reduced: pd.DataFrame,
                        y_train: pd.Series) -> dict:
    """Compare model performance before and after feature selection."""
    print("\n" + "=" * 60)
    print("COMPARING BEFORE VS AFTER FEATURE SELECTION")
    print("=" * 60)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    
    # Before (full features) - use subset to avoid memory issues
    print("\nEvaluating with ALL features (may take a moment)...")
    scores_before = cross_val_score(model, X_train_full, y_train, cv=5, scoring='accuracy')
    
    # After (selected features)
    print("Evaluating with SELECTED features...")
    scores_after = cross_val_score(model, X_train_reduced, y_train, cv=5, scoring='accuracy')
    
    comparison = {
        'before': {
            'n_features': X_train_full.shape[1],
            'accuracy_mean': scores_before.mean(),
            'accuracy_std': scores_before.std()
        },
        'after': {
            'n_features': X_train_reduced.shape[1],
            'accuracy_mean': scores_after.mean(),
            'accuracy_std': scores_after.std()
        }
    }
    
    print("\n" + "-" * 40)
    print(f"BEFORE (Full features: {comparison['before']['n_features']})")
    print(f"  Accuracy: {comparison['before']['accuracy_mean']:.4f} (+/- {comparison['before']['accuracy_std']*2:.4f})")
    print(f"\nAFTER (Selected features: {comparison['after']['n_features']})")
    print(f"  Accuracy: {comparison['after']['accuracy_mean']:.4f} (+/- {comparison['after']['accuracy_std']*2:.4f})")
    
    improvement = comparison['after']['accuracy_mean'] - comparison['before']['accuracy_mean']
    print(f"\nImprovement: {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    if improvement > 0:
        print("✓ Feature selection IMPROVED performance!")
    elif improvement < -0.01:
        print("⚠ Feature selection decreased performance. Consider adjusting k.")
    else:
        print("→ Similar performance with fewer features (good for interpretability)")
    
    return comparison


def plot_feature_scores(feature_scores: pd.DataFrame, 
                       top_n: int = 30,
                       save_path: str = None):
    """Plot top feature scores."""
    plt.figure(figsize=(12, 8))
    
    top_features = feature_scores.head(top_n)
    
    plt.barh(range(len(top_features)), top_features['score'].values, color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('F-Score')
    plt.ylabel('Gene')
    plt.title(f'Top {top_n} Genes by F-Score (ANOVA)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def feature_selection_pipeline(k: int = 100, 
                               evaluate_k_values: bool = True,
                               save_results: bool = True) -> dict:
    """
    Complete feature selection pipeline.
    
    Parameters:
    -----------
    k : int
        Number of features to select (default: 100)
    evaluate_k_values : bool
        Whether to evaluate multiple k values
    save_results : bool
        Whether to save selector and selected features
        
    Returns:
    --------
    dict with X_train_selected, X_test_selected, selector, feature_names, etc.
    """
    print("=" * 60)
    print("FEATURE SELECTION PIPELINE")
    print("=" * 60)
    
    # Step 1: Load preprocessed data
    print("\nStep 1: Loading preprocessed data...")
    data = preprocess_pipeline(save_scaler=False)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print(f"\nOriginal feature count: {X_train.shape[1]}")
    
    # Step 2: Evaluate different k values (optional)
    if evaluate_k_values:
        print("\nStep 2: Evaluating different k values...")
        k_results = evaluate_feature_selection(X_train, y_train, k_values=[50, 100, 200, 500])
    else:
        k_results = None
    
    # Step 3: Select features with chosen k
    print(f"\nStep 3: Selecting top {k} features...")
    selector, selected_features, feature_scores = select_k_best_features(
        X_train, y_train, k=k, score_func=f_classif
    )
    
    # Step 4: Transform datasets
    print("\nStep 4: Transforming datasets...")
    X_train_selected = transform_with_selector(selector, X_train, selected_features)
    X_test_selected = transform_with_selector(selector, X_test, selected_features)
    
    print(f"  X_train: {X_train.shape} → {X_train_selected.shape}")
    print(f"  X_test: {X_test.shape} → {X_test_selected.shape}")
    
    # Step 5: Compare before vs after
    print("\nStep 5: Comparing performance before vs after selection...")
    comparison = compare_before_after(X_train, X_train_selected, y_train)
    
    # Step 6: Save results
    if save_results:
        print("\nStep 6: Saving results...")
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Save selector
        joblib.dump(selector, models_dir / "feature_selector.joblib")
        print(f"  Selector saved to: models/feature_selector.joblib")
        
        # Save selected feature names
        pd.Series(selected_features).to_csv(models_dir / "selected_features.csv", index=False)
        print(f"  Selected features saved to: models/selected_features.csv")
        
        # Save feature scores
        feature_scores.to_csv(models_dir / "feature_scores.csv", index=False)
        print(f"  Feature scores saved to: models/feature_scores.csv")
    
    print("\n" + "=" * 60)
    print("FEATURE SELECTION COMPLETE")
    print("=" * 60)
    print(f"\nReduced from {X_train.shape[1]} to {X_train_selected.shape[1]} features")
    print(f"Dimensionality reduction: {(1 - X_train_selected.shape[1]/X_train.shape[1])*100:.1f}%")
    
    return {
        'X_train': X_train_selected,
        'X_test': X_test_selected,
        'y_train': y_train,
        'y_test': y_test,
        'selector': selector,
        'selected_features': selected_features,
        'feature_scores': feature_scores,
        'k_results': k_results,
        'comparison': comparison
    }


if __name__ == "__main__":
    # Run feature selection pipeline
    results = feature_selection_pipeline(k=100, evaluate_k_values=True)
    
    # Display top selected genes
    print("\n" + "=" * 60)
    print("TOP 20 SELECTED GENES")
    print("=" * 60)
    top_genes = results['feature_scores'].head(20)
    for i, (_, row) in enumerate(top_genes.iterrows(), 1):
        print(f"{i:2d}. {row['feature']}: score={row['score']:.2f}")
    
    print("\nData is ready for model training!")
