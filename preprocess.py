"""
Data Preprocessing for Gene Expression Cancer Classification

This script performs:
1. Loading the raw dataset
2. Handling missing values
3. Removing duplicate samples
4. Separating features and labels
5. Normalizing gene expression values (StandardScaler)
6. Train-test split (stratified)

Output: Preprocessed data ready for feature selection and model training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib


def load_data(data_path: str = "data/dataset.csv") -> pd.DataFrame:
    """Load the gene expression dataset."""
    print("Loading dataset...")
    df = pd.read_csv(data_path, index_col='sample_id')
    print(f"  Loaded {df.shape[0]} samples with {df.shape[1]} columns")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    print("\nHandling missing values...")
    
    # Check for missing values
    missing_total = df.isnull().sum().sum()
    print(f"  Total missing values: {missing_total}")
    
    if missing_total > 0:
        # For gene expression data, fill with median of each gene
        gene_cols = [col for col in df.columns if col != 'label']
        for col in gene_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        print(f"  Filled missing values with column medians")
    else:
        print("  No missing values found")
    
    return df


def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate samples from the dataset."""
    print("\nHandling duplicates...")
    
    initial_count = len(df)
    df = df.drop_duplicates()
    removed = initial_count - len(df)
    
    if removed > 0:
        print(f"  Removed {removed} duplicate samples")
    else:
        print("  No duplicates found")
    
    print(f"  Final sample count: {len(df)}")
    return df


def separate_features_labels(df: pd.DataFrame) -> tuple:
    """Separate features (X) and labels (y) from the dataset."""
    print("\nSeparating features and labels...")
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    print(f"  Features shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")
    print(f"  Class distribution:")
    print(f"    - Class 0 (No Relapse): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    print(f"    - Class 1 (Relapse): {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
    
    return X, y


def normalize_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """Normalize gene expression values using StandardScaler.
    
    Fits on training data only to prevent data leakage.
    """
    print("\nNormalizing features with StandardScaler...")
    
    scaler = StandardScaler()
    
    # Fit on training data, transform both
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    print(f"  Training set - Mean: {X_train_scaled.mean().mean():.6f}, Std: {X_train_scaled.std().mean():.6f}")
    print(f"  Test set - Mean: {X_test_scaled.mean().mean():.6f}, Std: {X_test_scaled.std().mean():.6f}")
    
    return X_train_scaled, X_test_scaled, scaler


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """Split data into training and test sets with stratification."""
    print(f"\nSplitting data (test_size={test_size})...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Maintain class balance
    )
    
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"    - Class 0: {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
    print(f"    - Class 1: {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")
    print(f"  Test set: {X_test.shape[0]} samples")
    print(f"    - Class 0: {(y_test == 0).sum()} ({(y_test == 0).mean()*100:.1f}%)")
    print(f"    - Class 1: {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test


def preprocess_pipeline(data_path: str = "data/dataset.csv", 
                        test_size: float = 0.2,
                        random_state: int = 42,
                        save_scaler: bool = True) -> dict:
    """
    Complete preprocessing pipeline for gene expression data.
    
    Returns a dictionary containing:
    - X_train, X_test: Normalized feature matrices
    - y_train, y_test: Label vectors
    - scaler: Fitted StandardScaler
    - feature_names: List of gene names
    """
    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load data
    df = load_data(data_path)
    
    # Step 2: Handle missing values
    df = handle_missing_values(df)
    
    # Step 3: Handle duplicates
    df = handle_duplicates(df)
    
    # Step 4: Separate features and labels
    X, y = separate_features_labels(df)
    
    # Step 5: Train-test split
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    
    # Step 6: Normalize features
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)
    
    # Save scaler for later use
    if save_scaler:
        scaler_path = Path("models/scaler.joblib")
        scaler_path.parent.mkdir(exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"\nScaler saved to: {scaler_path}")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nFinal dataset shapes:")
    print(f"  X_train: {X_train_scaled.shape}")
    print(f"  X_test: {X_test_scaled.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test: {y_test.shape}")
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': list(X.columns)
    }


if __name__ == "__main__":
    # Run preprocessing pipeline
    data = preprocess_pipeline()
    
    # Display summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total features (genes): {len(data['feature_names'])}")
    print(f"Training samples: {len(data['y_train'])}")
    print(f"Test samples: {len(data['y_test'])}")
    print("\nData is ready for feature selection and model training!")
