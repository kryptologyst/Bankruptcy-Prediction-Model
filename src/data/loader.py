"""Data loading and preprocessing utilities for bankruptcy prediction."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class BankruptcyDataLoader:
    """Data loader for bankruptcy prediction datasets."""

    def __init__(self, data_path: Optional[Union[str, Path]] = None):
        """Initialize the data loader.
        
        Args:
            data_path: Path to the data directory. If None, uses synthetic data.
        """
        self.data_path = Path(data_path) if data_path else None
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.target_column: str = "bankruptcy"

    def generate_synthetic_data(
        self, 
        n_samples: int = 1000, 
        random_state: int = 42
    ) -> pd.DataFrame:
        """Generate synthetic financial data for demonstration.
        
        Args:
            n_samples: Number of samples to generate.
            random_state: Random seed for reproducibility.
            
        Returns:
            DataFrame with synthetic financial ratios and bankruptcy labels.
        """
        np.random.seed(random_state)
        
        # Generate realistic financial ratios with correlations
        data = {
            # Liquidity ratios
            'current_ratio': np.random.normal(1.2, 0.3, n_samples),
            'quick_ratio': np.random.normal(0.8, 0.2, n_samples),
            'cash_ratio': np.random.normal(0.3, 0.1, n_samples),
            
            # Leverage ratios
            'debt_to_equity': np.random.normal(1.5, 0.5, n_samples),
            'debt_to_assets': np.random.normal(0.4, 0.1, n_samples),
            'interest_coverage': np.random.normal(3.0, 1.5, n_samples),
            
            # Profitability ratios
            'return_on_assets': np.random.normal(0.05, 0.02, n_samples),
            'return_on_equity': np.random.normal(0.10, 0.05, n_samples),
            'gross_margin': np.random.normal(0.25, 0.08, n_samples),
            
            # Activity ratios
            'asset_turnover': np.random.normal(1.0, 0.3, n_samples),
            'inventory_turnover': np.random.normal(6.0, 2.0, n_samples),
            'receivables_turnover': np.random.normal(8.0, 3.0, n_samples),
            
            # Market ratios
            'price_to_earnings': np.random.normal(15.0, 5.0, n_samples),
            'price_to_book': np.random.normal(1.5, 0.5, n_samples),
            'market_to_book': np.random.normal(1.2, 0.4, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Ensure non-negative ratios
        df = df.clip(lower=0.01)
        
        # Generate bankruptcy labels based on financial distress indicators
        # Higher debt ratios and lower profitability increase bankruptcy risk
        bankruptcy_score = (
            -0.3 * df['current_ratio'] +
            -0.2 * df['quick_ratio'] +
            0.4 * df['debt_to_equity'] +
            0.3 * df['debt_to_assets'] +
            -0.2 * df['interest_coverage'] +
            -0.4 * df['return_on_assets'] +
            -0.3 * df['return_on_equity'] +
            -0.2 * df['gross_margin'] +
            np.random.normal(0, 0.1, n_samples)
        )
        
        # Convert to binary labels (top 20% are bankrupt)
        bankruptcy_threshold = np.percentile(bankruptcy_score, 80)
        df[self.target_column] = (bankruptcy_score > bankruptcy_threshold).astype(int)
        
        logger.info(f"Generated synthetic dataset with {n_samples} samples")
        logger.info(f"Bankruptcy rate: {df[self.target_column].mean():.2%}")
        
        return df

    def load_data(self, filename: Optional[str] = None) -> pd.DataFrame:
        """Load bankruptcy prediction data.
        
        Args:
            filename: Name of the data file. If None, generates synthetic data.
            
        Returns:
            DataFrame with financial ratios and bankruptcy labels.
        """
        if self.data_path and filename:
            file_path = self.data_path / filename
            if file_path.exists():
                df = pd.read_csv(file_path)
                logger.info(f"Loaded data from {file_path}")
                return df
            else:
                logger.warning(f"File {file_path} not found, generating synthetic data")
        
        return self.generate_synthetic_data()

    def preprocess_data(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Preprocess the data for model training.
        
        Args:
            df: Input DataFrame.
            test_size: Proportion of data to use for testing.
            random_state: Random seed for reproducibility.
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        # Identify feature columns (exclude target and metadata)
        self.feature_columns = [
            col for col in df.columns 
            if col != self.target_column and df[col].dtype in ['float64', 'int64']
        ]
        
        X = df[self.feature_columns]
        y = df[self.target_column]
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Preprocessed data: {len(X_train)} train, {len(X_test)} test samples")
        logger.info(f"Features: {self.feature_columns}")
        
        return X_train, X_test, y_train, y_test

    def get_feature_names(self) -> List[str]:
        """Get the list of feature column names."""
        return self.feature_columns.copy()

    def transform_new_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using the fitted scaler.
        
        Args:
            X: New data to transform.
            
        Returns:
            Transformed data.
        """
        if not hasattr(self.scaler, 'mean_'):
            raise ValueError("Scaler must be fitted before transforming new data")
        
        X_transformed = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        return X_transformed
