"""Feature engineering utilities for bankruptcy prediction."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for bankruptcy prediction."""

    def __init__(self):
        """Initialize the feature engineer."""
        self.poly_features = None
        self.feature_selector = None
        self.feature_names: List[str] = []

    def create_financial_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional financial ratios from basic metrics.
        
        Args:
            df: DataFrame with basic financial metrics.
            
        Returns:
            DataFrame with additional financial ratios.
        """
        df_enhanced = df.copy()
        
        # Working capital ratios
        if 'current_assets' in df.columns and 'current_liabilities' in df.columns:
            df_enhanced['working_capital'] = df['current_assets'] - df['current_liabilities']
            df_enhanced['working_capital_ratio'] = (
                df_enhanced['working_capital'] / df.get('total_assets', 1)
            )
        
        # Efficiency ratios
        if 'sales' in df.columns and 'total_assets' in df.columns:
            df_enhanced['asset_turnover'] = df['sales'] / df['total_assets']
        
        if 'sales' in df.columns and 'inventory' in df.columns:
            df_enhanced['inventory_turnover'] = df['sales'] / df['inventory']
        
        # Coverage ratios
        if 'ebit' in df.columns and 'interest_expense' in df.columns:
            df_enhanced['interest_coverage'] = df['ebit'] / (df['interest_expense'] + 1e-8)
        
        # Growth ratios (if time series data available)
        for col in ['sales', 'net_income', 'total_assets']:
            if col in df.columns:
                df_enhanced[f'{col}_growth'] = df[col].pct_change()
        
        logger.info(f"Created {len(df_enhanced.columns) - len(df.columns)} additional features")
        
        return df_enhanced

    def create_interaction_features(
        self, 
        df: pd.DataFrame, 
        feature_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> pd.DataFrame:
        """Create interaction features between financial ratios.
        
        Args:
            df: DataFrame with financial features.
            feature_pairs: List of feature pairs to create interactions for.
            
        Returns:
            DataFrame with interaction features.
        """
        df_enhanced = df.copy()
        
        if feature_pairs is None:
            # Default interaction pairs for bankruptcy prediction
            feature_pairs = [
                ('debt_to_equity', 'return_on_assets'),
                ('current_ratio', 'debt_to_assets'),
                ('interest_coverage', 'return_on_equity'),
                ('gross_margin', 'asset_turnover'),
            ]
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                df_enhanced[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                df_enhanced[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-8)
        
        logger.info(f"Created {len(df_enhanced.columns) - len(df.columns)} interaction features")
        
        return df_enhanced

    def create_polynomial_features(
        self, 
        X: pd.DataFrame, 
        degree: int = 2,
        include_bias: bool = False
    ) -> pd.DataFrame:
        """Create polynomial features for non-linear relationships.
        
        Args:
            X: Input features.
            degree: Degree of polynomial features.
            include_bias: Whether to include bias term.
            
        Returns:
            DataFrame with polynomial features.
        """
        self.poly_features = PolynomialFeatures(
            degree=degree, 
            include_bias=include_bias,
            interaction_only=False
        )
        
        X_poly = self.poly_features.fit_transform(X)
        
        # Create feature names
        feature_names = self.poly_features.get_feature_names_out(X.columns)
        
        df_poly = pd.DataFrame(X_poly, columns=feature_names, index=X.index)
        
        logger.info(f"Created polynomial features: {X.shape[1]} -> {df_poly.shape[1]}")
        
        return df_poly

    def create_risk_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create categorical risk indicators from continuous ratios.
        
        Args:
            df: DataFrame with financial ratios.
            
        Returns:
            DataFrame with risk category features.
        """
        df_enhanced = df.copy()
        
        # Liquidity risk categories
        if 'current_ratio' in df.columns:
            df_enhanced['liquidity_risk'] = pd.cut(
                df['current_ratio'],
                bins=[0, 1.0, 1.5, 2.0, float('inf')],
                labels=['High', 'Medium', 'Low', 'Very Low'],
                include_lowest=True
            )
        
        # Leverage risk categories
        if 'debt_to_equity' in df.columns:
            df_enhanced['leverage_risk'] = pd.cut(
                df['debt_to_equity'],
                bins=[0, 0.5, 1.0, 2.0, float('inf')],
                labels=['Low', 'Medium', 'High', 'Very High'],
                include_lowest=True
            )
        
        # Profitability risk categories
        if 'return_on_assets' in df.columns:
            df_enhanced['profitability_risk'] = pd.cut(
                df['return_on_assets'],
                bins=[-float('inf'), 0, 0.05, 0.10, float('inf')],
                labels=['Very High', 'High', 'Medium', 'Low']
            )
        
        logger.info(f"Created risk category features")
        
        return df_enhanced

    def select_features(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        k: int = 20,
        score_func: str = 'f_classif'
    ) -> pd.DataFrame:
        """Select the most important features using statistical tests.
        
        Args:
            X: Input features.
            y: Target variable.
            k: Number of features to select.
            score_func: Scoring function for feature selection.
            
        Returns:
            DataFrame with selected features.
        """
        if score_func == 'f_classif':
            score_function = f_classif
        else:
            raise ValueError(f"Unsupported score function: {score_func}")
        
        self.feature_selector = SelectKBest(score_func=score_function, k=k)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[self.feature_selector.get_support()]
        self.feature_names = selected_features.tolist()
        
        df_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        logger.info(f"Selected {len(selected_features)} features from {X.shape[1]}")
        
        return df_selected

    def get_feature_importance_scores(self) -> Dict[str, float]:
        """Get feature importance scores from the selector.
        
        Returns:
            Dictionary mapping feature names to importance scores.
        """
        if self.feature_selector is None:
            raise ValueError("Feature selector not fitted yet")
        
        scores = self.feature_selector.scores_
        feature_names = self.feature_names
        
        return dict(zip(feature_names, scores))

    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted transformers.
        
        Args:
            X: Input features to transform.
            
        Returns:
            Transformed features.
        """
        X_transformed = X.copy()
        
        # Apply polynomial features if fitted
        if self.poly_features is not None:
            X_poly = self.poly_features.transform(X)
            feature_names = self.poly_features.get_feature_names_out(X.columns)
            X_transformed = pd.DataFrame(X_poly, columns=feature_names, index=X.index)
        
        # Apply feature selection if fitted
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_transformed)
            selected_features = X_transformed.columns[self.feature_selector.get_support()]
            X_transformed = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        return X_transformed
