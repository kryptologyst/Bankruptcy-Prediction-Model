"""Unit tests for bankruptcy prediction models."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src.data.loader import BankruptcyDataLoader
from src.features.engineering import FeatureEngineer
from src.models.predictors import (
    LogisticRegressionPredictor,
    RandomForestPredictor,
    XGBoostPredictor,
    LightGBMPredictor,
    CatBoostPredictor,
    ModelEnsemble
)
from src.models.evaluation import BankruptcyEvaluator
from src.models.explainability import BankruptcyExplainer


class TestBankruptcyDataLoader:
    """Test cases for BankruptcyDataLoader."""

    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        loader = BankruptcyDataLoader()
        df = loader.generate_synthetic_data(n_samples=100)
        
        assert len(df) == 100
        assert 'bankruptcy' in df.columns
        assert df['bankruptcy'].dtype in ['int64', 'int32']
        assert df['bankruptcy'].isin([0, 1]).all()

    def test_data_preprocessing(self):
        """Test data preprocessing."""
        loader = BankruptcyDataLoader()
        df = loader.generate_synthetic_data(n_samples=100)
        
        X_train, X_test, y_train, y_test = loader.preprocess_data(df)
        
        assert len(X_train) + len(X_test) == len(df)
        assert len(y_train) + len(y_test) == len(df)
        assert X_train.shape[1] == X_test.shape[1]

    def test_feature_names(self):
        """Test feature name extraction."""
        loader = BankruptcyDataLoader()
        df = loader.generate_synthetic_data(n_samples=100)
        loader.preprocess_data(df)
        
        feature_names = loader.get_feature_names()
        assert len(feature_names) > 0
        assert all(isinstance(name, str) for name in feature_names)


class TestFeatureEngineer:
    """Test cases for FeatureEngineer."""

    def test_interaction_features(self):
        """Test interaction feature creation."""
        engineer = FeatureEngineer()
        
        # Create simple test data
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [2, 3, 4, 5],
            'feature3': [1, 1, 2, 2]
        })
        
        df_enhanced = engineer.create_interaction_features(df)
        
        assert len(df_enhanced.columns) > len(df.columns)
        assert 'feature1_x_feature2' in df_enhanced.columns

    def test_polynomial_features(self):
        """Test polynomial feature creation."""
        engineer = FeatureEngineer()
        
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [2, 3, 4]
        })
        
        df_poly = engineer.create_polynomial_features(df, degree=2)
        
        assert len(df_poly.columns) > len(df.columns)
        assert df_poly.shape[0] == df.shape[0]

    def test_feature_selection(self):
        """Test feature selection."""
        engineer = FeatureEngineer()
        
        # Create test data with clear signal
        X, y = make_classification(n_samples=100, n_features=10, n_informative=5, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        y_series = pd.Series(y)
        
        X_selected = engineer.select_features(X_df, y_series, k=5)
        
        assert X_selected.shape[1] == 5
        assert X_selected.shape[0] == X_df.shape[0]


class TestBankruptcyPredictors:
    """Test cases for bankruptcy prediction models."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        y_series = pd.Series(y)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_series, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test

    def test_logistic_regression(self, sample_data):
        """Test LogisticRegressionPredictor."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = LogisticRegressionPredictor(random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        assert len(predictions) == len(X_test)
        assert probabilities.shape[0] == len(X_test)
        assert probabilities.shape[1] == 2
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_random_forest(self, sample_data):
        """Test RandomForestPredictor."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = RandomForestPredictor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        assert len(predictions) == len(X_test)
        assert probabilities.shape[0] == len(X_test)
        assert probabilities.shape[1] == 2

    def test_xgboost(self, sample_data):
        """Test XGBoostPredictor."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = XGBoostPredictor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        assert len(predictions) == len(X_test)
        assert probabilities.shape[0] == len(X_test)
        assert probabilities.shape[1] == 2

    def test_lightgbm(self, sample_data):
        """Test LightGBMPredictor."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = LightGBMPredictor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        assert len(predictions) == len(X_test)
        assert probabilities.shape[0] == len(X_test)
        assert probabilities.shape[1] == 2

    def test_catboost(self, sample_data):
        """Test CatBoostPredictor."""
        X_train, X_test, y_train, y_test = sample_data
        
        model = CatBoostPredictor(iterations=10, random_seed=42, verbose=False)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        assert len(predictions) == len(X_test)
        assert probabilities.shape[0] == len(X_test)
        assert probabilities.shape[1] == 2

    def test_model_ensemble(self, sample_data):
        """Test ModelEnsemble."""
        X_train, X_test, y_train, y_test = sample_data
        
        models = [
            LogisticRegressionPredictor(random_state=42),
            RandomForestPredictor(n_estimators=10, random_state=42)
        ]
        
        ensemble = ModelEnsemble(models)
        ensemble.fit(X_train, y_train)
        
        predictions = ensemble.predict(X_test)
        probabilities = ensemble.predict_proba(X_test)
        
        assert len(predictions) == len(X_test)
        assert probabilities.shape[0] == len(X_test)
        assert probabilities.shape[1] == 2


class TestBankruptcyEvaluator:
    """Test cases for BankruptcyEvaluator."""

    def test_ml_metrics_calculation(self):
        """Test ML metrics calculation."""
        evaluator = BankruptcyEvaluator()
        
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.3, 0.8])
        
        metrics = evaluator.calculate_ml_metrics(y_true, y_pred, y_proba)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'auc_roc' in metrics
        assert 'auc_pr' in metrics

    def test_credit_metrics_calculation(self):
        """Test credit risk metrics calculation."""
        evaluator = BankruptcyEvaluator()
        
        y_true = np.array([0, 1, 0, 1, 0])
        y_proba = np.array([0.1, 0.9, 0.2, 0.3, 0.8])
        
        metrics = evaluator.calculate_credit_metrics(y_true, y_proba)
        
        assert 'ks_statistic' in metrics
        assert 'gini_coefficient' in metrics
        assert 'brier_score' in metrics
        assert 'population_stability_index' in metrics

    def test_business_metrics_calculation(self):
        """Test business metrics calculation."""
        evaluator = BankruptcyEvaluator()
        
        y_true = np.array([0, 1, 0, 1, 0])
        y_proba = np.array([0.1, 0.9, 0.2, 0.3, 0.8])
        
        metrics = evaluator.calculate_business_metrics(y_true, y_proba)
        
        assert 'true_positive_rate' in metrics
        assert 'false_positive_rate' in metrics
        assert 'bankruptcy_capture_rate' in metrics
        assert 'false_alarm_rate' in metrics

    def test_evaluation_report_generation(self):
        """Test evaluation report generation."""
        evaluator = BankruptcyEvaluator()
        
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.3, 0.8])
        
        report = evaluator.generate_evaluation_report(y_true, y_pred, y_proba)
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "BANKRUPTCY PREDICTION EVALUATION REPORT" in report


class TestBankruptcyExplainer:
    """Test cases for BankruptcyExplainer."""

    @pytest.fixture
    def sample_model_and_data(self):
        """Create sample model and data for testing."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        y_series = pd.Series(y)
        
        # Train a simple model
        model = LogisticRegressionPredictor(random_state=42)
        model.fit(X_df, y_series)
        
        return model, X_df

    def test_explainer_initialization(self, sample_model_and_data):
        """Test explainer initialization."""
        model, X_df = sample_model_and_data
        
        explainer = BankruptcyExplainer(model, X_df)
        
        assert explainer.model == model
        assert explainer.X_train.equals(X_df)
        assert explainer.explainer is not None

    def test_prediction_explanation(self, sample_model_and_data):
        """Test prediction explanation."""
        model, X_df = sample_model_and_data
        
        explainer = BankruptcyExplainer(model, X_df, X_df.iloc[:10])
        
        shap_values = explainer.explain_predictions()
        
        assert shap_values is not None
        assert shap_values.shape[0] == 10
        assert shap_values.shape[1] == X_df.shape[1]

    def test_feature_contributions(self, sample_model_and_data):
        """Test feature contributions calculation."""
        model, X_df = sample_model_and_data
        
        explainer = BankruptcyExplainer(model, X_df, X_df.iloc[:10])
        explainer.explain_predictions()
        
        contributions = explainer.get_feature_contributions(0)
        
        assert isinstance(contributions, pd.DataFrame)
        assert 'feature' in contributions.columns
        assert 'shap_value' in contributions.columns
        assert len(contributions) == X_df.shape[1]

    def test_explanation_report(self, sample_model_and_data):
        """Test explanation report generation."""
        model, X_df = sample_model_and_data
        
        explainer = BankruptcyExplainer(model, X_df, X_df.iloc[:10])
        
        report = explainer.generate_explanation_report()
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "SHAP EXPLANATION REPORT" in report


# Integration tests
class TestIntegration:
    """Integration tests for the complete pipeline."""

    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # Generate data
        loader = BankruptcyDataLoader()
        df = loader.generate_synthetic_data(n_samples=200)
        
        # Feature engineering
        engineer = FeatureEngineer()
        df_enhanced = engineer.create_interaction_features(df)
        
        # Preprocess data
        X_train, X_test, y_train, y_test = loader.preprocess_data(df_enhanced)
        
        # Train model
        model = XGBoostPredictor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        
        # Evaluate model
        evaluator = BankruptcyEvaluator()
        report = evaluator.generate_evaluation_report(
            y_test, predictions, probabilities, model_name="XGBoost"
        )
        
        # Test explainability
        explainer = BankruptcyExplainer(model, X_train, X_test)
        shap_values = explainer.explain_predictions()
        
        # Assertions
        assert len(predictions) == len(y_test)
        assert len(probabilities) == len(y_test)
        assert shap_values.shape[0] == len(X_test)
        assert shap_values.shape[1] == X_test.shape[1]
        assert isinstance(report, str)
        assert len(report) > 0


if __name__ == "__main__":
    pytest.main([__file__])
