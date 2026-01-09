"""Main training script for bankruptcy prediction models."""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Import our custom modules
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_deterministic_seeds(seed: int = 42) -> None:
    """Set up deterministic seeds for reproducibility."""
    np.random.seed(seed)
    logger.info(f"Set random seed to {seed}")


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models_config: Optional[Dict] = None
) -> Dict[str, any]:
    """Train multiple models and return results.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        models_config: Configuration for models.
        
    Returns:
        Dictionary with trained models and results.
    """
    if models_config is None:
        models_config = {
            'logistic_regression': {'random_state': 42},
            'random_forest': {'n_estimators': 100, 'random_state': 42, 'class_weight': 'balanced'},
            'xgboost': {'n_estimators': 100, 'random_state': 42},
            'lightgbm': {'n_estimators': 100, 'random_state': 42, 'class_weight': 'balanced'},
            'catboost': {'iterations': 100, 'random_seed': 42, 'verbose': False}
        }
    
    models = {
        'Logistic Regression': LogisticRegressionPredictor(**models_config['logistic_regression']),
        'Random Forest': RandomForestPredictor(**models_config['random_forest']),
        'XGBoost': XGBoostPredictor(**models_config['xgboost']),
        'LightGBM': LightGBMPredictor(**models_config['lightgbm']),
        'CatBoost': CatBoostPredictor(**models_config['catboost'])
    }
    
    results = {}
    evaluator = BankruptcyEvaluator()
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]  # Probability of bankruptcy
        
        # Evaluate model
        report = evaluator.generate_evaluation_report(
            y_test, y_pred, y_proba, model_name=name
        )
        
        # Store results
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_proba,
            'evaluation_report': report,
            'metrics': evaluator.metrics.copy(),
            'feature_importance': model.get_feature_importance()
        }
        
        logger.info(f"Completed training {name}")
    
    return results


def create_ensemble_model(results: Dict[str, any]) -> ModelEnsemble:
    """Create an ensemble model from trained models.
    
    Args:
        results: Dictionary with trained models and results.
        
    Returns:
        Ensemble model.
    """
    # Select top performing models for ensemble
    model_scores = {}
    for name, result in results.items():
        if 'auc_roc' in result['metrics']:
            model_scores[name] = result['metrics']['auc_roc']
    
    # Sort by AUC score and select top 3
    top_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    
    logger.info(f"Creating ensemble with top models: {[name for name, _ in top_models]}")
    
    # Create ensemble with equal weights
    ensemble_models = [results[name]['model'] for name, _ in top_models]
    ensemble = ModelEnsemble(ensemble_models)
    
    return ensemble


def generate_visualizations(
    results: Dict[str, any],
    y_test: pd.Series,
    output_dir: Path
) -> None:
    """Generate evaluation visualizations.
    
    Args:
        results: Dictionary with model results.
        y_test: Test labels.
        output_dir: Output directory for plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator = BankruptcyEvaluator()
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC Curves
    ax1 = axes[0, 0]
    for name, result in results.items():
        fpr, tpr, _ = evaluator.roc_curve_data if hasattr(evaluator, 'roc_curve_data') else (None, None, None)
        if fpr is None:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
        auc_score = result['metrics'].get('auc_roc', 0)
        ax1.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
    
    ax1.plot([0, 1], [0, 1], 'k--', label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall Curves
    ax2 = axes[0, 1]
    for name, result in results.items():
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision, recall, _ = precision_recall_curve(y_test, result['probabilities'])
        ap_score = average_precision_score(y_test, result['probabilities'])
        ax2.plot(recall, precision, label=f'{name} (AP = {ap_score:.3f})')
    
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Feature Importance (for tree-based models)
    ax3 = axes[1, 0]
    tree_models = ['Random Forest', 'XGBoost', 'LightGBM', 'CatBoost']
    for name in tree_models:
        if name in results and results[name]['feature_importance']:
            importance = results[name]['feature_importance']
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            features, scores = zip(*top_features)
            ax3.barh(features, scores, alpha=0.7, label=name)
    
    ax3.set_xlabel('Feature Importance')
    ax3.set_title('Top 10 Feature Importance')
    ax3.legend()
    
    # Model Performance Comparison
    ax4 = axes[1, 1]
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    model_names = list(results.keys())
    
    for metric in metrics_to_plot:
        scores = [results[name]['metrics'].get(metric, 0) for name in model_names]
        ax4.plot(model_names, scores, marker='o', label=metric)
    
    ax4.set_xlabel('Models')
    ax4.set_ylabel('Score')
    ax4.set_title('Model Performance Comparison')
    ax4.legend()
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visualizations to {output_dir}")


def save_results(
    results: Dict[str, any],
    ensemble_model: Optional[ModelEnsemble],
    output_dir: Path
) -> None:
    """Save model results and artifacts.
    
    Args:
        results: Dictionary with model results.
        ensemble_model: Trained ensemble model.
        output_dir: Output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual models
    for name, result in results.items():
        model_path = output_dir / f'{name.lower().replace(" ", "_")}_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(result['model'], f)
        
        # Save evaluation report
        report_path = output_dir / f'{name.lower().replace(" ", "_")}_report.txt'
        with open(report_path, 'w') as f:
            f.write(result['evaluation_report'])
    
    # Save ensemble model
    if ensemble_model:
        ensemble_path = output_dir / 'ensemble_model.pkl'
        with open(ensemble_path, 'wb') as f:
            pickle.dump(ensemble_model, f)
    
    # Save metrics summary
    metrics_summary = {}
    for name, result in results.items():
        metrics_summary[name] = result['metrics']
    
    metrics_df = pd.DataFrame(metrics_summary).T
    metrics_df.to_csv(output_dir / 'metrics_summary.csv')
    
    logger.info(f"Saved results to {output_dir}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train bankruptcy prediction models')
    parser.add_argument('--data-path', type=str, help='Path to data directory')
    parser.add_argument('--output-dir', type=str, default='assets', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--n-samples', type=int, default=1000, help='Number of synthetic samples')
    
    args = parser.parse_args()
    
    # Setup
    setup_deterministic_seeds(args.seed)
    output_dir = Path(args.output_dir)
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    data_loader = BankruptcyDataLoader(args.data_path)
    df = data_loader.load_data()
    
    # Feature engineering
    feature_engineer = FeatureEngineer()
    df_enhanced = feature_engineer.create_interaction_features(df)
    
    # Preprocess data
    X_train, X_test, y_train, y_test = data_loader.preprocess_data(
        df_enhanced, test_size=args.test_size, random_state=args.seed
    )
    
    logger.info(f"Data shape: {X_train.shape} train, {X_test.shape} test")
    logger.info(f"Bankruptcy rate: {y_train.mean():.2%}")
    
    # Train models
    logger.info("Training models...")
    results = train_models(X_train, y_train, X_test, y_test)
    
    # Create ensemble
    logger.info("Creating ensemble model...")
    ensemble_model = create_ensemble_model(results)
    
    # Evaluate ensemble
    ensemble_pred = ensemble_model.predict(X_test)
    ensemble_proba = ensemble_model.predict_proba(X_test)[:, 1]
    
    evaluator = BankruptcyEvaluator()
    ensemble_report = evaluator.generate_evaluation_report(
        y_test, ensemble_pred, ensemble_proba, model_name="Ensemble"
    )
    
    results['Ensemble'] = {
        'model': ensemble_model,
        'predictions': ensemble_pred,
        'probabilities': ensemble_proba,
        'evaluation_report': ensemble_report,
        'metrics': evaluator.metrics.copy(),
        'feature_importance': ensemble_model.get_feature_importance()
    }
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    generate_visualizations(results, y_test, output_dir)
    
    # Save results
    logger.info("Saving results...")
    save_results(results, ensemble_model, output_dir)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("="*60)
    
    for name, result in results.items():
        auc_score = result['metrics'].get('auc_roc', 0)
        logger.info(f"{name:20}: AUC = {auc_score:.4f}")
    
    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
