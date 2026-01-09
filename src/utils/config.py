"""Configuration management for bankruptcy prediction models."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    n_samples: int = 1000
    test_size: float = 0.2
    random_state: int = 42
    data_path: Optional[str] = None


@dataclass
class ModelConfig:
    """Configuration for individual models."""
    logistic_regression: Dict[str, Any] = None
    random_forest: Dict[str, Any] = None
    xgboost: Dict[str, Any] = None
    lightgbm: Dict[str, Any] = None
    catboost: Dict[str, Any] = None
    
    def __post_init__(self):
        """Set default model configurations."""
        if self.logistic_regression is None:
            self.logistic_regression = {
                'random_state': 42,
                'max_iter': 1000,
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'liblinear'
            }
        
        if self.random_forest is None:
            self.random_forest = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42,
                'class_weight': 'balanced'
            }
        
        if self.xgboost is None:
            self.xgboost = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        
        if self.lightgbm is None:
            self.lightgbm = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'class_weight': 'balanced'
            }
        
        if self.catboost is None:
            self.catboost = {
                'iterations': 100,
                'depth': 6,
                'learning_rate': 0.1,
                'random_seed': 42,
                'verbose': False
            }


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    create_interactions: bool = True
    interaction_pairs: Optional[list] = None
    create_polynomial: bool = False
    polynomial_degree: int = 2
    create_risk_categories: bool = False
    feature_selection: bool = False
    n_features_select: int = 20


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    cv_folds: int = 5
    scoring_metrics: list = None
    generate_plots: bool = True
    save_plots: bool = True
    plot_format: str = 'png'
    plot_dpi: int = 300
    
    def __post_init__(self):
        if self.scoring_metrics is None:
            self.scoring_metrics = [
                'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
            ]


@dataclass
class ExplainabilityConfig:
    """Configuration for model explainability."""
    use_shap: bool = True
    shap_background_samples: int = 100
    max_display_features: int = 20
    generate_plots: bool = True
    save_explanations: bool = True


@dataclass
class TrainingConfig:
    """Main training configuration."""
    data: DataConfig = None
    models: ModelConfig = None
    features: FeatureConfig = None
    evaluation: EvaluationConfig = None
    explainability: ExplainabilityConfig = None
    output_dir: str = "assets"
    experiment_name: str = "bankruptcy_prediction"
    save_models: bool = True
    save_predictions: bool = True
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.models is None:
            self.models = ModelConfig()
        if self.features is None:
            self.features = FeatureConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
        if self.explainability is None:
            self.explainability = ExplainabilityConfig()


class ConfigManager:
    """Configuration manager for loading and saving configurations."""
    
    @staticmethod
    def load_config(config_path: str) -> TrainingConfig:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file.
            
        Returns:
            TrainingConfig object.
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return ConfigManager._dict_to_config(config_dict)
    
    @staticmethod
    def save_config(config: TrainingConfig, config_path: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            config: TrainingConfig object to save.
            config_path: Path to save configuration file.
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = ConfigManager._config_to_dict(config)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @staticmethod
    def create_default_config(config_path: str) -> None:
        """Create default configuration file.
        
        Args:
            config_path: Path to save default configuration.
        """
        config = TrainingConfig()
        ConfigManager.save_config(config, config_path)
    
    @staticmethod
    def _config_to_dict(config: TrainingConfig) -> Dict[str, Any]:
        """Convert TrainingConfig to dictionary."""
        return asdict(config)
    
    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> TrainingConfig:
        """Convert dictionary to TrainingConfig."""
        # Handle nested dataclasses
        if 'data' in config_dict:
            config_dict['data'] = DataConfig(**config_dict['data'])
        
        if 'models' in config_dict:
            config_dict['models'] = ModelConfig(**config_dict['models'])
        
        if 'features' in config_dict:
            config_dict['features'] = FeatureConfig(**config_dict['features'])
        
        if 'evaluation' in config_dict:
            config_dict['evaluation'] = EvaluationConfig(**config_dict['evaluation'])
        
        if 'explainability' in config_dict:
            config_dict['explainability'] = ExplainabilityConfig(**config_dict['explainability'])
        
        return TrainingConfig(**config_dict)


# Default configuration files
DEFAULT_CONFIG = {
    'data': {
        'n_samples': 1000,
        'test_size': 0.2,
        'random_state': 42,
        'data_path': None
    },
    'models': {
        'logistic_regression': {
            'random_state': 42,
            'max_iter': 1000,
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'liblinear'
        },
        'random_forest': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'class_weight': 'balanced'
        },
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        },
        'lightgbm': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'class_weight': 'balanced'
        },
        'catboost': {
            'iterations': 100,
            'depth': 6,
            'learning_rate': 0.1,
            'random_seed': 42,
            'verbose': False
        }
    },
    'features': {
        'create_interactions': True,
        'interaction_pairs': None,
        'create_polynomial': False,
        'polynomial_degree': 2,
        'create_risk_categories': False,
        'feature_selection': False,
        'n_features_select': 20
    },
    'evaluation': {
        'cv_folds': 5,
        'scoring_metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        'generate_plots': True,
        'save_plots': True,
        'plot_format': 'png',
        'plot_dpi': 300
    },
    'explainability': {
        'use_shap': True,
        'shap_background_samples': 100,
        'max_display_features': 20,
        'generate_plots': True,
        'save_explanations': True
    },
    'output_dir': 'assets',
    'experiment_name': 'bankruptcy_prediction',
    'save_models': True,
    'save_predictions': True
}
