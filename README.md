# Bankruptcy Prediction Model

A comprehensive machine learning system for predicting corporate bankruptcy risk using financial ratios and advanced ML techniques.

## ⚠️ IMPORTANT DISCLAIMER

**This is a research demonstration project only.**

- This model is for educational and research purposes only
- It should NOT be used for actual investment decisions
- Predictions may be inaccurate and should not be relied upon
- Past performance does not guarantee future results
- Always consult with qualified financial professionals

## Overview

This project implements a modern bankruptcy prediction system using multiple machine learning algorithms including Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost. The system includes comprehensive evaluation metrics, SHAP explainability, and an interactive web demo.

## Features

- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost
- **Ensemble Learning**: Combines top-performing models for improved accuracy
- **Comprehensive Evaluation**: AUC, KS statistic, Gini coefficient, Brier score, calibration curves
- **SHAP Explainability**: Feature importance and individual prediction explanations
- **Interactive Demo**: Streamlit web application for easy model interaction
- **Credit Risk Metrics**: Specialized metrics for financial risk assessment
- **Feature Engineering**: Advanced feature creation and selection
- **Reproducible**: Deterministic seeding and proper train/test splits

## Project Structure

```
bankruptcy-prediction/
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── features/          # Feature engineering
│   ├── models/            # Model implementations and evaluation
│   └── utils/             # Utility functions
├── scripts/               # Training and evaluation scripts
├── demo/                  # Streamlit demo application
├── configs/               # Configuration files
├── data/                  # Data directory
├── assets/                # Model artifacts and visualizations
├── tests/                 # Unit tests
├── notebooks/             # Jupyter notebooks for analysis
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project configuration
└── README.md              # This file
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Bankruptcy-Prediction-Model.git
cd Bankruptcy-Prediction-Model
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Train Models

Run the training script to train all models:

```bash
python scripts/train.py --output-dir assets --seed 42
```

This will:
- Generate synthetic financial data
- Train multiple ML models
- Create an ensemble model
- Generate evaluation metrics and visualizations
- Save trained models to the `assets/` directory

### 2. Launch Interactive Demo

Start the Streamlit demo:

```bash
streamlit run demo/app.py
```

The demo provides:
- Single company bankruptcy risk assessment
- Batch analysis of multiple companies
- Model performance comparison
- Feature importance visualization

### 3. Use Models Programmatically

```python
from src.data.loader import BankruptcyDataLoader
from src.models.predictors import XGBoostPredictor
from src.models.evaluation import BankruptcyEvaluator

# Load data
data_loader = BankruptcyDataLoader()
df = data_loader.generate_synthetic_data(n_samples=1000)
X_train, X_test, y_train, y_test = data_loader.preprocess_data(df)

# Train model
model = XGBoostPredictor()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# Evaluate
evaluator = BankruptcyEvaluator()
report = evaluator.generate_evaluation_report(y_test, predictions, probabilities)
print(report)
```

## Data Schema

The model expects financial ratio data with the following structure:

### Required Features

- **Liquidity Ratios**: current_ratio, quick_ratio, cash_ratio
- **Leverage Ratios**: debt_to_equity, debt_to_assets, interest_coverage
- **Profitability Ratios**: return_on_assets, return_on_equity, gross_margin
- **Activity Ratios**: asset_turnover, inventory_turnover, receivables_turnover
- **Market Ratios**: price_to_earnings, price_to_book, market_to_book

### Target Variable

- `bankruptcy`: Binary indicator (0 = Non-bankrupt, 1 = Bankrupt)

### Data Format

```csv
current_ratio,debt_to_equity,return_on_assets,interest_coverage,bankruptcy
1.2,1.5,0.05,3.0,0
0.8,2.1,0.02,1.5,1
...
```

## Model Performance

The models are evaluated using both standard ML metrics and credit risk specific metrics:

### Machine Learning Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve
- **AUC-PR**: Area under the Precision-Recall curve

### Credit Risk Metrics
- **KS Statistic**: Kolmogorov-Smirnov test statistic
- **Gini Coefficient**: Measure of model discrimination
- **Brier Score**: Calibration quality measure
- **Population Stability Index**: Distribution stability measure

### Business Metrics
- **Bankruptcy Capture Rate**: Percentage of actual bankruptcies correctly identified
- **False Alarm Rate**: Percentage of non-bankrupt companies incorrectly flagged
- **True Positive Rate**: Sensitivity of the model
- **True Negative Rate**: Specificity of the model

## Configuration

Model parameters can be configured in the training script or through command-line arguments:

```bash
python scripts/train.py \
    --data-path data/ \
    --output-dir assets/ \
    --seed 42 \
    --test-size 0.2 \
    --n-samples 1000
```

## Advanced Usage

### Feature Engineering

```python
from src.features.engineering import FeatureEngineer

# Create interaction features
engineer = FeatureEngineer()
df_enhanced = engineer.create_interaction_features(df)

# Create polynomial features
df_poly = engineer.create_polynomial_features(df_enhanced, degree=2)

# Select important features
df_selected = engineer.select_features(df_poly, y, k=20)
```

### SHAP Explainability

```python
from src.models.explainability import BankruptcyExplainer

# Create explainer
explainer = BankruptcyExplainer(model, X_train, X_test)

# Explain predictions
shap_values = explainer.explain_predictions()

# Generate explanation report
report = explainer.generate_explanation_report()
print(report)

# Plot feature importance
fig = explainer.plot_feature_importance()
plt.show()
```

### Custom Model Training

```python
from src.models.predictors import ModelEnsemble

# Create custom ensemble
models = [
    XGBoostPredictor(n_estimators=200),
    LightGBMPredictor(n_estimators=200),
    CatBoostPredictor(iterations=200)
]

ensemble = ModelEnsemble(models, weights=[0.4, 0.3, 0.3])
ensemble.fit(X_train, y_train)
```

## Testing

Run the test suite:

```bash
pytest tests/ -v --cov=src
```

## Development

### Code Quality

The project uses several tools for code quality:

- **Black**: Code formatting
- **Ruff**: Linting and import sorting
- **Pytest**: Testing framework
- **Pre-commit**: Git hooks for quality checks

Setup pre-commit hooks:

```bash
pre-commit install
```

### Adding New Models

To add a new model:

1. Create a new predictor class inheriting from `BankruptcyPredictor`
2. Implement the required methods (`fit`, `predict`, `predict_proba`)
3. Add the model to the training script
4. Update the demo application if needed

Example:

```python
class CustomPredictor(BankruptcyPredictor):
    def __init__(self, **kwargs):
        super().__init__("CustomModel")
        self.model = CustomModel(**kwargs)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{bankruptcy_prediction,
  title={Bankruptcy Prediction Model},
  author={AI Research Team},
  year={2024},
  url={https://github.com/your-repo/bankruptcy-prediction}
}
```

## Support

For questions or issues:

1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed description
4. Provide sample code and error messages

## Changelog

### Version 1.0.0
- Initial release
- Multiple ML models (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost)
- Ensemble learning
- SHAP explainability
- Streamlit demo
- Comprehensive evaluation metrics
- Credit risk specific metrics

---

**Remember**: This is a research demonstration tool only. Do not use for actual investment decisions.
# Bankruptcy-Prediction-Model
