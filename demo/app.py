"""Streamlit demo application for bankruptcy prediction."""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our custom modules
from src.data.loader import BankruptcyDataLoader
from src.features.engineering import FeatureEngineer
from src.models.predictors import BankruptcyPredictor
from src.models.evaluation import BankruptcyEvaluator
from src.models.explainability import BankruptcyExplainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Bankruptcy Prediction Model",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .feature-importance {
        background-color: #e8f4fd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <h4>⚠️ IMPORTANT DISCLAIMER</h4>
    <p><strong>This is a research demonstration tool only.</strong></p>
    <ul>
        <li>This model is for educational and research purposes only</li>
        <li>It should NOT be used for actual investment decisions</li>
        <li>Predictions may be inaccurate and should not be relied upon</li>
        <li>Past performance does not guarantee future results</li>
        <li>Always consult with qualified financial professionals</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🏦 Bankruptcy Prediction Model</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Model Configuration")

# Load models
@st.cache_data
def load_models(models_dir: Path) -> Dict[str, BankruptcyPredictor]:
    """Load trained models from disk."""
    models = {}
    
    if not models_dir.exists():
        st.error(f"Models directory {models_dir} not found. Please train models first.")
        return models
    
    model_files = {
        'Logistic Regression': 'logistic_regression_model.pkl',
        'Random Forest': 'random_forest_model.pkl',
        'XGBoost': 'xgboost_model.pkl',
        'LightGBM': 'lightgbm_model.pkl',
        'CatBoost': 'catboost_model.pkl',
        'Ensemble': 'ensemble_model.pkl'
    }
    
    for name, filename in model_files.items():
        model_path = models_dir / filename
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    models[name] = pickle.load(f)
                st.sidebar.success(f"✓ Loaded {name}")
            except Exception as e:
                st.sidebar.error(f"✗ Failed to load {name}: {e}")
    
    return models

# Load data
@st.cache_data
def load_sample_data() -> Tuple[pd.DataFrame, List[str]]:
    """Load sample data for demonstration."""
    data_loader = BankruptcyDataLoader()
    df = data_loader.generate_synthetic_data(n_samples=1000)
    
    # Get feature columns
    feature_columns = [col for col in df.columns if col != 'bankruptcy']
    
    return df, feature_columns

# Load models and data
models_dir = Path("assets")
models = load_models(models_dir)
df_sample, feature_columns = load_sample_data()

if not models:
    st.error("No trained models found. Please run the training script first.")
    st.stop()

# Model selection
selected_model_name = st.sidebar.selectbox(
    "Select Model",
    options=list(models.keys()),
    help="Choose which trained model to use for predictions"
)

selected_model = models[selected_model_name]

# Prediction mode selection
prediction_mode = st.sidebar.radio(
    "Prediction Mode",
    ["Single Company", "Batch Analysis", "Model Comparison"],
    help="Choose how to use the model"
)

if prediction_mode == "Single Company":
    st.header("Single Company Analysis")
    
    # Input form
    with st.form("company_input"):
        st.subheader("Company Financial Ratios")
        
        # Create input fields for key financial ratios
        col1, col2 = st.columns(2)
        
        with col1:
            current_ratio = st.number_input(
                "Current Ratio",
                min_value=0.1,
                max_value=10.0,
                value=1.2,
                step=0.1,
                help="Current Assets / Current Liabilities"
            )
            
            debt_to_equity = st.number_input(
                "Debt-to-Equity Ratio",
                min_value=0.0,
                max_value=10.0,
                value=1.5,
                step=0.1,
                help="Total Debt / Total Equity"
            )
            
            return_on_assets = st.number_input(
                "Return on Assets (%)",
                min_value=-50.0,
                max_value=50.0,
                value=5.0,
                step=0.1,
                help="Net Income / Total Assets"
            )
        
        with col2:
            quick_ratio = st.number_input(
                "Quick Ratio",
                min_value=0.1,
                max_value=10.0,
                value=0.8,
                step=0.1,
                help="(Current Assets - Inventory) / Current Liabilities"
            )
            
            interest_coverage = st.number_input(
                "Interest Coverage Ratio",
                min_value=0.0,
                max_value=50.0,
                value=3.0,
                step=0.1,
                help="EBIT / Interest Expense"
            )
            
            gross_margin = st.number_input(
                "Gross Margin (%)",
                min_value=-100.0,
                max_value=100.0,
                value=25.0,
                step=0.1,
                help="(Revenue - COGS) / Revenue"
            )
        
        # Additional ratios
        st.subheader("Additional Financial Metrics")
        col3, col4 = st.columns(2)
        
        with col3:
            asset_turnover = st.number_input(
                "Asset Turnover",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="Revenue / Total Assets"
            )
            
            inventory_turnover = st.number_input(
                "Inventory Turnover",
                min_value=0.1,
                max_value=50.0,
                value=6.0,
                step=0.1,
                help="COGS / Average Inventory"
            )
        
        with col4:
            receivables_turnover = st.number_input(
                "Receivables Turnover",
                min_value=0.1,
                max_value=50.0,
                value=8.0,
                step=0.1,
                help="Revenue / Average Receivables"
            )
            
            price_to_earnings = st.number_input(
                "Price-to-Earnings Ratio",
                min_value=0.1,
                max_value=100.0,
                value=15.0,
                step=0.1,
                help="Stock Price / Earnings per Share"
            )
        
        submitted = st.form_submit_button("Predict Bankruptcy Risk", type="primary")
    
    if submitted:
        # Create input data
        input_data = pd.DataFrame({
            'current_ratio': [current_ratio],
            'quick_ratio': [quick_ratio],
            'debt_to_equity': [debt_to_equity],
            'interest_coverage': [interest_coverage],
            'return_on_assets': [return_on_assets / 100],  # Convert percentage to decimal
            'gross_margin': [gross_margin / 100],  # Convert percentage to decimal
            'asset_turnover': [asset_turnover],
            'inventory_turnover': [inventory_turnover],
            'receivables_turnover': [receivables_turnover],
            'price_to_earnings': [price_to_earnings],
            # Add default values for missing features
            'cash_ratio': [0.3],
            'debt_to_assets': [0.4],
            'return_on_equity': [0.10],
            'price_to_book': [1.5],
            'market_to_book': [1.2]
        })
        
        # Ensure all required features are present
        for feature in feature_columns:
            if feature not in input_data.columns:
                input_data[feature] = df_sample[feature].median()
        
        # Make prediction
        try:
            # Get probability
            proba = selected_model.predict_proba(input_data[feature_columns])[0, 1]
            prediction = selected_model.predict(input_data[feature_columns])[0]
            
            # Display results
            st.subheader("Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Bankruptcy Probability",
                    f"{proba:.1%}",
                    delta=f"{proba - 0.5:+.1%}" if proba != 0.5 else "0.0%"
                )
            
            with col2:
                risk_level = "High" if proba > 0.7 else "Medium" if proba > 0.3 else "Low"
                st.metric("Risk Level", risk_level)
            
            with col3:
                prediction_text = "Bankrupt" if prediction == 1 else "Non-Bankrupt"
                st.metric("Prediction", prediction_text)
            
            # Risk interpretation
            st.subheader("Risk Interpretation")
            
            if proba > 0.7:
                st.error("🚨 **High Risk**: This company shows strong indicators of potential bankruptcy.")
            elif proba > 0.3:
                st.warning("⚠️ **Medium Risk**: This company shows some concerning financial indicators.")
            else:
                st.success("✅ **Low Risk**: This company appears financially stable.")
            
            # Feature importance (if available)
            feature_importance = selected_model.get_feature_importance()
            if feature_importance:
                st.subheader("Feature Importance")
                
                # Create feature importance plot
                importance_df = pd.DataFrame(
                    list(feature_importance.items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=True)
                
                fig = px.bar(
                    importance_df.tail(10),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 10 Most Important Features",
                    color='Importance',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif prediction_mode == "Batch Analysis":
    st.header("Batch Analysis")
    
    # Upload CSV file
    uploaded_file = st.file_uploader(
        "Upload CSV file with company data",
        type=['csv'],
        help="CSV should contain financial ratios for multiple companies"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df_uploaded = pd.read_csv(uploaded_file)
            
            st.subheader("Uploaded Data Preview")
            st.dataframe(df_uploaded.head())
            
            # Check if required columns are present
            missing_columns = [col for col in feature_columns if col not in df_uploaded.columns]
            
            if missing_columns:
                st.warning(f"Missing columns: {missing_columns}")
                st.info("Using median values from training data for missing columns.")
                
                # Fill missing columns with median values
                for col in missing_columns:
                    df_uploaded[col] = df_sample[col].median()
            
            # Make predictions
            if st.button("Run Batch Predictions", type="primary"):
                with st.spinner("Making predictions..."):
                    # Get probabilities
                    probabilities = selected_model.predict_proba(df_uploaded[feature_columns])[:, 1]
                    predictions = selected_model.predict(df_uploaded[feature_columns])
                    
                    # Add results to dataframe
                    df_results = df_uploaded.copy()
                    df_results['bankruptcy_probability'] = probabilities
                    df_results['prediction'] = ['Bankrupt' if p == 1 else 'Non-Bankrupt' for p in predictions]
                    df_results['risk_level'] = [
                        'High' if p > 0.7 else 'Medium' if p > 0.3 else 'Low' 
                        for p in probabilities
                    ]
                    
                    st.subheader("Prediction Results")
                    st.dataframe(df_results)
                    
                    # Download results
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="Download Results",
                        data=csv,
                        file_name="bankruptcy_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Summary statistics
                    st.subheader("Summary Statistics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Companies", len(df_results))
                    
                    with col2:
                        high_risk_count = sum(1 for p in probabilities if p > 0.7)
                        st.metric("High Risk Companies", high_risk_count)
                    
                    with col3:
                        avg_prob = np.mean(probabilities)
                        st.metric("Average Risk", f"{avg_prob:.1%}")
                    
                    # Distribution plot
                    fig = px.histogram(
                        df_results,
                        x='bankruptcy_probability',
                        nbins=20,
                        title="Distribution of Bankruptcy Probabilities",
                        labels={'bankruptcy_probability': 'Bankruptcy Probability', 'count': 'Number of Companies'}
                    )
                    fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="50% Threshold")
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

elif prediction_mode == "Model Comparison":
    st.header("Model Comparison")
    
    if len(models) > 1:
        st.subheader("Model Performance Comparison")
        
        # Load metrics if available
        metrics_file = models_dir / "metrics_summary.csv"
        if metrics_file.exists():
            metrics_df = pd.read_csv(metrics_file, index_col=0)
            
            # Display metrics table
            st.dataframe(metrics_df.round(4))
            
            # Performance comparison plots
            col1, col2 = st.columns(2)
            
            with col1:
                # AUC comparison
                if 'auc_roc' in metrics_df.columns:
                    fig_auc = px.bar(
                        x=metrics_df.index,
                        y=metrics_df['auc_roc'],
                        title="AUC-ROC Comparison",
                        labels={'x': 'Model', 'y': 'AUC-ROC'}
                    )
                    fig_auc.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_auc, use_container_width=True)
            
            with col2:
                # F1 Score comparison
                if 'f1_score' in metrics_df.columns:
                    fig_f1 = px.bar(
                        x=metrics_df.index,
                        y=metrics_df['f1_score'],
                        title="F1 Score Comparison",
                        labels={'x': 'Model', 'y': 'F1 Score'}
                    )
                    fig_f1.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_f1, use_container_width=True)
            
            # Feature importance comparison
            st.subheader("Feature Importance Comparison")
            
            # Get feature importance from different models
            importance_data = []
            for model_name, model in models.items():
                importance = model.get_feature_importance()
                if importance:
                    for feature, score in importance.items():
                        importance_data.append({
                            'Model': model_name,
                            'Feature': feature,
                            'Importance': score
                        })
            
            if importance_data:
                importance_df = pd.DataFrame(importance_data)
                
                # Top features across all models
                top_features = importance_df.groupby('Feature')['Importance'].mean().sort_values(ascending=False).head(10).index
                
                fig_importance = px.bar(
                    importance_df[importance_df['Feature'].isin(top_features)],
                    x='Feature',
                    y='Importance',
                    color='Model',
                    title="Feature Importance Comparison (Top 10 Features)",
                    barmode='group'
                )
                fig_importance.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_importance, use_container_width=True)
        
        else:
            st.warning("No metrics file found. Please run the training script to generate model comparison data.")
    
    else:
        st.warning("Only one model available. Train multiple models to enable comparison.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>Bankruptcy Prediction Model - Research Demonstration Only</p>
    <p>⚠️ This tool is for educational purposes only and should not be used for investment decisions</p>
</div>
""", unsafe_allow_html=True)
