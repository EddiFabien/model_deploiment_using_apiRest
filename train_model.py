"""
Script to train and save the option classification model using PyCaret.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from pycaret.classification import setup, compare_models, create_model, pull, save_model
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger(__name__)

def load_data(file_path):
    """Load data from Excel file."""
    try:
        logger.info(f"Loading data from {file_path}")
        return pd.read_excel(file_path)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    """Preprocess the data with robust handling of missing values."""
    logger.info("Preprocessing data...")
    
    # Drop unnecessary columns
    columns_to_drop = [
        'NOM', 'Prénom', 'CIN', 'N° CIN', 'Téléphone', 'Email', 'Date de naissance',
        'Lieu de naissance', 'Adresse', 'Ville', 'Pays', 'Code postal', 'Nationalité',
        'N° de bac', 'Série de bac', 'Année de bac', 'Mention de bac', 'Lycée',
        'Ville du lycée', 'Académie', 'Moyenne de bac', 'Moyenne de la 1ère année',
        'Moyenne de la 2ème année', 'Moyenne de la 3ème année', 'Moyenne générale',
        'Rang', 'Année universitaire', 'Niveau d\'étude', 'Filière', 'Etablissement',
        'Ville de l\'établissement', 'Pays de l\'établissement', 'Année d\'obtention',
        'Diplôme', 'Spécialité', 'Etablissement de formation', 'Ville de formation',
        'Pays de formation', 'Année d\'obtention du diplôme', 'Mention', 'Moyenne',
        'Rang/Classement', 'Niveau d\'étude actuel', 'Etablissement actuel',
        'Ville actuelle', 'Pays actuel', 'Année en cours', 'Moyenne générale actuelle',
        'Rang/Classement actuel', 'Choix 1', 'Choix 2', 'Choix 3', 'Choix 4', 'Choix 5',
        'Choix 6', 'Choix 7', 'Choix 8', 'Choix 9', 'Choix 10', 'Moyenne de S1',
        'Moyenne de S2', 'Moyenne de S3', 'Moyenne de S4', 'Moyenne de S5', 'Moyenne de S6'
    ]
    
    # Drop specified columns
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns], errors='ignore')
    
    # Drop rows where target is missing
    data = data.dropna(subset=['Opt'])
    
    # Separate numeric and categorical columns
    numeric_cols = data.select_dtypes(include=['number']).columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    
    # Handle missing values in numeric columns
    for col in numeric_cols:
        if data[col].isna().any():
            # Fill with median for numeric columns
            data[col] = data[col].fillna(data[col].median())
    
    # Handle missing values in categorical columns (excluding target)
    for col in categorical_cols:
        if col != 'Opt' and data[col].isna().any():
            # Fill with mode for categorical columns
            data[col] = data[col].fillna(data[col].mode()[0])
    
    # Convert categorical variables to numeric codes
    for col in categorical_cols:
        if col != 'Opt':  # Don't encode target variable
            data[col] = data[col].astype('category').cat.codes
    
    # Convert target to numeric if it's not already
    if data['Opt'].dtype == 'object':
        data['Opt'] = data['Opt'].astype('category').cat.codes
    
    # Ensure all numeric columns are filled
    data = data.fillna(0)
    
    # Log basic info about the processed data
    logger.info(f"Processed data shape: {data.shape}")
    logger.info(f"Number of missing values after preprocessing: {data.isna().sum().sum()}")
    
    return data

def select_features(data, target_col='Opt', n_features=20):
    """Select top features using Recursive Feature Elimination."""
    logger.info("Selecting features...")
    
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    # Initialize the model
    model = ExtraTreesClassifier(random_state=42)
    
    # Initialize RFE
    selector = RFE(estimator=model, n_features_to_select=n_features, step=1)
    
    # Fit RFE
    selector = selector.fit(X, y)
    
    # Get selected features
    selected_features = X.columns[selector.support_].tolist()
    selected_features.append(target_col)
    
    return data[selected_features]

def train_and_evaluate_model(data, target_col='Opt'):
    """Train and evaluate the model using PyCaret."""
    logger.info("Setting up PyCaret environment...")
    
    # Setup PyCaret
    clf = setup(
        data=data,
        target=target_col,
        train_size=0.7,
        data_split_stratify=True,
        normalize=True,
        fold=5,
        session_id=1032,
        verbose=False
    )
    
    # Compare models
    logger.info("Comparing models...")
    top_models = compare_models(n_select=3, sort='Accuracy', exclude=['catboost'])
    
    # Get the best model (first one in the list)
    best_model = top_models[0]
    logger.info(f"Best model: {type(best_model).__name__}")
    
    # Tune the best model
    logger.info("Tuning the best model...")
    tuned_model = create_model(best_model)
    
    # Get model performance metrics
    metrics = pull().iloc[0].to_dict()
    
    return tuned_model, metrics

def save_model_pipeline(model, model_path):
    """Save the trained model."""
    # Remove .pkl extension if it's already in the path to prevent double extension
    if model_path.endswith('.pkl'):
        model_path = os.path.splitext(model_path)[0]
    logger.info(f"Saving model to {model_path}.pkl")
    save_model(model, model_path)

def main():
    """Main function to execute the training pipeline."""
    try:
        # Load configuration from environment variables
        from decouple import config
        
        # Configuration
        data_path = config('DATA_PATH', default='data/raw/base.xlsx')
        model_path = config('MODEL_PATH', default='models/option_classifier')
        processed_data_path = config('PROCESSED_DATA_PATH', default='data/processed/processed_data.csv')
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        
        # Load and preprocess data
        data = load_data(data_path)
        processed_data = preprocess_data(data)
        
        # Save the processed data
        logger.info(f"Saving processed data to {processed_data_path}")
        processed_data.to_csv(processed_data_path, index=False)
        
        # Select features
        selected_data = select_features(processed_data)
        
        # Train and evaluate model
        model, metrics = train_and_evaluate_model(selected_data)
        
        # Save the model
        save_model_pipeline(model, model_path)
        
        logger.info("Training completed successfully!")
        logger.info(f"Model performance metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Error during model training: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
