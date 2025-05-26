"""
Script to test the model's expected input format.
"""
import pandas as pd
from pycaret.classification import load_model, predict_model

# Load the model
print("Loading model...")
model = load_model('models/option_classifier')

# Get the model's expected features
try:
    # Get the preprocessing pipeline
    pipeline = model
    
    # Get the feature names
    if hasattr(pipeline, 'prep_pipe') and hasattr(pipeline.prep_pipe, 'feature_names_in_'):
        features = pipeline.prep_pipe.feature_names_in_
        print(f"\nModel expects {len(features)} features:")
        for i, feature in enumerate(features, 1):
            print(f"{i}. {feature}")
    else:
        print("\nCould not determine feature names from the pipeline.")
        
        # Try to get feature names from the model
        if hasattr(pipeline, 'feature_name_in_'):
            print(f"Model uses feature: {pipeline.feature_name_in_}")
        else:
            print("No feature names found in the model.")
            
except Exception as e:
    print(f"Error getting model features: {e}")

# Try to get the model's predict function signature if possible
try:
    if hasattr(model, 'predict'):
        print("\nModel has predict method")
        print(f"Predict method signature: {model.predict.__code__.co_varnames}")
except Exception as e:
    print(f"Error getting predict method info: {e}")
