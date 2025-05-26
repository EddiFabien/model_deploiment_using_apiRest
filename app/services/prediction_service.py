"""
Service for making predictions using the trained model.
"""
import logging
import pandas as pd
from pathlib import Path
import os
from typing import Dict, Any, List, Optional
from pycaret.classification import load_model, predict_model

logger = logging.getLogger(__name__)

class PredictionService:
    """Service for making predictions using the trained model."""
    
    def __init__(self, model_path: str):
        """Initialize the prediction service with the path to the model.
        
        Args:
            model_path (str): Path to the trained model file.
        """
        self.model_path = model_path
        self.model = None
        self._model_loaded = False
        self._load_model()
        
    @property
    def is_model_loaded(self) -> bool:
        """Check if the model is loaded.
        
        Returns:
            bool: True if the model is loaded, False otherwise.
        """
        return self._model_loaded
    

    def _load_model(self):
        """Load the model from disk."""
        try:
            # Handle PyCaret's automatic .pkl appending
            model_path = self.model_path
            
            # If the path doesn't exist, try removing .pkl if it's at the end
            if not os.path.exists(model_path) and model_path.endswith('.pkl'):
                model_path = model_path[:-4]  # Remove .pkl
            
            logger.info(f"Attempting to load model from: {model_path}")
            logger.info(f"Current working directory: {os.getcwd()}")
            
            # Check if file exists
            if not os.path.exists(model_path):
                # Try to find the model file in the same directory
                dir_path = os.path.dirname(model_path) or '.'
                files = os.listdir(dir_path)
                logger.error(f"Available files in {os.path.abspath(dir_path)}: {files}")
                raise FileNotFoundError(
                    f"Model file not found at {os.path.abspath(model_path)}. "
                    f"Available files: {files}"
                )
            
            # Load the model
            logger.info("Loading model...")
            
            # Load the model without the .pkl extension to prevent double extension
            self.model = load_model(model_path.replace('.pkl', ''))
            
            # Verify model has required attributes
            if not hasattr(self.model, 'predict'):
                raise AttributeError("Loaded model does not have a predict method")
                
            self._model_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            self.model = None
            self._model_loaded = False
            raise
    
    def _map_input_to_model_fields(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map API input fields to model's expected fields.
        
        Args:
            input_data: Dictionary containing the input features from the API
            
        Returns:
            Dictionary with field names mapped to the model's expected format
        """
        # Complete mapping of all model features with their default values
        all_features = {
            # Basic information
            'sexe': 1.0,  # Default to male (1) if not provided
            'nb frère': 0.0,
            'nb sœur': 0.0,
            "commune d'origine": 0.0,
            'habite avec les parents': 1.0,  # Default to living with parents
            'electricite': 1.0,  # Default to having electricity
            'conn sur les options': 1.0,  # Default to knowing the options
            
            # Academic grades
            'MLG': 0.0,
            'FRS': 0.0,
            'ANG': 0.0,
            'HG': 0.0,
            'SES': 0.0,
            'MATHS': 0.0,
            'PC': 0.0,
            'SVT': 0.0,
            'EPS': 0.0,
            '1°S': 0.0,  # prim_S
            '2°S': 0.0,  # sec_S
            'MOY AN': 0.0,  # MOY_AN
            
            # Additional fields required by the model
            'N°': 1.0,  # Student number
            'Nom et Prénoms': '1',  # Student name
            'prof père': '1',  # Father's profession
            'prof mère': '1',  # Mother's profession
            'RANG': 1.0  # Rank
        }
        
        # API field to model field mapping
        field_mapping = {
            'sexe': 'sexe',
            'nb_frere': 'nb frère',
            'nb_soeur': 'nb sœur',
            'commune_d_origine': "commune d'origine",
            'habite_avec_les_parents': 'habite avec les parents',
            'electricite': 'electricite',
            'conn_sur_les_options': 'conn sur les options',
            'MLG': 'MLG',
            'FRS': 'FRS',
            'ANG': 'ANG',
            'HG': 'HG',
            'SES': 'SES',
            'MATHS': 'MATHS',
            'PC': 'PC',
            'SVT': 'SVT',
            'EPS': 'EPS',
            'prim_S': '1°S',
            'sec_S': '2°S',
            'MOY_AN': 'MOY AN'
        }
        
        # Start with default values for all features
        mapped_data = all_features.copy()
        
        # Update with values from input data
        for api_field, model_field in field_mapping.items():
            if api_field in input_data and input_data[api_field] is not None:
                value = input_data[api_field]
                # Convert string numbers to appropriate numeric types
                if isinstance(value, str):
                    if value.replace('.', '', 1).isdigit():
                        value = float(value) if '.' in value else int(value)
                mapped_data[model_field] = value
                
        return mapped_data

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction using the loaded model.
        
        Args:
            input_data: Dictionary containing the input features for prediction
            
        Returns:
            Dictionary containing the prediction result with required fields:
            - prediction (str): The predicted class
            - confidence_scores (Dict[str, float]): Confidence scores for each class
        """
        try:
            # Map input fields to model's expected format
            mapped_input = self._map_input_to_model_fields(input_data)
            
            # Convert input data to DataFrame
            input_df = pd.DataFrame([mapped_input])
            
            # Make prediction
            logger.info(f"Input DataFrame columns: {input_df.columns.tolist()}")
            logger.info(f"Input DataFrame head (transposed):\n{input_df.head().T}")
            
            # Make prediction
            prediction = predict_model(self.model, data=input_df)
            
            # Debug: Print available columns in the prediction output
            logger.info(f"Prediction columns: {prediction.columns.tolist()}")
            logger.info(f"Prediction data (transposed):\n{prediction.head().T}")
            
            # Get the raw prediction label (might be numeric or string)
            raw_label = str(prediction['prediction_label'].iloc[0])
            
            # Map numeric labels to class names
            label_map = {
                '0': 'L',
                '1': 'S',
                '2': 'OSE',
                'L': 'L',
                'S': 'S',
                'O': 'OSE',
                'OSE': 'OSE'
            }
            
            # Get the mapped prediction label
            prediction_label = label_map.get(raw_label, 'OSE')
            
            # Initialize confidence scores
            confidence_scores = {'L': 0.0, 'S': 0.0, 'OSE': 0.0}
            
            # Try to get probability scores for each class
            # PyCaret might store them in different formats
            proba_columns = {
                'L': ['prediction_score_L', 'score_L', 'L', '0'],
                'S': ['prediction_score_S', 'score_S', 'S', '1'],
                'OSE': ['prediction_score_OSE', 'score_OSE', 'O', 'OSE', '2']
            }
            
            # First, check for probability columns
            proba_found = False
            
            # Look for probability columns in the prediction output
            for col in prediction.columns:
                if 'score_' in col.lower() or 'prob_' in col.lower():
                    # Try to extract class from column name
                    for cls in ['L', 'S', 'OSE']:
                        if cls in col.upper():
                            confidence_scores[cls] = float(prediction[col].iloc[0])
                            proba_found = True
                            break
            
            # If we found probability columns, use them directly
            if proba_found:
                # Ensure all classes have values (some might be missing)
                for cls in confidence_scores:
                    if confidence_scores[cls] == 0.0:
                        # If a class has 0 probability, set it to a small value
                        confidence_scores[cls] = 0.01
                
                # Normalize to ensure sum is 1.0
                total = sum(confidence_scores.values())
                if total > 0:
                    confidence_scores = {k: v/total for k, v in confidence_scores.items()}
            else:
                # Fallback: Use prediction score and distribute remaining probability
                # based on class distribution in the training data
                # These are example distributions - adjust based on your actual data
                class_distribution = {
                    'L': 0.3,  # 30% of training data is class L
                    'S': 0.4,  # 40% of training data is class S
                    'OSE': 0.3  # 30% of training data is class OSE
                }
                
                if 'prediction_score' in prediction.columns:
                    score = float(prediction['prediction_score'].iloc[0])
                    confidence_scores[prediction_label] = score
                    
                    # Distribute remaining probability according to class distribution
                    remaining = 1.0 - score
                    total_dist = sum(v for k, v in class_distribution.items() 
                                  if k != prediction_label)
                    
                    if total_dist > 0:
                        for cls in confidence_scores:
                            if cls != prediction_label:
                                confidence_scores[cls] = (class_distribution[cls] / total_dist) * remaining
                    else:
                        # If all other classes have 0 distribution, distribute equally
                        remaining_classes = [c for c in confidence_scores if c != prediction_label]
                        if remaining_classes:
                            per_class = remaining / len(remaining_classes)
                            for cls in remaining_classes:
                                confidence_scores[cls] = per_class
                else:
                    # Last resort: use class distribution directly
                    confidence_scores = class_distribution.copy()
            
            # Get the final prediction label (should match prediction_label)
            mapped_label = max(confidence_scores, key=confidence_scores.get)
            
            # Log the final confidence scores for debugging
            logger.info(f"Final confidence scores: {confidence_scores}")
            
            result = {
                'prediction': mapped_label,  # Use the mapped label
                'confidence_scores': confidence_scores,
                'status': 'success'
            }
            
            logger.info(f"Prediction made: {result}")
            return result
            
        except Exception as e:
            error_msg = f"Error making prediction: {str(e)}"
            logger.error(error_msg)
            # Return a valid response format even in case of error
            return {
                'prediction': 'error',
                'confidence_scores': {'L': 0.0, 'S': 0.0, 'OSE': 1.0},
                'status': 'error',
                'error': error_msg
            }
    
    def get_model_features(self) -> Dict[str, List[str]]:
        """
        Get the list of features expected by the model.
        
        Returns:
            Dict containing the list of features used by the model
        """
        # Define the core features used by the model (excluding metadata fields)
        model_features = [
            # Basic information
            'sexe',
            'nb frère',
            'nb sœur',
            "commune d'origine",
            'habite avec les parents',
            'electricite',
            'conn sur les options',
            
            # Academic grades
            'MLG',
            'FRS',
            'ANG',
            'HG',
            'SES',
            'MATHS',
            'PC',
            'SVT',
            'EPS',
            '1°S',  # prim_S
            '2°S',  # sec_S
            'MOY AN'  # MOY_AN
        ]
        
        # Return the model features in the expected format
        return {"features": model_features}
    
    @property
    def is_model_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.model is not None
    
    @classmethod
    def from_config(cls):
        """Create a PredictionService instance using configuration."""
        from decouple import config
        import os
        
        # Get the model path from config or use default
        model_path = config('MODEL_PATH', default='models/option_classifier')
        
        # Ensure the path is absolute
        if not os.path.isabs(model_path):
            # If it's a relative path, make it relative to the project root
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            model_path = os.path.abspath(os.path.join(project_root, model_path))
        
        # Check if file exists with or without .pkl extension
        model_path_to_use = None
        
        # Try the exact path first
        if os.path.exists(model_path):
            model_path_to_use = model_path
        # Try adding .pkl if not present
        elif not model_path.endswith('.pkl') and os.path.exists(f"{model_path}.pkl"):
            model_path_to_use = f"{model_path}.pkl"
        # Try removing .pkl if present
        elif model_path.endswith('.pkl') and os.path.exists(model_path[:-4]):
            model_path_to_use = model_path[:-4]
        
        if model_path_to_use is None:
            # List available files for debugging
            dir_path = os.path.dirname(model_path) or '.'
            try:
                files = os.listdir(dir_path)
                files_str = '\n'.join(f'  - {f}' for f in files)
            except Exception as e:
                files_str = f"Error listing directory: {str(e)}"
                
            raise FileNotFoundError(
                f"Model file not found at {model_path} (or with .pkl extension).\n"
                f"Directory contents ({os.path.abspath(dir_path)}):\n{files_str}\n"
                "Please train the model first using train_model.py"
            )
        
        logger.info(f"Using model path: {model_path_to_use}")
        return cls(model_path=model_path_to_use)

# Create a singleton instance of the prediction service
prediction_service = PredictionService.from_config()
