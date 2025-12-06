"""
Classification module for machine failure prediction
Handles model loading and prediction logic
"""

import joblib
import pandas as pd
from typing import Dict, Optional


# Mapping for Type ordinal encoding
TYPE_MAP = {'L': 0, 'M': 1, 'H': 2}

# Label mapping for predictions
LABEL_MAP = {
    0: 'No Failure',
    1: 'Heat Dissipation Failure',
    2: 'Power Failure',
    3: 'Overstrain Failure',
    4: 'Tool Wear Failure',
    5: 'Random Failures'
}


class MachineClassifier:
    """
    Machine failure classifier using XGBoost model
    """
    
    def __init__(self, model_path: str):
        """
        Initialize classifier with model file
        
        Args:
            model_path: Path to the trained model file (.h5)
        """
        self.model_path = model_path
        self.model = None
        self.expected_features = ['Air temperature', 'Process temperature', 
                                  'Rotational speed', 'Torque', 'Tool wear', 'Type']
        self.load_model()
    
    def load_model(self):
        """Load the trained model from file"""
        try:
            self.model = joblib.load(self.model_path)
            
            # Try to get feature names from XGBoost model
            try:
                booster = self.model.get_booster()
                model_feature_names = booster.feature_names
                if model_feature_names:
                    self.expected_features = list(model_feature_names)
                    print(f'✅ Model loaded with features: {self.expected_features}')
            except Exception:
                print(f'⚠️ Could not read feature names from model, using defaults')
            
            print(f'✅ Classifier model loaded from: {self.model_path}')
            
        except Exception as e:
            print(f'❌ Error loading model: {e}')
            self.model = None
    
    def is_loaded(self) -> bool:
        """Check if model is successfully loaded"""
        return self.model is not None
    
    def predict(
        self,
        air_temperature: float,
        process_temperature: float,
        rotational_speed: float,
        torque: float,
        tool_wear: float,
        machine_type: str
    ) -> Dict:
        """
        Predict machine failure status
        
        Args:
            air_temperature: Air temperature in Kelvin
            process_temperature: Process temperature in Kelvin
            rotational_speed: Rotational speed in rpm
            torque: Torque in Nm
            tool_wear: Tool wear in minutes
            machine_type: Machine type ('L', 'M', or 'H')
        
        Returns:
            Dictionary with prediction results:
            {
                'prediction_numeric': int,
                'prediction_label': str,
                'probabilities': list (optional)
            }
        
        Raises:
            ValueError: If model is not loaded or input is invalid
        """
        if not self.is_loaded():
            raise ValueError('Model not loaded')
        
        # Create DataFrame with input data
        df = pd.DataFrame([{
            'Air temperature': air_temperature,
            'Process temperature': process_temperature,
            'Rotational speed': rotational_speed,
            'Torque': torque,
            'Tool wear': tool_wear,
            'Type': machine_type
        }])
        
        # Map Type to numeric
        if df['Type'].dtype == object:
            df['Type'] = df['Type'].map(TYPE_MAP)
        
        # Validate all expected features are present
        missing = set(self.expected_features) - set(df.columns)
        if missing:
            raise ValueError(f'Missing features: {missing}')
        
        # Reorder columns to match model's expected order
        df = df[self.expected_features]
        
        # Convert to float
        df = df.astype(float)
        
        # Make prediction
        pred = self.model.predict(df)
        pred_numeric = int(pred[0])
        pred_label = LABEL_MAP.get(pred_numeric, str(pred_numeric))
        
        result = {
            'prediction_numeric': pred_numeric,
            'prediction_label': pred_label
        }
        
        # Add probabilities if available
        if hasattr(self.model, 'predict_proba'):
            try:
                probs = self.model.predict_proba(df)[0].tolist()
                result['probabilities'] = probs
            except Exception:
                result['probabilities'] = None
        
        return result


# Singleton instance (will be initialized in app.py)
classifier: Optional[MachineClassifier] = None


def initialize_classifier(model_path: str) -> MachineClassifier:
    """
    Initialize the global classifier instance
    
    Args:
        model_path: Path to model file
    
    Returns:
        MachineClassifier instance
    """
    global classifier
    classifier = MachineClassifier(model_path)
    return classifier


def get_classifier() -> Optional[MachineClassifier]:
    """Get the global classifier instance"""
    return classifier
