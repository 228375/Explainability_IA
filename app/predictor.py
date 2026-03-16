"""
HR Analytics Predictor Module

This module loads the pre-trained Random Forest model, SHAP explainer,
and HR dataset to provide employee turnover predictions with explanations.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd


# Define paths relative to the project root
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "rf_model.joblib"
EXPLAINER_PATH = PROJECT_ROOT / "shap_explainer.joblib"
LE_SOURCE_PATH = PROJECT_ROOT / "le_source.joblib"
LE_POS_PATH = PROJECT_ROOT / "le_pos.joblib"
DATA_PATH = PROJECT_ROOT / "HR_anonymized.csv"

# Feature columns used by the model (11 features)
FEATURE_ORDER = [
    'Salary', 'EngagementSurvey', 'EmpSatisfaction', 'SpecialProjectsCount',
    'Absences', 'DaysLateLast30', 'DeptID', 'PerfScoreID',
    'Tenure', 'RecruitmentSource_Enc', 'Position_Enc'
]


class HRPredictor:
    """
    HR Analytics Predictor class that handles employee turnover predictions
    using a pre-trained Random Forest model and SHAP explainer.
    """
    
    def __init__(self):
        """Initialize the predictor by loading model, explainer, and dataset."""
        self.model = None
        self.explainer = None
        self.le_source = None
        self.le_pos = None
        self.data = None
        self._load_components()
    
    def _load_components(self) -> None:
        """Load the model, explainer, label encoders, and dataset from disk."""
        # Load the Random Forest model
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
        self.model = joblib.load(MODEL_PATH)
        
        # Load the SHAP explainer
        if not EXPLAINER_PATH.exists():
            raise FileNotFoundError(f"Explainer not found at: {EXPLAINER_PATH}")
        self.explainer = joblib.load(EXPLAINER_PATH)
        
        # Load label encoders
        if not LE_SOURCE_PATH.exists():
            raise FileNotFoundError(f"Label encoder not found at: {LE_SOURCE_PATH}")
        self.le_source = joblib.load(LE_SOURCE_PATH)
        
        if not LE_POS_PATH.exists():
            raise FileNotFoundError(f"Label encoder not found at: {LE_POS_PATH}")
        self.le_pos = joblib.load(LE_POS_PATH)
        
        # Load the HR dataset
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")
        self.data = pd.read_csv(DATA_PATH)
    
    def prepare_employee_data(self, index: int) -> pd.DataFrame:
        """
        Prepare the feature data for a specific employee by index.
        
        Args:
            index: The row index of the employee in the dataset
            
        Returns:
            DataFrame with the prepared features for the model
        """
        if index < 0 or index >= len(self.data):
            raise IndexError(f"Employee index {index} out of range")
        
        emp_row = self.data.iloc[index].copy()
        
        # Calculate Tenure
        hire_date = pd.to_datetime(emp_row['DateofHire'])
        tenure = (pd.to_datetime('2020-01-01') - hire_date).days / 365.25
        
        # Encode categorical variables
        try:
            source_enc = self.le_source.transform([emp_row['RecruitmentSource']])[0]
        except:
            source_enc = 0
            
        try:
            pos_enc = self.le_pos.transform([emp_row['Position']])[0]
        except:
            pos_enc = 0
        
        # Prepare input data
        input_data = pd.DataFrame([{
            'Salary': emp_row['Salary'],
            'EngagementSurvey': emp_row['EngagementSurvey'],
            'EmpSatisfaction': emp_row['EmpSatisfaction'],
            'SpecialProjectsCount': emp_row['SpecialProjectsCount'],
            'Absences': emp_row['Absences'],
            'DaysLateLast30': emp_row['DaysLateLast30'],
            'DeptID': emp_row['DeptID'],
            'PerfScoreID': emp_row['PerfScoreID'],
            'Tenure': tenure,
            'RecruitmentSource_Enc': source_enc,
            'Position_Enc': pos_enc
        }])[FEATURE_ORDER]
        
        return input_data
    
    def predict(self, index: int) -> Dict[str, Any]:
        """
        Get a complete prediction with SHAP explanations for an employee.
        
        Args:
            index: The row index of the employee in the dataset
            
        Returns:
            Dictionary containing prediction, probability, and SHAP factors
        """
        # Prepare employee data
        employee_data = self.prepare_employee_data(index)
        emp_row = self.data.iloc[index].copy()
        
        # Get prediction probability (class 1 = leaving)
        prob = self.model.predict_proba(employee_data)[0][1]
        
        # Get SHAP values
        shap_result = self.explainer(employee_data)
        
        # Handle different SHAP value shapes
        if len(shap_result.values.shape) == 3:
            # Shape: (1, n_features, n_classes)
            shap_values = shap_result.values[0, :, 1]
        else:
            # Shape: (1, n_features)
            shap_values = shap_result.values[0]
        
        # Create feature importance dictionary
        feature_importance = dict(zip(FEATURE_ORDER, shap_values))
        
        # Separate positive and negative factors
        positive_factors = []
        negative_factors = []
        
        for feature, value in feature_importance.items():
            # Get the actual feature value from model input
            actual_value = employee_data[feature].values[0]
            
            # Convert numpy types to native Python types for JSON serialization
            if hasattr(actual_value, 'item'):
                actual_value = actual_value.item()
            
            # Get original values from emp_row for display
            display_value = actual_value  # Default to actual_value
            
            if feature == 'RecruitmentSource_Enc':
                display_value = str(emp_row['RecruitmentSource'])
            elif feature == 'Position_Enc':
                display_value = str(emp_row['Position'])
            elif feature == 'DeptID':
                # Map DeptID to department names
                dept_map = {1: 'Admin Offices', 2: 'IT/IS', 3: 'Sales', 4: 'Software Engineering', 5: 'Production'}
                display_value = dept_map.get(int(actual_value), f"Dept {actual_value}")
            elif feature == 'PerfScoreID':
                # Map PerfScoreID to performance names
                perf_map = {1: 'Needs Improvement', 2: 'PIP', 3: 'Fully Meets', 4: 'Exceeds', 5: 'Exceptional'}
                display_value = perf_map.get(int(actual_value), f"Score {actual_value}")
            
            factor = {
                "feature": feature,
                "value": actual_value,
                "display_value": display_value,
                "shap_value": float(value)
            }
            
            if value > 0:
                positive_factors.append(factor)
            else:
                negative_factors.append(factor)
        
        # Sort by absolute value and get top factors
        positive_factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        negative_factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        
        # Determine prediction label
        prediction = "High Risk" if prob > 0.5 else "Low Risk"
        
        # Calculate tenure for response
        hire_date = pd.to_datetime(emp_row['DateofHire'])
        tenure = (pd.to_datetime('2020-01-01') - hire_date).days / 365.25
        
        return {
            "employee_index": index,
            "prediction": prediction,
            "probability": float(prob),
            "top_positive_factors": positive_factors[:5],
            "top_negative_factors": negative_factors[:5],
            "details": {
                "position": emp_row['Position'],
                "tenure": round(tenure, 1),
                "satisfaction": emp_row['EmpSatisfaction']
            }
        }
    
    def get_employee_count(self) -> int:
        """Return the total number of employees in the dataset."""
        return len(self.data)


# Global predictor instance (lazy loaded)
_predictor: Optional[HRPredictor] = None


def get_predictor() -> HRPredictor:
    """
    Get the global predictor instance (singleton pattern).
    
    Returns:
        The HRPredictor instance
    """
    global _predictor
    if _predictor is None:
        _predictor = HRPredictor()
    return _predictor


def get_employee_prediction(index: int) -> Dict[str, Any]:
    """
    Convenience function to get prediction for a specific employee.
    
    Args:
        index: The row index of the employee in the dataset
        
    Returns:
        Dictionary containing prediction and SHAP factors
    """
    predictor = get_predictor()
    return predictor.predict(index)


def get_prediction_for_employee_id(employee_id: str) -> Dict[str, Any]:
    """
    Get prediction for an employee by their ID string.
    
    Args:
        employee_id: String identifier for the employee
        
    Returns:
        Dictionary containing prediction and SHAP factors
    """
    try:
        index = int(employee_id)
    except ValueError:
        raise ValueError(f"Invalid employee_id: {employee_id}. Must be a numeric string.")
    
    return get_employee_prediction(index)
