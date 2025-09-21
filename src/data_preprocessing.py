"""
Data Preprocessing Utilities for PAH Removal Analysis
Author: [Your Name]
Date: 2025
Description: Comprehensive data preprocessing, cleaning, and transformation utilities
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing for PAH removal analysis.
    """
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.target_name = None
        self.preprocessing_params = {}
        
    def load_data(self, filepath, sheet_name=None, sep=','):
        """
        Load data from various file formats.
        
        Args:
            filepath: Path to data file
            sheet_name: Sheet name for Excel files
            sep: Separator for CSV files
            
        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading data from {filepath}")
        
        if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
            df = pd.read_excel(filepath, sheet_name=sheet_name)
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath, sep=sep)
        elif filepath.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def clean_data(self, df, remove_duplicates=True, handle_missing='drop',
                  missing_threshold=0.5):
        """
        Clean the dataset.
        
        Args:
            df: Input DataFrame
            remove_duplicates: Whether to remove duplicate rows
            handle_missing: How to handle missing values ('drop', 'impute_mean', 
                          'impute_median', 'impute_knn')
            missing_threshold: Drop columns with more than this fraction missing
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data...")
        df_clean = df.copy()
        
        # Remove duplicates
        if remove_duplicates:
            initial_rows = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            logger.info(f"Removed {initial_rows - len(df_clean)} duplicate rows")
        
        # Handle missing values
        missing_cols = df_clean.columns[df_clean.isnull().mean() > missing_threshold]
        if len(missing_cols) > 0:
            logger.info(f"Dropping {len(missing_cols)} columns with >{missing_threshold*100}% missing")
            df_clean = df_clean.drop(columns=missing_cols)
        
        if handle_missing == 'drop':
            initial_rows = len(df_clean)
            df_clean = df_clean.dropna()
            logger.info(f"Dropped {initial_rows - len(df_clean)} rows with missing values")
        
        elif handle_missing.startswith('impute'):
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            
            if handle_missing == 'impute_mean':
                imputer = SimpleImputer(strategy='mean')
            elif handle_missing == 'impute_median':
                imputer = SimpleImputer(strategy='median')
            elif handle_missing == 'impute_knn':
                imputer = KNNImputer(n_neighbors=5)
            
            df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
            logger.info(f"Imputed missing values using {handle_missing}")
        
        return df_clean
    
    def encode_categorical(self, df, categorical_columns=None, encoding_type='onehot'):
        """
        Encode categorical variables.
        
        Args:
            df: Input DataFrame
            categorical_columns: List of categorical columns (auto-detect if None)
            encoding_type: 'onehot' or 'label'
            
        Returns:
            DataFrame with encoded variables
        """
        df_encoded = df.copy()
        
        # Auto-detect categorical columns if not specified
        if categorical_columns is None:
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        if len(categorical_columns) == 0:
            logger.info("No categorical columns to encode")
            return df_encoded
        
        logger.info(f"Encoding {len(categorical_columns)} categorical columns")
        
        if encoding_type == 'onehot':
            # One-hot encoding
            for col in categorical_columns:
                encoder = OneHotEncoder(sparse_output=False, drop='first')
                encoded = encoder.fit_transform(df_encoded[[col]])
                
                # Create column names
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
                
                # Create DataFrame with encoded columns
                encoded_df = pd.DataFrame(encoded, columns=feature_names, 
                                         index=df_encoded.index)
                
                # Drop original column and concat encoded
                df_encoded = pd.concat([df_encoded.drop(col, axis=1), encoded_df], axis=1)
                
                # Store encoder
                self.encoders[col] = encoder
        
        elif encoding_type == 'label':
            # Label encoding
            for col in categorical_columns:
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df_encoded[col])
                self.encoders[col] = encoder
        
        return df_encoded
    
    def handle_outliers(self, df, columns=None, method='iqr', threshold=1.5):
        """
        Handle outliers in the dataset.
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers (None for all numeric)
            method: 'iqr', 'zscore', or 'isolation_forest'
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers handled
        """
        df_clean = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Handling outliers using {method} method")
        outliers_count = 0
        
        if method == 'iqr':
            for col in columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                outliers_count += outliers.sum()
                
                # Cap outliers instead of removing
                df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
        
        elif method == 'zscore':
            from scipy import stats
            for col in columns:
                z_scores = np.abs(stats.zscore(df_clean[col]))
                outliers = z_scores > threshold
                outliers_count += outliers.sum()
                
                # Remove outliers
                df_clean = df_clean[z_scores <= threshold]
        
        elif method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            iso = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso.fit_predict(df_clean[columns]) == -1
            outliers_count = outliers.sum()
            df_clean = df_clean[~outliers]
        
        logger.info(f"Handled {outliers_count} outlier values")
        return df_clean
    
    def scale_features(self, X, method='standard', feature_range=(0, 1)):
        """
        Scale features using various methods.
        
        Args:
            X: Feature matrix
            method: 'standard', 'minmax', or 'robust'
            feature_range: Range for minmax scaling
            
        Returns:
            Scaled features
        """
        logger.info(f"Scaling features using {method} scaler")
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler(feature_range=feature_range)
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        X_scaled = scaler.fit_transform(X)
        self.scalers['feature_scaler'] = scaler
        
        return X_scaled
    
    def create_polynomial_features(self, df, columns, degree=2, include_bias=False):
        """
        Create polynomial features.
        
        Args:
            df: Input DataFrame
            columns: Columns to create polynomial features from
            degree: Polynomial degree
            include_bias: Whether to include bias term
            
        Returns:
            DataFrame with polynomial features
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        logger.info(f"Creating polynomial features of degree {degree}")
        
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        poly_features = poly.fit_transform(df[columns])
        
        # Get feature names
        poly_names = poly.get_feature_names_out(columns)
        
        # Create DataFrame
        poly_df = pd.DataFrame(poly_features, columns=poly_names, index=df.index)
        
        # Combine with original DataFrame (excluding original columns)
        df_result = pd.concat([df.drop(columns, axis=1), poly_df], axis=1)
        
        return df_result
    
    def select_features(self, X, y, method='mutual_info', k=10):
        """
        Select top k features.
        
        Args:
            X: Feature matrix
            y: Target values
            method: 'mutual_info', 'f_regression', or 'rfe'
            k: Number of features to select
            
        Returns:
            Selected features and selector object
        """
        logger.info(f"Selecting top {k} features using {method}")
        
        if method == 'mutual_info':
            selector = SelectKBest(mutual_info_regression, k=k)
        elif method == 'f_regression':
            selector = SelectKBest(f_regression, k=k)
        elif method == 'rfe':
            from sklearn.feature_selection import RFE
            from sklearn.linear_model import LinearRegression
            estimator = LinearRegression()
            selector = RFE(estimator, n_features_to_select=k)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature indices
        selected_indices = selector.get_support(indices=True)
        
        return X_selected, selector, selected_indices
    
    def create_interaction_features(self, df, columns):
        """
        Create interaction features between specified columns.
        
        Args:
            df: Input DataFrame
            columns: List of columns to create interactions for
            
        Returns:
            DataFrame with interaction features
        """
        logger.info(f"Creating interaction features for {len(columns)} columns")
        
        df_result = df.copy()
        
        # Create pairwise interactions
        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                col1, col2 = columns[i], columns[j]
                interaction_name = f"{col1}_x_{col2}"
                df_result[interaction_name] = df[col1] * df[col2]
        
        return df_result
    
    def prepare_for_modeling(self, df, target_column, test_size=0.2, 
                            validation_size=0.1, random_state=42,
                            scale_features=True, encoding_type='onehot'):
        """
        Complete preprocessing pipeline for modeling.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            test_size: Proportion for test set
            validation_size: Proportion for validation set
            random_state: Random seed
            scale_features: Whether to scale features
            encoding_type: Type of categorical encoding
            
        Returns:
            Dictionary with train, validation, and test sets
        """
        logger.info("Preparing data for modeling...")
        
        # Separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Store feature and target names
        self.feature_names = X.columns.tolist()
        self.target_name = target_column
        
        # Encode categorical variables
        X = self.encode_categorical(X, encoding_type=encoding_type)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        if validation_size > 0:
            val_size_adjusted = validation_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
            )
        else:
            X_train, X_val, y_train, y_val = X_temp, None, y_temp, None
        
        # Scale features if requested
        if scale_features:
            X_train = self.scale_features(X_train)
            X_test = self.scalers['feature_scaler'].transform(X_test)
            if X_val is not None:
                X_val = self.scalers['feature_scaler'].transform(X_val)
        
        # Store preprocessing parameters
        self.preprocessing_params = {
            'test_size': test_size,
            'validation_size': validation_size,
            'random_state': random_state,
            'scaled': scale_features,
            'encoding': encoding_type,
            'n_features': X_train.shape[1],
            'n_samples_train': X_train.shape[0],
            'n_samples_test': X_test.shape[0],
            'n_samples_val': X_val.shape[0] if X_val is not None else 0
        }
        
        logger.info(f"Data split - Train: {X_train.shape[0]}, "
                   f"Val: {X_val.shape[0] if X_val is not None else 0}, "
                   f"Test: {X_test.shape[0]}")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'preprocessing_params': self.preprocessing_params
        }
    
    def save_preprocessor(self, filepath):
        """
        Save preprocessing objects.
        
        Args:
            filepath: Path to save preprocessor
        """
        import joblib
        
        preprocessor_dict = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'preprocessing_params': self.preprocessing_params
        }
        
        joblib.dump(preprocessor_dict, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath):
        """
        Load preprocessing objects.
        
        Args:
            filepath: Path to load preprocessor from
        """
        import joblib
        
        preprocessor_dict = joblib.load(filepath)
        self.scalers = preprocessor_dict['scalers']
        self.encoders = preprocessor_dict['encoders']
        self.feature_names = preprocessor_dict['feature_names']
        self.target_name = preprocessor_dict['target_name']
        self.preprocessing_params = preprocessor_dict['preprocessing_params']
        
        logger.info(f"Preprocessor loaded from {filepath}")


def create_train_test_splits(df, target_column, n_splits=5, test_size=0.2):
    """
    Create multiple train-test splits for cross-validation.
    
    Args:
        df: Input DataFrame
        target_column: Target column name
        n_splits: Number of splits
        test_size: Test set size
        
    Returns:
        List of (train_data, test_data) tuples
    """
    splits = []
    
    for i in range(n_splits):
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42+i
        )
        splits.append((train_df, test_df))
    
    return splits


def main():
    """Main execution function for data preprocessing."""
    logger.info("Starting data preprocessing...")
    
    # Example usage
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'PAH_Compound': np.random.choice(['Naphthalene', 'Anthracene', 'Pyrene'], n_samples),
        'EDTA_Concentration': np.random.uniform(10, 20, n_samples),
        'Fe_Concentration': np.random.uniform(0.5, 2, n_samples),
        'H2O2_Concentration': np.random.uniform(5, 20, n_samples),
        'Time': np.random.uniform(10, 30, n_samples),
        'Temperature': np.random.uniform(20, 40, n_samples),
        'pH': np.random.uniform(6, 8, n_samples),
        'Removal_Rate': np.random.uniform(70, 99, n_samples)
    })
    
    # Add some missing values
    df.loc[np.random.choice(df.index, 50), 'Temperature'] = np.nan
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Clean data
    df_clean = preprocessor.clean_data(df, handle_missing='impute_mean')
    
    # Handle outliers
    df_clean = preprocessor.handle_outliers(df_clean)
    
    # Prepare for modeling
    data_dict = preprocessor.prepare_for_modeling(
        df_clean, 
        target_column='Removal_Rate',
        test_size=0.2,
        validation_size=0.1
    )
    
    # Save preprocessor
    preprocessor.save_preprocessor('results/preprocessor.joblib')
    
    logger.info("Data preprocessing completed!")
    
    return data_dict


if __name__ == "__main__":
    main()