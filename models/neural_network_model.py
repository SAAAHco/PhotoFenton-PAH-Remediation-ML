"""
Neural Network Model for PAH Removal Rate Prediction
Author: [Your Name]
Date: 2025
Description: Deep learning model for predicting PAH removal rates in Photo-Fenton systems
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scikeras.wrappers import KerasRegressor
import matplotlib.pyplot as plt
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PAHRemovalPredictor:
    """
    Neural Network model for predicting PAH removal rates based on treatment parameters.
    
    Attributes:
        model: Keras Sequential model
        scaler: StandardScaler for feature normalization
        encoder: OneHotEncoder for categorical variables
        history: Training history
    """
    
    def __init__(self, learning_rate=0.00005, dropout_rate=0.3, batch_size=32):
        """
        Initialize the PAH Removal Predictor.
        
        Args:
            learning_rate: Learning rate for Adam optimizer
            dropout_rate: Dropout rate for regularization
            batch_size: Batch size for training
        """
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.model = None
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False)
        self.history = None
        self.feature_names = None
        
    def preprocess_data(self, df):
        """
        Preprocess the data for training.
        
        Args:
            df: DataFrame with columns ['PAH Compound', 'EDTA Concentration', 
                'Fe Concentration', 'H2O2 Concentration', 'Time', 'Removal Rate']
        
        Returns:
            X: Preprocessed features
            y: Target values
        """
        logger.info("Preprocessing data...")
        
        # One-hot encode the categorical 'PAH Compound' column
        encoded_pah = self.encoder.fit_transform(df[['PAH Compound']])
        encoded_pah_df = pd.DataFrame(
            encoded_pah, 
            columns=self.encoder.get_feature_names_out(['PAH Compound'])
        )
        
        # Combine encoded PAH with numerical features
        numerical_features = ['EDTA Concentration', 'Fe Concentration', 
                            'H2O2 Concentration', 'Time']
        df_combined = pd.concat([
            encoded_pah_df, 
            df[numerical_features + ['Removal Rate']]
        ], axis=1)
        
        # Separate features and target
        X = df_combined.drop('Removal Rate', axis=1).values
        y = df_combined['Removal Rate'].values
        
        # Store feature names
        self.feature_names = df_combined.drop('Removal Rate', axis=1).columns.tolist()
        
        # Handle outliers using IQR method
        q1 = np.percentile(y, 25)
        q3 = np.percentile(y, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        mask = (y >= lower_bound) & (y <= upper_bound)
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Data shape after preprocessing: {X.shape}")
        return X, y
    
    def build_model(self, input_shape):
        """
        Build the neural network architecture.
        
        Args:
            input_shape: Shape of input features
        """
        self.model = Sequential([
            Input(shape=(input_shape,)),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')  # For regression
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error',
            metrics=['mean_squared_error', 'mae']
        )
        
        logger.info("Model architecture built successfully")
        return self.model
    
    def train(self, X_train, y_train, epochs=1000, validation_split=0.1):
        """
        Train the neural network model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            epochs: Number of training epochs
            validation_split: Fraction of training data for validation
        
        Returns:
            history: Training history object
        """
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Build model if not already built
        if self.model is None:
            self.build_model(X_train_scaled.shape[1])
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=10,
            min_lr=0.00001,
            verbose=1
        )
        
        # Train the model
        logger.info("Starting model training...")
        self.history = self.model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        logger.info("Model training completed")
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
        
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        X_test_scaled = self.scaler.transform(X_test)
        test_loss, test_mse, test_mae = self.model.evaluate(X_test_scaled, y_test, verbose=0)
        
        # Make predictions
        predictions = self.model.predict(X_test_scaled, verbose=0)
        
        # Calculate R-squared
        from sklearn.metrics import r2_score
        r2 = r2_score(y_test, predictions)
        
        metrics = {
            'test_loss': test_loss,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'r2_score': r2
        }
        
        logger.info(f"Test MSE: {test_mse:.4f}, R²: {r2:.4f}")
        return metrics
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Features for prediction
        
        Returns:
            predictions: Predicted removal rates
        """
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled, verbose=0)
        return predictions
    
    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            logger.warning("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot MSE
        ax1.plot(self.history.history['mean_squared_error'], label='Train MSE')
        ax1.plot(self.history.history['val_mean_squared_error'], label='Validation MSE')
        ax1.set_title('Model MSE During Training')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Mean Squared Error')
        ax1.legend()
        ax1.grid(True)
        
        # Plot MAE
        ax2.plot(self.history.history['mae'], label='Train MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE During Training')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, y_true, y_pred):
        """
        Plot actual vs predicted values.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
        """
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Add diagonal line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        plt.xlabel('Actual Removal Rate (%)')
        plt.ylabel('Predicted Removal Rate (%)')
        plt.title('Actual vs Predicted PAH Removal Rate')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
        
        # Save scaler and encoder
        import joblib
        base_path = os.path.splitext(filepath)[0]
        joblib.dump(self.scaler, f"{base_path}_scaler.joblib")
        joblib.dump(self.encoder, f"{base_path}_encoder.joblib")
        joblib.dump(self.feature_names, f"{base_path}_features.joblib")
        logger.info("Scaler, encoder, and feature names saved")
    
    def load_model(self, filepath):
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
        
        # Load scaler and encoder
        import joblib
        base_path = os.path.splitext(filepath)[0]
        self.scaler = joblib.load(f"{base_path}_scaler.joblib")
        self.encoder = joblib.load(f"{base_path}_encoder.joblib")
        self.feature_names = joblib.load(f"{base_path}_features.joblib")
        logger.info("Scaler, encoder, and feature names loaded")


def perform_hyperparameter_tuning(X_train, y_train):
    """
    Perform hyperparameter tuning using GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training targets
    
    Returns:
        best_params: Best hyperparameters found
        best_model: Best model from grid search
    """
    def create_model(learning_rate=0.00005, dropout_rate=0.3):
        model = Sequential([
            Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(64, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mean_squared_error',
            metrics=['mean_squared_error']
        )
        return model
    
    # Wrap the Keras model for scikit-learn
    model = KerasRegressor(model=create_model, verbose=0)
    
    # Define parameter grid
    param_grid = {
        'model__learning_rate': [0.0001, 0.00005, 0.00001],
        'model__dropout_rate': [0.3, 0.4, 0.5],
        'batch_size': [16, 32, 64],
        'epochs': [500, 1000]
    }
    
    # Grid search
    logger.info("Starting hyperparameter tuning...")
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        verbose=2
    )
    
    grid_result = grid.fit(X_train, y_train)
    
    logger.info(f"Best parameters: {grid_result.best_params_}")
    logger.info(f"Best score: {-grid_result.best_score_:.4f}")
    
    return grid_result.best_params_, grid_result.best_estimator_


def main(data_filepath=None, config_path=None):
    """
    Main execution function.
    
    Args:
        data_filepath: Path to data file (if None, uses example data)
        config_path: Path to configuration file
    """
    # Load configuration if provided
    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            'data': {
                'filepath': data_filepath or 'data/pah_removal_data.xlsx',
                'sheet_name': 'Sheet1',
                'target_column': 'Removal_Rate'
            },
            'model': {
                'learning_rate': 0.00005,
                'dropout_rate': 0.3,
                'batch_size': 32,
                'epochs': 1000
            },
            'split': {
                'test_size': 0.2,
                'random_state': 42
            }
        }
    
    # Load data
    logger.info("Loading data...")
    if data_filepath:
        df = pd.read_excel(data_filepath, sheet_name=config['data'].get('sheet_name', None))
    else:
        # Create example data if no file provided
        logger.info("No data file provided, creating example data...")
        np.random.seed(42)
        n_samples = 500
        df = pd.DataFrame({
            'PAH_Compound': np.random.choice(['Naphthalene', 'Anthracene', 'Pyrene', 'Fluorene'], n_samples),
            'EDTA_Concentration': np.random.uniform(10, 20, n_samples),
            'Fe_Concentration': np.random.uniform(0.5, 2, n_samples),
            'H2O2_Concentration': np.random.uniform(5, 20, n_samples),
            'Time': np.random.uniform(10, 30, n_samples),
            'Removal_Rate': np.random.uniform(70, 99, n_samples)
        })
    
    # Initialize predictor
    predictor = PAHRemovalPredictor()
    
    # Preprocess data
    X, y = predictor.preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    predictor.train(X_train, y_train, epochs=1000)
    
    # Evaluate model
    metrics = predictor.evaluate(X_test, y_test)
    print(f"Model Performance: {metrics}")
    
    # Plot results
    predictor.plot_training_history()
    
    # Make predictions and plot
    predictions = predictor.predict(X_test)
    predictor.plot_predictions(y_test, predictions.flatten())
    
    # Save model
    predictor.save_model('results/models/pah_removal_model.h5')


if __name__ == "__main__":
    main()