"""
Molecular Fingerprint Analysis for PAH Compounds
Author: [Your Name]
Date: 2025
Description: Generate and analyze molecular fingerprints (ECFP4, MACCS, Atom Pairs) for PAH prediction
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem, Descriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MolecularFingerprintPredictor:
    """
    Molecular fingerprint-based predictor for PAH removal rates.
    Supports ECFP4, MACCS keys, and Atom Pair fingerprints.
    """
    
    def __init__(self, fingerprint_types=['ECFP4', 'MACCS', 'AtomPair'], 
                 ecfp_radius=2, ecfp_bits=1024, atompair_bits=1024):
        """
        Initialize the fingerprint predictor.
        
        Args:
            fingerprint_types: List of fingerprint types to use
            ecfp_radius: Radius for ECFP fingerprints
            ecfp_bits: Number of bits for ECFP fingerprints
            atompair_bits: Number of bits for Atom Pair fingerprints
        """
        self.fingerprint_types = fingerprint_types
        self.ecfp_radius = ecfp_radius
        self.ecfp_bits = ecfp_bits
        self.atompair_bits = atompair_bits
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.history = None
        
    def get_maccs_key_descriptions(self):
        """
        Returns descriptions for MACCS keys.
        """
        descriptions = {
            1: "ISOTOPE",
            2: ">=32 atoms",
            3: ">=16 atoms",
            4: ">=8 atoms",
            5: ">=4 atoms",
            6: ">1 rotatable bond",
            7: ">1 aromatic ring",
            8: ">=8 ring atoms",
            9: ">1 ring",
            10: ">3 heteroatoms",
            11: ">1 heteroatom",
            12: ">=4 H-bond acceptors",
            13: ">2 H-bond acceptors",
            14: ">=1 H-bond acceptor",
            15: ">=4 H-bond donors",
            16: ">2 H-bond donors",
            17: ">=1 H-bond donor",
            18: ">=1 halogen",
            19: ">=1 S",
            20: ">=1 N",
            21: ">=2 O",
            22: ">=1 O",
            23: ">=1 high electronegativity atom",
            24: ">=1 acid group",
            25: ">=2 aromatic rings",
            26: ">=1 di-substituted benzene",
            27: ">=1 hydroxy group",
            28: ">=1 methyl group",
            29: ">=1 methoxy group",
            30: ">=1 oxo group (C=O)",
            # Add more as needed
        }
        return descriptions
    
    def calculate_molecular_descriptors(self, mol):
        """
        Calculate molecular descriptors for a molecule.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary of molecular descriptors
        """
        descriptors = {}
        descriptors['MolWeight'] = Descriptors.MolWt(mol)
        descriptors['LogP'] = Descriptors.MolLogP(mol)
        descriptors['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
        descriptors['NumHDonors'] = Descriptors.NumHDonors(mol)
        descriptors['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
        descriptors['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
        descriptors['TPSA'] = Descriptors.TPSA(mol)
        descriptors['NumHeavyAtoms'] = Descriptors.HeavyAtomCount(mol)
        descriptors['NumHeteroatoms'] = Descriptors.NumHeteroatoms(mol)
        descriptors['RingCount'] = Descriptors.RingCount(mol)
        descriptors['MolMR'] = Descriptors.MolMR(mol)
        descriptors['BalabanJ'] = Descriptors.BalabanJ(mol)
        descriptors['BertzCT'] = Descriptors.BertzCT(mol)
        descriptors['Chi0'] = Descriptors.Chi0(mol)
        return descriptors
    
    def generate_fingerprints(self, smiles_list):
        """
        Generate molecular fingerprints from SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            numpy array of concatenated fingerprints
        """
        all_fingerprints = []
        self.feature_names = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Could not parse SMILES: {smiles}")
                continue
            
            fingerprints = []
            
            # ECFP4 fingerprints
            if 'ECFP4' in self.fingerprint_types:
                ecfp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.ecfp_radius, nBits=self.ecfp_bits
                )
                ecfp_array = np.zeros((self.ecfp_bits,))
                DataStructs.ConvertToNumpyArray(ecfp, ecfp_array)
                fingerprints.append(ecfp_array)
                
                if len(self.feature_names) == 0:
                    self.feature_names.extend([f'ECFP4_bit_{i}' for i in range(self.ecfp_bits)])
            
            # MACCS keys
            if 'MACCS' in self.fingerprint_types:
                maccs = MACCSkeys.GenMACCSKeys(mol)
                maccs_array = np.array([int(maccs[i]) for i in range(167)])
                fingerprints.append(maccs_array)
                
                if 'MACCS_0' not in self.feature_names:
                    self.feature_names.extend([f'MACCS_{i}' for i in range(167)])
            
            # Atom Pair fingerprints
            if 'AtomPair' in self.fingerprint_types:
                atompair = AllChem.GetHashedAtomPairFingerprintAsBitVect(
                    mol, nBits=self.atompair_bits
                )
                atompair_array = np.zeros((self.atompair_bits,))
                DataStructs.ConvertToNumpyArray(atompair, atompair_array)
                fingerprints.append(atompair_array)
                
                if 'AtomPair_bit_0' not in self.feature_names:
                    self.feature_names.extend([f'AtomPair_bit_{i}' for i in range(self.atompair_bits)])
            
            # Concatenate all fingerprints
            combined_fingerprint = np.concatenate(fingerprints)
            all_fingerprints.append(combined_fingerprint)
        
        return np.array(all_fingerprints)
    
    def generate_features_with_parameters(self, smiles_list, parameters_df):
        """
        Combine molecular fingerprints with experimental parameters.
        
        Args:
            smiles_list: List of SMILES strings
            parameters_df: DataFrame with experimental parameters
            
        Returns:
            Combined feature matrix
        """
        # Generate fingerprints
        fingerprints = self.generate_fingerprints(smiles_list)
        
        # Add experimental parameters
        param_names = parameters_df.columns.tolist()
        self.feature_names.extend(param_names)
        
        # Combine features
        combined_features = np.hstack([fingerprints, parameters_df.values])
        
        return combined_features
    
    def build_model(self, input_shape):
        """
        Build the neural network model for fingerprint-based prediction.
        
        Args:
            input_shape: Shape of input features
        """
        self.model = Sequential([
            Dense(512, activation='relu', input_shape=(input_shape,)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='mean_squared_error',
            metrics=['mean_squared_error', 'mae']
        )
        
        logger.info("Fingerprint model architecture built successfully")
    
    def train(self, X_train, y_train, epochs=500, batch_size=32, validation_split=0.1):
        """
        Train the fingerprint model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction for validation
            
        Returns:
            Training history
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Build model if not already built
        if self.model is None:
            self.build_model(X_train_scaled.shape[1])
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=15,
            min_lr=0.00001,
            verbose=1
        )
        
        # Train model
        logger.info("Training fingerprint model...")
        self.history = self.model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return self.history
    
    def cross_validate(self, X, y, n_folds=5):
        """
        Perform k-fold cross-validation.
        
        Args:
            X: Features
            y: Targets
            n_folds: Number of folds
            
        Returns:
            Dictionary with CV results
        """
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_scores = []
        fold_predictions = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
            logger.info(f"Training fold {fold}/{n_folds}")
            
            # Split data
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train_fold)
            X_val_scaled = self.scaler.transform(X_val_fold)
            
            # Build and train model
            self.build_model(X_train_scaled.shape[1])
            
            history = self.model.fit(
                X_train_scaled, y_train_fold,
                epochs=200,
                batch_size=32,
                validation_data=(X_val_scaled, y_val_fold),
                verbose=0
            )
            
            # Evaluate
            val_loss = self.model.evaluate(X_val_scaled, y_val_fold, verbose=0)[0]
            cv_scores.append(val_loss)
            
            # Store predictions
            predictions = self.model.predict(X_val_scaled, verbose=0)
            fold_predictions.append((y_val_fold, predictions))
        
        results = {
            'cv_scores': cv_scores,
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores),
            'fold_predictions': fold_predictions
        }
        
        logger.info(f"Cross-validation MSE: {results['mean_cv_score']:.4f} ± {results['std_cv_score']:.4f}")
        
        return results
    
    def analyze_feature_importance(self, X_test, y_test, n_repeats=10):
        """
        Analyze feature importance using permutation importance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            n_repeats: Number of permutation repeats
            
        Returns:
            DataFrame with feature importances
        """
        X_test_scaled = self.scaler.transform(X_test)
        
        # Calculate permutation importance
        result = permutation_importance(
            self.model, X_test_scaled, y_test,
            n_repeats=n_repeats,
            random_state=42,
            scoring='neg_mean_squared_error'
        )
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance_mean', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self, importance_df, top_n=30):
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with feature importances
            top_n: Number of top features to plot
        """
        # Get top features
        top_features = importance_df.head(top_n)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.barh(range(len(top_features)), 
                top_features['importance_mean'].values,
                xerr=top_features['importance_std'].values)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values)
        ax.set_xlabel('Permutation Importance')
        ax.set_title(f'Top {top_n} Most Important Features')
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def analyze_maccs_keys(self, importance_df):
        """
        Analyze which MACCS keys are most important.
        
        Args:
            importance_df: DataFrame with feature importances
            
        Returns:
            DataFrame with MACCS key analysis
        """
        # Filter MACCS features
        maccs_features = importance_df[importance_df['feature'].str.startswith('MACCS_')]
        
        # Get descriptions
        descriptions = self.get_maccs_key_descriptions()
        
        # Add descriptions
        maccs_features['key_number'] = maccs_features['feature'].str.extract(r'MACCS_(\d+)').astype(int)
        maccs_features['description'] = maccs_features['key_number'].map(descriptions)
        
        # Sort by importance
        maccs_features = maccs_features.sort_values('importance_mean', ascending=False)
        
        return maccs_features
    
    def save_model(self, filepath):
        """
        Save the trained model and associated files.
        
        Args:
            filepath: Base filepath for saving
        """
        import os
        
        # Save model
        self.model.save(f"{filepath}_model.h5")
        
        # Save scaler and feature names
        joblib.dump(self.scaler, f"{filepath}_scaler.joblib")
        joblib.dump(self.feature_names, f"{filepath}_features.joblib")
        
        # Save fingerprint configuration
        config = {
            'fingerprint_types': self.fingerprint_types,
            'ecfp_radius': self.ecfp_radius,
            'ecfp_bits': self.ecfp_bits,
            'atompair_bits': self.atompair_bits
        }
        joblib.dump(config, f"{filepath}_config.joblib")
        
        logger.info(f"Model and configuration saved to {filepath}")


def compare_fingerprint_types(smiles_list, targets, parameters_df=None):
    """
    Compare performance of different fingerprint types.
    
    Args:
        smiles_list: List of SMILES strings
        targets: Target values
        parameters_df: Optional experimental parameters
        
    Returns:
        DataFrame with comparison results
    """
    fingerprint_configs = [
        ['ECFP4'],
        ['MACCS'],
        ['AtomPair'],
        ['ECFP4', 'MACCS'],
        ['ECFP4', 'AtomPair'],
        ['MACCS', 'AtomPair'],
        ['ECFP4', 'MACCS', 'AtomPair']
    ]
    
    results = []
    
    for fp_types in fingerprint_configs:
        logger.info(f"Testing fingerprint configuration: {fp_types}")
        
        # Create predictor
        predictor = MolecularFingerprintPredictor(fingerprint_types=fp_types)
        
        # Generate features
        if parameters_df is not None:
            features = predictor.generate_features_with_parameters(smiles_list, parameters_df)
        else:
            features = predictor.generate_fingerprints(smiles_list)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )
        
        # Train model
        predictor.train(X_train, y_train, epochs=200, validation_split=0.1)
        
        # Evaluate
        X_test_scaled = predictor.scaler.transform(X_test)
        test_loss, test_mse, test_mae = predictor.model.evaluate(X_test_scaled, y_test, verbose=0)
        
        # Calculate R2
        predictions = predictor.model.predict(X_test_scaled, verbose=0)
        r2 = r2_score(y_test, predictions)
        
        results.append({
            'fingerprints': '+'.join(fp_types),
            'test_mse': test_mse,
            'test_mae': test_mae,
            'r2_score': r2,
            'num_features': len(predictor.feature_names)
        })
    
    return pd.DataFrame(results)


def main():
    """Main execution function for fingerprint analysis."""
    # Example usage
    logger.info("Starting molecular fingerprint analysis...")
    
    # Example SMILES for PAH compounds
    pah_smiles = {
        'Naphthalene': 'c1ccc2c(c1)cccc2',
        'Anthracene': 'c1ccc2c(c1)cc3c(c2)cccc3',
        'Phenanthrene': 'c1ccc2c(c1)ccc3c2cccc3',
        'Pyrene': 'c1cc2c3c(c1)cccc3c1c2cccc1',
        'Fluorene': 'c1ccc2c(c1)Cc3c2cccc3',
        # Add more PAH SMILES
    }
    
    # Convert to lists
    compound_names = list(pah_smiles.keys())
    smiles_list = list(pah_smiles.values())
    
    # Example targets (replace with actual data)
    removal_rates = np.random.uniform(70, 99, len(smiles_list))
    
    # Example experimental parameters (replace with actual data)
    parameters_df = pd.DataFrame({
        'EDTA_Concentration': np.random.uniform(10, 20, len(smiles_list)),
        'Fe_Concentration': np.random.uniform(0.5, 2, len(smiles_list)),
        'H2O2_Concentration': np.random.uniform(5, 20, len(smiles_list)),
        'Time': np.random.uniform(10, 30, len(smiles_list))
    })
    
    # Initialize predictor
    predictor = MolecularFingerprintPredictor()
    
    # Generate features
    features = predictor.generate_features_with_parameters(smiles_list, parameters_df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, removal_rates, test_size=0.2, random_state=42
    )
    
    # Train model
    predictor.train(X_train, y_train, epochs=300)
    
    # Evaluate
    X_test_scaled = predictor.scaler.transform(X_test)
    test_loss, test_mse, test_mae = predictor.model.evaluate(X_test_scaled, y_test)
    
    logger.info(f"Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}")
    
    # Feature importance
    importance_df = predictor.analyze_feature_importance(X_test, y_test)
    predictor.plot_feature_importance(importance_df)
    
    # Save model
    predictor.save_model('results/models/fingerprint_model')
    
    logger.info("Fingerprint analysis completed!")


if __name__ == "__main__":
    main()