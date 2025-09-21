# PhotoFenton-PAH-Remediation-ML

## Machine Learning Framework for Chemical Remediation Prediction

This repository contains a generalized machine learning framework for predicting removal rates in chemical remediation processes, developed for research in Photo-Fenton treatment systems.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Code Structure](#code-structure)
- [How to Use](#how-to-use)
- [Data Format Requirements](#data-format-requirements)
- [Examples with Sample Data](#examples-with-sample-data)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## Overview

This framework provides machine learning models for predicting treatment efficiency based on:
- Compound characteristics
- Treatment parameters
- Reaction conditions
- Time factors

The code is designed to be adaptable to various chemical remediation scenarios.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. Clone or download this repository
```bash
git clone https://github.com/SAAAHco/PhotoFenton-PAH-Remediation-ML.git
cd PhotoFenton-PAH-Remediation-ML
```

2. Create a virtual environment (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

3. Install required packages
```bash
pip install -r requirements.txt
```

## Code Structure

```
PhotoFenton-PAH-Remediation-ML/
│
├── models/
│   ├── neural_network_model.py      # Deep learning prediction model
│   └── fingerprint_analysis.py      # Molecular structure analysis
│
├── src/
│   ├── data_preprocessing.py        # Data preparation utilities
│   ├── feature_analysis.py          # Feature importance analysis
│   └── visualization.py             # Plotting and visualization tools
│
├── requirements.txt                  # Package dependencies
└── README.md                        # Documentation
```

## Data Format Requirements

### Required Column Structure

Your data file should be in Excel (.xlsx) or CSV format with the following column types:

| Column Type | Data Type | Description | Example Name |
|------------|-----------|-------------|--------------|
| Compound Identifier | Text | Name or ID of compound | "Compound_Name" |
| Parameter 1 | Numeric | First treatment parameter | "Concentration_A" |
| Parameter 2 | Numeric | Second treatment parameter | "Concentration_B" |
| Parameter 3 | Numeric | Third treatment parameter | "Concentration_C" |
| Time | Numeric | Time parameter | "Treatment_Time" |
| Target | Numeric | Outcome to predict (0-100) | "Removal_Rate" |

Note: You can have any number of parameters. Column names can be customized.

### Data Preparation Example

```python
import pandas as pd

# Example of how your data should be structured
example_data = pd.DataFrame({
    'Compound': ['Type_A', 'Type_B', 'Type_C'],
    'Param_1': [10.0, 15.0, 20.0],
    'Param_2': [1.0, 1.5, 2.0],
    'Param_3': [5.0, 10.0, 15.0],
    'Time': [30, 45, 60],
    'Target_Value': [85.0, 90.0, 95.0]
})

# Save as Excel for use with the models
example_data.to_excel('my_data.xlsx', index=False)
```

## How to Use

### Step 1: Prepare Your Data

Ensure your data file has:
- Column headers in the first row
- No missing values in required columns
- Numeric columns contain only numbers
- Target values are in appropriate range (e.g., 0-100 for percentages)

### Step 2: Basic Model Training

```python
from models.neural_network_model import PAHRemovalPredictor
from src.data_preprocessing import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Load your data
data = preprocessor.load_data('path_to_your_data.xlsx')

# Clean data (handles missing values, outliers)
data_clean = preprocessor.clean_data(data)

# Prepare for modeling
prepared_data = preprocessor.prepare_for_modeling(
    data_clean,
    target_column='Your_Target_Column',  # Change this!
    test_size=0.2  # 80% training, 20% testing
)

# Create and train model
model = PAHRemovalPredictor()
model.train(
    prepared_data['X_train'],
    prepared_data['y_train'],
    epochs=500  # Adjust based on your needs
)

# Evaluate performance
metrics = model.evaluate(
    prepared_data['X_test'],
    prepared_data['y_test']
)

print(f"Model R² Score: {metrics['r2_score']:.4f}")
```

### Step 3: Making Predictions

```python
# After training, make predictions on new data
import pandas as pd

# Create new data in the same format as training data
new_data = pd.DataFrame({
    'Compound': ['New_Type'],
    'Param_1': [12.5],
    'Param_2': [1.2],
    'Param_3': [8.0],
    'Time': [35]
})

# Process and predict
X_new = preprocessor.transform(new_data)
predictions = model.predict(X_new)
print(f"Predicted value: {predictions[0]:.2f}")
```

## Examples with Sample Data

### Example 1: Creating Sample Data for Testing

```python
import numpy as np
import pandas as pd

# Generate synthetic data for testing
np.random.seed(42)
n_samples = 200

# Create synthetic dataset
sample_data = pd.DataFrame({
    'Compound_Type': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
    'Parameter_X': np.random.uniform(5, 25, n_samples),
    'Parameter_Y': np.random.uniform(0.5, 3.0, n_samples),
    'Parameter_Z': np.random.uniform(2, 20, n_samples),
    'Duration': np.random.uniform(10, 60, n_samples),
    'Efficiency': np.random.uniform(70, 99, n_samples)  # Target variable
})

# Save for use
sample_data.to_excel('sample_data.xlsx', index=False)
print(f"Created sample data with {n_samples} rows")
```

### Example 2: Feature Importance Analysis

```python
from src.feature_analysis import FeatureAnalyzer

# After training your model
analyzer = FeatureAnalyzer()

# Analyze which features are most important
importance_results = analyzer.random_forest_importance(
    prepared_data['X_train'],
    prepared_data['y_train'],
    feature_names=prepared_data.get('feature_names')
)

print("\nMost Important Features:")
print(importance_results.head(10))
```

### Example 3: Visualization

```python
from src.visualization import AdvancedVisualizer

# Create visualizer
viz = AdvancedVisualizer()

# Plot actual vs predicted values
predictions = model.predict(prepared_data['X_test'])
model.plot_predictions(
    prepared_data['y_test'], 
    predictions.flatten()
)
```

### Example 4: Hyperparameter Optimization

```python
# Test different model configurations
configurations = [
    {'learning_rate': 0.001, 'dropout_rate': 0.2},
    {'learning_rate': 0.0001, 'dropout_rate': 0.3},
    {'learning_rate': 0.00001, 'dropout_rate': 0.4}
]

results = []
for config in configurations:
    model = PAHRemovalPredictor(**config)
    # Train and evaluate
    # ... training code ...
    results.append({'config': config, 'score': metrics['r2_score']})

# Find best configuration
best = max(results, key=lambda x: x['score'])
print(f"Best configuration: {best['config']}")
```

## Customization Guide

### Adapting to Your Data

1. **Different Column Names**: Update the column names in the preprocessing step
2. **Different Ranges**: Adjust scaling parameters in DataPreprocessor
3. **More Features**: The model automatically adapts to the number of input features
4. **Different Target Range**: Modify the output activation if not using 0-100 range

### Model Architecture Customization

```python
# Modify the neural network architecture
model = PAHRemovalPredictor(
    learning_rate=0.0001,      # Adjust learning speed
    dropout_rate=0.3,          # Prevent overfitting (0.2-0.5)
    batch_size=32,             # Samples per training step (16-64)
)
```

## Troubleshooting

### Common Issues

**Import Errors**
```
ModuleNotFoundError: No module named 'tensorflow'
```
Solution: Ensure all packages are installed: `pip install -r requirements.txt`

**Data Format Errors**
```
KeyError: 'column_name'
```
Solution: Check that your column names match exactly (case-sensitive)

**Memory Issues**
```
ResourceExhaustedError
```
Solution: Reduce batch_size or use fewer epochs

**Poor Model Performance**
- Ensure sufficient data (minimum 100 samples recommended)
- Check for data quality issues (outliers, errors)
- Try different hyperparameters
- Increase training epochs

## Understanding Results

### Model Metrics

- **R² Score**: Proportion of variance explained (0-1, higher is better)
- **MSE**: Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error (average prediction error)

### Typical Performance Ranges

- **Excellent**: R² > 0.90
- **Good**: R² = 0.80-0.90
- **Acceptable**: R² = 0.70-0.80
- **Poor**: R² < 0.70

## Advanced Features

### Molecular Fingerprinting (Optional)

If working with chemical compounds with known structures:

```python
from models.fingerprint_analysis import MolecularFingerprintPredictor

# Requires SMILES strings of molecules
predictor = MolecularFingerprintPredictor()
# See fingerprint_analysis.py for detailed usage
```

### Statistical Analysis

```python
from src.feature_analysis import FeatureAnalyzer

analyzer = FeatureAnalyzer()
# Perform comprehensive statistical analysis
report = analyzer.create_feature_report(X, y)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{photofenton_ml_2025,
  title={Efficacy of Advanced Fenton-Photo Systems for the Degradation of Petroleum Hydrocarbons Using Complex Neural Networks},
  author={Zainab Ashkanania, Rabi Mohtar, Salah Al-Enezi, Faten Khalil, Muthanna Al-Momin, Xingmao Ma, Patricia K. Smith, Salvatore Calabrese, Meshal Abdullah, and Najeeb Aladwani},
  year={2025},
  url={https://github.com/SAAAHco/PhotoFenton-PAH-Remediation-ML}
}
```

## License

MIT License - Free to use, modify, and distribute with attribution.

## Contact

For questions about the code implementation: Ashkanani@tamu.edu

---

**Version**: 1.0.0  
**Last Updated**: 2025