# NHANES Diabetes Prediction - Machine Learning Analysis

## Project Overview

This project develops a comprehensive machine learning pipeline to predict diabetes diagnosis using the National Health and Nutrition Examination Survey (NHANES) data. The analysis integrates multiple health survey modules to build robust predictive models for diabetes classification.

### Dataset

The project utilizes NHANES dataset (.XPT format) covering multiple health domains:
- **Primary Module**: Diabetes diagnosis and treatment
- **Health Modules**: Blood pressure/cholesterol, alcohol use, weight history, physical activity
- **Socioeconomic**: Income, health insurance coverage
- **Lifestyle**: Diet/nutrition, smoking habits (cigarette use, household exposure)
- **Medical**: Kidney conditions, comorbidities

### Target Variable

- **EverTold_Diabetes**: Binary classification (1 = Diabetes, 2 = No Diabetes)
  - Mapped to: 1 = Diabetes, 0 = No Diabetes

## Pipeline Architecture

### 1. Data Loading and Integration

**Purpose**: Load and merge multiple NHANES health survey modules

**Key Components**:
- `load_dataset()`: Loads .XPT format files from `data/raw/`
- Variable mapping using `data/extra/variables.csv`
- Merging datasets on participant ID (SEQN)

### 2. Data Cleaning

**Cleaning Strategy** (`clean_nhanes_module`):
- Handles NHANES-specific missing value codes (7, 9, 77, 99, 777, 999, etc.)
- Separates variables by type:
  - Binary variables (1=Yes, 2=No)
  - Ordinal categorical variables
  - Continuous numeric variables
- Converts coded missing values to `pd.NA`

**Data Leakage Prevention**:
- Removes insulin and diabetic pills columns (treatment information that would leak diagnosis)

### 3. Feature Engineering

**Transformations**:
- Time conversion (sleep/wake times → minutes since midnight)
- BMI calculation from height and weight
- Age group categorization

### 4. Missing Value Handling

**4-Step Imputation Strategy**:
1. **Binary/Categorical**: Mode imputation
2. **Numeric Variables**: Median imputation
3. **Age-based**: Group median for age-related features
4. **Feature-specific**: Custom strategies for domain knowledge

**Goal**: Preserve data distribution while avoiding bias from missing patterns

### 5. Exploratory Data Analysis (EDA)

**Univariate Analysis**:
- Histograms for continuous variables
- Bar plots for categorical variables
- Distribution assessment

**Bivariate Analysis**:
- Violin plots comparing diabetic vs. non-diabetic groups
- Feature-target relationships
- Correlation analysis

### 6. Model Training

The project implements 4 machine learning models with specific optimization strategies:

#### Logistic Regression
- **Approach**: Baseline linear model with regularization
- **Class Weights**: {0: 1, 1: 3} to address class imbalance
- **Optimization**: Hyperparameter tuning for C (regularization strength)

#### Random Forest
- **Approach**: Ensemble decision trees
- **Class Weights**: {0: 1, 1: 3}
- **Optimization**: 
  - F2-score threshold tuning (prioritizes recall)
  - Number of trees, max depth, min samples split

#### XGBoost with BorderlineSMOTE
- **Approach**: Gradient boosting with balanced training data
- **Pipeline**: BorderlineSMOTE → XGBoost
- **Optimization**:
  - SMOTE oversampling for minority class
  - F2-score threshold optimization
  - Learning rate, max depth, n_estimators

#### Support Vector Machine (SVM)
- **Approach**: Linear SVM with standardization
- **Pipeline**: StandardScaler → LinearSVM
- **Class Weights**: {0: 1, 1: 3}
- **Optimization**: C parameter tuning

### 7. Model Evaluation

**Metrics**:
- ROC-AUC Score
- Recall (Sensitivity) - Priority metric for medical diagnosis
- F1-Score
- Accuracy
- Balanced Accuracy
- Confusion Matrix

**Threshold Analysis**:
- Threshold vs. Recall/F1/Accuracy curves
- Optimal threshold selection using F2-score (emphasizes recall over precision)

### 8. Feature Importance

- XGBoost feature importance ranking
- Top 20 most influential features visualization
- Domain interpretation of key predictors

## Key Findings

### Class Imbalance Strategy
- Addressed through:
  - Class weighting (1:3 ratio)
  - SMOTE oversampling (BorderlineSMOTE variant)
  - F2-score optimization (recall-weighted metric)

### Medical ML Considerations
- **Why prioritize recall?**: In diabetes screening, false negatives (missing diabetic cases) are more costly than false positives
- **F2-score**: Weighs recall 4x higher than precision
- **Threshold optimization**: Adjusts decision boundary to maximize recall while maintaining acceptable precision

## Setup and Installation

The project uses `uv` package manager. Install [uv](https://docs.astral.sh/uv/#installation) first.

Run the following in your console:

```bash
uv sync
```

## Starting

**Jupyter Lab**

```bash
uv run jupyter lab
```

**VSCode**

Open the notebook and select the right kernel (VSCode auto-detects `uv` projects)

## Notebook Structure

**main.ipynb** (77 cells):
1. **Cells 1-3**: Project description and objectives
2. **Cells 4-8**: Data loading from .XPT files
3. **Cells 9-36**: Data cleaning (9 health modules)
4. **Cells 37-41**: Dataset merging and leakage prevention
5. **Cells 42-51**: Missing value handling and BMI engineering
6. **Cells 52-58**: EDA (univariate and bivariate visualizations)
7. **Cells 59-63**: Train-test split and baseline
8. **Cells 64-77**: Model training and feature importance

## Dependencies

**Core Libraries**:
- pandas, numpy
- scikit-learn
- xgboost
- imbalanced-learn (BorderlineSMOTE)
- seaborn, matplotlib

**Data Format**:
- NHANES .XPT files (SAS transport format)

## Results Interpretation

### Why This Approach?

1. **Multiple Data Sources**: NHANES provides comprehensive health data beyond simple glucose measurements
2. **Class Imbalance**: Real-world diabetes prevalence (~10-15%) requires specialized handling
3. **Medical Context**: Prioritizing recall over precision aligns with screening goals
4. **Threshold Tuning**: Fixed 0.5 threshold suboptimal for imbalanced medical data

### Model Selection Rationale

- **Logistic Regression**: Interpretable baseline
- **Random Forest**: Captures non-linear relationships
- **XGBoost + SMOTE**: State-of-the-art for imbalanced data
- **SVM**: Alternative linear approach with different decision boundary


