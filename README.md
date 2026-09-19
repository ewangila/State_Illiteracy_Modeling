# U.S. State Illiteracy Regression Analysis

Comprehensive regression analysis of U.S. state-level illiteracy rates using the classic `statedata` dataset from the `faraway` package.

This project systematically compares traditional Ordinary Least Squares (OLS) models against regularized machine learning approaches (Ridge, Lasso, and ElasticNet). It includes feature engineering, multicollinearity diagnostics, residual analysis, formal assumption testing, cross-validation, and rigorous out-of-sample evaluation.

---

## Key Features

- Multiple OLS specifications:
  - Simple linear model
  - Quadratic and cubic polynomial terms
  - Interaction terms
  - Logarithmic transformations
- Regularized models with automatic hyperparameter tuning:
  - Ridge Regression
  - Lasso Regression
  - ElasticNet
- Feature engineering (polynomial terms, log transformations, interactions, binary indicators)
- Multicollinearity detection using Variance Inflation Factor (VIF)
- Residual diagnostics and assumption testing:
  - Breusch-Pagan test (homoscedasticity)
  - Jarque-Bera test (normality)
  - Residual vs Fitted, Q-Q, Scale-Location, and Histogram plots
- 5-fold cross-validation
- Train-test split evaluation with R², RMSE, and MAE
- Overfitting gap analysis

---

## Dataset

The analysis uses the classic **statedata** dataset (50 U.S. states), which contains socioeconomic indicators including:

- Illiteracy rate
- Income
- Population
- Other related variables

Source: `faraway` package

---

## Project Structure
```
├── State_Illetracy.ipynb     # Main analysis notebook
├── State_Illetracy.py        # Clean Python script version
├── requirements.txt          # Project dependencies
├── LICENSE                   # MIT License
└── README.md
```
---
## Installation

1. Clone the repository:
```bash
git clone https://github.com/ewangila/State_Illetracy_Analysis.git
cd State_Illetracy_Analysis
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
## Usage

Option 1: Jupyter Notebook
```Bash
jupyter notebook State_Illetracy.ipynb
```

Option 2: Python Script
```Bash
python State_Illetracy.py
```

## Requirements

- Python 3.9+
- pandas
- numpy
- statsmodels
- scikit-learn
- matplotlib
- seaborn
- scipy
- faraway

See `requirements.txt` for the full list.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Author

**Eugin Wangila**
