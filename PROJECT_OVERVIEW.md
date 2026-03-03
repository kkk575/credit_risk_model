# Loan Default Prediction System

A statistical model to assess the likelihood of loan default using customer financial data and credit history.

## Project Overview

This project develops a predictive model to evaluate borrower creditworthiness and estimate default probability. The model uses logistic regression combined with Weight of Evidence (WOE) transformation to create interpretable risk scores.

## Key Components

### Data Processing
- Combines applicant information, payment patterns, and credit bureau data
- Handles missing values and creates derived features
- Bins continuous variables for better interpretability

### Statistical Methods
- **Weight of Evidence (WOE)**: Measures predictive power of each variable
- **Information Value (IV)**: Quantifies variable importance
- **Logistic Regression**: Estimates probability of default
- **Score Scaling**: Converts probabilities to 300-900 point scale

### Model Validation
- **AUC-ROC**: Measures discrimination ability
- **KS Statistic**: Tests separation between good and bad customers
- **Gini Coefficient**: Alternative measure of model strength

## Technical Approach

The model follows standard credit risk modeling practices:

1. **Data Integration**: Merge multiple data sources
2. **Target Definition**: Define default based on 90+ days past due
3. **Feature Engineering**: Create bins and calculate WOE
4. **Model Training**: Fit logistic regression on training data
5. **Validation**: Test on holdout sample
6. **Score Creation**: Transform probabilities to scores

## Files Structure

- `loan_default_analysis.ipynb` - Main analysis notebook
- `customer_scorecard_input.xlsx` - Applicant and behavioral data
- `bureau_data.xlsx` - Credit bureau information
- `model_predictions.xlsx` - Model output with scores
- `feature_importance.xlsx` - Variable importance rankings

## Usage

1. Ensure all data files are in the same directory
2. Open `loan_default_analysis.ipynb` in Jupyter
3. Run cells sequentially
4. Review model performance metrics
5. Check output files for predictions

## Model Performance

The model is evaluated using:
- **Discrimination**: How well it separates good from bad customers
- **Calibration**: How accurate the probability estimates are
- **Stability**: How consistent it performs over time

## Limitations

- Model trained on historical data may not capture future trends
- Requires periodic retraining as customer behavior changes
- Should be combined with business judgment for final decisions

## Next Steps

- Monitor model performance on new data
- Retrain periodically with updated information
- Consider additional data sources for improvement
- Implement automated scoring system

---

*This is an educational project demonstrating credit risk modeling techniques commonly used in financial institutions.*
