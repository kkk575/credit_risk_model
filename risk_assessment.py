"""
Loan Default Risk Assessment Model
A simple implementation for predicting loan defaults
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


class DefaultRiskModel:
    """
    A class to build and evaluate loan default prediction models
    """
    
    def __init__(self, base_score=600, pdo=50):
        """
        Initialize the model
        
        Parameters:
        -----------
        base_score : int
            Base credit score (default 600)
        pdo : int
            Points to double the odds (default 50)
        """
        self.base_score = base_score
        self.pdo = pdo
        self.model = None
        self.scaler = None
        self.feature_names = None
        
    def load_data(self, applicant_file, bureau_file):
        """
        Load and merge data from multiple sources
        
        Parameters:
        -----------
        applicant_file : str
            Path to applicant data file
        bureau_file : str
            Path to bureau data file
            
        Returns:
        --------
        pd.DataFrame
            Merged dataset
        """
        # Load applicant data
        app_data = pd.read_excel(applicant_file, sheet_name='Application_Data')
        behavior_data = pd.read_excel(applicant_file, sheet_name='Behavioral_Data')
        
        # Load bureau data
        bureau_data = pd.read_excel(bureau_file, sheet_name='Bureau_Data')
        
        # Merge datasets
        combined = app_data.merge(behavior_data, on='Customer_ID', how='inner')
        combined = combined.merge(bureau_data, on='Customer_ID', how='inner')
        
        return combined
    
    def prepare_target(self, data, dpd_column='DPD_90', threshold=1):
        """
        Create binary target variable
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        dpd_column : str
            Column name for days past due
        threshold : int
            Threshold for defining default
            
        Returns:
        --------
        pd.Series
            Binary target (1=default, 0=no default)
        """
        return (data[dpd_column] >= threshold).astype(int)
    
    def select_features(self, data):
        """
        Select relevant features for modeling
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
            
        Returns:
        --------
        list
            List of feature names
        """
        # Define features to use
        features = [
            'Age', 'Income_INR', 'Employment_Years', 'Credit_History_Length',
            'Outstanding_Loans', 'Loan_Amount', 'Loan_Tenure_Months',
            'Savings_Account_Balance', 'Checking_Account_Balance',
            'Delinquency_12M', 'Credit_Card_Utilization',
            'Behavior_Spending_Score', 'Behavior_Repayment_Score',
            'No_of_Open_Accounts', 'No_of_Closed_Accounts',
            'Total_Credit_Limit', 'Total_Current_Balance',
            'Credit_Utilization_Ratio', 'No_of_Inquiries_6M', 'No_of_Inquiries_12M',
            'DPD_30', 'DPD_60', 'Worst_Current_Status',
            'Months_Since_Most_Recent_Delinquency', 'Max_Credit_Exposure',
            'Oldest_Trade_Open_Months', 'Newest_Trade_Open_Months'
        ]
        
        # Keep only features that exist in data
        available_features = [f for f in features if f in data.columns]
        
        return available_features
    
    def train(self, X_train, y_train):
        """
        Train the logistic regression model
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        """
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        print("Model training completed")
    
    def predict_probability(self, X):
        """
        Predict default probability
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features to predict
            
        Returns:
        --------
        np.array
            Predicted probabilities
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def convert_to_score(self, probabilities):
        """
        Convert probabilities to credit scores
        
        Parameters:
        -----------
        probabilities : np.array
            Default probabilities
            
        Returns:
        --------
        np.array
            Credit scores
        """
        # Convert probability to odds
        odds = (1 - probabilities) / probabilities
        
        # Calculate score
        factor = self.pdo / np.log(2)
        offset = self.base_score - factor * np.log(20)  # base odds = 20
        scores = offset + factor * np.log(odds)
        
        return scores
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target
            
        Returns:
        --------
        dict
            Performance metrics
        """
        # Get predictions
        probs = self.predict_probability(X_test)
        
        # Calculate AUC
        auc = roc_auc_score(y_test, probs)
        
        # Calculate KS
        ks = self._calculate_ks(y_test, probs)
        
        # Calculate Gini
        gini = 2 * auc - 1
        
        metrics = {
            'AUC': auc,
            'KS': ks,
            'Gini': gini
        }
        
        return metrics
    
    def _calculate_ks(self, y_true, y_pred_proba):
        """
        Calculate Kolmogorov-Smirnov statistic
        """
        df = pd.DataFrame({
            'actual': y_true,
            'predicted': y_pred_proba
        })
        
        df = df.sort_values('predicted', ascending=False).reset_index(drop=True)
        
        df['bad_cumsum'] = df['actual'].cumsum()
        df['good_cumsum'] = (1 - df['actual']).cumsum()
        
        total_bad = df['actual'].sum()
        total_good = len(df) - total_bad
        
        df['bad_pct'] = df['bad_cumsum'] / total_bad
        df['good_pct'] = df['good_cumsum'] / total_good
        
        df['ks'] = abs(df['bad_pct'] - df['good_pct'])
        
        return df['ks'].max()
    
    def plot_roc_curve(self, X_test, y_test):
        """
        Plot ROC curve
        """
        probs = self.predict_probability(X_test)
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'Model (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
    
    def get_feature_importance(self):
        """
        Get feature importance from model coefficients
        
        Returns:
        --------
        pd.DataFrame
            Feature importance rankings
        """
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_[0]
        })
        
        importance_df = importance_df.sort_values('coefficient', key=abs, ascending=False)
        
        return importance_df


def main():
    """
    Main execution function
    """
    print("Starting Loan Default Risk Assessment...")
    
    # Initialize model
    risk_model = DefaultRiskModel(base_score=600, pdo=50)
    
    # Load data
    print("\n1. Loading data...")
    data = risk_model.load_data(
        'customer_scorecard_input.xlsx',
        'bureau_data.xlsx'
    )
    print(f"   Loaded {len(data)} records")
    
    # Prepare target
    print("\n2. Preparing target variable...")
    data['target'] = risk_model.prepare_target(data)
    print(f"   Default rate: {data['target'].mean()*100:.2f}%")
    
    # Select features
    print("\n3. Selecting features...")
    features = risk_model.select_features(data)
    print(f"   Selected {len(features)} features")
    
    # Prepare modeling data
    model_data = data[features + ['target']].dropna()
    X = model_data[features]
    y = model_data['target']
    
    # Split data
    print("\n4. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train model
    print("\n5. Training model...")
    risk_model.train(X_train, y_train)
    
    # Evaluate
    print("\n6. Evaluating model...")
    metrics = risk_model.evaluate(X_test, y_test)
    print(f"   AUC: {metrics['AUC']:.4f}")
    print(f"   KS: {metrics['KS']:.4f}")
    print(f"   Gini: {metrics['Gini']:.4f}")
    
    # Generate scores
    print("\n7. Generating risk scores...")
    test_probs = risk_model.predict_probability(X_test)
    test_scores = risk_model.convert_to_score(test_probs)
    
    # Save results
    print("\n8. Saving results...")
    results = pd.DataFrame({
        'actual_default': y_test.values,
        'predicted_probability': test_probs,
        'risk_score': test_scores
    })
    results.to_excel('model_output.xlsx', index=False)
    
    # Save feature importance
    importance = risk_model.get_feature_importance()
    importance.to_excel('variable_importance.xlsx', index=False)
    
    print("\n✓ Analysis complete!")
    print("\nTop 5 Most Important Features:")
    print(importance.head())
    
    # Plot ROC curve
    risk_model.plot_roc_curve(X_test, y_test)


if __name__ == "__main__":
    main()
