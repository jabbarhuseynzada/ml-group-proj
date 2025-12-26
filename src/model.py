import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

class SalaryPredictor:
    def __init__(self):
        self.model = None
        self.feature_importance = None

    def train(self, X, y, model_type='random_forest'):
        """Train the salary prediction model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Initialize model
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.1,
                random_state=42
            )

        # Train
        print(f"Training {model_type} model...")
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        print(f"\nTraining Results:")
        print(f"Train MAE: ${train_mae:,.2f}")
        print(f"Test MAE: ${test_mae:,.2f}")
        print(f"Train R²: {train_r2:.3f}")
        print(f"Test R²: {test_r2:.3f}")

        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }

    def predict(self, X):
        """Predict salary for new data"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        return self.model.predict(X)

    def get_feature_importance(self, top_n=10):
        """Get top N important features"""
        return self.feature_importance.head(top_n)

    def save_model(self, filepath):
        """Save trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {filepath}")
