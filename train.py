import pandas as pd
from src.preprocessing import DataPreprocessor
from src.model import SalaryPredictor

def main():
    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Load data
    print("Loading data...")
    df = preprocessor.load_data('data/raw/ai_job_dataset.csv')

    # Clean and prepare data
    print("\nCleaning data...")
    df = preprocessor.clean_data(df)

    # Engineer features
    print("Engineering features...")
    df = preprocessor.engineer_features(df)

    # Encode categorical variables
    print("Encoding features...")
    df = preprocessor.encode_features(df, is_training=True)

    # Save encoders
    preprocessor.save_encoders('models/label_encoders.pkl')
    print("Label encoders saved")

    # Prepare features and target
    feature_cols = preprocessor.get_feature_columns()
    X = df[feature_cols]
    y = df['salary_usd']

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")

    # Train model
    print("\n" + "="*50)
    print("Training Salary Prediction Model")
    print("="*50)

    predictor = SalaryPredictor()
    metrics = predictor.train(X, y, model_type='random_forest')

    # Show feature importance
    print("\nTop 10 Most Important Features:")
    print(predictor.get_feature_importance(10))

    # Save model
    predictor.save_model('models/salary_model.pkl')

    # Save processed data for the app
    df.to_csv('data/processed_data.csv', index=False)
    print("\nProcessed data saved")

    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)

if __name__ == "__main__":
    main()
