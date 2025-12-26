import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}

    def load_data(self, filepath):
        """Load and combine datasets"""
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records")
        return df

    def clean_data(self, df):
        """Clean and prepare data with intelligent missing value handling"""
        print(f"Initial missing values:\n{df.isnull().sum()}")

        # Handle missing job_title - fill with 'Not Specified'
        df['job_title'] = df['job_title'].fillna('Not Specified')

        # Handle missing required_skills - fill with empty string
        df['required_skills'] = df['required_skills'].fillna('')

        # Handle missing salary_usd with intelligent imputation
        # Strategy: Use median salary based on experience_level and employment_type
        if df['salary_usd'].isnull().any():
            print(f"Imputing {df['salary_usd'].isnull().sum()} missing salary values...")

            # Create groups for imputation
            for idx, row in df[df['salary_usd'].isnull()].iterrows():
                # Try to find similar jobs by experience level and employment type
                similar_jobs = df[
                    (df['experience_level'] == row['experience_level']) &
                    (df['employment_type'] == row['employment_type']) &
                    (df['salary_usd'].notnull())
                ]

                if len(similar_jobs) > 0:
                    df.loc[idx, 'salary_usd'] = similar_jobs['salary_usd'].median()
                else:
                    # Fallback to overall median if no similar jobs found
                    df.loc[idx, 'salary_usd'] = df['salary_usd'].median()

        # Handle other numeric columns with median imputation
        numeric_cols = ['remote_ratio', 'years_experience', 'job_description_length', 'benefits_score']
        for col in numeric_cols:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        # Handle categorical columns with mode or 'Unknown'
        categorical_cols = ['experience_level', 'employment_type', 'company_location',
                           'company_size', 'education_required', 'industry']
        for col in categorical_cols:
            if col in df.columns and df[col].isnull().any():
                mode_value = df[col].mode()
                if len(mode_value) > 0:
                    df[col] = df[col].fillna(mode_value[0])
                else:
                    df[col] = df[col].fillna('Unknown')

        # Convert dates
        df['posting_date'] = pd.to_datetime(df['posting_date'], errors='coerce')
        df['application_deadline'] = pd.to_datetime(df['application_deadline'], errors='coerce')

        # Handle missing dates - use a default or forward fill
        if df['posting_date'].isnull().any():
            df['posting_date'] = df['posting_date'].fillna(pd.Timestamp.now())
        if df['application_deadline'].isnull().any():
            # Set deadline to 30 days after posting date if missing
            df['application_deadline'] = df['application_deadline'].fillna(
                df['posting_date'] + pd.Timedelta(days=30)
            )

        # Extract skills as list (handle empty strings)
        df['skills_list'] = df['required_skills'].apply(
            lambda x: [s.strip() for s in x.split(',') if s.strip()] if x else []
        )
        df['num_skills'] = df['skills_list'].apply(len)

        print(f"Final missing values:\n{df.isnull().sum()}")
        print(f"Retained {len(df)} records (no rows dropped!)")

        return df

    def engineer_features(self, df):
        """Create new features"""
        # Extract year and month from posting date
        df['posting_year'] = df['posting_date'].dt.year
        df['posting_month'] = df['posting_date'].dt.month

        # Days until deadline
        df['days_to_apply'] = (df['application_deadline'] - df['posting_date']).dt.days

        # Create binary features for common skills
        common_skills = ['Python', 'AWS', 'Docker', 'Kubernetes', 'SQL',
                        'Deep Learning', 'NLP', 'TensorFlow', 'PyTorch']

        for skill in common_skills:
            df[f'has_{skill.lower().replace(" ", "_")}'] = df['required_skills'].str.contains(skill, case=False, na=False).astype(int)

        return df

    def encode_features(self, df, is_training=True):
        """Encode categorical variables"""
        categorical_cols = ['experience_level', 'employment_type', 'company_location',
                           'company_size', 'education_required', 'industry']

        for col in categorical_cols:
            if is_training:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                df[f'{col}_encoded'] = df[col].apply(lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1)

        return df

    def get_feature_columns(self):
        """Return list of feature columns for modeling"""
        feature_cols = [
            'experience_level_encoded', 'employment_type_encoded',
            'company_location_encoded', 'company_size_encoded',
            'education_required_encoded', 'industry_encoded',
            'remote_ratio', 'years_experience', 'num_skills',
            'job_description_length', 'benefits_score',
            'posting_month', 'days_to_apply'
        ]

        # Add skill binary features
        common_skills = ['Python', 'AWS', 'Docker', 'Kubernetes', 'SQL',
                        'Deep Learning', 'NLP', 'TensorFlow', 'PyTorch']
        for skill in common_skills:
            feature_cols.append(f'has_{skill.lower().replace(" ", "_")}')

        return feature_cols

    def save_encoders(self, filepath):
        """Save label encoders"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.label_encoders, f)

    def load_encoders(self, filepath):
        """Load label encoders"""
        with open(filepath, 'rb') as f:
            self.label_encoders = pickle.load(f)
