"""Utility functions for the AI Job Market Analyzer"""

def get_experience_label(code):
    """Convert experience level code to readable label"""
    mapping = {
        'EN': 'Entry Level',
        'MI': 'Mid Level',
        'SE': 'Senior',
        'EX': 'Executive'
    }
    return mapping.get(code, code)

def get_employment_label(code):
    """Convert employment type code to readable label"""
    mapping = {
        'FT': 'Full-time',
        'PT': 'Part-time',
        'CT': 'Contract',
        'FL': 'Freelance'
    }
    return mapping.get(code, code)

def get_company_size_label(code):
    """Convert company size code to readable label"""
    mapping = {
        'S': 'Small',
        'M': 'Medium',
        'L': 'Large'
    }
    return mapping.get(code, code)

def create_readable_dataframe(df):
    """Add readable label columns to dataframe"""
    df_copy = df.copy()
    df_copy['experience_label'] = df_copy['experience_level'].apply(get_experience_label)
    df_copy['employment_label'] = df_copy['employment_type'].apply(get_employment_label)
    df_copy['company_size_label'] = df_copy['company_size'].apply(get_company_size_label)
    return df_copy
