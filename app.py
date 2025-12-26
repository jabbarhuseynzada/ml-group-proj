import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.preprocessing import DataPreprocessor
from src.model import SalaryPredictor
from src.analysis import JobMarketAnalyzer
from src.utils import get_experience_label, get_employment_label, get_company_size_label
import os

# Page config
st.set_page_config(page_title="AI Job Market Analyzer", layout="wide")

# Load data and models
@st.cache_data
def load_data():
    if os.path.exists('data/processed_data.csv'):
        return pd.read_csv('data/processed_data.csv')
    else:
        st.error("Please run train.py first to process the data!")
        return None

@st.cache_resource
def load_model():
    if os.path.exists('models/salary_model.pkl'):
        predictor = SalaryPredictor()
        predictor.load_model('models/salary_model.pkl')
        return predictor
    else:
        st.error("Please run train.py first to train the model!")
        return None

@st.cache_resource
def load_preprocessor():
    if os.path.exists('models/label_encoders.pkl'):
        preprocessor = DataPreprocessor()
        preprocessor.load_encoders('models/label_encoders.pkl')
        return preprocessor
    else:
        st.error("Please run train.py first!")
        return None

@st.cache_data
def get_all_skills(df):
    """Extract all unique skills from the dataset"""
    all_skills = set()
    for skills_str in df['skills_list'].dropna():
        # Parse the string representation of list
        if isinstance(skills_str, str):
            import ast
            skills_list = ast.literal_eval(skills_str)
        else:
            skills_list = skills_str
        all_skills.update(skills_list)
    return sorted(list(all_skills))

# Main app
def main():
    st.title("AI Job Market Analyzer")
    st.markdown("Analyze the AI job market, predict salaries, and forecast skills demand")

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page",
                           ["Salary Predictor", "Job Market Analysis", "Skills Demand Forecast"])

    # Load data
    df = load_data()
    if df is None:
        return

    analyzer = JobMarketAnalyzer(df)

    # Page routing
    if page == "Salary Predictor":
        salary_predictor_page(df, load_model(), load_preprocessor())
    elif page == "Job Market Analysis":
        job_market_analysis_page(analyzer, df)
    elif page == "Skills Demand Forecast":
        skills_forecast_page(analyzer, df)

def salary_predictor_page(df, model, preprocessor):
    st.header("Salary Predictor")
    st.markdown("Predict your potential salary based on job parameters")

    # Create readable mappings
    experience_options = {get_experience_label(x): x for x in df['experience_level'].unique()}
    employment_options = {get_employment_label(x): x for x in df['employment_type'].unique()}
    company_size_options = {get_company_size_label(x): x for x in df['company_size'].unique()}

    col1, col2 = st.columns(2)

    with col1:
        experience_label = st.selectbox("Experience Level", sorted(experience_options.keys()))
        experience = experience_options[experience_label]

        employment_label = st.selectbox("Employment Type", sorted(employment_options.keys()))
        employment = employment_options[employment_label]

        location = st.selectbox("Company Location", sorted(df['company_location'].unique()))

        company_size_label = st.selectbox("Company Size", sorted(company_size_options.keys()))
        company_size = company_size_options[company_size_label]

    with col2:
        education = st.selectbox("Education Required", df['education_required'].unique())
        industry = st.selectbox("Industry", sorted(df['industry'].unique()))
        years_exp = st.slider("Years of Experience", 0, 20, 3)
        remote_ratio = st.slider("Remote Work %", 0, 100, 50)

    st.subheader("Skills")
    col3, col4 = st.columns(2)

    with col3:
        python = st.checkbox("Python", value=True)
        aws = st.checkbox("AWS")
        docker = st.checkbox("Docker")
        kubernetes = st.checkbox("Kubernetes")
        sql = st.checkbox("SQL")

    with col4:
        deep_learning = st.checkbox("Deep Learning")
        nlp = st.checkbox("NLP")
        tensorflow = st.checkbox("TensorFlow")
        pytorch = st.checkbox("PyTorch")

    num_skills = sum([python, aws, docker, kubernetes, sql, deep_learning, nlp, tensorflow, pytorch])

    # Use average values for job_desc_length and benefits_score
    job_desc_length = int(df['job_description_length'].mean())
    benefits_score = df['benefits_score'].mean()

    if st.button("Predict Salary", type="primary"):
        if model and preprocessor:
            # Prepare input data
            input_data = pd.DataFrame({
                'experience_level': [experience],
                'employment_type': [employment],
                'company_location': [location],
                'company_size': [company_size],
                'education_required': [education],
                'industry': [industry],
                'remote_ratio': [remote_ratio],
                'years_experience': [years_exp],
                'num_skills': [num_skills],
                'job_description_length': [job_desc_length],
                'benefits_score': [benefits_score],
                'posting_month': [6],  # Default value
                'days_to_apply': [30],  # Default value
                'has_python': [int(python)],
                'has_aws': [int(aws)],
                'has_docker': [int(docker)],
                'has_kubernetes': [int(kubernetes)],
                'has_sql': [int(sql)],
                'has_deep_learning': [int(deep_learning)],
                'has_nlp': [int(nlp)],
                'has_tensorflow': [int(tensorflow)],
                'has_pytorch': [int(pytorch)]
            })

            # Encode categorical features
            input_data = preprocessor.encode_features(input_data, is_training=False)

            # Get feature columns
            feature_cols = preprocessor.get_feature_columns()
            X_pred = input_data[feature_cols]

            # Predict
            prediction = model.predict(X_pred)[0]

            # Display result
            st.success(f"### Predicted Salary: ${prediction:,.2f} USD/year")

            # Show similar jobs
            similar_jobs = df[
                (df['experience_level'] == experience) &
                (df['industry'] == industry)
            ]['salary_usd']

            if len(similar_jobs) > 0:
                st.info(f"Similar jobs in {industry} with {experience_label} level earn: ${similar_jobs.mean():,.2f} (avg) | ${similar_jobs.median():,.2f} (median)")

def job_market_analysis_page(analyzer, df):
    st.header("Job Market Analysis")

    # Salary statistics
    st.subheader("Salary Statistics")
    stats = analyzer.salary_statistics()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Salary", f"${stats['overall']['mean']:,.0f}")
    col2.metric("Median Salary", f"${stats['overall']['median']:,.0f}")
    col3.metric("Min Salary", f"${stats['overall']['min']:,.0f}")
    col4.metric("Max Salary", f"${stats['overall']['max']:,.0f}")

    # Top skills
    st.subheader("Top In-Demand Skills")
    skills_data = analyzer.top_skills_analysis(15)
    skills_df = pd.DataFrame([
        {'Skill': skill, 'Count': data['count'], 'Avg Salary': data['avg_salary']}
        for skill, data in skills_data.items()
    ])

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(skills_df, x='Skill', y='Count', title='Most Demanded Skills')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(skills_df, x='Skill', y='Avg Salary', title='Average Salary by Skill')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    # Salary by experience
    st.subheader("Salary by Experience Level")
    exp_data = pd.DataFrame(stats['by_experience']).T
    exp_data.index = [get_experience_label(x) for x in exp_data.index]
    fig = px.bar(exp_data, x=exp_data.index, y='mean', title='Average Salary by Experience')
    st.plotly_chart(fig, use_container_width=True)

    # Remote work trends
    st.subheader("Remote Work Analysis")
    remote_stats = analyzer.remote_work_trends()

    col1, col2 = st.columns(2)
    with col1:
        remote_dist = pd.DataFrame(list(remote_stats['distribution'].items()),
                                   columns=['Remote %', 'Count'])
        fig = px.pie(remote_dist, values='Count', names='Remote %',
                    title='Remote Work Distribution')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        remote_salary = pd.DataFrame(list(remote_stats['avg_salary_by_remote'].items()),
                                     columns=['Remote %', 'Avg Salary'])
        fig = px.bar(remote_salary, x='Remote %', y='Avg Salary',
                    title='Average Salary by Remote %')
        st.plotly_chart(fig, use_container_width=True)

    # Top paying locations
    st.subheader("Top Paying Locations")
    location_data = pd.DataFrame(stats['by_location']).T.head(10)
    fig = px.bar(location_data, x=location_data.index, y='mean',
                title='Top 10 Highest Paying Locations')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

def skills_forecast_page(analyzer, df):
    st.header("Skills Demand Forecast")
    st.markdown("Select your skills to see job opportunities and recommendations")

    # Get all available skills
    all_skills = get_all_skills(df)

    st.subheader("Available Skills in Dataset")
    st.markdown(f"**{len(all_skills)} skills available:** {', '.join(all_skills)}")

    st.write("")  # Add spacing

    # Create checkboxes in a grid layout for better UX
    st.subheader("Select Your Skills")

    # Create columns for checkboxes
    cols = st.columns(4)
    selected_skills = []

    for idx, skill in enumerate(all_skills):
        col_idx = idx % 4
        with cols[col_idx]:
            if st.checkbox(skill, key=f"skill_{skill}"):
                selected_skills.append(skill)

    st.write("")  # Add spacing

    # Show selected skills count
    if selected_skills:
        st.info(f"Selected {len(selected_skills)} skill(s): {', '.join(selected_skills)}")

    if st.button("Analyze Selected Skills", type="primary"):
        if selected_skills:
            user_skills = ', '.join(selected_skills)
            forecast = analyzer.skills_demand_forecast(user_skills)

            if forecast['matching_jobs'] > 0:
                st.success(f"### Found {forecast['matching_jobs']} matching jobs!")

                col1, col2, col3 = st.columns(3)
                col1.metric("Market Share", f"{forecast['percentage_of_market']:.1f}%")
                col2.metric("Avg Salary", f"${forecast['avg_salary']:,.0f}")
                col3.metric("Remote Jobs", forecast['remote_opportunities'])

                st.subheader("Salary Range")
                st.info(f"${forecast['salary_range']['min']:,.0f} - ${forecast['salary_range']['max']:,.0f}")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Top Industries")
                    industries_df = pd.DataFrame(list(forecast['top_industries'].items()),
                                                columns=['Industry', 'Jobs'])
                    fig = px.bar(industries_df, x='Jobs', y='Industry', orientation='h')
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("Top Locations")
                    locations_df = pd.DataFrame(list(forecast['top_locations'].items()),
                                               columns=['Location', 'Jobs'])
                    fig = px.bar(locations_df, x='Jobs', y='Location', orientation='h')
                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("Recommended Skills to Learn")
                if forecast['recommended_skills']:
                    rec_skills_df = pd.DataFrame(list(forecast['recommended_skills'].items()),
                                                columns=['Skill', 'Frequency'])
                    fig = px.bar(rec_skills_df, x='Skill', y='Frequency',
                               title='Skills often paired with yours')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No additional skill recommendations at this time")

                st.subheader("Experience Level Distribution")
                exp_df = pd.DataFrame(list(forecast['experience_levels'].items()),
                                     columns=['Level', 'Count'])
                exp_df['Level'] = exp_df['Level'].apply(get_experience_label)
                fig = px.pie(exp_df, values='Count', names='Level')
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("No matching jobs found. Try selecting more skills!")
        else:
            st.warning("Please select at least one skill from the list")

if __name__ == "__main__":
    main()
