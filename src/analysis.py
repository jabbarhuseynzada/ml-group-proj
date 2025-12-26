import pandas as pd
import numpy as np
from collections import Counter

class JobMarketAnalyzer:
    def __init__(self, df):
        self.df = df

    def salary_statistics(self):
        """Get salary statistics by various factors"""
        stats = {
            'overall': {
                'mean': self.df['salary_usd'].mean(),
                'median': self.df['salary_usd'].median(),
                'min': self.df['salary_usd'].min(),
                'max': self.df['salary_usd'].max(),
                'std': self.df['salary_usd'].std()
            },
            'by_experience': self.df.groupby('experience_level')['salary_usd'].agg(['mean', 'median', 'count']).to_dict('index'),
            'by_location': self.df.groupby('company_location')['salary_usd'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False).head(10).to_dict('index'),
            'by_industry': self.df.groupby('industry')['salary_usd'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False).to_dict('index')
        }
        return stats

    def top_skills_analysis(self, top_n=15):
        """Analyze most in-demand skills"""
        all_skills = []
        for skills_list in self.df['skills_list']:
            all_skills.extend(skills_list)

        skill_counts = Counter(all_skills)
        top_skills = skill_counts.most_common(top_n)

        # Calculate average salary for each skill
        skill_salary = {}
        for skill, count in top_skills:
            salaries = self.df[self.df['required_skills'].str.contains(skill, case=False, na=False, regex=False)]['salary_usd']
            skill_salary[skill] = {
                'count': count,
                'avg_salary': salaries.mean(),
                'median_salary': salaries.median()
            }

        return skill_salary

    def remote_work_trends(self):
        """Analyze remote work patterns"""
        remote_stats = {
            'distribution': self.df['remote_ratio'].value_counts().to_dict(),
            'avg_salary_by_remote': self.df.groupby('remote_ratio')['salary_usd'].mean().to_dict(),
            'remote_by_industry': self.df.groupby('industry')['remote_ratio'].mean().sort_values(ascending=False).to_dict()
        }
        return remote_stats

    def job_growth_trends(self):
        """Analyze job posting trends over time"""
        self.df['year_month'] = self.df['posting_date'].dt.to_period('M')
        trends = {
            'monthly_postings': self.df.groupby('year_month').size().to_dict(),
            'monthly_avg_salary': self.df.groupby('year_month')['salary_usd'].mean().to_dict()
        }
        return trends

    def education_impact(self):
        """Analyze impact of education on salary"""
        edu_stats = self.df.groupby('education_required').agg({
            'salary_usd': ['mean', 'median', 'count'],
            'years_experience': 'mean'
        }).round(2)
        return edu_stats.to_dict()

    def skills_demand_forecast(self, user_skills):
        """Forecast demand for user's skills"""
        user_skills_list = [s.strip() for s in user_skills.split(',')]

        # Find jobs matching user skills
        matching_jobs = self.df[
            self.df['required_skills'].apply(
                lambda x: any(skill.lower() in x.lower() for skill in user_skills_list)
            )
        ]

        if len(matching_jobs) == 0:
            return {
                'matching_jobs': 0,
                'message': 'No jobs found matching your skills'
            }

        # Calculate statistics
        forecast = {
            'matching_jobs': len(matching_jobs),
            'percentage_of_market': (len(matching_jobs) / len(self.df)) * 100,
            'avg_salary': matching_jobs['salary_usd'].mean(),
            'median_salary': matching_jobs['salary_usd'].median(),
            'salary_range': {
                'min': matching_jobs['salary_usd'].min(),
                'max': matching_jobs['salary_usd'].max()
            },
            'top_industries': matching_jobs['industry'].value_counts().head(5).to_dict(),
            'top_locations': matching_jobs['company_location'].value_counts().head(5).to_dict(),
            'experience_levels': matching_jobs['experience_level'].value_counts().to_dict(),
            'remote_opportunities': (matching_jobs['remote_ratio'] > 0).sum(),
            'recommended_skills': self._get_recommended_skills(matching_jobs, user_skills_list)
        }

        return forecast

    def _get_recommended_skills(self, matching_jobs, user_skills):
        """Get skills commonly paired with user's skills"""
        all_paired_skills = []
        for skills_list in matching_jobs['skills_list']:
            for skill in skills_list:
                if skill.lower() not in [s.lower() for s in user_skills]:
                    all_paired_skills.append(skill)

        if all_paired_skills:
            skill_counts = Counter(all_paired_skills)
            return dict(skill_counts.most_common(5))
        return {}
