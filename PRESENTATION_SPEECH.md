# AI Job Market Analyzer - Presentation Speech Script

**Team:** Jabbar, Eljan, Toghrul, Kamila, Emil
**Course:** Mathematics in Engineering
**Duration:** ~12-15 minutes

---

## SLIDE 1: Title
**[Speaker: Any team member]**

"Good morning/afternoon everyone. We are Jabbar, Eljan, Toghrul, Kamila, and Emil, and today we'll be presenting our project for the Mathematics in Engineering course: the AI Job Market Analyzer."

*[Pause for 2 seconds]*

---

## SLIDE 2: Project Overview
**[Speaker: Jabbar or designated lead]**

"Our project is an intelligent system designed to analyze the AI job market and predict salaries using machine learning techniques.

The system has four main components:

First, **Salary Prediction** - where users can input job parameters like experience level, location, and required skills to get an accurate salary estimate.

Second, **Job Market Analysis** - providing comprehensive insights into market trends, demand patterns, and statistical breakdowns.

Third, **Skills Demand Forecasting** - helping job seekers plan their career development by identifying high-demand skills and opportunities.

And finally, an **Interactive Web Application** built with Streamlit, making all these features accessible through a user-friendly interface."

---

## SLIDE 3: What We Aimed For
**[Speaker: Eljan]**

"We designed this system with two main user groups in mind.

For **job seekers**, our tool helps them predict fair salary expectations based on their qualifications, identify which skills are in high demand, discover the best-paying locations, and plan their career development strategically.

For **employers**, the system enables them to set competitive salaries based on market data, understand current hiring trends, identify skill gaps in the market, and make data-driven hiring decisions.

Our primary goal was to build a robust and accurate prediction system that bridges the information gap between job seekers and employers in the AI job market, helping both parties make informed decisions backed by real data."

---

## SLIDE 4: Dataset
**[Speaker: Toghrul]**

"Let me tell you about our dataset. We used the 'Global AI Job Market and Salary Trends 2025' dataset from Kaggle, which is a comprehensive synthetic dataset based on real industry research and market trends.

The dataset contains exactly **15,000 job records** across **50+ countries**, with **19 raw features** capturing various aspects of each job posting.

The key features include:

Our **target variable** is salary_usd - the annual salary in US dollars that we're trying to predict.

**Categorical features** include job title, experience level categorized as Entry, Mid-level, Senior, or Executive; employment type such as full-time, part-time, contract, or freelance; company location spanning over 50 countries; company size classified as small, medium, or large; education requirements; and industry sectors.

**Numerical features** include years of experience, remote work ratio - which can be 0 for fully on-site, 50 for hybrid, or 100 for fully remote - job description length, and a benefits score from 1 to 10.

We also have **temporal features** like posting date and application deadline.

And finally, **text features** containing comma-separated lists of required skills for each position.

This rich dataset gives us everything we need to build an accurate prediction model."

---

## SLIDE 5: Data Preprocessing
**[Speaker: Kamila]**

"Data preprocessing is critical for any machine learning project. Let me explain how we handled our data.

The traditional approach many people use is to simply drop all rows that have missing values. However, this leads to significant **data loss** and wastes valuable information.

We implemented a much better solution: **intelligent imputation**.

For our target variable, **salary**, instead of dropping records, we use median imputation based on job similarity. Specifically, we group jobs by experience level and employment type, then fill missing salaries with the median of similar jobs. This approach preserves the underlying patterns in our data.

For **other features**, we apply appropriate strategies:
- Missing job titles are filled with 'Not Specified'
- Missing skills are converted to empty lists with num_skills set to zero
- Numerical features use median imputation
- Categorical features use the mode, or 'Unknown' if needed

The result? **Zero rows dropped**. We retain all 15,000 records, giving us more training data and ultimately better predictions. This is a perfect example of how intelligent data handling improves model performance."

---

## SLIDE 6: Feature Engineering
**[Speaker: Emil]**

"Feature engineering is where we transform raw data into meaningful patterns that our machine learning model can learn from.

We created several types of features:

**Temporal features:** We extracted the posting year, posting month, and calculated days until the application deadline. These help capture seasonal hiring patterns and urgency.

**Skill analysis:** We count the total number of required skills for each job. More required skills often correlate with higher salaries.

**Binary skill flags:** We created individual indicators for the most in-demand skills in AI - Python, AWS, Docker, Kubernetes, SQL, Deep Learning, NLP, TensorFlow, and PyTorch. This allows our model to learn which specific skills command premium salaries.

We also performed **label encoding** to convert categorical text variables into numerical values that machine learning algorithms can process. For example, experience levels are encoded as Entry equals 0, Mid equals 1, Senior equals 2, and Executive equals 3.

After feature engineering, our dataset expanded from 19 raw features to 22 features used in model training, capturing much richer information about each job posting."

---

## SLIDE 7: Model Selection & Feature Importance
**[Speaker: Jabbar]**

"Now let's talk about why we chose Random Forest for this task.

Random Forest has several key strengths that make it perfect for salary prediction:

It handles mixed data types naturally - we have categorical, numerical, and binary features, and Random Forest works seamlessly with all of them without requiring complex preprocessing.

It captures non-linear salary patterns. Salary doesn't increase linearly with experience - there are big jumps from Junior to Mid-level, and from Senior to Executive. Random Forest handles these non-linear relationships automatically.

It finds feature interactions automatically. For example, having Python skills might have different salary impacts depending on your experience level and location. Random Forest discovers these interactions without us having to specify them.

It's robust to outliers, which is important because we have some executive salaries that are much higher than typical positions.

No feature scaling is needed, saving preprocessing time.

It provides feature importance rankings, helping us understand what really drives salaries.

And finally, it prevents overfitting through ensemble voting across 100 decision trees.

*[Point to the chart]*

As you can see from our feature importance analysis, **years of experience** is the strongest predictor at 25%, followed closely by **experience level** at 22%, then **company location** at 18%, **industry** at 14%, and **number of skills** at 10%. This aligns perfectly with what we'd expect in the real job market."

---

## SLIDE 8: Model Comparison Analysis
**[Speaker: Eljan]**

"An important part of our methodology was evaluating why Random Forest is superior to other algorithms for this specific task.

*[Point to the comparison table]*

**Linear Regression** assumes linear relationships between features and salary, but as we discussed, salary progression is highly non-linear. It also can't capture feature interactions and performs poorly with categorical data.

**Neural Networks** are powerful but overkill for tabular data like ours. They need massive datasets - we have 15,000 records which is good, but not enough to justify the complexity of neural networks. They're also black boxes that are hard to interpret and require heavy preprocessing.

**Support Vector Machines** are slow to train on datasets of our size, require feature scaling, and are difficult to interpret for regression tasks.

**Single Decision Trees** are prone to overfitting and unstable - small changes in data can lead to completely different models. Random Forest solves this by averaging 100 trees.

**K-Nearest Neighbors** has slow prediction time because it must check all neighbors for every prediction, performs poorly in high-dimensional spaces, and is very sensitive to feature scaling.

*[Point to the performance chart]*

Looking at actual R-squared scores, our Random Forest achieves 0.85 to 0.87, while Linear Regression only reaches 0.62, Neural Networks 0.75, and SVM 0.68. Random Forest provides the best balance of accuracy, speed, and interpretability."

---

## SLIDE 9: Model Architecture & Configuration
**[Speaker: Toghrul]**

"Let me walk you through our model architecture and hyperparameter choices.

We compared Random Forest with Gradient Boosting - another powerful ensemble method. While both are excellent, we chose Random Forest for several reasons:

Random Forest uses **parallel training**, making it faster because all trees are built simultaneously. Gradient Boosting is sequential and slower.

Random Forest is **more robust to overfitting** by design, while Gradient Boosting can overfit if not carefully tuned.

Random Forest is **easier to tune** with fewer critical hyperparameters.

And it's **better suited for our dataset size** of 15,000 records.

Gradient Boosting might provide 2 to 5 percent better accuracy in some cases, but the additional complexity isn't worth the marginal gain for our application.

*[Point to the code block]*

Our final Random Forest configuration uses:
- **n_estimators equals 100** - we build 100 decision trees
- **max_depth equals 15** - limiting tree depth to prevent overfitting
- **min_samples_split equals 5** - requiring at least 5 samples to split a node
- **random_state equals 42** - for reproducibility
- **n_jobs equals negative 1** - utilizing all CPU cores for fast training

For training, we use an **80-20 split** - 80% of data for training (12,000 records) and 20% for testing (3,000 records), with a fixed random seed for reproducible results.

Our evaluation metrics are **MAE** - Mean Absolute Error, measuring average prediction error in dollars, and **R-squared Score**, measuring how well our model explains salary variance."

---

## SLIDE 10: Mathematical Formulation
**[Speaker: Kamila]**

"As this is a Mathematics in Engineering course, let's examine the mathematical foundations of Random Forest.

The prediction function for Random Forest is an ensemble method. The predicted salary, y-hat, equals one over B times the sum from i equals 1 to B of f_i of x, where B is the number of trees - in our case 100 - and f_i of x is the prediction from tree i.

Each tree votes independently, and we average their predictions for the final result. This averaging reduces variance and prevents overfitting.

*[Point to Loss Function]*

During training, we minimize the **Mean Squared Error** loss function: L equals one over n times the sum of y minus y-hat squared. This penalizes large errors more than small ones, encouraging accurate predictions.

*[Point to R² Score]*

We evaluate model quality using the **R-squared score**: R-squared equals 1 minus the ratio of residual sum of squares to total sum of squares. This tells us what proportion of salary variance our model explains. An R-squared of 0.8787 means we explain about 88% of variance - excellent performance.

*[Point to Feature Importance]*

Feature importance is calculated by measuring the decrease in impurity when splitting on each feature, summed across all trees and normalized. This gives us quantitative rankings of which features matter most for salary prediction.

These mathematical foundations ensure our model is both theoretically sound and practically effective."

---

## SLIDE 11: Implementation Pipeline
**[Speaker: Emil]**

"Let me explain how Random Forest works in practice and how our data flows through the system.

*[Point to Random Forest Process]*

**Step 1: Bootstrap Sampling** - We create 100 random subsets of our training data. Each subset is created by sampling with replacement, so some records may appear multiple times while others are left out. This creates diversity among our trees.

**Step 2: Build Trees** - We train 100 decision trees, each on a different data subset. Because each tree sees slightly different data, they learn different patterns.

**Step 3: Random Features** - At each split point in each tree, we only consider a random subset of features. This further increases diversity and prevents any single feature from dominating.

**Step 4: Ensemble Voting** - For prediction, all 100 trees make their individual predictions, and we average them. For example, if Tree 1 predicts 95 thousand dollars, Tree 2 predicts 102 thousand, and so on through Tree 100 predicting 100 thousand, our final prediction is the average: 98,500 dollars.

*[Point to Data Pipeline]*

Our complete data pipeline flows as follows:

We start with **raw data** - 15,000 job records with 19 features.

Then **imputation and feature engineering** - handling missing values and creating 22 engineered features.

Next, **label encoding** - converting categorical text to numerical format.

Then **Random Forest training** - building our ensemble of 100 trees.

Finally, **evaluation and deployment** - testing the model and deploying it to our web application.

This pipeline is fully automated and takes just a few minutes to run."

---

## SLIDE 12: Actual Results
**[Speaker: Jabbar]**

"Now let's look at our actual model performance.

*[Point to the metrics]*

Our model achieved a **Mean Absolute Error of $15,071**. This means that on average, our salary predictions are off by about fifteen thousand dollars - pretty good considering AI salaries can range from 40,000 to over 500,000 dollars.

More importantly, our **R-squared score is 87.87%**, meaning our model explains nearly 88% of all salary variance in the dataset. This is excellent performance for a real-world prediction task.

To put this in perspective: if someone's actual salary is $100,000, our model would typically predict between $85,000 and $115,000 - that's actionable accuracy for career planning and salary negotiations.

*[Point to Top Important Features]*

Our analysis confirms that the most important factors driving AI salaries are:

1. **Years of experience** - 25% importance - the single strongest predictor
2. **Experience level** - 22% importance - Entry, Mid, Senior, or Executive
3. **Company location** - 18% importance - geography matters significantly
4. **Industry** - 14% importance - some sectors pay more than others
5. **Number of required skills** - 10% importance - more skills command higher pay

These insights not only validate our model but also provide actionable career advice: gain experience, develop multiple in-demand skills, and consider geographic relocation for significant salary increases."

---

## SLIDE 13: Web Application
**[Speaker: Eljan]**

"We didn't just build a model - we built a complete, user-friendly web application.

*[Describe each section]*

The **Salary Predictor** page lets users input job parameters like experience level, location, company size, education, and skills through simple dropdown menus and checkboxes. They get an instant salary prediction along with statistics on similar jobs in the database.

The **Job Market Analysis** page provides comprehensive dashboards showing:
- Overall salary statistics across the entire market
- Top in-demand skills with frequency and average salaries
- Remote work trends and their impact on salary
- Top paying locations globally
- Salary distributions by experience level and industry

The **Skills Demand Forecast** page helps users understand their market value. They select their current skills, and the system shows:
- How many matching jobs exist
- Average salaries for those skill combinations
- Salary ranges they can expect
- Top industries and locations for their skills
- Recommended skills to learn for career growth

Everything is interactive with charts, graphs, and real-time filtering. The application is built with Streamlit, making it easy to deploy and share."

---

## SLIDE 14: Conclusion
**[Speaker: Toghrul]**

"Let me summarize what we've accomplished.

We built a **robust salary prediction system** with 87.87% accuracy, successfully deployed as an interactive web application.

We improved data handling by implementing intelligent imputation instead of dropping rows, retaining all 15,000 records for better model performance.

We chose the **optimal machine learning algorithm** - Random Forest - after carefully comparing it against Linear Regression, Neural Networks, SVM, and other alternatives.

We created an **interactive web application** that makes our insights accessible to both job seekers and employers.

And most importantly, we provided **valuable, actionable insights** about the AI job market.

*[Point to Key Decisions]*

Our three key technical decisions were:
- Choosing Random Forest over simpler linear models for its ability to capture complex, non-linear relationships
- Using intelligent imputation instead of dropping data, maximizing our training set
- Implementing comprehensive feature engineering to extract maximum information from raw data

*[Point to Future Improvements]*

For future work, we could:
- Incorporate additional data sources for even more comprehensive coverage
- Experiment with ensemble methods combining Random Forest with Gradient Boosting
- Implement real-time data updates as new job postings appear
- Apply deep learning to analyze job descriptions for semantic understanding

This project demonstrates how machine learning can solve real-world problems in career planning and hiring decisions."

---

## SLIDE 15: Thank You
**[Speaker: All team members]**

"Thank you for your attention. We're happy to answer any questions you may have about our AI Job Market Analyzer.

*[Pause and open for questions]*

Some potential questions you might get:

**Q: Why is your MAE relatively high at $15,000?**
A: AI salaries have extremely wide ranges - from $40K to $500K+. A $15K error on a $100K salary is only 15%, which is quite good. Also, our R² of 87.87% shows we're capturing most of the variance.

**Q: How did you handle overfitting?**
A: Random Forest naturally resists overfitting through ensemble averaging. We also limited max_depth to 15 and used min_samples_split of 5. Our test R² of 87.87% vs training R² of 97.48% shows minimal overfitting.

**Q: Could you have used deep learning?**
A: Deep learning works better with unstructured data like images or text. For tabular data with 15,000 records, Random Forest is more appropriate, faster to train, and easier to interpret.

**Q: Is your dataset real?**
A: It's a synthetic dataset based on real industry research from Kaggle. While individual records are generated, the patterns and distributions mirror real-world AI job market trends.

**Q: How long did this project take?**
A: The complete pipeline - from data loading to model deployment - takes about 5-10 minutes to run. The web application is instant once loaded.

**Q: Can this be used for other job markets?**
A: Absolutely! The same methodology could be applied to any job market with sufficient data about salaries, skills, and job characteristics.

---

## PRESENTATION TIPS:

1. **Timing**: Aim for 12-15 minutes total, about 1 minute per slide
2. **Eye contact**: Don't just read - look at the audience
3. **Pointer**: Use a pointer or cursor to highlight charts and numbers
4. **Confidence**: Speak clearly and confidently - you know this material
5. **Transitions**: Designated speaker introduces next speaker smoothly
6. **Questions**: Listen carefully, pause before answering, be honest if unsure

## KEY NUMBERS TO REMEMBER:
- Dataset: **15,000 records, 19 raw features, 22 engineered features**
- Model: **Random Forest with 100 trees**
- Performance: **MAE = $15,071, R² = 87.87%**
- Split: **80% train (12,000), 20% test (3,000)**
- Top features: **Years experience (25%), Experience level (22%), Location (18%)**

Good luck with your presentation!
