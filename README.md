Content Monetization Modeler

🏆 Project Overview:
This project predicts potential YouTube ad revenue using multiple regression models. By analyzing video performance metrics and contextual data, it helps content creators and media companies make data-driven decisions for content strategy, revenue forecasting, and ad campaign planning.

💡 Skills Learned:
Machine Learning: Regression models, predictive modeling, feature engineering

Data Processing: Data cleaning, handling missing values, outlier detection, categorical encoding

Data Analysis: Exploratory Data Analysis (EDA), regression metrics (R², RMSE, MAE), data visualization

Tools & Technologies: Python, Pandas, Scikit-learn, Streamlit

Domain Knowledge: Social media analytics, content monetization


📊 Problem Statement:
As video creators and media companies increasingly rely on platforms like YouTube for income, predicting potential ad revenue becomes essential for business planning and content strategy.

🗂 Dataset
Name: YouTube Monetization Modeler


Format: CSV (~122,000 rows)

Source: Synthetic (created for learning purposes)

Target Variable: ad_revenue_usd

Columns:

| Column                                   | Description                     |
| ---------------------------------------- | ------------------------------- |
| video_id                                 | Unique identifier for the video |
| date                                     | Upload or report date           |
| views, likes, comments                   | Performance metrics             |
| watch_time_minutes, video_length_minutes | Engagement and content length   |
| subscribers                              | Channel subscriber count        |
| category, device, country                | Contextual information          |
| ad_revenue_usd                           | Revenue generated (target)      |


🛠 Preprocessing Steps:

Handle ~5% missing values in key columns

Remove ~2% duplicated records

Encode categorical variables (category, device, country)

Normalize/scale features if necessary


🎯 Business Use Cases:

Content Strategy Optimization: Identify which content types yield the highest returns

Revenue Forecasting: Predict expected income from future video uploads

Creator Support Tools: Integrate insights into analytics platforms for YouTubers

Ad Campaign Planning: Forecast ROI for advertisers


🧩 Approach:

1.Understand the Dataset: Load and inspect the data

2.Exploratory Data Analysis (EDA): Identify trends, correlations, and outliers

3.Preprocessing: Handle missing values, remove duplicates, encode categorical features

4.Feature Engineering: Create new metrics like engagement rate ((likes + comments)/views)

5.Model Building: Train and compare 5 regression models to predict ad_revenue_usd

6.Model Evaluation: Use R², RMSE, MAE to select the best model

7.Streamlit App Development: Build an interactive app for predictions and visualizations

8.Interpretation & Insights: Highlight key drivers of ad revenue

9.Documentation: Ensure code and findings are clearly documented

✅ Results

A trained and evaluated regression model to predict YouTube ad revenue


A cleaned, preprocessed dataset ready for further analysis

Insights on features driving ad revenue

A Streamlit app to interactively test predictions

📂 File Structure

app.py                     # Streamlit app

decision_tree.pkl           # Trained decision tree model

gradient_boosting.pkl       # Trained gradient boosting model

lasso_regression.pkl        # Trained Lasso regression model

linear_regression.pkl       # Trained linear regression model

ridge_regression.pkl        # Trained Ridge regression model

youtube_ad_revenue_dataset.csv # Dataset

README.md                   # Project documentation


⚡ How to Run

1.Clone the repository:
git clone https://github.com/Keerthana-Mathaiyan/Content-Monetization-Modeler.git

2.Install dependencies:
pip install -r requirements.txt

3.Run the Streamlit app:
streamlit run app.py
