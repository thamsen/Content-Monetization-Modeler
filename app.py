import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.set_page_config(page_title="Content Monetization Modeler", layout="wide", page_icon="💰")
st.markdown("""
    <style>
    body {background-color:#0e1117; color:dark;}
    .sidebar .sidebar-content {background-color:#1c1c24;}
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Home", "Prediction", "About"])

# Load dataset
df = pd.read_csv(r"C:/Users/Thamaraiselvan M/OneDrive/Desktop/content monetization modele/youtube_ad_revenue_dataset.csv")

# Keep numeric cols only + fill NaN with mean
numeric_df = df.select_dtypes(include=np.number)
numeric_df = numeric_df.fillna(numeric_df.mean())

# Home Page
if menu == "Home":
    st.title("📊 Content Monetization Dashboard")
    tabs = st.tabs(["Heatmap", "Model Comparison", "YouTube Content Insights", "Insights"])
    # Heatmap
    with tabs[0]:
        st.subheader("Correlation Heatmap")
        if numeric_df.shape[1] < 2:
            st.warning("Not enough numeric columns for correlation heatmap.")
        else:
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(10,7))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
    
    # Model comparison
    with tabs[1]:
        st.subheader("Model Performance Comparison (R2 Score)")
        target_col = 'ad_revenue_usd'
        if target_col not in numeric_df.columns:
            st.warning(f"Target column '{target_col}' not found.")
        else:
            feature_cols = numeric_df.columns.drop(target_col)
            X = numeric_df[feature_cols]
            y = numeric_df[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor()
            }
            
            results = []
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                r2 = r2_score(y_test, preds)
                mae = mean_absolute_error(y_test, preds)
                mse = mean_squared_error(y_test, preds)
                rmse = np.sqrt(mse)
            # Display table of R2 scores
                results.append({
                    "Model": name,
                    "R2 Score": r2,
                    "MAE": mae,
                    "MSE": mse,
                    "RMSE": rmse
                })

            results_df = pd.DataFrame(results)
            results_df = results_df.round(4)
            st.write("### Performance Metrics for Models")
            st.table(results_df)
            #barplot
            fig, ax = plt.subplots(figsize=(10,5))
            sns.barplot(data=results_df, x="Model", y="R2 Score", palette="viridis", ax=ax)
            ax.set_ylim(0, 1)
            ax.set_ylabel("R2 Score")
            st.pyplot(fig)
            
            analysis_text = """
            🔎**Analysis:**

            
                ✨ Lasso has the lowest MAE, MSE, and RMSE and a top R² score (0.9482).

                ✨ Linear Regression and Ridge are almost identical to Lasso but slightly worse in MAE and RMSE.

                ✨ Decision Tree performs poorly compared to the others.

                ✨ Gradient Boosting is slightly worse than Lasso in all metrics.
                



            """

            st.markdown(analysis_text)




   
    with tabs[2]:
        st.subheader("📊 YouTube Content Insights")

        # Audience Retention proxy: average watch time percentage
        if 'watch_time_minutes' in df.columns and 'video_length_minutes' in df.columns:
            df['watch_time_pct'] = (df['watch_time_minutes'] / df['video_length_minutes']) * 100
            retention_category = df.groupby('category')['watch_time_pct'].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10,5))
            retention_category.plot(kind='bar', color='mediumpurple', ax=ax)
            ax.set_title("Average Watch Time % by Category")
            ax.set_ylabel("Watch Time Percentage")
            ax.set_xlabel("Category")
            st.pyplot(fig)
        else:
            st.info("Watch time or video length columns not found for retention analysis.")

        # Engagement: Likes and Comments vs Ad Revenue scatterplot
        if all(col in df.columns for col in ['likes', 'comments', 'ad_revenue_usd']):
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.scatterplot(data=df, x='likes', y='ad_revenue_usd', label='Likes', color='forestgreen', ax=ax)
            sns.scatterplot(data=df, x='comments', y='ad_revenue_usd', label='Comments', color='darkorange', ax=ax)
            ax.set_title("Likes & Comments vs Ad Revenue")
            ax.set_xlabel("Count")
            ax.set_ylabel("Ad Revenue (USD)")
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("Required engagement columns (likes/comments/ad_revenue_usd) not found.")

        # Watch Time by Device (average watch time)
        if 'device' in df.columns and 'watch_time_minutes' in df.columns:
            watch_time_device = df.groupby('device')['watch_time_minutes'].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8,4))
            watch_time_device.plot(kind='barh', color='deepskyblue', ax=ax)
            ax.set_title("Average Watch Time (minutes) by Device")
            ax.set_xlabel("Watch Time (minutes)")
            ax.set_ylabel("Device")
            st.pyplot(fig)
        else:
            st.info("Device or watch time data not found for Watch Time by Device.")
            
    # New Ideas tab with line charts
    with tabs[3]:
        st.subheader("💡 Line Graph")
         # Views vs Revenue analysis
        if "views" in numeric_df.columns and "ad_revenue_usd" in numeric_df.columns:
            fig, ax = plt.subplots(figsize=(8,5))
            sns.scatterplot(x=numeric_df["views"], y=numeric_df["ad_revenue_usd"], ax=ax, color="cyan")
            sns.regplot(x=numeric_df["views"], y=numeric_df["ad_revenue_usd"], scatter=False, ax=ax, color="red")
            ax.set_xlabel("Views")
            ax.set_ylabel("Ad Revenue (USD)")
            ax.set_title("Views vs Revenue with Regression Line")
            st.pyplot(fig)


            corr = numeric_df["views"].corr(numeric_df["ad_revenue_usd"])
            st.metric("📊 Correlation (Views ↔ Revenue)", f"{corr:.4f}")


        else:
            st.warning("❌ Columns 'views' and 'ad_revenue_usd' not found in dataset.")

    # ad_revenue vs category
        if 'category' in df.columns and 'ad_revenue_usd' in df.columns:
            category_revenue = df.groupby('category')['ad_revenue_usd'].mean().sort_values()
            fig, ax = plt.subplots()
            category_revenue.plot(kind='line', marker='o', ax=ax)
            ax.set_title("Average Ad Revenue by Category")
            ax.set_xlabel("Category")
            ax.set_ylabel("Average Ad Revenue (USD)")
            st.pyplot(fig)
        else:
            st.warning("Required columns 'category' or 'ad_revenue_usd' not found for Category vs Ad Revenue.")

        # ad_revenue vs device
        if 'device' in df.columns and 'ad_revenue_usd' in df.columns:
            device_revenue = df.groupby('device')['ad_revenue_usd'].mean().sort_values()
            fig, ax = plt.subplots()
            device_revenue.plot(kind='line', marker='o', color='orange', ax=ax)
            ax.set_title("Average Ad Revenue by Device")
            ax.set_xlabel("Device")
            ax.set_ylabel("Average Ad Revenue (USD)")
            st.pyplot(fig)
        else:
            st.warning("Required columns 'device' or 'ad_revenue_usd' not found for Device vs Ad Revenue.")

        #line graph  'category' and 'device'

        if 'device' in df.columns and 'category' in df.columns:
            counts = df.groupby(['device', 'category']).size().unstack().fillna(0)
            fig, ax = plt.subplots()
            counts.plot(kind='line', marker='o', ax=ax)
            ax.set_title("Category Counts by Device")
            ax.set_xlabel("Device")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.warning("Required columns 'device' or 'category' not found.")

        # category vs device counts (using count plot)
        if 'category' in df.columns and 'device' in df.columns:
            fig, ax = plt.subplots(figsize=(10,5))
            sns.countplot(data=df, x='category', hue='device', ax=ax)
            ax.set_title("Content Category Distribution by Device")
            ax.set_xlabel("Category")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.warning("Required columns 'category' and/or 'device' not found for Category vs Device distribution.")
# Prediction Page
elif menu == "Prediction":
    st.title("🔮 Predict YouTube Ad Revenue")
    target_col = 'ad_revenue_usd'
    if target_col not in numeric_df.columns:
        st.warning(f"Target column '{target_col}' not found.")
    else:
        feature_cols = numeric_df.columns.drop(target_col)
        st.subheader("Enter Feature Values")
        input_data = {}
        for col in feature_cols:
            input_data[col] = st.number_input(f"{col}", value=float(numeric_df[col].mean()))
        
        if st.button("Predict"):
            X = numeric_df[feature_cols]
            y = numeric_df[target_col]
            model =  LinearRegression()
            model.fit(X, y)  # Safe: NaNs filled
            input_df = pd.DataFrame([input_data])
            pred = model.predict(input_df)[0]
            st.success(f"Predicted Ad Revenue: **${pred:,.2f}**")
            
            

# # About Page
elif menu == "About":
    st.title("ℹ️ About This Dashboard")
    st.write("""
    The **Content Monetization Modeler** is a dark-themed Streamlit app 
    designed to analyze and predict **YouTube Ad Revenue**.  

    ### 🎯 Features:
    - 📊 Correlation Heatmap  
    - ⚡ Model Performance (R² score)  
    - 📈 Views vs Revenue Impact  
    - 🔮 Dynamic Revenue Prediction  

    ---
    ✅ Powered by **Python, Pandas, Scikit-learn, Seaborn, and Streamlit**    
    📌 Purpose: Helping creators and analysts make **data-driven content decisions**
    """)

    st.markdown("👨‍💻 Author: **Thamaraiselvan M**")

