# Swiggy Recommendation

**Objective:**

A machine learning-based recommendation system that suggests restaurants to users based on their preferences, location, and dining history using Streamlit.

**-  Features**

**Three Recommendation Methods:**

🏷️ Cluster-based recommendations
🔍 Preference-based filtering
📊 Cosine similarity matching

**Smart Filtering Options:**

📍 City selection
🍲 Cuisine preferences
⭐ Minimum rating
💰 Maximum budget

**User-Friendly Interface:**

🎨 Clean Streamlit interface

**📱 Responsive design**

⚡ Real-time results

**🧠 Methodology**

**Data Processing**

Data Cleaning: Handle missing values, remove duplicates
Encoding: One-Hot Encoding for categorical features
Clustering: K-Means clustering for grouping similar restaurants

**Evaluation report:**

📊 BASIC METRICS:
   Number of clusters: 50
   Number of restaurants: 148,541
   Inertia (WCSS): 434361887.72
   Inertia per sample: 2924.19

**Recommendation Algorithms**

Cluster-based: Find restaurants in the same cluster
Preference-based: Filter by user criteria
Similarity-based: Cosine similarity between feature vectors

**Technologies**

Frontend: Streamlit
Backend: Python, Pandas, NumPy
ML: Scikit-learn (K-Means, Cosine Similarity)

**🎯 How to Use**

By Restaurant: Select a restaurant you like, get similar ones
By Preferences: Choose city, cuisine, rating, and budget
Advanced Similarity: Get mathematically similar restaurants

Run a code:
1.Data_cleaning_impute.py
2.Data_cleaning_drop.py
3.Data_processing.py
4.indices_validate.py
5.recommendation_method.py
6.evaluate.py
7.stream_app.py

Note: Added encoded1_data.parquet as encoded csv file size is more than 100mb which will not able to push in Github
