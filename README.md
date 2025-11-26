# Netflix-Recommendation-Engine
End-to-End Machine Learning + NLP + Collaborative Filtering Recommender System

This project is a full-stack movie recommendation system built using:

🧹 Data Cleaning & Feature Engineering

📝 NLP (TF-IDF + SVD) for movie descriptions

🤖 ML Models (LogReg, Random Forest, XGBoost/GBM)

🔧 Hyperparameter Tuning (GridSearchCV / RandomizedSearchCV)

🎯 Collaborative Filtering (NumPy SVD)

👤 User Profiles & Personalized Recommendations

🎞 Similar Movie Search (Cosine Similarity)

🖥 Streamlit Web App Interface

This README provides everything you need to download, install, and run this project on your own system.

1. Project Structure
Netflix-Recommendation-Engine/
│
├── data/
│   ├── raw/
│   │   ├── ratings.csv
│   │   ├── movies.csv
│   │   ├── movie_metadata.csv
│   │   ├── genome-tags.csv
│   │   ├── genome-scores.csv
│   └── processed/   (created automatically)
│
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── models.py
│   ├── recommender.py
│   ├── config.py
│   └── __init__.py
│
├── app.py     ← Main Streamlit App
├── requirements.txt
└── README.md
All raw datasets go into data/raw/
✔ Processed files will be auto-generated in data/processed/

2. Installation Guide:
   Follow these steps to set up the project on your local system.

   Step 1: Clone the Repository
   git clone https://github.com/your-username/Netflix-Recommendation-Engine.git
   cd Netflix-Recommendation-Engine

   Step 2: Create a Virtual Environment
   python -m venv venv
   venv\Scripts\activate

   3. Install Dependencies
      Install all required libraries:  **pip install -r requirements.txt**

   4. Download Required Dataset

      Place all raw dataset CSV files into:
      data/raw/

      Required files:
      1. ratings.csv
      2. movies.csv
      3. movie_metadata.csv
      4. genome-tags.csv
      5. genome-scores.csv

      USE LINK: https://drive.google.com/drive/folders/1GOkIg2pe927JjE4HGAf8eLWJVVKe2NKq?usp=sharing
      To access the CSVs

   5. Running the Application

      The system runs through Streamlit.
      Launch the app using: **streamlit run app.py**

      Streamlit will automatically open in your browser:
      http://localhost:8501

   6. What Happens When You Run the App?

      When you execute: streamlit run app.py
      The application automatically:
      1. Loads raw data
         → ratings.csv, movies.csv, metadata, genome data (Handled in data_loader.py)
      2. Cleans and preprocesses everything
        → Fills missing values
        → Extracts release years
        → Merges metadata
        → Normalizes fields
        (Handled in features.py → clean_full_table())
      3. Builds NLP embeddings
        → TF-IDF on overview text
        → Dimensionality reduction using SVD
      4. Creates engineered user features & movie features
        → Average ratings
        → Like ratios
        → Popularity
        → Recency
        → Favourite genre per user
      5. Generates the ML training dataset
         → Each row = (user, movie) pair with features
      6. Trains Multiple ML Models
         a. Logistic Regression
         b. Random Forest
         c. XGBoost (if installed)
         d. Gradient Boosting(In models.py)
      7. Hyperparameter tuning
         a. GridSearchCV
         b. RandomizedSearchCV
      8. Builds SVD collaborative filtering
         → Full matrix factorization model
      9. Loads the UI with 4 tabs
          a. Data Exploration
          b. Model Metrics
          c. Personalized Recommendations
          d. Similar Movie Search

All of this runs automatically.
  7. Streamlit App Features
    📊 Tab 1: Data & EDA
        a. Dataset statistics
        b. Sample records
        c. Rating distribution chart
        d. Top genres bar chart

   Tab 2: Machine Learning Models
        Shows:
        a. Baseline & Tuned models
        b. Accuracy, Precision, Recall, F1, ROC-AUC
        c. Hyperparameter results
        d. Auto-generated charts

  👤 Tab 3: User Profiles & Recommendations
        For each selected user:
        a. Rating behaviour
        b. Top genres
        c. Top actors
        d. Top directors
        e. Recent-release preference
        f. Two types of recommendations
        g. ML-Based Personalized Recommendations
        h. SVD Collaborative Filtering Recommendations
        
  Each recommendation displays:
        a. Poster
        b. Title
        c. Genres
        d. Predicted probability/score

  🎞 Tab 4: Similar Movies
    → Pick a reference movie
    → View the most similar movies based on NLP embeddings
    → Uses cosine similarity


