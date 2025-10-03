# SMS Spam Classifier

A complete **SMS Spam Classification System** using **Python, Machine Learning, FastAPI, and Streamlit**.  
This project covers **data preprocessing, exploratory data analysis (EDA), machine learning model training, API creation, and interactive frontend**.

---

## Features & Steps Implemented

### 1️⃣ Data Exploration & Cleaning
- Loaded dataset `spam.csv` containing columns `label` (ham/spam) and `message`.
- Checked shape, columns, data types, missing values, and duplicates.
- Removed unnecessary columns (`Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`) and dropped duplicate rows.
- Result: Cleaned dataset with no missing values or duplicates.

### 2️⃣ Exploratory Data Analysis (EDA)
- Frequency counts of labels (`ham` vs `spam`).
- Calculated number of **characters, words, and sentences** per message.
- Statistical summary for all data, and separately for `ham` and `spam`.
- Visualizations:
  - Pie chart for label distribution
  - Histograms of number of characters and words by label
  - Pairplot for feature correlations
  - Correlation heatmap
  - WordCloud of all preprocessed messages

### 3️⃣ Text Preprocessing
- Converted text to **lowercase**.
- **Tokenized** messages into words.
- Removed **special characters and punctuation**.
- Removed **stopwords** (using NLTK).
- Applied **stemming** using PorterStemmer.
- Rejoined tokens to create `final_message` for ML.

### 4️⃣ Machine Learning Model
- Used **Multinomial Naive Bayes** for classification.
- Converted text to numeric using **TF-IDF vectorization**.
- Split data: **80% training, 20% testing**, stratified by label.
- Hyperparameter tuning with **GridSearchCV** for `alpha` values `[0.1, 0.5, 1.0]`.
- Saved the **best model** and **vectorizer** using `joblib`.

**Results:**
- **Accuracy:** 96.7%
- **Classification Report:**
            precision    recall  f1-score   support
Ham           0.98      0.99      0.98       903
Spam          0.90      0.83      0.87       131
accuracy                         0.97      1034
macro avg     0.94      0.91      0.92      1034
weighted avg  0.97      0.97      0.97      1034
[[891 12]
[ 22 109]]


### 5️⃣ FastAPI Backend
- Created **FastAPI app** with POST endpoint `/predict`.
- Input: JSON with `"message": "text here"`.
- Output: JSON with `"label"` (Ham/Spam) and `"probability"` (confidence score).
- Handled empty messages with proper HTTP error.

### 6️⃣ Streamlit Frontend
- Created **interactive web app**.
- Users can input an SMS message in a textbox and click **Predict**.
- Connects to **FastAPI backend** for real-time prediction.
- Displays:
- **Green** message if Ham
- **Red** message if Spam
- Confidence score

---
## Technologies Used

- **Python**  
- **Data Processing:** Pandas, NumPy  
- **Text Processing:** NLTK (tokenization, stopwords, stemming)  
- **Machine Learning:** Scikit-learn (MultinomialNB, TF-IDF, GridSearchCV)  
- **Visualization:** Matplotlib, Seaborn, WordCloud  
- **API:** FastAPI
- -------------------------------------
URL Linked in :[https://www.linkedin.com/in/engineer-youssef-mahmoud-63b243361?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BDDZI3no1Q1WilUjqSAM0Iw%3D%3D]
- **Frontend:** Streamlit  
- **Serialization:** Joblib  
