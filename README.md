# 🎵 YouTube Music Exploratory Data Analysis (EDA)

## 📌 Overview
This project performs an exploratory data analysis (EDA) on YouTube watch history data to understand user behavior and music consumption trends.

The analysis includes data preprocessing, statistical summaries, visualizations, and an interactive dashboard.

---

## 📊 Dataset
- Watch History: 49,400 entries  
- Search History: 6,912 entries  

Each record contains:
- Video title
- Timestamp
- Channel name

Derived features:
- Year
- Month
- Day

---

## ⚙️ Features
- Data cleaning and preprocessing
- Summary statistics
- Visualizations:
  - Watch activity by year
  - Monthly trends
  - Top channels
  - Word cloud
- Interactive dashboard using Streamlit

---

## 🛠️ Technologies Used
- Python
- Pandas
- Matplotlib
- WordCloud
- Streamlit

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt

2. Run data analysis
python load_data.py
3. Run dashboard
python -m streamlit run dashboard.py