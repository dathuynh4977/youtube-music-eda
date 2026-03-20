import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

st.set_page_config(page_title="Music EDA Dashboard", layout="wide")

st.title("🎵 YouTube Music Data Dashboard")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    with open("watch-history.json", "r", encoding="utf-8") as f:
        watch_data = json.load(f)

    df = pd.DataFrame(watch_data)

    df = df[['title', 'time', 'subtitles']]

    def get_channel(x):
        try:
            return x[0]['name']
        except:
            return None

    df['channel'] = df['subtitles'].apply(get_channel)

    df['time'] = pd.to_datetime(df['time'], format='ISO8601', errors='coerce')

    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month

    return df

df = load_data()

# -------------------------------
# SIDEBAR FILTER
# -------------------------------
st.sidebar.header("Filters")

years = st.sidebar.multiselect(
    "Select Year",
    options=sorted(df['year'].dropna().unique()),
    default=sorted(df['year'].dropna().unique())
)

filtered_df = df[df['year'].isin(years)]

# -------------------------------
# METRICS
# -------------------------------
st.subheader("📊 Summary")

col1, col2 = st.columns(2)

col1.metric("Total Videos", len(filtered_df))
col2.metric("Unique Channels", filtered_df['channel'].nunique())

# -------------------------------
# YEARLY CHART
# -------------------------------
st.subheader("📈 Watch Activity by Year")

year_counts = filtered_df['year'].value_counts().sort_index()

fig, ax = plt.subplots()
year_counts.plot(kind='bar', ax=ax)
st.pyplot(fig)

# -------------------------------
# MONTHLY TREND
# -------------------------------
st.subheader("📅 Monthly Trend")

monthly = filtered_df.groupby(['year', 'month']).size().unstack()

fig, ax = plt.subplots()
monthly.T.plot(ax=ax)
st.pyplot(fig)

# -------------------------------
# TOP CHANNELS
# -------------------------------
st.subheader("🔥 Top Channels")

top_channels = filtered_df['channel'].value_counts().head(10)

fig, ax = plt.subplots()
top_channels.plot(kind='barh', ax=ax)
st.pyplot(fig)

# -------------------------------
# WORD CLOUD
# -------------------------------
st.subheader("☁️ Word Cloud")

stopwords = set(STOPWORDS)
stopwords.update(["watched", "youtube", "video"])

text = " ".join(filtered_df['title'].dropna().str.lower())

wc = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(text)

fig, ax = plt.subplots()
ax.imshow(wc)
ax.axis("off")

st.pyplot(fig)