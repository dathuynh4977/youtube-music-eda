import json

# Load watch history
with open(r"C:\Users\Admin\OneDrive\Documents\code\Music_data\watch-history.json", "r", encoding="utf-8") as f:
    watch_data = json.load(f)

# Load search history
with open(r"C:\Users\Admin\OneDrive\Documents\code\Music_data\search-history.json", "r", encoding="utf-8") as f:
    search_data = json.load(f)

print("Watch entries:", len(watch_data))
print("Search entries:", len(search_data))

# Show first entry
print("Sample entry:", watch_data[0])


import pandas as pd

# Convert to DataFrame
df = pd.DataFrame(watch_data)

# Keep only useful columns
df = df[['title', 'time', 'subtitles']]

# Extract channel name (artist)
def get_channel(x):
    try:
        return x[0]['name']
    except:
        return None

df['channel'] = df['subtitles'].apply(get_channel)

# Convert time to datetime
df['time'] = pd.to_datetime(df['time'], format='ISO8601')

# Extract date info
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month
df['day'] = df['time'].dt.day

print(df.head())

print("Total entries:", len(df))
print("Years:", df['year'].value_counts())

print("\nTop Channels:")
print(df['channel'].value_counts().head(10))


# Histogram (watch activity by year)
import matplotlib.pyplot as plt

df['year'].value_counts().sort_index().plot(kind='bar')
plt.title("Watch Activity by Year")
plt.xlabel("Year")
plt.ylabel("Number of Videos")
plt.show()

# Monthly Trend
monthly = df.groupby(['year', 'month']).size().unstack()

monthly.T.plot()
plt.title("Monthly Watching Trend")
plt.xlabel("Month")
plt.ylabel("Count")
plt.show()

# Box Plot
df.boxplot(column='month', by='year')
plt.title("Distribution of Watching Months by Year")
plt.suptitle("")
plt.show()

# Top Channels
df['channel'].value_counts().head(10).plot(kind='barh')
plt.title("Top 10 Most Watched Channels")
plt.xlabel("Count")
plt.show()

# Year Comparison
df_2025 = df[df['year'] == 2025]
df_2026 = df[df['year'] == 2026]

print("2025:", len(df_2025))
print("2026:", len(df_2026))

df_2025['channel'].value_counts().head(5).plot(kind='bar', title="Top Channels 2025")
plt.show()

df_2026['channel'].value_counts().head(5).plot(kind='bar', title="Top Channels 2026")
plt.show()

# Word Cloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Add custom stopwords
stopwords = set(STOPWORDS)
stopwords.update([
    "watched", "watch", "youtube", "video", "official",
    "music", "channel", "live"
])

text = " ".join(df['title'].dropna().str.lower())

wc = WordCloud(
    width=1000,
    height=500,
    background_color='white',
    stopwords=stopwords
).generate(text)

plt.imshow(wc)
plt.axis('off')
plt.title("Word Cloud of Video Titles (Filtered)")
plt.show()