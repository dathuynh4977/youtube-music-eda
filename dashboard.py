import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import re

st.set_page_config(page_title="Music EDA Dashboard", layout="wide")
st.title("🎵 YouTube Music Dashboard")

# -------------------------------
# LOAD DATA
# -------------------------------
def load_uploaded_files(uploaded_files):
    all_dfs = []

    for file in uploaded_files:
        data = json.load(file)
        df = pd.DataFrame(data)

        if 'title' not in df.columns or 'time' not in df.columns:
            continue

        cols = ['title', 'time']
        if 'subtitles' in df.columns:
            cols.append('subtitles')

        df = df[cols]

        def get_channel(x):
            try:
                return x[0]['name']
            except:
                return None

        if 'subtitles' in df.columns:
            df['channel'] = df['subtitles'].apply(get_channel)
            df['type'] = 'Watch'
        else:
            df['channel'] = "Search Activity"
            df['type'] = 'Search'

        df['time'] = pd.to_datetime(df['time'], format='ISO8601', errors='coerce')
        df['year'] = df['time'].dt.year
        df['month'] = df['time'].dt.month
        df['hour'] = df['time'].dt.hour

        # Clean user names
        df['user'] = (
            file.name.replace(".json", "")
            .replace("_watch-history", "")
            .replace("_search-history", "")
        )

        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame(columns=["title", "time", "channel", "type", "year", "month", "hour", "user"])

    return pd.concat(all_dfs, ignore_index=True)


# -------------------------------
# FILE UPLOAD
# -------------------------------
st.sidebar.header("📂 Upload Data")

uploaded_files = st.sidebar.file_uploader(
    "Upload JSON files",
    type="json",
    accept_multiple_files=True
)

if not uploaded_files:
    st.warning("Upload at least one dataset")
    st.stop()

df = load_uploaded_files(uploaded_files)

# -------------------------------
# FILTERS
# -------------------------------
st.sidebar.header("Filters")

users = st.sidebar.multiselect(
    "Users",
    options=sorted(df['user'].dropna().unique()),
    default=sorted(df['user'].dropna().unique())
)

years = st.sidebar.multiselect(
    "Years",
    options=sorted(df['year'].dropna().unique()),
    default=sorted(df['year'].dropna().unique())
)

data_type = st.sidebar.selectbox("Data Type", ["All", "Watch", "Search"])

filtered_df = df[df['user'].isin(users) & df['year'].isin(years)]

if data_type != "All":
    filtered_df = filtered_df[filtered_df['type'] == data_type]

# -------------------------------
# CLEAN WATCH DATA FOR ML
# -------------------------------
watch_df = filtered_df[
    (filtered_df['type'] == 'Watch') &
    (filtered_df['channel'].notna()) &
    (~filtered_df['channel'].str.contains("Search", na=False))
].copy()

# -------------------------------
# TABS UI
# -------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Overview",
    "👥 Comparison",
    "☁️ Word Cloud",
    "✔ Similarity",
    "✔ Classification",
    "✔ Clustering",
    "✔ Outliers",
    "🎯 Recommender"
])

# -------------------------------
# TAB 1: OVERVIEW (IMPROVED A+)
# -------------------------------
with tab1:
    st.subheader("Summary")

    # -------- GLOBAL --------
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", len(filtered_df))
    col2.metric("Watch Rows", len(watch_df))
    col3.metric("Channels", watch_df['channel'].nunique() if not watch_df.empty else 0)

    st.divider()

    # -------- PER USER --------
    st.subheader("Per User Summary")

    if watch_df.empty:
        st.warning("No watch data available")
    else:
        users_list = sorted(watch_df['user'].unique())

        cols = st.columns(len(users_list))

        for i, user in enumerate(users_list):
            user_df = watch_df[watch_df['user'] == user]

            with cols[i]:
                st.markdown(f"### {user}")

                st.metric("Watch Count", len(user_df))
                st.metric("Unique Channels", user_df['channel'].nunique())

                # Optional bonus: peak hour
                if not user_df.empty:
                    peak_hour = user_df['hour'].mode()[0]
                    st.metric("Peak Hour", int(peak_hour))

# -------------------------------
# TAB 2: COMPARISON
# -------------------------------
with tab2:
    st.subheader("User Activity Comparison")

    if watch_df.empty:
        st.warning("No watch data available")
    else:
        fig, ax = plt.subplots()
        watch_df.groupby('user').size().plot(kind='bar', ax=ax)
        ax.set_xlabel("User")
        ax.set_ylabel("Watch Count")
        ax.set_title("Watch Count per User")
        st.pyplot(fig)

# -------------------------------
# TAB 3: WORD CLOUD (TWO SEPARATE CHARTS)
# -------------------------------
with tab3:
    st.subheader("Word Cloud by User")

    if watch_df.empty:
        st.warning("No watch data available")
    else:
        stopwords = set(STOPWORDS)
        stopwords.update([
            "https", "http", "www",
            "watch", "watched",
            "youtube", "video",
            "official", "mv", "audio", "lyrics", "lyric",
            "short", "shorts", "remix",
            "ft", "feat"
        ])

        def clean_text(text):
            text = re.sub(r"http\S+", "", text)
            text = re.sub(r"\b(watched|watch|youtube|video)\b", "", text, flags=re.IGNORECASE)
            text = re.sub(r"[^a-zA-ZÀ-ỹ\s]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        selected_users = sorted(watch_df['user'].dropna().unique())

        if len(selected_users) == 0:
            st.warning("No valid users in watch data")
        else:
            cols = st.columns(2)

            for idx, user in enumerate(selected_users[:2]):
                user_text = " ".join(
                    watch_df[watch_df['user'] == user]['title']
                    .dropna()
                    .apply(clean_text)
                    .str.lower()
                )

                with cols[idx]:
                    st.write(f"### {user}")
                    if not user_text.strip():
                        st.info("No text available for this user")
                    else:
                        wc = WordCloud(
                            width=900,
                            height=400,
                            background_color='white',
                            stopwords=stopwords,
                            collocations=False
                        ).generate(user_text)

                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)

            # If more than 2 users somehow appear, show remaining below
            if len(selected_users) > 2:
                for user in selected_users[2:]:
                    st.write(f"### {user}")
                    user_text = " ".join(
                        watch_df[watch_df['user'] == user]['title']
                        .dropna()
                        .apply(clean_text)
                        .str.lower()
                    )
                    if not user_text.strip():
                        st.info("No text available for this user")
                    else:
                        wc = WordCloud(
                            width=900,
                            height=400,
                            background_color='white',
                            stopwords=stopwords,
                            collocations=False
                        ).generate(user_text)

                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)

# -------------------------------
# TAB 4: SIMILARITY (HEATMAP)
# -------------------------------
with tab4:
    st.subheader("User Similarity (Heatmap)")

    if watch_df.empty:
        st.warning("No watch data available")
    else:
        matrix = pd.crosstab(watch_df['user'], watch_df['channel'])

        if matrix.shape[0] < 2 or matrix.shape[1] == 0:
            st.warning("Need at least 2 users and valid channels")
        else:
            sim = cosine_similarity(matrix)
            sim_df = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)

            # 🔥 HEATMAP (BEST VISUAL)
            fig, ax = plt.subplots(figsize=(6, 5))

            sns.heatmap(
                sim_df,
                annot=True,
                fmt=".2f",
                linewidths=0.5,
                square=True,
                cbar=True,
                ax=ax
            )

            ax.set_title("User Similarity Heatmap")

            st.pyplot(fig)

            # Optional: still show table
            with st.expander("Show raw similarity values"):
                st.dataframe(sim_df)

# -------------------------------
# TAB 5: CLASSIFICATION
# -------------------------------
with tab5:
    st.subheader("Classification Accuracy (Normalized)")

    if watch_df.empty or len(watch_df['user'].unique()) < 2:
        st.warning("Need at least 2 users")
    else:
        X = watch_df[['hour', 'month']].fillna(0)
        y = watch_df['user']

        model = DecisionTreeClassifier(random_state=42)
        model.fit(X, y)
        preds = model.predict(X)

        labels = sorted(y.unique())
        cm = confusion_matrix(y, preds, labels=labels, normalize='true')

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )
        ax.set_title("Classification Accuracy (Normalized)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

# -------------------------------
# TAB 6: CLUSTERING
# -------------------------------
with tab6:
    st.subheader("User Clustering (PCA)")

    if watch_df.empty:
        st.warning("No watch data available")
    else:
        matrix = pd.crosstab(watch_df['user'], watch_df['channel'])

        if matrix.shape[0] < 2:
            st.warning("Need multiple users")
        else:
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(matrix)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(reduced[:, 0], reduced[:, 1])

            for i, user in enumerate(matrix.index):
                ax.text(reduced[i, 0], reduced[i, 1], user, fontsize=12)

            ax.set_title("User Clustering (PCA)")
            st.pyplot(fig)

# -------------------------------
# TAB 7: OUTLIERS (FIXED A+)
# -------------------------------
with tab7:
    st.subheader("Outlier Detection (Clean Visualization)")

    if watch_df.empty:
        st.warning("No watch data available")
    else:
        from sklearn.preprocessing import StandardScaler

        # Better features
        watch_df['dayofweek'] = watch_df['time'].dt.dayofweek

        X = watch_df[['hour','month','dayofweek']].fillna(0)

        # Normalize
        X_scaled = StandardScaler().fit_transform(X)

        # Better model
        model = IsolationForest(
            contamination=0.05,   # only 5% anomalies
            random_state=42
        )

        preds = model.fit_predict(X_scaled)

        watch_df['anomaly'] = preds

        # Split data
        normal = watch_df[watch_df['anomaly'] == 1]
        anomaly = watch_df[watch_df['anomaly'] == -1]

        # Plot clean
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(
            normal['hour'],
            normal['month'],
            alpha=0.4,
            s=20,
            label="Normal"
        )

        ax.scatter(
            anomaly['hour'],
            anomaly['month'],
            alpha=0.9,
            s=40,
            label="Anomaly"
        )

        ax.set_title("Outlier Detection (Improved)")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Month")
        ax.legend()

        st.pyplot(fig)

# -------------------------------
# TAB 8: RECOMMENDER
# -------------------------------
with tab8:
    st.subheader("Personalized Recommendations")

    if watch_df.empty or len(watch_df['user'].unique()) < 2:
        st.warning("Need at least 2 users")
    else:
        matrix = pd.crosstab(watch_df['user'], watch_df['channel'])

        if matrix.shape[0] < 2:
            st.warning("Not enough data")
        else:
            similarity = cosine_similarity(matrix)
            sim_df = pd.DataFrame(similarity, index=matrix.index, columns=matrix.index)

            user = st.selectbox("Select User", matrix.index)

            similar = sim_df[user].sort_values(ascending=False)[1:3].index.tolist()
            st.write("Most similar users:", similar)

            user_channels = set(matrix.columns[matrix.loc[user] > 0])

            recs = {}
            for u in similar:
                for c in matrix.columns[matrix.loc[u] > 0]:
                    if c not in user_channels:
                        recs[c] = recs.get(c, 0) + 1

            rec_df = pd.DataFrame(
                sorted(recs.items(), key=lambda x: x[1], reverse=True),
                columns=["Channel", "Score"]
            ).head(10)

            st.dataframe(rec_df)