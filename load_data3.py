import json
import pandas as pd


# ==========================================
# LOAD SINGLE USER DATA
# ==========================================
def load_user(watch_path, search_path, user_id):
    # ---------- Load watch history ----------
    with open(watch_path, "r", encoding="utf-8") as f:
        watch_data = json.load(f)

    # ---------- Load search history ----------
    with open(search_path, "r", encoding="utf-8") as f:
        search_data = json.load(f)

    print(f"\n[{user_id}] Watch entries: {len(watch_data)}")
    print(f"[{user_id}] Search entries: {len(search_data)}")

    # ---------- Convert to DataFrame ----------
    df = pd.DataFrame(watch_data)

    # Keep only useful columns (safe check)
    df = df[['title', 'time', 'subtitles']].copy()

    # ---------- Extract artist/channel ----------
    def get_artist(x):
        try:
            if isinstance(x, list) and len(x) > 0:
                return x[0].get('name', None)
        except:
            pass
        return None

    df['artist'] = df['subtitles'].apply(get_artist)

    # ---------- Convert time ----------
    df['time'] = pd.to_datetime(df['time'], errors='coerce')

    # Drop invalid times
    df = df.dropna(subset=['time'])

    # ---------- Extract time features ----------
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour
    df['weekday'] = df['time'].dt.day_name()

    # ---------- Season feature (IMPORTANT for similarity) ----------
    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"

    df['season'] = df['month'].apply(get_season)

    # ---------- Clean title ----------
    df['title'] = df['title'].astype(str).str.replace("Watched ", "", regex=False)

    # ---------- Add user ----------
    df['user'] = user_id

    # ---------- Drop missing artist ----------
    df = df.dropna(subset=['artist'])

    return df


# ==========================================
# LOAD BOTH USERS
# ==========================================
def load_all_users():
    user1 = load_user(
        "user1_watch-history.json",
        "user1_search-history.json",
        "user1"
    )

    user2 = load_user(
        "user2_watch-history.json",
        "user2_search-history.json",
        "user2"
    )

    df = pd.concat([user1, user2], ignore_index=True)

    print("\nTotal combined entries:", len(df))
    print("Users distribution:\n", df['user'].value_counts())

    return df


# ==========================================
# SUMMARY (FOR DEBUGGING / REPORT)
# ==========================================
def summarize(df):
    print("\n========== DATA SUMMARY ==========")

    print("\nTotal rows:", len(df))

    print("\nUsers:")
    print(df['user'].value_counts())

    print("\nYears distribution:")
    print(df['year'].value_counts())

    print("\nTop Artists:")
    print(df['artist'].value_counts().head(10))

    print("\nSeason Distribution:")
    print(df['season'].value_counts())

    print("\nHourly Distribution:")
    print(df['hour'].value_counts().sort_index())


# ==========================================
# RUN TEST (ONLY WHEN RUN DIRECTLY)
# ==========================================
if __name__ == "__main__":
    df = load_all_users()
    summarize(df)

    print("\nSample data:")
    print(df.head())