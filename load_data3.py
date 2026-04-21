import json
import os
import pandas as pd


def clean_user_name(filename):
    user = filename.replace(".json", "")
    user = user.replace("_watch-history", "")
    user = user.replace("_search-history", "")
    return user


def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"


def extract_artist(x):
    try:
        if isinstance(x, list) and len(x) > 0:
            return x[0].get("name", None)
    except Exception:
        return None
    return None


def process_history_file(path, user_id, data_type):
    """
    data_type should be:
    - Watch
    - Search
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[{user_id}] {data_type} entries:", len(data))

    df = pd.DataFrame(data)

    if "title" not in df.columns or "time" not in df.columns:
        print(f"Skipping {path}: missing title/time")
        return pd.DataFrame()

    cols = ["title", "time"]
    if "subtitles" in df.columns:
        cols.append("subtitles")

    df = df[cols].copy()

    if "subtitles" in df.columns:
        df["artist"] = df["subtitles"].apply(extract_artist)
    else:
        df["artist"] = None

    df["time"] = pd.to_datetime(df["time"], format="ISO8601", errors="coerce")
    df = df.dropna(subset=["time"])

    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["day"] = df["time"].dt.day
    df["hour"] = df["time"].dt.hour
    df["weekday"] = df["time"].dt.day_name()
    df["season"] = df["month"].apply(get_season)

    df["title"] = (
        df["title"]
        .astype(str)
        .str.replace("Watched ", "", regex=False)
        .str.replace("Searched for ", "", regex=False)
    )

    df["user"] = user_id
    df["type"] = data_type

    return df


def load_all_data():
    """
    Automatically loads every pair:
    user*_watch-history.json
    user*_search-history.json

    Search file is optional, but watch file is required for ML.
    """
    all_dfs = []

    watch_files = sorted([
        f for f in os.listdir(".")
        if f.startswith("user") and f.endswith("_watch-history.json")
    ])

    if not watch_files:
        raise FileNotFoundError("No user*_watch-history.json files found in this folder.")

    print("\nUsers detected:")

    for watch_file in watch_files:
        user_id = clean_user_name(watch_file)
        search_file = f"{user_id}_search-history.json"

        print(f"\nLoading {user_id}:")
        print("-", watch_file)

        watch_df = process_history_file(watch_file, user_id, "Watch")
        if not watch_df.empty:
            all_dfs.append(watch_df)

        if os.path.exists(search_file):
            print("-", search_file)
            search_df = process_history_file(search_file, user_id, "Search")
            if not search_df.empty:
                all_dfs.append(search_df)
        else:
            print(f"- {search_file} not found, skipping search history")

    if not all_dfs:
        raise ValueError("No valid data loaded.")

    combined = pd.concat(all_dfs, ignore_index=True)

    print("\nTotal combined entries:", len(combined))

    print("\nUsers distribution:")
    print(combined["user"].value_counts())

    print("\nData type distribution:")
    print(combined["type"].value_counts())

    return combined


def get_watch_data(df):
    """
    Use this helper in ML files so search history does not pollute:
    similarity, classification, clustering, outliers, recommender.
    """
    return df[
        (df["type"] == "Watch") &
        (df["artist"].notna())
    ].copy()


def summarize(df):
    print("\n========== DATA SUMMARY ==========")

    print("\nTotal rows:", len(df))

    print("\nUsers:")
    print(df["user"].value_counts())

    print("\nData Types:")
    print(df["type"].value_counts())

    print("\nYears distribution:")
    print(df["year"].value_counts())

    watch_df = get_watch_data(df)

    print("\nTop Artists / Channels:")
    print(watch_df["artist"].value_counts().head(10))

    print("\nSeason Distribution:")
    print(df["season"].value_counts())

    print("\nHourly Distribution:")
    print(df["hour"].value_counts().sort_index())


if __name__ == "__main__":
    df = load_all_data()
    summarize(df)

    print("\nSample data:")
    print(df.head())