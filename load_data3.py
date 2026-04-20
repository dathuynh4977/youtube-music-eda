import json
import pandas as pd

# -----------------------------------------
# HELPER: LOAD ONE USER
# -----------------------------------------
def load_user(watch_path, search_path, user_name):
    with open(watch_path, "r", encoding="utf-8") as f:
        watch_data = json.load(f)

    print(f"[{user_name}] Watch entries:", len(watch_data))

    df = pd.DataFrame(watch_data)

    # Keep useful columns
    df = df[['title', 'time', 'subtitles']]

    # Extract artist (channel)
    def get_artist(x):
        try:
            return x[0]['name']
        except:
            return None

    df['artist'] = df['subtitles'].apply(get_artist)

    # Convert time
    df['time'] = pd.to_datetime(df['time'], format='ISO8601')

    # Extract features
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour
    df['weekday'] = df['time'].dt.day_name()

    # Season
    def get_season(m):
        if m in [12, 1, 2]:
            return 'Winter'
        elif m in [3, 4, 5]:
            return 'Spring'
        elif m in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    df['season'] = df['month'].apply(get_season)

    df['user'] = user_name

    return df


# -----------------------------------------
# MAIN FUNCTION (THIS WAS MISSING)
# -----------------------------------------
def load_all_data():
    df1 = load_user(
        "user1_watch-history.json",
        "user1_search-history.json",
        "user1"
    )

    df2 = load_user(
        "user2_watch-history.json",
        "user2_search-history.json",
        "user2"
    )

    df = pd.concat([df1, df2], ignore_index=True)

    print("\nTotal combined entries:", len(df))
    print("Users distribution:\n", df['user'].value_counts())

    return df