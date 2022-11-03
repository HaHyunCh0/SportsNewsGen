from datetime import datetime

import pandas as pd


def drop_na(df):
    # remove rows with NaN
    na_dropped = df.dropna()
    print(na_dropped)
    na_dropped.reset_index(drop=True, inplace=True)
    print(na_dropped)
    # na_dropped.to_csv('../data/raw_na_dropped.csv', index=False, encoding='utf-8')
    return na_dropped


def clean_timestamp(df):
    # Datetime to date
    # Remove time as it's unnecessary
    df['Date'] = pd.to_datetime(df['DateTime']).dt.date
    print(df)
    return df


def correct_season_value(df):
    def calc_season(date_str):
        date_datetime = datetime.strptime(date_str, '%Y-%m-%d')
        year, month, day = date_str.split('-')

        if date_datetime < datetime.strptime(f"{year}-08-01", '%Y-%m-%d'):
            season_start = str(int(year) - 1)
            season_end = year
        else:
            season_start = year
            season_end = str(int(year) + 1)

        return season_start + '-' + season_end[2:]
    df['Season'] = df['Date'].apply(lambda x: calc_season(str(x)))
    return df


def save_df_csv(df, f_name):
    df.to_csv(f_name, index=False, encoding='utf-8')


if __name__ == '__main__':
    df = pd.read_csv('../data/raw.csv', encoding='utf-8')
    df = drop_na(df)
    df = clean_timestamp(df)
    df = correct_season_value(df)
    save_df_csv(df, "../data/raw_cleaned.csv")

