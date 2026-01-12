import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering used by the model.
    """
    df = df.copy()

    df = df.sort_values(["driverId", "raceId"])

    df["avg_finish_last_5"] = (
        df.groupby("driverId")["positionOrder"]
          .rolling(5)
          .mean()
          .reset_index(level=0, drop=True)
    )

    # Add more features EXACTLY as in notebook
    # team_avg_finish_last_5, avg_points_last_5, etc.

    return df
