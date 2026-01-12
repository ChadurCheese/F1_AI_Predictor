import pandas as pd

def load_race_data(path: str) -> pd.DataFrame:
    """
    Load preprocessed F1 race data.
    """
    df = pd.read_csv(path)
    return df
