import pandas as pd

def predict_positions(model, df, feature_cols):
    """
    Predict finishing positions.
    """
    X = df[feature_cols]
    preds = model.predict(X)

    df = df.copy()
    df["predicted_position"] = preds

    return df.sort_values("predicted_position")
