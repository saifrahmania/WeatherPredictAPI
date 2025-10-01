# app/utils.py
from datetime import date, timedelta

def plus_days(d: date, n: int) -> date:
    """Return a new date that is n days after d."""
    return d + timedelta(days=n)


def align_to_model_columns(df, model):
    """
    Ensure df has the same columns (and order) the model was trained with.
    Missing columns are added with 0. Extra columns are dropped.
    """
    expected = getattr(model, "feature_names_in_", None)
    if expected is None:  # model wasn't fit with column names
        return df
    X = df.copy()
    for col in expected:
        if col not in X.columns:
            X[col] = 0
    # reorder and drop extras
    X = X.loc[:, list(expected)]
    return X