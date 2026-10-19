"""
src/features.py
Shared helper code used by training, prediction AND the web app.
Keeping feature-building in ONE place guarantees that a new profile is
transformed exactly the same way as the training data (no mismatch bugs).
"""
import numpy as np
import pandas as pd

# The CSV has awkward column names ("nums/length username"). We rename them
# to simple snake_case names that are easy to type in Python.
RENAME_MAP = {
    "profile pic": "profile_pic",
    "nums/length username": "username_digit_ratio",
    "fullname words": "fullname_words",
    "nums/length fullname": "fullname_digit_ratio",
    "name==username": "name_equals_username",
    "description length": "bio_length",
    "external URL": "external_url",
    "private": "private",
    "#posts": "posts",
    "#followers": "followers",
    "#follows": "following",
    "fake": "fake",
}

# The 11 original (raw) input features
RAW_FEATURES = [
    "profile_pic", "username_digit_ratio", "fullname_words",
    "fullname_digit_ratio", "name_equals_username", "bio_length",
    "external_url", "private", "posts", "followers", "following",
]

# The 5 features we create ourselves
ENGINEERED_FEATURES = [
    "followers_to_following_ratio", "posts_per_follower",
    "has_bio", "log_followers", "log_following",
]

# The full list the model is trained on (order matters!)
FEATURE_COLUMNS = RAW_FEATURES + ENGINEERED_FEATURES


def clean_column_names(df):
    """Rename the CSV columns to snake_case."""
    return df.rename(columns=RENAME_MAP)


def add_features(df):
    """
    Add the engineered features. We add +1 in every denominator so we never
    divide by zero (an account can have 0 following or 0 followers).
    """
    df = df.copy()
    df["followers_to_following_ratio"] = df["followers"] / (df["following"] + 1)
    df["posts_per_follower"] = df["posts"] / (df["followers"] + 1)
    df["has_bio"] = (df["bio_length"] > 0).astype(int)
    # log1p = log(1 + x). Follower counts range from 0 to 15 million, so we
    # squash them with a log to stop giant numbers dominating.
    df["log_followers"] = np.log1p(df["followers"])
    df["log_following"] = np.log1p(df["following"])
    return df


# ---- Text helpers (used by the web app to compute inputs from typed text) ----
def digit_ratio(text):
    """Fraction of characters that are digits, e.g. 'john_99' -> 2/7 = 0.29"""
    text = text.strip()
    if len(text) == 0:
        return 0.0
    return round(sum(ch.isdigit() for ch in text) / len(text), 2)


def name_equals_username(fullname, username):
    """1 if the full name and username are the same (ignoring case/spaces)."""
    a = fullname.strip().lower().replace(" ", "")
    b = username.strip().lower()
    return int(len(a) > 0 and a == b)
