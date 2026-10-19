"""
src/preprocess.py  -  Step 3: Cleaning, feature engineering, train/test split
Run from the project root:  python src/preprocess.py
"""
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(__file__))
from features import clean_column_names, add_features, FEATURE_COLUMNS

DATA_PATH = "data/Instagram_fake_profile_dataset.csv"
os.makedirs("data/processed", exist_ok=True)

df = clean_column_names(pd.read_csv(DATA_PATH))
print("Rows before cleaning:", len(df))

# 1) Drop duplicates BEFORE splitting. If we split first, the same profile could
#    land in both train and test, and the model would be "tested" on data it has
#    already seen (data leakage) -> fake-high accuracy.
df = df.drop_duplicates().reset_index(drop=True)
print("Rows after dropping duplicates:", len(df))

# 2) Feature engineering
df = add_features(df)

# 3) 80/20 split. stratify keeps the fake/real ratio the same in both parts.
X = df[FEATURE_COLUMNS]
y = df["fake"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

train = X_train.assign(fake=y_train)
test = X_test.assign(fake=y_test)
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print(f"\nTrain size: {len(train)}  (fake share {train.fake.mean():.1%})")
print(f"Test size:  {len(test)}  (fake share {test.fake.mean():.1%})")
print("\nFeatures used by the model:", FEATURE_COLUMNS)
print("Saved: data/processed/train.csv and data/processed/test.csv")
