"""
src/explore.py  -  Step 2: Data exploration
Run from the project root:  python src/explore.py
"""
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # save plots to files, don't open windows
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(__file__))
from features import clean_column_names

DATA_PATH = "data/Instagram_fake_profile_dataset.csv"
os.makedirs("reports", exist_ok=True)

df = clean_column_names(pd.read_csv(DATA_PATH))
sns.set_theme(style="whitegrid")
COLORS = {0: "#2a9d8f", 1: "#d62828"}      # teal = real, red = fake
NAMES = {0: "Real", 1: "Fake"}

# ---------------- Basic facts ----------------
print("Shape (rows, columns):", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values per column:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nClass balance (fake: 1 = fake, 0 = real):\n", df["fake"].value_counts())

real, fake = df[df.fake == 0], df[df.fake == 1]

# ---------------- Plot 1: class distribution ----------------
fig, ax = plt.subplots(figsize=(5, 4))
counts = df["fake"].value_counts().sort_index()
ax.bar([NAMES[i] for i in counts.index], counts.values,
       color=[COLORS[i] for i in counts.index])
ax.set_title("Class distribution")
ax.set_ylabel("Number of profiles")
plt.tight_layout(); plt.savefig("reports/01_class_distribution.png", dpi=150); plt.close()
print("\n[Plot 1] class distribution")
print("  -> The classes are perfectly balanced (2500 vs 2500), so plain accuracy is")
print("     a fair metric and we do not need resampling.")

# ---------------- Plot 2: followers vs following ----------------
fig, ax = plt.subplots(figsize=(6.5, 5))
for label, group in df.groupby("fake"):
    ax.scatter(group["following"] + 1, group["followers"] + 1, s=10, alpha=0.35,
               color=COLORS[label], label=NAMES[label])
ax.set_xscale("log"); ax.set_yscale("log")   # +1 so zeros can be shown on a log axis
ax.set_xlabel("Following (+1, log scale)"); ax.set_ylabel("Followers (+1, log scale)")
ax.set_title("Followers vs following by class"); ax.legend()
plt.tight_layout(); plt.savefig("reports/02_followers_vs_following.png", dpi=150); plt.close()
print("\n[Plot 2] followers vs following (log scale)")
print(f"  -> Median followers: real = {real.followers.median():.0f}, fake = {fake.followers.median():.0f}")
print(f"  -> Median following: real = {real.following.median():.0f}, fake = {fake.following.median():.0f}")

# ---------------- Plot 3: bio length ----------------
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for label, group in df.groupby("fake"):
    ax.hist(group["bio_length"], bins=30, alpha=0.6, color=COLORS[label], label=NAMES[label])
ax.set_xlabel("Bio length (characters)"); ax.set_ylabel("Number of profiles")
ax.set_title("Bio length by class"); ax.legend()
plt.tight_layout(); plt.savefig("reports/03_bio_length.png", dpi=150); plt.close()
print("\n[Plot 3] bio length")
print(f"  -> Share with an EMPTY bio: real = {(real.bio_length==0).mean():.1%}, fake = {(fake.bio_length==0).mean():.1%}")
print(f"  -> Mean bio length: real = {real.bio_length.mean():.1f}, fake = {fake.bio_length.mean():.1f}")

# ---------------- Plot 4: correlation heatmap ----------------
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
            annot_kws={"size": 7})
ax.set_title("Correlation heatmap (raw columns)")
plt.tight_layout(); plt.savefig("reports/04_correlation_heatmap.png", dpi=150); plt.close()
corr = df.corr()["fake"].drop("fake").sort_values()
print("\n[Plot 4] correlation heatmap - correlation of each column with 'fake':")
print(corr.round(2).to_string())
print("  -> Negative = more common in REAL accounts, positive = more common in FAKE.")

# ---------------- Extra facts used in the report ----------------
print("\nOther group facts:")
print(f"  Has profile picture: real = {real.profile_pic.mean():.1%}, fake = {fake.profile_pic.mean():.1%}")
print(f"  Has external URL:    real = {real.external_url.mean():.1%}, fake = {fake.external_url.mean():.1%}")
print(f"  Median #posts:       real = {real.posts.median():.0f}, fake = {fake.posts.median():.0f}")
print(f"  Mean username digit ratio: real = {real.username_digit_ratio.mean():.2f}, fake = {fake.username_digit_ratio.mean():.2f}")
print("\nPlots saved to reports/")
