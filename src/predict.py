"""
src/predict.py  -  Step 6: Predict ONE profile and explain why
Run from the project root:  python src/predict.py
(also imported by app/app.py)

HOW THE "REASONS" WORK (simple what-if idea, easy to explain in a viva):
  For every input (profile picture, bio, posts, ...) we ask the model:
  "What if THIS ONE thing looked like a typical REAL account?"
  If the fake-probability drops a lot, that input was a strong reason for FAKE.
  For accounts predicted REAL we do the opposite (swap in a typical FAKE value) to
  find what is convincing the model the account is real.
"""
import os
import sys
import json
import joblib
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import add_features, FEATURE_COLUMNS, RAW_FEATURES

# Paths are built from this file's location so it also works when called from the app
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = joblib.load(os.path.join(ROOT, "models", "best_model.joblib"))
SAVED_FEATURES = joblib.load(os.path.join(ROOT, "models", "feature_columns.joblib"))
with open(os.path.join(ROOT, "models", "reference_profiles.json")) as f:
    REFERENCE = json.load(f)          # typical real / fake values

BORDERLINE_BELOW = 0.65               # confidence under this = "borderline / not sure"


# ---------- Plain-English wording for each input ----------
def _words(key, v):
    """Turn one raw value into a short human-readable phrase."""
    if key == "profile_pic":
        return "no profile picture" if v == 0 else "has a profile picture"
    if key == "username_digit_ratio":
        if v >= 0.3: return f"many digits in the username ({v:.0%} of characters)"
        if v > 0:    return f"a few digits in the username ({v:.0%} of characters)"
        return "no digits in the username"
    if key == "fullname_words":
        if v == 0: return "no full name given"
        if v == 1: return "full name is a single word"
        return f"full name has {int(v)} words"
    if key == "fullname_digit_ratio":
        return "digits inside the full name" if v > 0 else "no digits in the full name"
    if key == "name_equals_username":
        return "full name is identical to the username" if v == 1 else "full name differs from the username"
    if key == "bio_length":
        if v == 0:  return "bio is empty"
        if v < 20:  return f"bio is very short ({int(v)} characters)"
        return f"bio has {int(v)} characters"
    if key == "external_url":
        return "has a link in the bio" if v == 1 else "no link in the bio"
    if key == "private":
        return "account is private" if v == 1 else "account is public"
    if key == "posts":
        if v == 0: return "no posts"
        if v < 10: return f"very few posts ({int(v)})"
        return f"{int(v):,} posts"
    if key == "followers":
        if v < 50: return f"very few followers ({int(v)})"
        return f"{int(v):,} followers"
    if key == "following":
        return f"follows {int(v):,} accounts"
    return key


def _with_context(key, value, label):
    """
    For counts (posts / followers / following) a bare number means little, so we add
    how it compares with a typical account of the opposite class from training data.
    """
    text = _words(key, value)
    if key in ("posts", "followers", "following"):
        other = "real" if label == "FAKE" else "fake"
        typical = int(REFERENCE[other][key])
        noun = key
        text += f"; a typical {other} account has about {typical:,} {noun}"
    return text


def format_percent(p):
    """Show probabilities honestly: a model should never claim 100% or 0% certainty."""
    if p > 0.999: return ">99.9%"
    if p < 0.001: return "<0.1%"
    return f"{p:.1%}"


def _fake_probability(profile):
    """Build the SAME features as training and return P(fake)."""
    row = pd.DataFrame([profile])[RAW_FEATURES]
    row = add_features(row)[SAVED_FEATURES]
    return float(MODEL.predict_proba(row)[0, 1])


def predict_profile(profile_dict):
    """
    profile_dict keys (all required):
      profile_pic (0/1), username_digit_ratio (0-1), fullname_words (int),
      fullname_digit_ratio (0-1), name_equals_username (0/1), bio_length (int),
      external_url (0/1), private (0/1), posts, followers, following (ints)

    Returns a dict: label, fake_probability, confidence, borderline, reasons (top 3)
    """
    missing = [k for k in RAW_FEATURES if k not in profile_dict]
    if missing:
        raise ValueError(f"Missing profile fields: {missing}")

    p_fake = _fake_probability(profile_dict)
    label = "FAKE" if p_fake >= 0.5 else "REAL"
    confidence = p_fake if label == "FAKE" else 1 - p_fake

    # --- What-if explanation ---
    # FAKE result: swap each input to the typical REAL value -> how much does P(fake) drop?
    # REAL result: swap each input to the typical FAKE value -> how much does P(fake) rise?
    swap_to = REFERENCE["real"] if label == "FAKE" else REFERENCE["fake"]
    effects = []
    for key in RAW_FEATURES:
        changed = dict(profile_dict)
        changed[key] = swap_to[key]
        new_p = _fake_probability(changed)
        effect = (p_fake - new_p) if label == "FAKE" else (new_p - p_fake)
        effects.append((effect, key))
    effects.sort(reverse=True)

    reasons = [_with_context(key, profile_dict[key], label)
               for effect, key in effects[:3] if effect > 0.01]
    return {
        "label": label,
        "fake_probability": round(p_fake, 4),
        "confidence": round(confidence, 4),
        "borderline": bool(confidence < BORDERLINE_BELOW),
        "reasons": reasons,
    }


# ---------------- 3 test profiles ----------------
TEST_PROFILES = {
    "Clearly FAKE": dict(
        profile_pic=0, username_digit_ratio=0.45, fullname_words=0, fullname_digit_ratio=0.0,
        name_equals_username=0, bio_length=0, external_url=0, private=0,
        posts=0, followers=12, following=25),
    "Clearly REAL": dict(
        profile_pic=1, username_digit_ratio=0.0, fullname_words=2, fullname_digit_ratio=0.0,
        name_equals_username=0, bio_length=70, external_url=1, private=0,
        posts=240, followers=1500, following=480),
    # Found by searching for a realistic profile the model is unsure about (about 50/50)
    "BORDERLINE": dict(
        profile_pic=1, username_digit_ratio=0.30, fullname_words=2, fullname_digit_ratio=0.0,
        name_equals_username=0, bio_length=10, external_url=0, private=0,
        posts=15, followers=150, following=100),
}

if __name__ == "__main__":
    for title, profile in TEST_PROFILES.items():
        result = predict_profile(profile)
        print(f"\n--- {title} ---")
        print("Input:", profile)
        print(f"Prediction : {result['label']}  (P(fake) = {format_percent(result['fake_probability'])}, "
              f"confidence = {format_percent(result['confidence'])}, borderline = {result['borderline']})")
        print("Top reasons:")
        for r in result["reasons"]:
            print("   -", r)
