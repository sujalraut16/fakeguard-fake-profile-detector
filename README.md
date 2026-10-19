# Fake Instagram Profile Detector

A college Cyber Security project. Given the public details of an Instagram profile, a Random Forest
model predicts **REAL or FAKE**, shows a confidence score and explains the top reasons in plain English.

## 1. Problem
Fake accounts are used for scams, phishing, impersonation and spam. Users cannot easily tell a fake
from a real profile at a glance. This project builds a tool that **warns** users, using only
information visible on a public profile.

## 2. Dataset
- File: `data/Instagram_fake_profile_dataset.csv` (5,000 rows, 12 columns)
- Balanced: 2,500 fake and 2,500 real. No missing values. 66 duplicate rows (removed, leaving 4,934).
- Target: `fake` (1 = fake, 0 = real). There is no account-age column, so account age is not used.
- Columns: profile pic, nums/length username, fullname words, nums/length fullname, name==username,
  description length, external URL, private, #posts, #followers, #follows, fake.

What the data shows (from `src/explore.py`):

| Signal | Real accounts | Fake accounts |
|---|---|---|
| Has a profile picture | 98.3% | 21.4% |
| Empty bio | 15.8% | 83.3% |
| Median followers | 651 | 42 |
| Median posts | 76 | 0 |
| Has a link in the bio | 9.9% | 0.0% |
| Average share of digits in username | 0.04 | 0.30 |

## 3. Method
1. **Clean:** drop 66 duplicates *before* splitting (prevents the same row appearing in train and test).
2. **Features:** the 11 original columns plus 5 new ones: `followers_to_following_ratio`,
   `posts_per_follower`, `has_bio`, `log_followers`, `log_following` (denominators use +1 to avoid division by zero).
3. **Split:** 80/20, stratified, `random_state=42` -> 3,947 training rows, 987 test rows.
4. **Models:** Random Forest (RandomizedSearchCV, 30 settings x 5-fold CV, scored on F1), plus Logistic
   Regression and a tuned Decision Tree as baselines. The best model is chosen by **cross-validation**
   score on the training data, never by the test score.
5. **Explanations:** for each input, swap it for a typical value of the opposite class and measure how much
   the fake-probability moves (a simple "what-if" explanation).

In the app, bio length is counted automatically from pasted text or approximated from a size choice,
and the text is never stored.

Best Random Forest settings: `n_estimators=300, max_depth=12, min_samples_split=5, min_samples_leaf=1`.

## 4. Results (test set, 987 profiles, never used for tuning)

| Model | CV F1 | Test Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| **Random Forest** | 0.9893 | 0.9828 | 0.9897 | 0.9756 | 0.9826 | 0.9995 |
| Decision Tree | 0.9770 | 0.9737 | 0.9794 | 0.9675 | 0.9734 | 0.9866 |
| Logistic Regression | 0.9671 | 0.9595 | 0.9748 | 0.9431 | 0.9587 | 0.9930 |

Confusion matrix (Random Forest): 490 real correct, **5 real wrongly flagged fake**, **12 fake missed**, 480 fake caught.

Most important features: profile picture, username digit ratio, followers, posts, followers-to-following ratio.

### Honest note on the high score
98.3% accuracy is very high, so it was checked rather than celebrated:

| Check | Result | Meaning |
|---|---|---|
| Rows in both train and test | 0 | no duplicate leakage |
| Train / 5-fold CV / test accuracy | 99.85% / 98.94% / 98.28% | small gap (1.6 points): mild overfitting only |
| Accuracy with shuffled labels | 49.4% | pipeline does not "cheat"; it learns real patterns |
| Best single feature alone (profile picture) | 88.4% | no one column gives the answer away |
| Majority-class baseline | 50.2% | the task is not trivially easy by class imbalance |

With 987 test rows the 95% margin of error on accuracy is about +/-0.8 points.
**No leakage was found, but the score probably overstates real-world performance.** In this dataset
fake accounts are crude (79% have no profile picture, median 0 posts), so they are easy to separate.
Modern fakes with photos, bios and bought followers would be harder. Treat 98% as "performance on this
dataset", not "performance on Instagram today".

## 5. How to run
```bash
python -m venv venv
venv\Scripts\activate            # Windows   |   source venv/bin/activate   (macOS/Linux)
pip install -r requirements.txt
python src/explore.py
python src/preprocess.py
python src/train.py
python src/evaluate.py
python src/predict.py
streamlit run app/app.py
```
Run every command from the project root. Plots and tables are written to `reports/`, models to `models/`.

## 6. Project structure
```
fake_profile_detector/
  data/       raw CSV (+ data/processed/ created by preprocess.py)
  src/        features.py (shared), explore.py, preprocess.py, train.py, evaluate.py, predict.py
  models/     best_model.joblib, random_forest.joblib, feature_columns.joblib, reference_profiles.json
  app/        app.py (Streamlit)
  reports/    plots, model_comparison.csv, metrics.json, run logs
```

## 7. Limitations and Ethics
- **Dataset age and bias.** The CSV has no collection date and no source details inside it, so its age cannot
  be verified; cite the original source in your report. Instagram and fake-account tactics change quickly, so
  patterns learned from older data may not hold. The labels were assigned by someone (method unknown) and may
  contain mistakes. The data may also not represent all regions, languages or account types.
- **Real users wrongly flagged.** 5 of 495 real test accounts (1.0%) were called fake. New users, private
  users, people with no photo or no bio, and people with numbers in their username look "fake-like" to this
  model. At Instagram's scale, even 1% false positives means a very large number of real people.
- **Scammers adapt.** Once the signals are known, a scammer can add a photo, write a bio, post a few pictures
  and buy followers. This is an arms race: the model needs regular retraining on fresh data.
- **Privacy.** The tool uses only public, aggregate counts and simple text patterns. It never reads photos,
  messages or private content. Any real deployment must follow platform terms and privacy law (for example
  India's DPDP Act), avoid storing the profiles people check, and avoid mass collection of personal data.
- **Warn, don't ban.** The output is a probability, not proof. It should warn users and prioritise accounts for
  *human review*, never trigger automatic bans. People who are flagged should have a way to appeal.
- **The explanations are simple.** The "Why?" reasons are what-if effects, not a proof of cause, and they
  describe what the model learned from this dataset.
- **Not tested in a live browser here.** The Streamlit page was written and its logic tested, but the visual
  layout should be checked on your own machine.
