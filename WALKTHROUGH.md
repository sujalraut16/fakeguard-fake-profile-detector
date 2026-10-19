# Step-by-step walkthrough (easy language, for your viva)

All numbers below come from the real run on your CSV.

## Step 1 - Project setup
Folders: `data/` (CSV), `src/` (code), `models/` (saved models), `app/` (web app), `reports/` (plots, tables).
```bash
python -m venv venv
venv\Scripts\activate          # Windows;  macOS/Linux: source venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit
pip freeze > requirements.txt  # or use the provided requirements.txt
```
A **virtual environment** keeps this project's libraries separate from other projects.

**Assumption:** I added one extra file, `src/features.py`. It holds the feature-building code that training,
prediction and the app all share, so a new profile is always processed exactly like the training data.

## Step 2 - Data exploration (`src/explore.py`)
Result: 5,000 rows x 12 columns, all numbers, no missing values, 66 duplicates, 2,500 / 2,500 classes.
- **Class distribution plot:** the classes are perfectly balanced, so accuracy is a fair metric and no resampling is needed.
- **Followers vs following (log scale):** fake accounts sit in the bottom-left corner: median 42 followers and 68 following,
  against 651 and 442 for real accounts. The log scale is needed because counts range from 0 to 15 million.
- **Bio length:** 83.3% of fake accounts have an empty bio versus 15.8% of real ones. The average bio is 4 characters for
  fake and 38 for real.
- **Correlation heatmap:** strongest links with `fake` are username digit ratio (+0.66, fakes have more digits)
  and profile picture (-0.78, fakes lack pictures), then bio length (-0.51). Correlation is not causation.

## Step 3 - Cleaning and feature engineering (`src/preprocess.py`)
1. **Drop duplicates (66)** before splitting, so no identical profile sits in both train and test.
2. New features, and why each helps:
   - `followers_to_following_ratio = followers / (following + 1)`: fakes follow many accounts but few follow back, so the ratio is low.
   - `posts_per_follower = posts / (followers + 1)`: shows an account with a profile but no activity (or a huge audience with no content).
   - `has_bio`: a simple yes/no version of "did they bother to write a bio" (83% of fakes did not).
   - `log_followers`, `log_following`: shrink huge numbers so a few celebrity accounts do not dominate.
3. **Split 80/20 with `stratify` and `random_state=42`:** stratify keeps the same fake/real mix in both parts; the fixed seed makes results repeatable.
   Train = 3,947 rows, test = 987 rows.

## Step 4 - Training (`src/train.py`)
**How a Random Forest works (simple words):** a *decision tree* is a flowchart of yes/no questions
("has a profile picture? -> posts more than 5? -> ..."). One tree can memorise the training data. A Random Forest
builds hundreds of trees, each on a random sample of the rows and a random subset of features at each split, then
lets all trees **vote**. Individual trees make different mistakes, and the vote averages them out, so the forest is
more accurate and more stable than one tree. The share of trees voting "fake" gives the probability.

**Tuning:** `RandomizedSearchCV` tried 30 combinations of `n_estimators` (number of trees), `max_depth` (how deep a tree grows),
`min_samples_split` and `min_samples_leaf` (stop trees from memorising tiny groups). Each combination was scored with
**5-fold cross-validation** (split training data in 5 parts; train on 4, check on 1; repeat 5 times).
Best: 300 trees, depth 12, split 5, leaf 1.

| Model | CV F1 | Test Acc | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Random Forest | 0.9893 | 0.9828 | 0.9897 | 0.9756 | 0.9826 | 0.9995 |
| Decision Tree | 0.9770 | 0.9737 | 0.9794 | 0.9675 | 0.9734 | 0.9866 |
| Logistic Regression | 0.9671 | 0.9595 | 0.9748 | 0.9431 | 0.9587 | 0.9930 |

The best model was picked by CV score, not test score, so the test set stays an honest final exam.
The Random Forest wins, but even simple logistic regression reaches 96%: this tells us the data is easy to separate.

## Step 5 - Evaluation (`src/evaluate.py`)
Confusion matrix: TN 490, FP 5, FN 12, TP 480.
- **False positive** = a real person wrongly flagged (5 cases, 1.0% of real accounts).
- **False negative** = a fake that slips through (12 cases, 2.4% of fake accounts).

**Which metric matters most?** It depends on what happens after a flag. If flags lead to punishment (ban), **precision** matters most,
because every false positive hurts a real person. If flags only warn users, **recall** matters more, because every missed fake can scam someone.
This project only *warns*, so we watch both and use **F1** as the main number, plus ROC-AUC (quality across all thresholds).
The threshold table shows the trade-off:

| Threshold | Precision | Recall | False positives | False negatives |
|---|---|---|---|---|
| 0.3 | 0.974 | 0.996 | 13 | 2 |
| 0.5 (default) | 0.990 | 0.976 | 5 | 12 |
| 0.7 | 0.998 | 0.963 | 1 | 18 |
| 0.9 | 1.000 | 0.923 | 0 | 38 |

**Is 98% suspicious? Yes, so we checked.** See the table in README section 4: no duplicate rows across train/test, a small train-test gap,
shuffled labels give ~49% (the pipeline is not cheating), and no single feature gives the answer away (best alone: 88%).
**What to say in your report:** "No data leakage was detected, but the dataset's fake accounts are fairly crude, so 98% is performance on
this dataset and probably overstates real-world performance. The model should be retrained and re-tested on newer, independently collected data."

Feature importance: profile picture, followers, posts and username digit ratio lead. `external_url` and `name==username` score near zero even
though no fake has a URL: they are rare, and other features already carry the same signal. Correlated features (e.g. `followers` and `log_followers`)
share importance, so permutation importance (right-hand plot) spreads it differently. Mention this if asked.

## Step 6 - Prediction with reasons (`src/predict.py`)
`predict_profile(dict)` builds the same 16 features, returns label, probability and the top 3 reasons.
**Reasons use a what-if test:** for each input, replace it with a typical value from the opposite class and see how far the fake-probability falls.
The biggest drops are the reasons. Test profiles on your model:

| Profile | Result | Reasons |
|---|---|---|
| Clearly fake (no picture, no bio, 0 posts, 12 followers) | FAKE, >99.9% | very few followers; no profile picture; no posts |
| Clearly real (picture, 70-char bio, link, 240 posts, 1,500 followers) | REAL, >99.9% | 1,500 followers; has a profile picture; 240 posts |
| Borderline (picture, 30% digits, 10-char bio, 15 posts, 150 followers) | FAKE at 50.0%, flagged "borderline" | many digits in username; 150 followers; 15 posts |

The borderline profile was found by searching for a realistic profile the model is genuinely unsure about; it is not a typical case.

## Step 7 - Web app (`app/app.py`)
Run `streamlit run app/app.py`. The sidebar form takes the 9 inputs; the digit ratios, word count and name==username flag are calculated from the typed text.
The result page has a verdict, a gauge showing the fake-probability with a grey "unsure" zone (35-65%), a "Why?" section, safe-usage advice and a disclaimer.
Results between 35% and 65% are shown as "Not sure" instead of a hard label.
**Honest note:** Streamlit could not be installed in my sandbox, so I tested the app's logic (feature calculation, verdict, gauge, predictions) with a stand-in,
but I could not view the page. Please open it once and tell me if any layout needs fixing.
