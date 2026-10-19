# 6-slide presentation outline (with speaker notes)

## Slide 1 - Title and problem
**On slide:** "Fake Instagram Profile Detector" / name, course, guide / one line: "Warning users about suspicious accounts using machine learning."
**Notes:** Fake accounts are used for scams, phishing and impersonation. Users cannot easily tell them apart. My project predicts whether a profile is real or fake from public details and explains why, so people can be careful.

## Slide 2 - Data
**On slide:** 5,000 profiles, balanced 2,500/2,500, 11 features, 66 duplicates removed. Small table: profile picture 98% real vs 21% fake; empty bio 16% vs 83%; median posts 76 vs 0. Include `reports/02_followers_vs_following.png`.
**Notes:** The data is balanced, so accuracy is meaningful. Fake accounts clearly differ: they rarely have a picture or bio, post nothing and have few followers. There is no account-age column, so I did not use it. I added five features such as followers-to-following ratio and log counts.

## Slide 3 - Method
**On slide:** Flow: clean -> features -> 80/20 split -> tune Random Forest (5-fold CV) -> compare with Logistic Regression and Decision Tree. One-line explanation: "Many trees vote."
**Notes:** I removed duplicates before the split to avoid leakage. A Random Forest builds many decision trees on random parts of the data and takes a vote. I tuned it with RandomizedSearchCV and 5-fold cross-validation, and chose the best model using cross-validation, not the test set.

## Slide 4 - Results
**On slide:** Comparison table (RF: accuracy 98.3%, precision 99.0%, recall 97.6%, F1 98.3%, AUC 0.9995). Confusion matrix image `reports/05_confusion_matrix.png`. Feature importance image.
**Notes:** Random Forest beat both baselines. Out of 987 test profiles, 5 real accounts were wrongly flagged and 12 fakes were missed. Precision matters if we punish, recall if we only warn; our tool warns. The most important signals are profile picture, username digits, followers and posts.

## Slide 5 - Is 98% believable? and the demo
**On slide:** Leakage checks table (0 overlap rows; train 99.85% vs test 98.28%; shuffled labels 49%; best single feature 88%). Screenshot of the Streamlit app showing verdict, gauge and "Why?".
**Notes:** I did not trust the high score, so I tested for leakage and overfitting and found none. But the dataset's fake accounts are fairly crude, so real-world performance will likely be lower. Demo: enter a fake-looking profile, then a real one, then a borderline one that shows "Not sure". The app explains its top 3 reasons and gives safety advice.

## Slide 6 - Limitations, ethics and future work
**On slide:** Old/biased data; real users wrongly flagged; scammers adapt; privacy; warn, don't ban. Future: newer data, account age and behaviour over time, human review and appeals.
**Notes:** This is a prediction, not proof. 1% of real accounts were wrongly flagged, which is many people at Instagram's scale. So the tool should warn users and support human reviewers, never auto-ban. Next steps are fresher data, behavioural features, and regular retraining.
