# 15 likely viva questions with short answers

1. **What problem does your project solve?**
   It warns users about suspicious Instagram profiles by predicting REAL or FAKE from public details such as profile picture, bio, posts and follower counts.

2. **Why a Random Forest?**
   It handles numbers and yes/no features without much preparation, copes with skewed data like follower counts, resists overfitting better than one tree, and gives probabilities and feature importances. It also scored best in cross-validation (F1 0.989 against 0.977 and 0.967).

3. **How does a Random Forest work?**
   It builds many decision trees on random samples of the rows and features, and each tree votes. The majority vote is the prediction; the share of "fake" votes is the probability.

4. **What is cross-validation and why use it?**
   The training data is split into 5 parts; the model trains on 4 and is checked on the 5th, five times. It gives a more reliable score than one split and is used to pick settings without touching the test set.

5. **What is stratify and random_state=42?**
   `stratify` keeps the fake/real proportion the same in train and test. `random_state=42` fixes the random shuffle so anyone gets the same split and results.

6. **Why did you remove duplicates before splitting?**
   66 rows were duplicates. If identical rows land in both train and test, the model is tested on data it has already seen, which inflates accuracy (data leakage).

7. **What new features did you create and why?**
   Followers-to-following ratio, posts per follower, has-bio flag, and log followers/following. They capture behaviour (fakes follow many, are followed by few, post little, write no bio) and tame huge counts. Every denominator has +1 to avoid division by zero.

8. **Why not just use accuracy?**
   The classes are balanced, so accuracy is fair here, but it hides *which* mistakes are made. Precision shows false alarms on real people, recall shows missed fakes, F1 balances them, and ROC-AUC shows quality at every threshold.

9. **Which matters more: precision or recall?**
   If flags lead to bans, precision (never punish real people). If flags only warn users, recall (catch more fakes). Our tool only warns, so we report both and use F1. We got precision 0.990, recall 0.976.

10. **Your accuracy is 98%. Is that too good to be true?**
    It is high, so I tested it: no rows shared between train and test, train/CV/test accuracy of 99.85/98.94/98.28 (small gap), shuffled labels gave 49%, and no single feature gave the answer (best alone 88%). No leakage was found, but the dataset's fakes are crude, so real-world accuracy would probably be lower.

11. **What is overfitting and did it happen?**
    Overfitting is memorising training data instead of learning general patterns. Train accuracy (99.85%) is 1.6 points above test (98.28%), which is mild and normal for a forest. CV and test agree, so it is not serious.

12. **What is the confusion matrix result?**
    490 real correct, 5 real wrongly flagged fake, 12 fake missed, 480 fake caught.

13. **How does your tool explain a prediction?**
    For each input, it swaps in a typical value from the opposite class and measures how much the fake-probability changes. The biggest changes become the top 3 reasons. It is a simple what-if method, not a proof of cause.

14. **What are the limitations and ethical risks?**
    Old and possibly biased data; the dataset has no collection date; real users with no photo or bio may be wrongly flagged; scammers adapt; privacy must be protected; and the output is a prediction, not proof, so it should warn users and support human review rather than auto-ban.

15. **How would you improve it?**
    Collect newer, independently labelled data; add behaviour over time (account age, posting pattern, engagement), which this dataset lacks; test on a second dataset; calibrate probabilities; add a human-review and appeal process; and retrain regularly.
