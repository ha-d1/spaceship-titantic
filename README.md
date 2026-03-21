# spaceship-titanic
**Kaggle Competition:** https://www.kaggle.com/competitions/spaceship-titanic

---

## 1. Problem Statement

It is the year 2912. The interstellar passenger liner *Spaceship Titanic* has collided with a spacetime anomaly hidden within a dust cloud. Though the ship remained intact, almost half of the passengers were mysteriously transported to an alternate dimension.

Using records recovered from the ship's damaged computer system, our task is to build a machine learning model that predicts which passengers were transported to the alternate dimension and which were not.

This is a **binary classification problem** — for each passenger, we predict one of two outcomes:
- `True` — the passenger was transported
- `False` — the passenger was not transported

---

## 2. Data Format

The competition provides three CSV files:

- **train.csv** — labelled passenger data used to train the model (~8,700 rows)
- **test.csv** — unlabelled passenger data to generate predictions for (~4,300 rows)
- **sample_submission.csv** — an example of the required submission format

### Features Provided

| Feature | Type | Description |
|---|---|---|
| PassengerId | Categorical | Unique ID in the format gggg_pp, where gggg is the passenger's travel group |
| HomePlanet | Categorical | The planet the passenger departed from |
| CryoSleep | Boolean | Whether the passenger was in suspended animation |
| Cabin | Categorical | Cabin number in the format deck/num/side |
| Destination | Categorical | The planet the passenger is travelling to |
| Age | Numerical | Age of the passenger |
| VIP | Boolean | Whether the passenger paid for VIP service |
| RoomService | Numerical | Amount billed at the room service amenity |
| FoodCourt | Numerical | Amount billed at the food court amenity |
| ShoppingMall | Numerical | Amount billed at the shopping mall amenity |
| Spa | Numerical | Amount billed at the spa amenity |
| VRDeck | Numerical | Amount billed at the VR deck amenity |
| Name | Categorical | Full name of the passenger |
| **Transported** | **Target** | **Whether the passenger was transported (True/False)** |

---

## 3. Challenges

**Missing Values:**
Several columns contain missing data. Rather than blindly filling all nulls, we first look for logical relationships between columns. For example, if a passenger's CryoSleep status is `True`, they were in suspended animation — their spending at FoodCourt, Spa, RoomService, ShoppingMall, and VRDeck should logically be 0. After applying these contextual rules, remaining nulls are filled with the median (numerical) or mode (categorical).

**Categorical Encoding:**
Features like HomePlanet, Destination, CryoSleep, and VIP are stored as text or boolean values. Machine learning models require numerical inputs, so these must be encoded using One-Hot Encoding or Label Encoding.

**Feature Engineering:**
The Cabin column encodes three pieces of information — deck, number, and side — packed into one string. Splitting this into three separate features (Deck, CabinNum, Side) can meaningfully improve model performance. Similarly, PassengerId encodes a group number (gggg_pp format) — we extract the group number as a new feature before discarding the original ID column, since passengers travelling in the same group may share the same fate.

**Skewed Distributions & Outliers:**
The spending amenity columns (RoomService, Spa, VRDeck, etc.) are heavily skewed, with many zero values and occasional large outliers. Feature scaling (e.g. StandardScaler or MinMaxScaler) is required so these don't distort the model.

---

## 4. Skills We Will Use

| Skill / Tool | Purpose |
|---|---|
| Python (pandas, NumPy) | Data loading, manipulation, and cleaning |
| Matplotlib / Seaborn | Exploratory data analysis and visualisation |
| Exploratory Data Analysis (EDA) | Understanding distributions and relationships between features and the target |
| Data Cleaning | Handling null/missing values |
| Feature Engineering | Splitting Cabin, extracting group from PassengerId |
| Categorical Encoding | Converting text/boolean features into numerical format |
| Feature Scaling | Normalising numerical columns so large values don't dominate |
| Model Training & Evaluation | Fitting a classifier, measuring performance, avoiding overfitting |
| Submission Formatting | Producing a correctly structured CSV file for Kaggle |

---

## 5. Type of Model

Because this is a **binary classification** problem, we use classification algorithms rather than regression:

| Model | Difficulty | Notes |
|---|---|---|
| Logistic Regression | Beginner | Good interpretable baseline — start here |
| Random Forest Classifier | Beginner–Intermediate | Handles mixed data types well; explore if time allows |
| XGBoost / Gradient Boosting | Intermediate | Higher accuracy potential; requires more tuning |

We will train Logistic Regression first as a benchmark, then Random Forest if time allows, and compare results between the two.

---

## 6. Feature Selection

Choosing the right features is one of the most important steps. We will use a restricted model approach — adding features one by one and keeping the ones that improve validation accuracy — alongside these three techniques:

**Correlation Analysis**
Plot a heatmap of correlations between numerical features and the target. Features with higher correlation values are stronger candidates to include.

**Feature Importance (from Random Forest)**
After training a Random Forest, extract which features the model found most useful. Drop features with very low importance scores.

**Domain Knowledge + Visual EDA**
Visually inspect how each feature relates to the target. Key observations from this dataset:
- Passengers in CryoSleep were significantly more likely to be transported
- Cabin deck and side show meaningful differences in transport rates
- Passengers with high amenity spending were less likely to be transported

### Feature Summary

| Action | Features |
|---|---|
| Extract group number, then drop | PassengerId |
| Drop — no predictive value | Name |
| Keep & Encode | HomePlanet, Destination, CryoSleep, VIP |
| Engineer — split into Deck, CabinNum, Side | Cabin |
| Keep & Scale | RoomService, FoodCourt, ShoppingMall, Spa, VRDeck, Age |

---

## 7. Evaluation Metrics

We will track two metrics locally during development:

| Metric | Why We Use It |
|---|---|
| Accuracy | Overall percentage of correct predictions — the primary Kaggle leaderboard metric |
| F1 Score | Balances false positives and false negatives — appropriate since both error types carry roughly equal cost in this problem |

---

## 8. Validation Strategy

The test.csv file provided by Kaggle is unlabelled — we cannot measure accuracy on it locally. To evaluate and tune our model, we split train.csv into two parts:

- **Training set (~80%)** — used to fit the model
- **Validation set (~20%)** — used to measure accuracy and F1 score locally

This allows us to tune hyperparameters and compare models without wasting Kaggle submission attempts. Only once we are satisfied with local performance do we generate predictions on test.csv for the final submission.

