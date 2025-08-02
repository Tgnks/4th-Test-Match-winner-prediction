# Cricket Match Outcome Prediction Using Machine Learning

## Overview

This project predicts the result of an upcoming Test cricket match between **England** and **India** at Manchester using **Random Forest classification**. It leverages historical match data, recent head-to-head statistics, and ground-specific trends to train a model that forecasts whether England, India, or a Draw is the most likely outcome.

---

## Requirements

- **Python 3.x**
- **Pandas** (for data manipulation)
- **scikit-learn** (for the Random Forest Classifier)

Install the dependencies using:

```bash
pip install pandas scikit-learn
```

---

## Data Inputs

- `df1`, `df2`, `df3`: Ball-by-ball DataFrames for the last 3 England vs India matches.
- `ground_df`: DataFrame of last 10 Test matches played at Manchester, with columns like `'Team 1'`, `'Team 2'`, `'Winner'`.

---

## Methodology

1. **Statistical Feature Engineering:**
    - For each team, calculate total matches played, won, lost, win and draw rates at the ground.
    - Calculate head-to-head (H2H) batting stats from recent matches: total runs, wickets, average, and strike rate.

2. **Dataset Construction:**
    - Each row represents a past Manchester Test, with statistics for both teams prior to that game (excluding the game itself to avoid data leakage).
    - The result is encoded as `1` (Team 1 wins), `2` (Team 2 wins), or `0` (Draw).

3. **Model Training:**
    - Use the constructed feature dataset to train a `RandomForestClassifier` to predict the match result.

4. **Prediction:**
    - Prepare the features for the next England vs India Test and use the trained model to predict the outcome.

---

## Code Breakdown

### 1. Importing Libraries

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
```

### 2. Team Statistics Function

Calculates matches played, won, lost, win/draw rates for a team at Manchester:

```python
def get_team_stats(hist_df, team):
    played = ((hist_df['Team 1'] == team) | (hist_df['Team 2'] == team)).sum()
    won = (hist_df['Winner'] == team).sum()
    drawn = (hist_df['Winner'] == 'drawn').sum()
    lost = played - won - drawn
    win_rate = won / played if played > 0 else 0
    draw_rate = drawn / played if played > 0 else 0
    return played, won, lost, win_rate, draw_rate
```

### 3. Head-to-Head Batting Stats Function

For a given team, computes runs, wickets, average and strike rate over multiple recent matches:

```python
def get_h2h_stats(dlf_list, team):
    balls = pd.concat(dlf_list)
    team_batting = balls[balls['team'] == team]
    runs = team_batting['runs_batter'].sum()
    wickets = pd.to_numeric(team_batting['wicket'], errors='coerce').fillna(0).astype(int).sum()
    balls_faced = team_batting.shape[0]
    avg = runs / wickets if wickets > 0 else runs
    sr = runs / balls_faced * 100 if balls_faced > 0 else 0
    return runs, wickets, avg, sr
```

### 4. Feature Construction

Iterate over each match in `ground_df` and build a feature row with statistics **before** that match:

```python
rows = []
for i, row in ground_df.reset_index(drop=True).iterrows():
    hist = ground_df.iloc[:i]
    t1 = row['Team 1']
    t2 = row['Team 2']
    t1_stats = get_team_stats(hist, t1)
    t2_stats = get_team_stats(hist, t2)
    # Result encoding
    if row['Winner'] == t1:
        result = 1
    elif row['Winner'] == t2:
        result = 2
    else:
        result = 0
    # Head-to-head stats placeholders (not used for historical rows)
    rows.append({
        't1_played': t1_stats[0],
        't1_won': t1_stats[1],
        't1_lost': t1_stats[2],
        't1_win_rate': t1_stats[3],
        't1_draw_rate': t1_stats[4],
        't2_played': t2_stats[0],
        't2_won': t2_stats[1],
        't2_lost': t2_stats[2],
        't2_win_rate': t2_stats[3],
        't2_draw_rate': t2_stats[4],
        't1_h2h_runs': 0,
        't1_h2h_wickets': 0,
        't1_h2h_avg': 0,
        't1_h2h_sr': 0,
        't2_h2h_runs': 0,
        't2_h2h_wickets': 0,
        't2_h2h_avg': 0,
        't2_h2h_sr': 0,
        'result': result,
        'team1': t1,
        'team2': t2,
    })
```

### 5. Model Training

```python
df_features = pd.DataFrame(rows)
feature_cols = [
    't1_played', 't1_won', 't1_lost', 't1_win_rate', 't1_draw_rate',
    't2_played', 't2_won', 't2_lost', 't2_win_rate', 't2_draw_rate',
    't1_h2h_runs', 't1_h2h_wickets', 't1_h2h_avg', 't1_h2h_sr',
    't2_h2h_runs', 't2_h2h_wickets', 't2_h2h_avg', 't2_h2h_sr'
]
X = df_features[feature_cols]
y = df_features['result']

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)
```

### 6. Preparing Features for Next Match & Prediction

Use current statistics and recent H2H data to build the input row:

```python
eng_stats = get_team_stats(ground_df, 'England')
ind_stats = get_team_stats(ground_df, 'India')
eng_h2h = get_h2h_stats([df1, df2, df3], 'England')
ind_h2h = get_h2h_stats([df1, df2, df3], 'India')

next_match_features = pd.DataFrame([{
    't1_played': eng_stats[0],
    't1_won': eng_stats[1],
    't1_lost': eng_stats[2],
    't1_win_rate': eng_stats[3],
    't1_draw_rate': eng_stats[4],
    't2_played': ind_stats[0],
    't2_won': ind_stats[1],
    't2_lost': ind_stats[2],
    't2_win_rate': ind_stats[3],
    't2_draw_rate': ind_stats[4],
    't1_h2h_runs': eng_h2h[0],
    't1_h2h_wickets': eng_h2h[1],
    't1_h2h_avg': eng_h2h[2],
    't1_h2h_sr': eng_h2h[3],
    't2_h2h_runs': ind_h2h[0],
    't2_h2h_wickets': ind_h2h[1],
    't2_h2h_avg': ind_h2h[2],
    't2_h2h_sr': ind_h2h[3],
}])

pred = clf.predict(next_match_features)[0]
if pred == 1:
    prediction = 'England'
elif pred == 2:
    prediction = 'India'
else:
    prediction = 'Draw'

print(f"Predicted winner for next England vs India Test at Manchester: {prediction}")
```

---

## Interpretation

- The model uses historical performance at the ground and recent H2H batting stats as input features.
- The output is a prediction (`England`, `India`, or `Draw`) for the upcoming Test match at Manchester.

---

## Notes & Improvements

- This is a basic approach; accuracy can be improved by including more features (bowling stats, player form, weather, toss, etc.).
- The model assumes the past is indicative of the future, which may not always hold true in sports.
- Ensure dataframes are cleaned and columns have expected names and types.
