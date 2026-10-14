"""
Check functions for the Agentic Coding workshop.

Run these after the agent finishes each part. They don't fix anything.
They tell you whether the agent's work holds up.

Please don't ask the agent to edit this file. The checks only help if
they stay the same for every team.
"""

import hashlib

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_URL = (
    "https://raw.githubusercontent.com/Break-Through-Tech/"
    "ai-studio-agentic-coding-workshop/main/data/hotel_bookings_sample.csv"
)
TARGET = "is_canceled"
LEAK_COLUMNS = ["reservation_status", "reservation_status_date"]
RECALL_GOAL = 0.75

NUMERIC = [
    "lead_time", "stays_in_weekend_nights", "stays_in_week_nights", "adults",
    "children", "babies", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "booking_changes", "days_in_waiting_list",
    "adr", "required_car_parking_spaces", "total_of_special_requests",
]
CATEGORICAL = [
    "hotel", "arrival_date_month", "meal", "market_segment",
    "distribution_channel", "deposit_type", "customer_type",
    "reserved_room_type",
]

_RAW_FINGERPRINT = None


def _line():
    print("-" * 60)


def _fingerprint(df):
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def load_data():
    """Load the workshop dataset and remember what the raw data looked like."""
    global _RAW_FINGERPRINT
    df = pd.read_csv(DATA_URL)
    _RAW_FINGERPRINT = _fingerprint(df)
    print(f"Loaded {len(df):,} bookings with {df.shape[1]} columns.")
    return df


# ----------------------------------------------------------------------
# Part 1: Explore
# ----------------------------------------------------------------------
def check_quality_report(raw_df):
    """Show the known data quality issues so you can compare with the agent's list."""
    _line()
    print("CHECK 1: What's actually in this data?")
    _line()
    dupes = raw_df.duplicated().sum()
    missing = raw_df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(f"Exact duplicate rows: {dupes:,}")
    print("Missing values:")
    for col, n in missing.items():
        print(f"  {col}: {n:,} ({n / len(raw_df):.0%})")
    print("Month is stored as text (for example 'July'), not a number.")
    print("Note: a missing 'agent' means the guest booked directly.")
    print("It is not an error, so dropping those rows throws away real bookings.")
    print()
    print("Compare this with the agent's report.")
    print("What did it catch? What did it miss? Did it list anything we didn't?")
    print("(This list isn't everything. There's more to find later.)")


# ----------------------------------------------------------------------
# Part 2: Plan, Build, Check
# ----------------------------------------------------------------------
def check_cleaned(raw_df, clean_df):
    """Check that cleaning worked and the raw data was left alone."""
    _line()
    print("CHECK 2: Did the cleaning hold up?")
    _line()
    ok = True
    if _RAW_FINGERPRINT and _fingerprint(raw_df) != _RAW_FINGERPRINT:
        print("WARNING: raw_df changed. The agent edited the raw data in place.")
        print("Ask it to rerun the cleaning on a copy and keep raw_df untouched.")
        ok = False
    else:
        print("PASS: raw_df is unchanged.")

    left = clean_df.isna().sum()
    left = left[left > 0]
    if len(left):
        print("WARNING: these columns still have missing values:")
        for col, n in left.items():
            print(f"  {col}: {n:,}")
        ok = False
    else:
        print("PASS: no missing values left.")

    n = len(clean_df)
    if n < 15_000:
        print(f"WARNING: only {n:,} rows left out of {len(raw_df):,}.")
        print("That's a lot to lose. Ask the agent which steps dropped rows and why.")
        ok = False
    else:
        print(f"PASS: {n:,} rows kept out of {len(raw_df):,}.")

    for col in [TARGET, "arrival_date_year", "hotel"]:
        if col not in clean_df.columns:
            print(f"WARNING: clean_df is missing the '{col}' column. Keep it.")
            ok = False
    print()
    print("Nice work. Move on to the model." if ok else "Fix the warnings before you build the model.")


def check_model(model, X_test, y_test):
    """Score the model and look for results that are too good to be true."""
    _line()
    print("CHECK 3: How good is the model, really?")
    _line()
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print()
    if acc > 0.95:
        print("RED FLAG: this score is too good to be true.")
        print("Predicting hotel cancellations is hard. Real models land well below this.")
        print("Ask the agent which features the model uses, and whether any of them")
        print("could only be known AFTER a booking is canceled. Then fix it and rerun.")
    else:
        print("This looks realistic for this problem.")
        print("Next, let's see how the model holds up on future bookings.")


# ----------------------------------------------------------------------
# Part 3: Find it yourself
# ----------------------------------------------------------------------
def _simple_pipeline(cols_num, cols_cat):
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]), cols_num),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cols_cat),
    ])
    return Pipeline([("prep", pre), ("model", LogisticRegression(max_iter=2000))])


def check_future(clean_df):
    """Train on 2015 and 2016 bookings, then test on 2017 bookings."""
    _line()
    print("CHECK 4: Does the model work on future bookings?")
    _line()
    if "arrival_date_year" not in clean_df.columns:
        print("This check needs the 'arrival_date_year' column. Keep it in clean_df.")
        return
    cols_num = [c for c in NUMERIC if c in clean_df.columns]
    cols_cat = [c for c in CATEGORICAL if c in clean_df.columns]
    past = clean_df[clean_df["arrival_date_year"] < 2017]
    future = clean_df[clean_df["arrival_date_year"] == 2017]
    if len(future) < 5000:
        print(f"WARNING: clean_df only has {len(future):,} bookings from 2017.")
        print("The raw data has about 6,000 even after removing duplicates.")
        print("A cleaning step removed a lot of 2017 bookings. Find out which one, and why.")
        print("Removing rows can hide a problem instead of fixing it.")
        print()
    pipe = _simple_pipeline(cols_num, cols_cat)

    # Accuracy on held-out PAST bookings, for comparison.
    shuffled = past.sample(frac=1, random_state=0)
    cut = int(len(shuffled) * 0.8)
    pipe.fit(shuffled.iloc[:cut][cols_num + cols_cat], shuffled.iloc[:cut][TARGET])
    past_acc = accuracy_score(
        shuffled.iloc[cut:][TARGET], pipe.predict(shuffled.iloc[cut:][cols_num + cols_cat])
    )

    pipe.fit(past[cols_num + cols_cat], past[TARGET])
    future_acc = accuracy_score(future[TARGET], pipe.predict(future[cols_num + cols_cat]))

    print(f"Accuracy on held-out 2015-2016 bookings: {past_acc:.3f}")
    print(f"Accuracy on 2017 bookings:               {future_acc:.3f}")
    gap = past_acc - future_acc
    print(f"Gap: {gap:.3f}")
    print()
    if gap > 0.08:
        print("The model does much worse on 2017 bookings.")
        print("Something about the 2017 data is different from earlier years.")
        print("Your job: find out what changed, and fix it.")
        print("The agent can help you look, but don't expect it to spot this one.")
        print("Tip: compare the same columns across years. Then across hotels.")
    else:
        print("PASS: the model holds up on future bookings.")
        print("If you fixed something to get here, write down what it was and how you found it.")


# ----------------------------------------------------------------------
# Challenge
# ----------------------------------------------------------------------
def _years_look_different(train_df, test_df):
    """True if any numeric column's average shifts a lot between the two splits."""
    for col in NUMERIC:
        if col in train_df.columns and col in test_df.columns:
            before, after = train_df[col].mean(), test_df[col].mean()
            if before > 1 and after / before > 3:
                return True
    return False


def check_challenge(pipeline, train_df, test_df, threshold=0.5):
    """Score a challenge submission. Read the rules in the notebook first."""
    _line()
    print("CHALLENGE CHECK")
    _line()
    ok = True
    if not isinstance(pipeline, Pipeline):
        print("WARNING: submit a scikit-learn Pipeline that includes your preprocessing.")
        ok = False
    if train_df["arrival_date_year"].max() > 2016 or (test_df["arrival_date_year"] != 2017).any():
        print("WARNING: train on 2015 and 2016 only, and test on 2017 only.")
        ok = False
    used = list(getattr(pipeline, "feature_names_in_", []))
    if not used:
        print("WARNING: fit the pipeline on a DataFrame (not a NumPy array),")
        print("so this check can see which columns it uses.")
        ok = False
    leaks = [c for c in used if c in LEAK_COLUMNS]
    if leaks:
        print(f"WARNING: the pipeline uses leaky columns: {leaks}")
        ok = False
    if _years_look_different(train_df, test_df):
        print("WARNING: something in the 2017 data still looks wrong.")
        print("Go back to Part 3 and run CHECK 4.")
        ok = False
    if not ok:
        print("\nFix the warnings above, then run this check again.")
        return None

    missing = [c for c in used if c not in test_df.columns]
    if missing:
        print(f"WARNING: test_df is missing columns the pipeline needs: {missing}")
        return None
    probs = pipeline.predict_proba(test_df[used])[:, 1]
    preds = (probs >= threshold).astype(int)
    y = test_df[TARGET]
    rec = recall_score(y, preds, zero_division=0)
    prec = precision_score(y, preds, zero_division=0)
    print(f"Threshold: {threshold:.2f}")
    print(f"Recall on 2017 bookings:    {rec:.3f}  (goal: {RECALL_GOAL:.2f} or higher)")
    print(f"Precision on 2017 bookings: {prec:.3f}")
    print()
    if rec >= RECALL_GOAL:
        print(f"PASS. Your team's score is {prec:.3f}.")
        print("Higher is better. Share it in the debrief.")
        return round(prec, 3)
    print("Recall is below the goal. Try lowering the threshold.")
    return None
