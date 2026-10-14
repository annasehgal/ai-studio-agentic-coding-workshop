# AI Studio Workshop: Agentic Coding

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Break-Through-Tech/ai-studio-agentic-coding-workshop/blob/main/agentic_coding_workshop.ipynb)

Let's use an AI agent to build a machine learning model, and learn how to steer it and check its work. We'll follow one framework the whole way: Explore, Plan, Build, Check.

We'll use Gemini in Google Colab. There's nothing to install. Click the badge above, sign in with a personal Google account, and you're ready.

## What's in this repo

| File | What it's for |
|---|---|
| `agentic_coding_workshop.ipynb` | The fellow notebook. Open it in Colab. |
| `workshop_checks.py` | Check functions the notebook runs after each part. Please don't edit it. |
| `data/hotel_bookings_sample.csv` | 20,000 hotel bookings from 2015 to 2017. |

## The task

Can we predict which hotel bookings will be canceled? Your team directs Gemini to clean the data and build a model. After each part, a check cell tells you whether the work holds up.

- **Part 1: Explore.** Get a data quality report. Fix nothing yet.
- **Part 2: Plan, Build, Check.** Plan, clean, and train a baseline model.
- **Part 3: Find it yourself.** Test the model on future bookings and track down what's wrong.
- **Challenge.** Build a time-split pipeline that catches at least 75% of cancellations.

## Data

Sampled from the Hotel Booking Demand dataset: Nuno Antonio, Ana de Almeida, and Luis Nunes, "Hotel booking demand datasets," *Data in Brief* 22 (2019): 41-49. Licensed under CC BY 4.0. We used the TidyTuesday copy (2020-02-11).

We drew a random sample (`random_state=42`) and made changes for teaching purposes.
