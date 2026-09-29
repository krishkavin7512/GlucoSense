# GlucoSense

Diabetes risk screening from 21 simple health questions, no blood test needed. A Naive Bayes model, written from
scratch in NumPy and trained on 253,680 real survey responses, turns your answers into a calibrated risk, a
test / no-test recommendation, and an explanation of which answers mattered.

## Features

- **Screener**: 21 questions in; out come a calibrated risk %, a blood-test recommendation at a sensitivity you choose
  (catch 50–95% of cases), and every answer's effect shown as an odds multiplier.
- **Labs**: a Bayes lab (prior → posterior icon arrays), BMI distributions with a quantile slider, and a curve-fitting
  lab where a polynomial overfits live.
- **Graphs explained** (`/graphs`): all 32 training graphs, each with what it shows, how it's computed, how to read it
  and where it's used.
- **Report**: test-set metrics, ROC, confusion matrix and calibration.
- Animated, accessible frontend with no framework (HTML, CSS, JavaScript modules, Plotly.js bundled locally).

## Results (38,052 held-out people)

| Metric | Value |
|---|---|
| ROC AUC | 0.807 (5-fold CV 0.811 ± 0.003) |
| Recall at the screening threshold | 79.4% of real cases caught |
| Precision | 28.5% (2× the 13.9% base rate) |
| People referred for a blood test | 38.8% |
| Log-loss, raw → calibrated | 0.512 → 0.327 |

## How it works

| Technique | Implementation (`backend/`) |
|---|---|
| Naive Bayes classifier | `naive_bayes.py`: Bernoulli + categorical likelihoods with Laplace smoothing, log-normal BMI, computed in log space |
| Probability calibration | `calibration.py`: equal-count binning, Laplace's rule of succession, pool-adjacent-violators |
| Decision threshold | Highest score that still catches ≥ 80% of cases on the validation set |
| Polynomial curve fitting | `polyfit.py`: least squares and ridge, (ΦᵀΦ + λI)w = Φᵀt, E_RMS, degree and λ sweeps |
| Bayes' rule, sum and product rules | `probability.joint_table`, `probability.bayes_rule` |
| Independence and conditional independence | `probability.mutual_information`, `conditional_mutual_information`, `chi_square_independence` |
| Quantiles, mean, variance | `probability.describe`, `quantile`, `ecdf`, `normal_ppf` |
| Densities, expectation, covariance | `probability.fit_gaussian` / `fit_lognormal` (maximum likelihood), `total_expectation`, `covariance_matrix` |
| Evaluation | `metrics.py`: ROC, precision–recall, confusion matrix, log-loss, Brier score, calibration bins |

Every model is implemented in NumPy. scikit-learn is used only in the tests, to confirm the results match.

## Data

CDC Behavioral Risk Factor Surveillance System 2015, cleaned release: 253,680 respondents, 21 features and a
diabetes / prediabetes label. Source: [UCI Machine Learning Repository #891](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
(CC BY 4.0), included in `data/`. Split 70 / 15 / 15 (stratified, seed 42) into training, validation and test.

## Run it

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
python run.py
```

The app opens at http://127.0.0.1:8001 (API docs at `/docs`). The trained models are included in `artifacts/`, so it
starts immediately.

- `python run.py --retrain`: retrain everything from scratch (about 30 s).
- `python -m backend.train --charts-only`: rebuild the graphs without retraining.
- `python -m pytest`: 17 tests; every from-scratch function is checked against NumPy, SciPy or scikit-learn.

## Layout

```
backend/   config, data, probability, naive_bayes, calibration, polyfit, metrics, train, charts, api
frontend/  index.html (app), graphs.html (graphs explained), assets/ (css, js, bundled plotly.js)
artifacts/ trained model, full report and graphs (JSON, written by backend/train.py)
data/      the dataset (CSV)
tests/     from-scratch vs reference-library checks, API tests
```

An educational tool, not a medical device: it suggests who should get a blood test; it does not diagnose diabetes.
