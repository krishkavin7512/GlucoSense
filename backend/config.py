"""Paths, constants and the feature dictionary for the CDC BRFSS 2015 diabetes data.

The feature dictionary drives three things at once: how Naive Bayes models each
column (Bernoulli / categorical / log-normal), how the frontend renders the
questionnaire, and the human-readable labels used in every chart.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "cdc_diabetes_health_indicators.csv"
ARTIFACTS = ROOT / "artifacts"
FRONTEND = ROOT / "frontend"

TARGET = "Diabetes_binary"
SEED = 42
SPLIT = (0.70, 0.15, 0.15)  # train / validation / test
TARGET_RECALL = 0.80        # screening operating point chosen on the validation set
SMOOTHING_ALPHA = 1.0       # Laplace smoothing for discrete likelihoods

# Days-in-the-last-30 answers are zero-inflated (70% say "0"), so a Gaussian is
# a poor density. They are binned and modelled as categorical instead.
DAY_BINS = [0, 1, 3, 8, 15, 30, 31]  # left-closed edges -> 0 | 1-2 | 3-7 | 8-14 | 15-29 | 30
DAY_BIN_LABELS = ["0 days", "1-2 days", "3-7 days", "8-14 days", "15-29 days", "Every day (30)"]

AGE_LABELS = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
              "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]
EDU_LABELS = ["Never attended school", "Grades 1-8", "Grades 9-11", "High school graduate",
              "Some college", "College graduate"]
INCOME_LABELS = ["< $10k", "$10k-15k", "$15k-20k", "$20k-25k", "$25k-35k", "$35k-50k",
                 "$50k-75k", "$75k+"]
GENHLTH_LABELS = ["Excellent", "Very good", "Good", "Fair", "Poor"]

YES_NO = ["No", "Yes"]

# kind: binary -> Bernoulli likelihood, ordinal -> categorical likelihood,
#       days -> binned categorical, continuous -> log-normal density.
FEATURES = [
    {"key": "HighBP", "label": "High blood pressure", "kind": "binary", "group": "Medical history",
     "question": "Has a doctor ever told you that you have high blood pressure?", "options": YES_NO},
    {"key": "HighChol", "label": "High cholesterol", "kind": "binary", "group": "Medical history",
     "question": "Has a doctor ever told you that your cholesterol is high?", "options": YES_NO},
    {"key": "CholCheck", "label": "Cholesterol check (5 yrs)", "kind": "binary", "group": "Medical history",
     "question": "Have you had your cholesterol checked in the last 5 years?", "options": YES_NO},
    {"key": "Stroke", "label": "Past stroke", "kind": "binary", "group": "Medical history",
     "question": "Have you ever had a stroke?", "options": YES_NO},
    {"key": "HeartDiseaseorAttack", "label": "Heart disease / attack", "kind": "binary", "group": "Medical history",
     "question": "Have you ever had coronary heart disease or a heart attack?", "options": YES_NO},
    {"key": "BMI", "label": "Body mass index", "kind": "continuous", "group": "Body",
     "question": "What is your body mass index (kg/m²)?", "min": 12, "max": 98, "default": 27},
    {"key": "GenHlth", "label": "General health", "kind": "ordinal", "group": "Body",
     "question": "How would you rate your general health?", "levels": [1, 2, 3, 4, 5],
     "options": GENHLTH_LABELS},
    {"key": "PhysHlth", "label": "Poor physical-health days", "kind": "days", "group": "Body",
     "question": "In the last 30 days, how many days was your physical health not good?"},
    {"key": "MentHlth", "label": "Poor mental-health days", "kind": "days", "group": "Body",
     "question": "In the last 30 days, how many days was your mental health not good?"},
    {"key": "DiffWalk", "label": "Difficulty walking", "kind": "binary", "group": "Body",
     "question": "Do you have serious difficulty walking or climbing stairs?", "options": YES_NO},
    {"key": "Smoker", "label": "Smoker (100+ cigarettes)", "kind": "binary", "group": "Lifestyle",
     "question": "Have you smoked at least 100 cigarettes in your entire life?", "options": YES_NO},
    {"key": "PhysActivity", "label": "Physically active", "kind": "binary", "group": "Lifestyle",
     "question": "Did you do any physical activity or exercise in the last 30 days (not counting your job)?",
     "options": YES_NO},
    {"key": "Fruits", "label": "Eats fruit daily", "kind": "binary", "group": "Lifestyle",
     "question": "Do you eat fruit at least once a day?", "options": YES_NO},
    {"key": "Veggies", "label": "Eats vegetables daily", "kind": "binary", "group": "Lifestyle",
     "question": "Do you eat vegetables at least once a day?", "options": YES_NO},
    {"key": "HvyAlcoholConsump", "label": "Heavy drinker", "kind": "binary", "group": "Lifestyle",
     "question": "Heavy drinking? (men 14+ drinks a week, women 7+)", "options": YES_NO},
    {"key": "Sex", "label": "Sex", "kind": "binary", "group": "About you",
     "question": "Sex", "options": ["Female", "Male"]},
    {"key": "Age", "label": "Age group", "kind": "ordinal", "group": "About you",
     "question": "Which age group are you in?", "levels": list(range(1, 14)), "options": AGE_LABELS},
    {"key": "Education", "label": "Education", "kind": "ordinal", "group": "About you",
     "question": "Highest level of education completed", "levels": list(range(1, 7)), "options": EDU_LABELS},
    {"key": "Income", "label": "Household income", "kind": "ordinal", "group": "About you",
     "question": "Annual household income (US dollars, as surveyed)", "levels": list(range(1, 9)),
     "options": INCOME_LABELS},
    {"key": "AnyHealthcare", "label": "Has health cover", "kind": "binary", "group": "About you",
     "question": "Do you have any kind of health-care coverage?", "options": YES_NO},
    {"key": "NoDocbcCost", "label": "Skipped doctor (cost)", "kind": "binary", "group": "About you",
     "question": "In the last year, did you skip seeing a doctor because of cost?", "options": YES_NO},
]

FEATURE_KEYS = [f["key"] for f in FEATURES]
FEATURE_BY_KEY = {f["key"]: f for f in FEATURES}
BINARY_KEYS = [f["key"] for f in FEATURES if f["kind"] == "binary"]
LABEL = {f["key"]: f["label"] for f in FEATURES} | {TARGET: "Diabetes / prediabetes"}

# A typical low-risk respondent; the screener form starts here.
DEFAULT_ANSWERS = {
    "HighBP": 0, "HighChol": 0, "CholCheck": 1, "Stroke": 0, "HeartDiseaseorAttack": 0,
    "BMI": 26, "GenHlth": 2, "PhysHlth": 0, "MentHlth": 0, "DiffWalk": 0, "Smoker": 0,
    "PhysActivity": 1, "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0, "Sex": 0,
    "Age": 5, "Education": 5, "Income": 6, "AnyHealthcare": 1, "NoDocbcCost": 0,
}
