# retrieval_agent.py
from typing import List, Tuple, Optional
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Hardcoded few-shot examples (replacing ingest_rules import)
FEWSHOT_CLASSIFICATION = """
# Pattern: Robust classification pipeline (fits Code Contract)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

X = df.drop(columns=[target_column])
y = df[target_column]

num_cols = X.select_dtypes(include=['number','bool']).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

pre = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols),
    ]
)

model = Pipeline(steps=[('pre', pre), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))])
_ = model.predict(X.head(1))
"""

FEWSHOT_REGRESSION = """
# Pattern: Robust regression pipeline (fits Code Contract)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

X = df.drop(columns=[target_column])
y = df[target_column]

num_cols = X.select_dtypes(include=['number','bool']).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

pre = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols),
    ]
)

model = Pipeline(steps=[('pre', pre), ('clf', RandomForestRegressor(n_estimators=300, random_state=42))])
_ = model.predict(X.head(1))
"""

def fewshot_snippets(profile: dict) -> str:
    """Return a task-appropriate few-shot snippet based on the dataset profile."""
    t = (profile or {}).get("target_type", "classification").lower()
    return FEWSHOT_REGRESSION if "regress" in t else FEWSHOT_CLASSIFICATION

def _append_fewshot(profile: dict, rules: List[str]) -> None:
    """Always add a compact few-shot snippet based on the current task."""
    try:
        snippet = fewshot_snippets(profile)
        if snippet and snippet.strip():
            rules.append(snippet.strip())
    except Exception:
        # Non-fatal: we still return whatever we have
        pass

def retrieve_rules(
    profile: dict,
    chroma_client: Optional[object] = None,
    max_distance: float = 0.30,
    k: int = 5,
) -> Tuple[List[str], bool]:
    """
    Simplified retrieval for Streamlit deployment.
    Skips vector database and uses only few-shot examples.
    
    Returns
    -------
    (rules, fallback_mode)
      rules         : list of rule strings (few-shot snippet is always included)
      fallback_mode : True (always in fallback mode without vector database)
    """
    
    # Skip vector search entirely for Streamlit deployment
    retrieved_rules: List[str] = []
    fallback_mode = True
    
    # Add basic ML guidelines as text rules
    basic_rules = [
        "Use appropriate preprocessing for numeric and categorical features",
        "Apply proper train/test splitting with stratification for classification",
        "Choose suitable algorithms based on dataset size and task type",
        "Ensure robust pipeline with proper error handling",
        "Include model validation and performance evaluation"
    ]
    
    retrieved_rules.extend(basic_rules)
    
    # Always append a compact few-shot pattern grounded in the current task
    _append_fewshot(profile, retrieved_rules)
    
    return retrieved_rules, fallback_mode