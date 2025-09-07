import re
import json
import math
import numpy as np
import pandas as pd
from typing import List, Optional, Dict

def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r'\s+', '_', c.strip()).lower() for c in df.columns]
    return df

def first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    for cand in candidates:
        for c in cols:
            if cand == c:
                return c
    # fuzzy contains
    for cand in candidates:
        for c in cols:
            if cand in c:
                return c
    return None

def to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors='coerce')

def safe_bool_from_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False})

def summarize_mapping(mapping: Dict[str, Optional[str]]) -> str:
    lines = ["Column mapping:"]
    for k,v in mapping.items():
        lines.append(f"  {k:30s} -> {v}")
    return "\n".join(lines)
