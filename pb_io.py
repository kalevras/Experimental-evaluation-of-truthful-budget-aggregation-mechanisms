"""Input loaders and input-validation helpers for budget allocation profiles."""

from __future__ import annotations
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

SUPPORTED = {'.csv', '.xlsx', '.json', '.xml'}


def _as_numeric_matrix(obj, source='input'):
    """Convert raw input to a finite two-dimensional numerical matrix."""
    try:
        P = np.asarray(obj, dtype=float)
    except (TypeError, ValueError) as e:
        raise ValueError(f'{source}: data must be numeric.') from e
    if P.ndim != 2 or P.shape[0] == 0 or P.shape[1] == 0:
        raise ValueError(f'{source}: expected a non-empty 2D voter-by-project matrix.')
    return P


def _load_csv(path: Path):
    # sep=None accepts comma, semicolon, tab, etc.; files are expected without headers.
    """Load one preference matrix from a CSV file."""
    return pd.read_csv(path, header=None, sep=None, engine='python').to_numpy(float)


def _load_excel(path: Path, sheet=0):
    """Read preference data from the selected Excel worksheet."""
    return pd.read_excel(path, header=None, sheet_name=sheet).to_numpy(float)


def _load_json(path: Path):
    """Parse budget allocations stored as JSON."""
    with path.open('r', encoding='utf-8-sig') as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ('profile', 'preferences', 'votes', 'voters', 'data'):
            if key in data:
                data = data[key]
                break
        else:
            raise ValueError("JSON object must contain one of: profile, preferences, votes, voters, data.")
    # Also accept [{"preferences":[...]}, ...]
    if isinstance(data, list) and data and isinstance(data[0], dict):
        rows = []
        for voter in data:
            found = None
            for key in ('preferences', 'profile', 'votes', 'values'):
                if key in voter:
                    found = voter[key]; break
            if found is None:
                raise ValueError('JSON voter objects must contain preferences/profile/votes/values.')
            rows.append(found)
        data = rows
    return _as_numeric_matrix(data, 'JSON')


def _numbers_from_text(text):
    """Extract numerical values from supported text representations."""
    if text is None:
        return []
    clean = text.replace(';', ' ').replace(',', ' ')
    return [float(x) for x in clean.split()]


def _load_xml(path: Path):
    """Load preference allocations from the supported XML schema."""
    root = ET.parse(path).getroot()
    # Preferred forms:
    # <profile><voter><preference>0.2</preference>...</voter>...</profile>
    # <profile><voter>0.2 0.3 0.5</voter>...</profile>
    voters = root.findall('.//voter') or root.findall('.//row')
    rows = []
    for voter in voters:
        children = list(voter)
        if children:
            vals = []
            for child in children:
                raw = child.get('value') if child.get('value') is not None else child.text
                if raw is not None and str(raw).strip() != '':
                    vals.append(float(str(raw).strip()))
            if vals:
                rows.append(vals)
        else:
            vals = _numbers_from_text(voter.text)
            if vals:
                rows.append(vals)
    if not rows:
        # Fallback: root contains row-like children directly.
        for child in list(root):
            vals = _numbers_from_text(child.text)
            if vals:
                rows.append(vals)
    if not rows:
        raise ValueError('XML contains no readable voter/row preference matrix.')
    return _as_numeric_matrix(rows, 'XML')


def load_profile_file(path, sheet=0):
    """Load a voter-by-project preference matrix from CSV/XLSX/JSON/XML.

    This function parses only. Mathematical validation (non-negativity, row sums,
    dimensions) remains centralized in pb_core.validate_profile().
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Input file not found: {p}')
    ext = p.suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported input type '{ext}'. Supported: {sorted(SUPPORTED)}")
    if ext == '.csv':
        P = _load_csv(p)
    elif ext == '.xlsx':
        P = _load_excel(p, sheet=sheet)
    elif ext == '.json':
        P = _load_json(p)
    else:
        P = _load_xml(p)
    return _as_numeric_matrix(P, p.name)
