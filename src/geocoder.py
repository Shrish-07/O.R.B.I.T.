from typing import Optional, Dict
import urllib.request
import urllib.parse
import json
import re
from pathlib import Path


def _extract_bbl_from_props(props: dict) -> Optional[str]:
    """Robustly find a BBL value inside a feature's properties dict."""
    if not isinstance(props, dict):
        return None
    # Common nested path: properties.addendum.pad.bbl
    try:
        addendum = props.get('addendum', {})
        if isinstance(addendum, dict):
            pad = addendum.get('pad', {})
            if isinstance(pad, dict):
                # look for standard keys
                for key in ('bbl', 'BBL', 'bbl10'):
                    if key in pad and pad[key]:
                        return pad[key]
                # fallback: any key containing 'bbl'
                for k, v in pad.items():
                    if 'bbl' in k.lower() and v:
                        return v
    except Exception:
        pass

    # Generic recursive search for key named 'bbl'
    def _find_bbl(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.lower() == 'bbl' and v:
                    return v
                res = _find_bbl(v)
                if res:
                    return res
        elif isinstance(obj, list):
            for item in obj:
                res = _find_bbl(item)
                if res:
                    return res
        return None

    return _find_bbl(props)


def geocode_address(address: str) -> Optional[Dict]:
    """
    Call NYC GeoSearch API with the given address string.
    Return a dict with keys: bbl, latitude, longitude, label (display address), borough
    Return None if no result found or API fails.
    Handles timeouts (5s), HTTP errors, and empty results gracefully.
    Never raises — always return None on failure.
    """
    try:
        if not address or not address.strip():
            return None
        base = 'https://geosearch.planninglabs.nyc/v2/search'
        # GeoSearch expects the address in the `text` parameter
        params = {'text': address}
        url = base + '?' + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={
            'Accept': 'application/json',
            'User-Agent': 'O.R.B.I.T geocoder/1.0'
        })
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw = resp.read()
            try:
                data = json.loads(raw)
            except Exception:
                data = json.loads(raw.decode('utf-8', errors='replace'))
    except Exception:
        return None

    features = data.get('features') or []
    if not features:
        return None

    # Prefer a feature that contains a BBL; otherwise use first feature
    chosen = None
    for f in features:
        props = f.get('properties', {})
        b = _extract_bbl_from_props(props)
        if b:
            chosen = f
            break
    if chosen is None:
        chosen = features[0]

    props = chosen.get('properties', {}) or {}
    bbl = _extract_bbl_from_props(props)

    # geometry coordinates are [lon, lat]
    lon = lat = None
    geom = chosen.get('geometry') or {}
    coords = geom.get('coordinates') if isinstance(geom, dict) else None
    if coords and isinstance(coords, (list, tuple)) and len(coords) >= 2:
        try:
            lon, lat = coords[0], coords[1]
        except Exception:
            lon = lat = None

    label = props.get('label') or props.get('display_name') or props.get('name') or address

    # try to find borough from properties, else derive from BBL prefix
    borough = None
    try:
        addendum = props.get('addendum', {})
        if isinstance(addendum, dict):
            pad = addendum.get('pad', {})
            if isinstance(pad, dict):
                borough = pad.get('borough') or pad.get('boro')
    except Exception:
        borough = None

    if not borough:
        borough = props.get('borough') or props.get('borough_name')

    if not borough and bbl:
        try:
            bstr = str(bbl)
            m = re.match(r'\s*([1-5])', bstr)
            if m:
                bmap = {'1': 'Manhattan', '2': 'Bronx', '3': 'Brooklyn', '4': 'Queens', '5': 'Staten Island'}
                borough = bmap.get(m.group(1))
        except Exception:
            borough = None

    # normalize bbl to string without .0
    if bbl is not None:
        try:
            if isinstance(bbl, (int, float)):
                bbl = str(int(bbl))
            else:
                bbl = str(bbl)
            bbl = bbl.replace('.0', '')
        except Exception:
            bbl = str(bbl)

    return {
        'bbl': bbl,
        'latitude': lat,
        'longitude': lon,
        'label': label,
        'borough': borough,
    }


def lookup_property_by_bbl(bbl: str, canonical_df=None) -> Optional[Dict]:
    """
    Given a BBL string and the canonical dataframe, find the matching row.
    Return a dict of the row's values for the matching property.
    Return None if BBL not found.
    Tries both the test split and the full canonical dataset when available.
    BBL may be stored as int or float — normalize before comparing.
    """
    try:
        import pandas as pd
    except Exception:
        return None

    if bbl is None:
        return None
    bbl_norm = str(bbl).strip().replace('.0', '')

    candidates = []
    test_path = Path('data/splits/all_years_test.parquet')
    if test_path.exists():
        try:
            candidates.append(pd.read_parquet(test_path))
        except Exception:
            pass

    if canonical_df is not None and hasattr(canonical_df, 'columns'):
        candidates.append(canonical_df)
    else:
        canon_path = Path('data/canonical/modeling_dataset_canonical_v2.parquet')
        if canon_path.exists():
            try:
                candidates.append(pd.read_parquet(canon_path))
            except Exception:
                pass

    for df in candidates:
        # find plausible bbl columns
        cols = [c for c in df.columns if c.lower() == 'bbl' or 'bbl' in c.lower()]
        if not cols and 'BBL' in df.columns:
            cols = ['BBL']
        for col in cols:
            try:
                ser = df[col].astype(str).str.replace('.0', '', regex=False)
                mask = ser == bbl_norm
                if mask.any():
                    row = df.loc[mask].iloc[0]
                    return row.to_dict()
            except Exception:
                continue

    return None
