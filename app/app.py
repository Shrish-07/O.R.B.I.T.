import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from pathlib import Path
import sys
from datetime import datetime

st.set_page_config(page_title='O.R.B.I.T — Forecast Explorer', layout='wide')

# Inject a minimal dark theme CSS to keep UI consistent
st.markdown(
    """
    <style>
    .stApp { background-color: #0b1020; color: #e6eef8; }
    .block-container { background-color: #0b1020; }
    .css-18e3th9 { background-color: #0b1020; }
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT = Path(__file__).resolve().parents[1]
# Prefer loading the champion experiment (pipeline_v2 if present). Fallback to political model hardcode.
MODEL_PATH = ROOT / 'models' / 'lgbm_all_years_political.txt'
FEATURES_PATH = ROOT / 'models' / 'artifacts' / 'lgbm_all_years_political_features.json'
TEST_SPLIT = ROOT / 'data' / 'splits' / 'all_years_test.parquet'
SHAP_SUM = ROOT / 'models' / 'artifacts' / 'lgbm_all_years_political_shap_summary.json'

try:
    champ = json.loads((ROOT / 'experiments' / 'champion.json').read_text())
    reg = json.loads((ROOT / 'experiments' / 'registry.json').read_text())
    entry = next((e for e in reg if e.get('id') == champ.get('selected_experiment')), None)
    if entry:
        mp = entry.get('model_path')
        fp = entry.get('features_path')
        # prefer pipeline_v2 experiments if flagged; otherwise use champion entry
        if entry.get('pipeline_v2') and mp:
            mpath = Path(mp)
            fpath = Path(fp) if fp else None
            # resolve relative paths against project root
            if not mpath.is_absolute():
                mpath = ROOT / mpath
            if fpath is not None and not fpath.is_absolute():
                fpath = ROOT / fpath
            if mpath.exists():
                MODEL_PATH = mpath
            if fpath is not None and fpath.exists():
                FEATURES_PATH = fpath
            # adjust SHAP summary path to follow model stem
            SHAP_SUM = ROOT / 'models' / 'artifacts' / f"{MODEL_PATH.stem}_shap_summary.json"
except Exception:
    pass

st.title('O.R.B.I.T — Forecast Explorer')

st.markdown('Interactive single-property forecast explorer. Select a sample property from the test set, adjust ideology sliders, and observe counterfactual predictions.')


@st.cache_data
def load_model():
    model = lgb.Booster(model_file=str(MODEL_PATH))
    with open(FEATURES_PATH) as f:
        features = json.load(f)
    return model, features


@st.cache_data
def load_test():
    return pd.read_parquet(TEST_SPLIT)


model, features = load_model()
test = load_test()

# Authentication: require login before using pages
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = None


def login_widget():
    st.header('Login / Sign up')
    email = st.text_input('Email')
    pwd = st.text_input('Password', type='password')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Log in'):
            try:
                sys.path.append(str(ROOT))
                from src.auth import login_user

                uid = login_user(email, pwd)
                if uid:
                    st.session_state['user_id'] = uid
                    st.success('Logged in')
                else:
                    st.error('Login failed')
            except Exception:
                st.error('Auth error')
    with col2:
        if st.button('Sign up'):
            try:
                from src.auth import create_user

                ok = create_user(email, pwd)
                if ok:
                    st.success('User created — please log in')
                else:
                    st.error('User exists')
            except Exception:
                st.error('Signup error')


if not st.session_state.get('user_id'):
    login_widget()
    st.stop()


def home_page():
    st.header('Home — Dashboard')
    st.markdown('Summary of canonical dataset and champion model')
    # champion info
    try:
        champ = json.loads((ROOT / 'experiments' / 'champion.json').read_text())
        st.write('Champion experiment:', champ.get('selected_experiment'))
        st.write('Champion MAE:', champ.get('metric'))
    except Exception:
        st.write('No champion found')

    # Median price trend by borough (2003–Present)
    try:
        canon_path = ROOT / 'data' / 'canonical' / 'modeling_dataset_canonical_v2.parquet'
        if canon_path.exists():
            canon = pd.read_parquet(canon_path)
            # detect price and year columns
            price_col = next((c for c in canon.columns if 'log' in c.lower() and 'price' in c.lower()), None)
            if price_col is None:
                price_col = next((c for c in canon.columns if 'price' in c.lower()), None)
            year_col = next((c for c in canon.columns if 'year' in c.lower()), None)
            if price_col and year_col and 'BOROUGH' in canon.columns:
                canon['sale_year'] = canon[year_col].astype(int)
                med = canon.groupby(['sale_year', 'BOROUGH'])[price_col].median().reset_index()
                import plotly.express as px
                fig = px.line(med, x='sale_year', y=price_col, color='BOROUGH', title='Median Price Trend by Borough (2003–Present)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write('Canonical dataset missing year/price/borough columns for dashboard chart')
    except Exception:
        st.write('Unable to load canonical dataset for dashboard')


def property_page():
    st.header('Individual Property Analysis')
    st.markdown('Select a property and explore predictions, SHAP, and counterfactuals.')
    if 'BBL' not in test.columns:
        st.error('Test split missing BBL column; explorer requires BBL to select samples.')
        return
    bbls = test['BBL'].astype(str).tolist()
    sel = st.selectbox('Pick a sample property (BBL)', options=bbls)
    row = test[test['BBL'].astype(str) == str(sel)].iloc[0]

    st.subheader('Selected property')
    left, right = st.columns([2, 1])
    with left:
        st.write(row[features].to_frame().T)
    with right:
        if 'target_log_price' in row.index:
            st.metric('Observed log price', float(row['target_log_price']))

    # Ideology sliders
    ideo_cols = [c for c in features if 'ideology' in c.lower() or c in ('dem_share', 'rep_share', 'turnout')]
    st.subheader('Political scenario (neutral modeling layer)')
    ideo_vals = {}
    for c in ideo_cols:
        v = float(row.get(c, 0.0))
        ideo_vals[c] = st.slider(f'{c}', min_value=0.0, max_value=1.0, value=float(v), step=0.01)

    st.write('---')
    st.subheader('Prediction')
    X = row[features].copy()
    for k, v in ideo_vals.items():
        if k in X.index:
            X.loc[k] = v
    for f in features:
        try:
            X.loc[f] = pd.to_numeric(X.loc[f], errors='coerce')
        except Exception:
            pass
    X = X.fillna(0)
    X_df = X.to_frame().T
    # apply serialized preprocessor if available
    try:
        from joblib import load
        preproc_path = ROOT / 'models' / 'artifacts' / f"{MODEL_PATH.stem}_preproc.joblib"
        if preproc_path.exists():
            preproc = load(preproc_path)
            Xp = preproc.transform(X_df)
            pred_log = float(model.predict(Xp)[0])
        else:
            pred_log = float(model.predict(X_df)[0])
    except Exception:
        pred_log = float(model.predict(X_df)[0])
    pred_price = float(np.exp(pred_log)) if pred_log is not None else None
    st.metric('Predicted log price', round(pred_log, 4))
    st.metric('Predicted price (exp)', f"${pred_price:,.0f}")
    # save prediction to user history
    try:
        from src.auth import get_user_data, save_user_data

        uid = st.session_state.get('user_id')
        if uid:
            data = get_user_data(uid) or {}
            data.setdefault('predictions', [])
            data['predictions'].append({'bbl': str(row['BBL']), 'pred_log': float(pred_log), 'pred_price': float(pred_price)})
            save_user_data(uid, data)
    except Exception:
        pass

    # Counterfactual panel
    st.write('---')
    st.subheader('Counterfactual Analysis')
    try:
        sys.path.append(str(ROOT))
        from src.counterfactual import run_counterfactual

        cf_available = True
    except Exception:
        cf_available = False
    if cf_available:
        cf_col1, cf_col2 = st.columns(2)
        with cf_col1:
            zoning_field = None
            for candidate in ['zoning', 'ZONING', 'zoning_field', 'zoning_code']:
                if candidate in row.index:
                    zoning_field = candidate
                    break
            if zoning_field is not None:
                new_zoning = st.text_input('Zoning mutation', value=str(row.get(zoning_field, '')))
            else:
                new_zoning = st.text_input('Zoning mutation', value='')
            new_far = st.number_input('FAR mutation (multiply factor)', value=1.0, step=0.1)
        with cf_col2:
            st.write('Ideology adjustments used above will also be applied.')
            if st.button('Run counterfactual'):
                muts = {}
                if zoning_field is not None and new_zoning != '':
                    muts[zoning_field] = new_zoning
                for candidate in ['FAR', 'far', 'gross_floor_area', 'FAR_RATIO']:
                    if candidate in row.index:
                        try:
                            muts[candidate] = float(row.get(candidate, 0)) * float(new_far)
                        except Exception:
                            pass
                muts.update(ideo_vals)
                res = run_counterfactual(row, muts)
                st.metric('Original log-price', round(res['original_log_price'], 4))
                st.metric('Mutated log-price', round(res['mutated_log_price'], 4))
                st.metric('Delta (log-price)', round(res['delta_log_price'], 4))
                if res.get('pct_price_change') is not None:
                    st.metric('Pct price change', f"{res['pct_price_change']*100:.2f}%")
                # save counterfactual to user history
                try:
                    from src.auth import get_user_data, save_user_data

                    uid = st.session_state.get('user_id')
                    if uid:
                        data = get_user_data(uid) or {}
                        data.setdefault('counterfactuals', [])
                        data['counterfactuals'].append({'bbl': str(row['BBL']), 'mutations': muts, 'result': res})
                        save_user_data(uid, data)
                except Exception:
                    pass
    # What-If analysis (single + batch)
        st.write('---')
        st.subheader('What-If Analysis')
        with st.expander('Configure What-If mutations', expanded=True):
            # ideology sliders (±0.30)
            ideo_candidates = [c for c in features if 'ideology' in c.lower() or c in ('ideology_score', 'dem_share', 'rep_share')]
            whatif_ideos = {}
            for c in ideo_candidates:
                curr = float(row.get(c, 0.0))
                minv = max(0.0, curr - 0.30)
                maxv = min(1.0, curr + 0.30)
                whatif_ideos[c] = st.slider(f'{c} (what-if)', min_value=minv, max_value=maxv, value=curr, step=0.01)

            # zonedist1 selector
            zone_candidates = [c for c in row.index if 'zonedist1' in c.lower() or 'zonedist' in c.lower()]
            zone_field = zone_candidates[0] if zone_candidates else None
            if zone_field:
                unique_z = sorted(test[zone_field].dropna().astype(str).unique().tolist())
                zone_choice = st.selectbox('Zoning (what-if)', options=['(no change)'] + unique_z, index=0)
            else:
                zone_choice = st.text_input('Zoning (what-if)', value='')

            # gross sqft (±50%)
            gross_candidates = [c for c in features if 'gross' in c.lower() or 'sqft' in c.lower() or 'gross_floor' in c.lower()]
            gross_field = gross_candidates[0] if gross_candidates else None
            if gross_field:
                curr = float(row.get(gross_field, 0.0))
                minv = max(0.0, curr * 0.5)
                maxv = curr * 1.5 if curr > 0 else curr + 1000
                step = max(1.0, (maxv - minv) / 100.0)
                gross_new = st.slider(f'{gross_field} (what-if)', min_value=minv, max_value=maxv, value=curr, step=step)
            else:
                gross_new = st.number_input('Gross sqft (what-if)', value=float(row.get('GROSS_SQFT', 0.0)))

            # numfloors (±10%)
            floor_candidates = [c for c in features if 'floor' in c.lower() or 'numfloors' in c.lower()]
            floor_field = floor_candidates[0] if floor_candidates else None
            if floor_field:
                curr = float(row.get(floor_field, 1.0))
                minv = max(0.0, curr * 0.9)
                maxv = curr * 1.1
                floors_new = st.slider(f'{floor_field} (what-if)', min_value=minv, max_value=maxv, value=curr, step=1.0)
            else:
                floors_new = st.number_input('Num floors (what-if)', value=float(row.get('numfloors', 1.0)))

        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button('Apply What-If to this property'):
                muts = {}
                muts.update(whatif_ideos)
                if zone_field:
                    if zone_choice and zone_choice != '(no change)':
                        muts[zone_field] = zone_choice
                else:
                    if zone_choice:
                        muts['zonedist1'] = zone_choice
                if gross_field:
                    muts[gross_field] = float(gross_new)
                if floor_field:
                    muts[floor_field] = float(floors_new)
                res2 = run_counterfactual(row, muts)
                st.metric('Original log-price', round(res2['original_log_price'], 4))
                st.metric('Mutated log-price', round(res2['mutated_log_price'], 4))
                st.metric('Delta (log-price)', round(res2['delta_log_price'], 4))
                if res2.get('pct_price_change') is not None:
                    st.metric('Pct price change', f"{res2['pct_price_change']*100:.2f}%")
                try:
                    from src.auth import get_user_data, save_user_data
                    uid = st.session_state.get('user_id')
                    if uid:
                        data = get_user_data(uid) or {}
                        data.setdefault('counterfactuals', [])
                        data['counterfactuals'].append({'bbl': str(row['BBL']), 'mutations': muts, 'result': res2})
                        save_user_data(uid, data)
                except Exception:
                    pass

        with action_col2:
            st.markdown('Batch mode: upload a CSV with `BBL` column to apply the same mutations to multiple properties')
            uploaded = st.file_uploader('Upload CSV (BBL)', type=['csv'])
            if uploaded is not None:
                try:
                    port = pd.read_csv(uploaded)
                except Exception:
                    st.error('Unable to read uploaded CSV')
                    port = None
                if port is not None:
                    if 'BBL' not in port.columns:
                        st.error('Uploaded CSV missing `BBL` column')
                    else:
                        bbllist = port['BBL'].astype(str).tolist()
                        df_port = test[test['BBL'].astype(str).isin(bbllist)].copy()
                        if df_port.empty:
                            st.warning('No matching BBLs found in test set')
                        else:
                            muts = {}
                            muts.update(whatif_ideos)
                            if zone_field and zone_choice and zone_choice != '(no change)':
                                muts[zone_field] = zone_choice
                            if gross_field:
                                muts[gross_field] = float(gross_new)
                            if floor_field:
                                muts[floor_field] = float(floors_new)
                            try:
                                from src.counterfactual import run_counterfactual_batch
                                res_df = run_counterfactual_batch(df_port, muts)
                                out_df = pd.concat([df_port.reset_index(drop=True), res_df.reset_index(drop=True)], axis=1)
                                csv_out = out_df.to_csv(index=False)
                                st.download_button('Download batch counterfactuals', csv_out, file_name='batch_counterfactuals.csv')
                            except Exception:
                                st.error('Error running batch counterfactuals')
    else:
        st.info('Counterfactual engine not available')


def portfolio_page():
    st.header('Portfolio Analysis')
    st.markdown('Upload a portfolio CSV with `BBL` to run batch predictions and scenario comparisons.')
    uploaded = st.file_uploader('Upload portfolio CSV with column `BBL`', type=['csv'])
    if uploaded is not None:
        try:
            port = pd.read_csv(uploaded)
        except Exception:
            st.error('Unable to read uploaded CSV')
            port = None
        if port is not None:
            if 'BBL' not in port.columns:
                st.error('Uploaded CSV missing `BBL` column')
            else:
                bbllist = port['BBL'].astype(str).tolist()
                df_port = test[test['BBL'].astype(str).isin(bbllist)].copy()
                if df_port.empty:
                    st.warning('No matching BBLs found in test set')
                else:
                    Xp_df = df_port[features].fillna(0)
                    try:
                        from joblib import load

                        preproc_path = ROOT / 'models' / 'artifacts' / f"{MODEL_PATH.stem}_preproc.joblib"
                        if preproc_path.exists():
                            preproc = load(preproc_path)
                            Xp = preproc.transform(Xp_df)
                        else:
                            Xp = Xp_df
                    except Exception:
                        Xp = Xp_df
                    preds = model.predict(Xp)
                    df_port['pred_log_price'] = preds
                    df_port['pred_price'] = np.exp(preds)

                    st.write('Portfolio predictions (sample)')
                    st.dataframe(df_port[['BBL', 'pred_price']].head(50))
                    csv_out = df_port.to_csv(index=False)
                    st.download_button('Download portfolio predictions', csv_out, file_name='portfolio_preds.csv')

                    # Scenario comparison block
                    st.subheader('Political Scenario Impact on Portfolio')
                    scen_dir2 = ROOT / 'results' / 'political_scenarios'
                    scen_results2 = {}
                    for scen2 in ['liberal_policy', 'conservative_policy', 'mixed_governance']:
                        f2 = scen_dir2 / f'{scen2}_by_council.csv'
                        if f2.exists():
                            sc2 = pd.read_csv(f2)
                            ccol2 = next((c for c in sc2.columns if 'coun' in c.lower()), None)
                            vcol2 = next((c for c in sc2.columns if 'pct' in c.lower()), None)
                            if ccol2 and vcol2:
                                scen_results2[scen2] = sc2.set_index(ccol2)[vcol2].to_dict()
                    if scen_results2 and 'Council District' in df_port.columns:
                        for scen2, lkup2 in scen_results2.items():
                            df_port[f'{scen2}_pct_impact'] = df_port['Council District'].map(lkup2)
                        impact_cols2 = ['BBL'] + [f'{s}_pct_impact' for s in scen_results2.keys()]
                        st.dataframe(df_port[impact_cols2].head(50))
                        st.download_button('Download with scenario impacts', df_port.to_csv(index=False), file_name='portfolio_scenario_impacts.csv')


def scenarios_page():
    st.header('Political Scenarios')
    st.markdown('Run pre-defined political scenarios and visualize impacts by council district.')
    scen_dir = ROOT / 'results' / 'political_scenarios'
    scenarios = ['liberal_policy', 'conservative_policy', 'mixed_governance']
    sel_label = st.radio('Select scenario', options=[s.replace('_', ' ').title() for s in scenarios], index=0)
    s_key = sel_label.lower().replace(' ', '_')

    # prefer by-council aggregate CSV if present
    by_file = scen_dir / f"{s_key}_by_council.csv"
    prop_file = scen_dir / f"{s_key}_all_properties.parquet"
    if by_file.exists():
        df_by = pd.read_csv(by_file)
    elif prop_file.exists():
        df_prop = pd.read_parquet(prop_file)
        council_col = next((c for c in df_prop.columns if 'council' in c.lower()), None)
        if council_col:
            agg = df_prop.groupby(council_col).agg({'pct_change': 'mean', 'target_log_price': 'mean'}).reset_index()
            agg.columns = [str(council_col), 'mean_pct_change', 'mean_log_price']
            df_by = agg.rename(columns={str(council_col): 'Council District'})
        else:
            st.warning('Scenario data lacks council district column; showing table preview')
            st.dataframe(df_prop.head(50))
            return
    else:
        st.warning('No scenario data found for selected option')
        return

    # Normalize council id column
    # Accept common column names
    col_candidates = [c for c in df_by.columns if 'coun' in c.lower() and 'dist' in c.lower()]
    if col_candidates:
        cid_col = col_candidates[0]
        df_by['CounDist'] = df_by[cid_col].astype(float).astype(int)
    elif 'Council District' in df_by.columns:
        df_by['CounDist'] = df_by['Council District'].astype(float).astype(int)
    else:
        # fallback to first column
        df_by['CounDist'] = df_by.iloc[:, 0].astype(float).astype(int)

    # Determine value column to color by
    val_col = None
    for candidate in ['mean_delta', 'mean_pct_change', 'mean_pct_change', 'pct_change', 'mean_delta', 'mean_log_price', 'mean_pct_change']:
        if candidate in df_by.columns:
            val_col = candidate
            break
    if val_col is None:
        # pick first numeric column that's not the id
        for c in df_by.columns:
            if c == 'CounDist':
                continue
            if pd.api.types.is_numeric_dtype(df_by[c]):
                val_col = c
                break
    if val_col is None:
        st.warning('No numeric value found to color map')
        st.dataframe(df_by.head(50))
        return

    # Prepare GeoJSON for council districts (convert from shapefile if needed)
    shp_dir = Path('data') / 'raw' / 'election_districts'
    shp_path = shp_dir / 'NYC_City_Council_Districts.shp'
    geojson_path = shp_dir / 'NYC_City_Council_Districts.geojson'
    if not geojson_path.exists():
        try:
            import geopandas as gpd
            gdf = gpd.read_file(str(shp_path))
            # write to geojson for faster reuse
            gdf.to_file(str(geojson_path), driver='GeoJSON')
        except Exception as e:
            st.warning('Unable to read/convert shapefile for council districts: ' + str(e))
            st.dataframe(df_by.head(50))
            return

    # Load geojson and enrich features with mean values
    import json
    import folium

    gj = json.loads(geojson_path.read_text())
    # find district property name in geojson properties (case-insensitive)
    sample_props = gj['features'][0].get('properties', {}) if gj.get('features') else {}
    district_prop = None
    for k in sample_props.keys():
        if 'coun' in k.lower() and 'dist' in k.lower():
            district_prop = k
            break
    if district_prop is None:
        # try other heuristics
        for k in sample_props.keys():
            if 'dist' in k.lower():
                district_prop = k
                break
    if district_prop is None:
        st.warning('Unable to determine district id property in geojson')
        st.dataframe(df_by.head(50))
        return

    # create lookup
    lookup = df_by.set_index('CounDist')[val_col].to_dict()
    # attach values to geojson features
    for feat in gj['features']:
        props = feat.setdefault('properties', {})
        raw_id = props.get(district_prop)
        try:
            cid = int(float(raw_id))
        except Exception:
            try:
                cid = int(str(raw_id))
            except Exception:
                cid = None
        props['CounDist'] = cid
        props['mean_value'] = lookup.get(cid)

    # build folium map
    center_lat, center_lon = 40.7128, -74.0060
    try:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles='CartoDB positron')
        folium.Choropleth(
            geo_data=gj,
            name='choropleth',
            data=df_by,
            columns=['CounDist', val_col],
            key_on='feature.properties.CounDist',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=f'Mean {val_col}'
        ).add_to(m)

        # add tooltips
        tooltip_fields = [district_prop, 'mean_value']
        tooltip_aliases = ['District', val_col.replace('_', ' ').title()]
        folium.GeoJson(
            gj,
            style_function=lambda feat: {'fillColor': 'transparent', 'color': 'black', 'weight': 0.5},
            tooltip=folium.features.GeoJsonTooltip(fields=['CounDist', 'mean_value'], aliases=['District', val_col.replace('_', ' ').title()], localize=True)
        ).add_to(m)

        # render
        html = m.get_root().render()
        st.components.v1.html(html, height=650)
    except Exception as e:
        st.warning('Error rendering folium map: ' + str(e))
        st.dataframe(df_by.head(50))


def model_explorer_page():
    st.header('Model Explorer')
    st.markdown('Experiment registry and model inspection')
    try:
        registry = json.loads((ROOT / 'experiments' / 'registry.json').read_text())
        df = pd.json_normalize(registry)
        st.dataframe(df[['id', 'name', 'mode', 'scope', 'results.r2', 'results.mae']].fillna(''))
    except Exception:
        st.write('No registry available')

    # model error spatial heatmap by borough
    st.write('---')
    st.subheader('Champion Error Heatmap by Borough')
    try:
        champ = json.loads((ROOT / 'experiments' / 'champion.json').read_text())
        reg = json.loads((ROOT / 'experiments' / 'registry.json').read_text())
        exp = next((e for e in reg if e.get('id') == champ.get('selected_experiment')), None)
        if exp:
            model_path = Path(exp.get('model_path'))
            feat_path = Path(exp.get('features_path'))
            feats = json.loads(feat_path.read_text())
            # load processed dataset
            dfn = pd.read_parquet(ROOT / 'data' / 'canonical' / 'modeling_dataset_canonical_v2.parquet')
            # compute predictions
            booster = lgb.Booster(model_file=str(model_path))
            dfn['pred'] = booster.predict(dfn[feats])
            if 'target_log_price' in dfn.columns and 'BOROUGH' in dfn.columns:
                dfn['abs_err'] = (dfn['pred'] - dfn['target_log_price']).abs()
                agg = dfn.groupby('BOROUGH').agg({'abs_err': 'mean', 'latitude': 'mean', 'longitude': 'mean'}).reset_index()
                try:
                    import pydeck as pdk
                    view = pdk.ViewState(latitude=agg['latitude'].mean(), longitude=agg['longitude'].mean(), zoom=10)
                    layer = pdk.Layer('HeatmapLayer', data=agg, get_position=['longitude', 'latitude'], get_weight='abs_err')
                    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view))
                except Exception:
                    st.dataframe(agg)
    except Exception:
        st.write('Unable to compute heatmap')
    # Borough MAE bar chart (sorted) — run unconditionally after attempting heatmap
    st.write('---')
    st.subheader('Borough MAE (Champion)')
    try:
        champ = json.loads((ROOT / 'experiments' / 'champion.json').read_text())
        reg = json.loads((ROOT / 'experiments' / 'registry.json').read_text())
        exp = next((e for e in reg if e.get('id') == champ.get('selected_experiment')), None)
        if exp:
            exp_name = exp.get('name')
            preds_dir = Path('experiments/predictions')
            preds_file = preds_dir / f"{exp_name}_test_preds.parquet"
            if not preds_file.exists():
                # try to find a matching file
                matches = list(preds_dir.glob(f"{exp_name}*test*.parquet"))
                if matches:
                    preds_file = matches[0]
            if not preds_file.exists():
                st.write('No test predictions file found for champion')
            else:
                try:
                    preds = pd.read_parquet(preds_file)
                    # find prediction and target columns
                    pred_col = next((c for c in preds.columns if 'pred' in c.lower()), None)
                    target_col = next((c for c in preds.columns if 'target' in c.lower() or ('log' in c.lower() and 'price' in c.lower())), None)
                    # ensure BOROUGH present, else join with test split
                    if 'BOROUGH' not in preds.columns:
                        test_df = pd.read_parquet(Path('data/splits/all_years_test.parquet'))
                        # try join by index
                        try:
                            joined = test_df[['BOROUGH', 'target_log_price']].join(preds, how='left')
                        except Exception:
                            # try merge on BBL
                            if 'BBL' in preds.columns and 'BBL' in test_df.columns:
                                joined = test_df.merge(preds, on='BBL', how='left')
                            else:
                                joined = preds
                    else:
                        joined = preds
                    if pred_col is None or target_col is None:
                        st.write('Prediction or target column not found in preds file')
                    else:
                        joined['abs_err'] = (joined[pred_col] - joined[target_col]).abs()
                        if 'BOROUGH' in joined.columns:
                            borough_mae = joined.groupby('BOROUGH')['abs_err'].mean().reset_index().sort_values('abs_err')
                            import plotly.express as px
                            fig = px.bar(borough_mae, x='BOROUGH', y='abs_err', title='Mean Absolute Error by Borough', labels={'abs_err': 'MAE'})
                            st.plotly_chart(fig, use_container_width=True)
                            # save borough MAE
                            outp = {row['BOROUGH']: float(row['abs_err']) for _, row in borough_mae.iterrows()}
                            Path('models/artifacts').mkdir(parents=True, exist_ok=True)
                            (Path('models/artifacts') / 'borough_mae.json').write_text(json.dumps(outp, indent=2))
                        else:
                            st.write('BOROUGH not available to compute MAE by borough')
                except Exception as e:
                    st.write('Error loading predictions:', e)
        else:
            st.write('No champion experiment available')
    except Exception:
        st.write('Unable to compute borough MAE')

    # time-evolution line chart of median predicted price by borough
    st.write('---')
    st.subheader('Time-evolution: Median predicted price by Borough')
    # Ensure booster and features are available for time-evolution plotting
    try:
        champ = json.loads((ROOT / 'experiments' / 'champion.json').read_text())
        reg = json.loads((ROOT / 'experiments' / 'registry.json').read_text())
        exp = next((e for e in reg if e.get('id') == champ.get('selected_experiment')), None)
        if exp:
            model_path = Path(exp.get('model_path'))
            feat_path = Path(exp.get('features_path'))
            feats = json.loads(feat_path.read_text())
            booster = lgb.Booster(model_file=str(model_path))
        else:
            booster = None
            feats = None
    except Exception:
        booster = None
        feats = None

    try:
        dfn = pd.read_parquet(ROOT / 'data' / 'canonical' / 'modeling_dataset_canonical_v2.parquet')
        year_col = next((c for c in dfn.columns if 'year' in c.lower()), None)
        if year_col and 'BOROUGH' in dfn.columns and booster is not None and feats is not None:
            dfn['pred'] = booster.predict(dfn[feats])
            dfn['pred_price'] = np.exp(dfn['pred'])
            evo = dfn.groupby([year_col, 'BOROUGH'])['pred_price'].median().reset_index()
            import altair as alt
            chart = alt.Chart(evo).mark_line().encode(x=year_col, y='pred_price', color='BOROUGH')
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write('No year or borough columns available for time-evolution')
    except Exception:
        st.write('Unable to generate time-evolution chart')


def research_page():
    st.header('Research Outputs')
    st.markdown('Publication-quality figures, model summaries, and downloadable artifacts.')
    summary_path = Path('docs/research_summary.md')
    if summary_path.exists():
        st.markdown(summary_path.read_text())
    st.subheader('Publication Figures')
    figs_dir = Path('docs/figures')
    if figs_dir.exists():
        for f in sorted(figs_dir.glob('*.png')):
            st.image(str(f), caption=f.stem.replace('_', ' ').title(), use_column_width=True)
    paper_path = Path('paper/orbit_paper.tex')
    if paper_path.exists():
        st.download_button('Download Research Paper (LaTeX)', paper_path.read_bytes(), file_name='orbit_paper.tex')
    st.subheader('Clean Model Comparison (Pipeline v2)')
    try:
        reg2 = json.loads((ROOT / 'experiments' / 'registry.json').read_text())
        clean = [e for e in reg2 if not e.get('tainted') and not e.get('stale') and e.get('pipeline_v2')]
        rows = []
        for e in clean:
            mae = (e.get('metrics') or {}).get('mae') or (e.get('metrics') or {}).get('mae_log_price') or (e.get('results') or {}).get('mae')
            r2 = (e.get('metrics') or {}).get('r2_temporal') or (e.get('results') or {}).get('r2')
            rows.append({'Model': e.get('name'), 'MAE': round(float(mae),4) if mae else None, 'R2': round(float(r2),4) if r2 else None})
        st.dataframe(pd.DataFrame(rows).sort_values('MAE'))
    except Exception as ex:
        st.write('Could not load model table:', ex)


def user_dashboard_page():
    st.header('User Dashboard')
    st.markdown('Your saved predictions, portfolios, and scenarios')
    try:
        from src.auth import get_user_data

        uid = st.session_state.get('user_id')
        data = get_user_data(uid) or {}
        st.subheader('Prediction History')
        preds = data.get('predictions', [])
        if preds:
            st.dataframe(pd.DataFrame(preds).tail(50))
        else:
            st.write('No predictions yet')
        st.subheader('Counterfactual History')
        cfs = data.get('counterfactuals', [])
        if cfs:
            st.dataframe(pd.json_normalize(cfs).tail(50))
        else:
            st.write('No counterfactuals yet')
    except Exception:
        st.write('User data not available')


page = st.sidebar.selectbox('Page', ['Home', 'Individual Property Analysis', 'Portfolio Analysis', 'Political Scenarios', 'Model Explorer', 'Research Outputs', 'User Dashboard'])

if page == 'Home':
    home_page()
elif page == 'Individual Property Analysis':
    property_page()
elif page == 'Portfolio Analysis':
    portfolio_page()
elif page == 'Political Scenarios':
    scenarios_page()
elif page == 'Model Explorer':
    model_explorer_page()
elif page == 'Research Outputs':
    research_page()
elif page == 'User Dashboard':
    user_dashboard_page()
                
