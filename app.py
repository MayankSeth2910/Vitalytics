"""
VITALYTICS – Multi-Organ Disease Prediction API
Handles: UTI (Kidney), Heart, Liver, COPD (Lung), Pancreatic
Google Sheets: 5 separate sheets in one spreadsheet
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle, numpy as np, pandas as pd, os, json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import gspread
from google.oauth2.service_account import Credentials

app = Flask(__name__, static_folder='static')
CORS(app)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SCOPES     = ['https://www.googleapis.com/auth/spreadsheets',
               'https://www.googleapis.com/auth/drive']

# ⬇️  Replace with YOUR Google Sheet ID (one sheet, 5 tabs)
SHEET_ID   = os.environ.get('SHEET_ID', '1c5Ha7R9POuKmjWzdrSeaQlDXaI5bSbmEdXuSjN3ZeyM')

# ─────────────────────────────────────────────────────────────────────────────
# MODELS REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
models = {
    'uti':        {'rf': None, 'lgbm_u': None, 'lgbm_t': None, 'scaler': None},
    'heart':      {'model': None, 'scaler': None},
    'liver':      {'model': None, 'scaler': None},
    'copd':       {'model': None, 'scaler': None},
    'pancreatic': {'model': None, 'encoder': None},
}

def _load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_models():
    d = MODELS_DIR
    try:
        m = models['uti']
        m['rf']     = _load(f'{d}/uti/random_forest_model.pkl')
        m['lgbm_u'] = _load(f'{d}/uti/lgbm_untuned_model.pkl')
        m['lgbm_t'] = _load(f'{d}/uti/lgbm_tuned_model.pkl')
        m['scaler'] = _load(f'{d}/uti/scaler.pkl')
        print("✅ UTI models loaded")
    except Exception as e:
        print(f"⚠️  UTI models: {e}")

    for key in ['heart', 'liver', 'copd']:
        try:
            models[key]['model']  = _load(f'{d}/{key}/model.pkl')
            models[key]['scaler'] = _load(f'{d}/{key}/scaler.pkl')
            print(f"✅ {key.capitalize()} model loaded")
        except Exception as e:
            print(f"⚠️  {key.capitalize()} model: {e}")

    try:
        models['pancreatic']['model']   = _load(f'{d}/pancreatic/model.pkl')
        models['pancreatic']['encoder'] = _load(f'{d}/pancreatic/encoder.pkl')
        print("✅ Pancreatic model loaded")
    except Exception as e:
        print(f"⚠️  Pancreatic model: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# GOOGLE SHEETS
# ─────────────────────────────────────────────────────────────────────────────
SHEET_HEADERS = {
    'UTI':        ['ID','Timestamp','Age','Gender','Color','Transparency','pH',
                   'Specific Gravity','WBC','RBC','Glucose','Protein',
                   'Epithelial Cells','Mucous Threads','Amorphous Urates',
                   'Bacteria','Prediction','Confidence','Feedback'],
    'Heart':      ['ID','Timestamp','Age','Sex','ChestPainType','RestingBP',
                   'Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina',
                   'Oldpeak','ST_Slope','Prediction','Confidence','Feedback'],
    'Liver':      ['ID','Timestamp','Age','Gender','TotalBilirubin',
                   'DirectBilirubin','AlkPhos','SGPT','SGOT','TotalProtiens',
                   'Albumin','AG_Ratio','Prediction','Confidence','Feedback'],
    'COPD':       ['ID','Timestamp','Age','Gender','PackHistory','MWT1Best',
                   'FEV1','FEV1PRED','FVC','FVCPRED','CAT','HAD','SGRQ',
                   'AGEquartiles','Smoking','Diabetes','Muscular','Hypertension',
                   'AtrialFib','IHD','Prediction','Confidence','Feedback'],
    'Pancreatic': ['ID','Timestamp','Country','Age','Gender','SmokingHistory',
                   'Obesity','Diabetes','ChronicPancreatitis','FamilyHistory',
                   'HereditaryCondition','Jaundice','AbdominalDiscomfort',
                   'BackPain','WeightLoss','StageAtDiagnosis','SurvivalTime',
                   'TreatmentType','SurvivalStatus','Prediction','Confidence','Feedback'],
}

worksheets = {}   # tab_name -> worksheet object

def init_sheets():
    global worksheets
    try:
        raw = os.environ.get('GOOGLE_CREDENTIALS')
        if not raw:
            print("⚠️  GOOGLE_CREDENTIALS env var not set – Sheets disabled")
            return
        creds_info = json.loads(raw)
        creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(SHEET_ID)

        existing = {ws.title for ws in sh.worksheets()}
        for tab, headers in SHEET_HEADERS.items():
            if tab not in existing:
                ws = sh.add_worksheet(title=tab, rows=1000, cols=len(headers))
                ws.append_row(headers)
            else:
                ws = sh.worksheet(tab)
                if not ws.row_values(1):
                    ws.append_row(headers)
            worksheets[tab] = ws
        print("✅ Google Sheets connected – 5 tabs ready")
    except Exception as e:
        print(f"⚠️  Sheets init error: {e}")

def save_row(tab: str, row: list):
    """Append a row; returns 1-based row index in sheet."""
    try:
        ws = worksheets.get(tab)
        if not ws:
            return None
        all_vals = ws.get_all_values()
        row_id = len(all_vals)         # ID = current row count (0-based after header)
        ws.append_row([row_id] + row)
        print(f"✅ New Entry Added in ({tab}) sheet")
        return row_id
    except Exception as e:
        print(f"⚠️  Sheets save error ({tab}): {e}")
        return None

def update_feedback(tab: str, row_id, feedback_val: str):
    try:
        ws = worksheets.get(tab)
        if not ws:
            return
        headers = SHEET_HEADERS[tab]
        fb_col  = headers.index('Feedback') + 1   # 1-based
        all_vals = ws.get_all_values()
        for i, row in enumerate(all_vals):
            if row and str(row[0]) == str(row_id):
                ws.update_cell(i + 1, fb_col, feedback_val)
                print(f"✅ Feedback Added in ({tab}) sheet")
                return
    except Exception as e:
        print(f"⚠️  Feedback update error ({tab}): {e}")

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING & PREDICTION – UTI
# ─────────────────────────────────────────────────────────────────────────────
TRANSPARENCY_MAP = {'CLEAR':0,'SLIGHTLY HAZY':1,'HAZY':2,'TURBID':3,'CLOUDY':4}
GLUCOSE_MAP      = {'NEGATIVE':0,'TRACE':1,'1+':2,'2+':3,'3+':4,'4+':5}
PROTEIN_MAP      = {'NEGATIVE':0,'TRACE':1,'1+':2,'2+':3,'3+':4}
EPITHELIAL_MAP   = {'NONE SEEN':0,'RARE':1,'OCCASIONAL':2,'FEW':3,'MODERATE':4,'PLENTY':5,'LOADED':6}
MUCOUS_MAP       = {'NONE SEEN':0,'RARE':1,'OCCASIONAL':2,'FEW':3,'MODERATE':4,'PLENTY':5}
AMORPHOUS_MAP    = {'NONE SEEN':0,'RARE':1,'OCCASIONAL':2,'FEW':3,'MODERATE':4,'PLENTY':5}
BACTERIA_MAP     = {'NONE SEEN':0,'RARE':1,'OCCASIONAL':2,'FEW':3,'MODERATE':4,'PLENTY':5,'LOADED':6}
GENDER_MAP       = {'FEMALE':0,'MALE':1}
COLOR_COLS = ['Color_AMBER','Color_BROWN','Color_DARK YELLOW','Color_LIGHT RED',
              'Color_LIGHT YELLOW','Color_RED','Color_REDDISH',
              'Color_REDDISH YELLOW','Color_STRAW','Color_YELLOW']
NUMERICAL_UTI = ['Age','pH','Specific Gravity','WBC','RBC',
                 'Transparency','Glucose','Protein','Epithelial Cells',
                 'Mucous Threads','Amorphous Urates','Bacteria']

def preprocess_uti(data: dict):
    print("✅ UTI Preprocess Starts")
    color = data['Color'].upper()
    row = {
        'Age': float(data['Age']),
        'Gender': GENDER_MAP[data['Gender'].upper()],
        'pH': float(data['pH']),
        'Specific Gravity': float(data['Specific Gravity']),
        'WBC': float(data['WBC']),
        'RBC': float(data['RBC']),
        'Transparency': TRANSPARENCY_MAP[data['Transparency'].upper()],
        'Glucose':        GLUCOSE_MAP[data['Glucose'].upper()],
        'Protein':        PROTEIN_MAP[data['Protein'].upper()],
        'Epithelial Cells': EPITHELIAL_MAP[data['Epithelial Cells'].upper()],
        'Mucous Threads':   MUCOUS_MAP[data['Mucous Threads'].upper()],
        'Amorphous Urates': AMORPHOUS_MAP[data['Amorphous Urates'].upper()],
        'Bacteria':         BACTERIA_MAP[data['Bacteria'].upper()],
    }
    for col in COLOR_COLS:
        row[col] = 1 if col == f'Color_{color}' else 0
    df = pd.DataFrame([row])
    df[NUMERICAL_UTI] = models['uti']['scaler'].transform(df[NUMERICAL_UTI])
    ordered = (['Age','Gender','Transparency','Glucose','Protein','pH',
                'Specific Gravity','WBC','RBC','Epithelial Cells',
                'Mucous Threads','Amorphous Urates','Bacteria'] + COLOR_COLS)
    print("✅ UTI Preprocess Ends")
    return df[ordered]

def predict_uti(df):
    print("✅ UTI Prediction Starts")
    m   = models['uti']
    p_rf    = m['rf'].predict(df)[0]
    p_lgbmu = m['lgbm_u'].predict(df)[0]
    p_lgbmt = m['lgbm_t'].predict(df)[0]
    votes   = [p_rf, p_lgbmu, p_lgbmt]
    pred    = 1 if sum(votes) >= 2 else 0
    try:
        conf = float(np.mean([
            m['rf'].predict_proba(df)[0][pred],
            m['lgbm_u'].predict_proba(df)[0][pred],
            m['lgbm_t'].predict_proba(df)[0][pred],
        ]))
    except:
        conf = None
    print("✅ UTI Prediction Ends")
    return pred, conf

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING & PREDICTION – HEART
# ─────────────────────────────────────────────────────────────────────────────
# Features (after one-hot): age, trestbps, chol, thalach, oldpeak,
#   sex_1, cp_1..cp_3, fbs_1, restecg_1..restecg_2, exang_1, slope_1..slope_2
HEART_CATEGORICAL = ['sex','cp','fbs','restecg','exang','slope']

def preprocess_heart(data: dict):
    print("✅ Heart Preprocess Starts")
    row = {
        'age':     float(data['age']),
        'sex':     int(data['sex']),         # 0/1
        'cp':      int(data['cp']),          # 0-3
        'trestbps': float(data['trestbps']),
        'chol':    float(data['chol']),
        'fbs':     int(data['fbs']),         # 0/1
        'restecg': int(data['restecg']),     # 0-2
        'thalach': float(data['thalach']),
        'exang':   int(data['exang']),       # 0/1
        'oldpeak': float(data['oldpeak']),
        'slope':   int(data['slope']),       # 0-2
    }
    df = pd.DataFrame([row])
    df = pd.get_dummies(df, columns=HEART_CATEGORICAL, drop_first=True)
    # Ensure all expected columns are present (fill missing with 0)
    expected = models['heart'].get('feature_names', df.columns.tolist())
    for col in expected:
        if col not in df.columns:
            df[col] = 0
    df = df.reindex(columns=expected, fill_value=0)
    num_cols = ['age','trestbps','chol','thalach','oldpeak']
    existing_num = [c for c in num_cols if c in df.columns]
    df[existing_num] = models['heart']['scaler'].transform(df[existing_num])
    print("✅ Heart Preprocess Ends")
    return df

def predict_single(key, df):
    print(f"✅ ({key}) Prediction Starts")
    m    = models[key]['model']
    pred = int(m.predict(df)[0])
    try:
        conf = float(m.predict_proba(df)[0][pred])
    except:
        conf = None
    print(f"✅ ({key}) Prediction Ends")
    return pred, conf

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING – LIVER
# ─────────────────────────────────────────────────────────────────────────────
LIVER_GENDER = {'Male': 1, 'Female': 0}
LIVER_FEATURES = ['Age of the patient','Gender of the patient','Total Bilirubin',
                  'Direct Bilirubin',' Alkphos Alkaline Phosphotase',
                  ' Sgpt Alamine Aminotransferase','Sgot Aspartate Aminotransferase',
                  'Total Protiens',' ALB Albumin','A/G Ratio Albumin and Globulin Ratio']

def preprocess_liver(data: dict):
    print("✅ Liver Preprocess Starts")
    row = {
        'Age of the patient':                    float(data['age']),
        'Gender of the patient':                 LIVER_GENDER.get(data['gender'], 0),
        'Total Bilirubin':                       float(data['total_bilirubin']),
        'Direct Bilirubin':                      float(data['direct_bilirubin']),
        ' Alkphos Alkaline Phosphotase':         float(data['alkphos']),
        ' Sgpt Alamine Aminotransferase':        float(data['sgpt']),
        'Sgot Aspartate Aminotransferase':       float(data['sgot']),
        'Total Protiens':                        float(data['total_proteins']),
        ' ALB Albumin':                          float(data['albumin']),
        'A/G Ratio Albumin and Globulin Ratio':  float(data['ag_ratio']),
    }
    df = pd.DataFrame([row])
    num_cols = [c for c in df.columns if c != 'Gender of the patient']
    df[num_cols] = models['liver']['scaler'].transform(df[num_cols])
    print("✅ Liver Preprocess Ends")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING – COPD
# ─────────────────────────────────────────────────────────────────────────────
COPD_FEATURES = ['AGE','PackHistory','MWT1Best','FEV1','FEV1PRED','FVC','FVCPRED',
                 'CAT','HAD','SGRQ','AGEquartiles','gender','smoking','Diabetes',
                 'muscular','hypertension','AtrialFib','IHD']
COPD_SEVERITY = {1:'MILD', 2:'MODERATE', 3:'SEVERE', 4:'VERY SEVERE'}

def preprocess_copd(data: dict):
    print("✅ COPD Preprocess Starts")
    row = {
        'AGE':          float(data['age']),
        'PackHistory':  float(data['pack_history']),
        'MWT1Best':     float(data['mwt1best']),
        'FEV1':         float(data['fev1']),
        'FEV1PRED':     float(data['fev1pred']),
        'FVC':          float(data['fvc']),
        'FVCPRED':      float(data['fvcpred']),
        'CAT':          float(data['cat']),
        'HAD':          float(data['had']),
        'SGRQ':         float(data['sgrq']),
        'AGEquartiles': int(data['age_quartiles']),
        'gender':       int(data['gender']),
        'smoking':      int(data['smoking']),
        'Diabetes':     int(data['diabetes']),
        'muscular':     int(data['muscular']),
        'hypertension': int(data['hypertension']),
        'AtrialFib':    int(data['atrial_fib']),
        'IHD':          int(data['ihd']),
    }
    df = pd.DataFrame([row])
    num_cols = ['AGE','PackHistory','MWT1Best','FEV1','FEV1PRED',
                'FVC','FVCPRED','CAT','HAD','SGRQ']
    df[num_cols] = models['copd']['scaler'].transform(df[num_cols])
    print("✅ COPD Preprocess Ends")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING – PANCREATIC
# ─────────────────────────────────────────────────────────────────────────────
YES_NO = {'Yes': 1, 'No': 0}
STAGE_MAP = {'Stage I':1,'Stage II':2,'Stage III':3,'Stage IV':4}
TREATMENT_MAP = {'Surgery':0,'Chemotherapy':1,'Radiation':2}

def preprocess_pancreatic(data: dict):
    print("✅ Pancreatic Preprocess Starts")
    row = {
        'Age':                   float(data['age']),
        'Gender':                1 if data['gender'] == 'Male' else 0,
        'SmokingHistory':        YES_NO.get(data['smoking_history'], 0),
        'Obesity':               YES_NO.get(data['obesity'], 0),
        'Diabetes':              YES_NO.get(data['diabetes'], 0),
        'ChronicPancreatitis':   YES_NO.get(data['chronic_pancreatitis'], 0),
        'FamilyHistory':         YES_NO.get(data['family_history'], 0),
        'HereditaryCondition':   YES_NO.get(data['hereditary_condition'], 0),
        'Jaundice':              YES_NO.get(data['jaundice'], 0),
        'AbdominalDiscomfort':   YES_NO.get(data['abdominal_discomfort'], 0),
        'BackPain':              YES_NO.get(data['back_pain'], 0),
        'WeightLoss':            YES_NO.get(data['weight_loss'], 0),
        'StageAtDiagnosis':      STAGE_MAP.get(data['stage_at_diagnosis'], 1),
        'SurvivalTime':          float(data['survival_time']),
        'TreatmentType':         TREATMENT_MAP.get(data['treatment_type'], 0),
        'SurvivalStatus':        YES_NO.get(data['survival_status'], 0),
    }
    df = pd.DataFrame([row])
    # If encoder stored (e.g. country dummies), apply here
    enc = models['pancreatic']['encoder']
    if enc is not None:
        country_encoded = enc.transform([[data.get('country','Unknown')]])
        # Append encoded country columns
        country_df = pd.DataFrame(country_encoded, columns=enc.get_feature_names_out(['Country']))
        df = pd.concat([df.reset_index(drop=True), country_df], axis=1)
    print("✅ Pancreatic Preprocess Ends")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES – STATIC PAGES
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    print("✅ Index Static Page Loaded")
    return send_from_directory('static', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    print(f"✅ ({filename}) Static Page Loaded")
    return send_from_directory('static', filename)

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES – PREDICTION ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/predict/uti', methods=['POST'])
def predict_uti_route():
    print("✅ Predict UTI Route Started")
    try:
        data = request.json
        df   = preprocess_uti(data)
        pred, conf = predict_uti(df)
        label = 'POSITIVE' if pred == 1 else 'NEGATIVE'
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        row = [ts, data.get('Age'), data.get('Gender'), data.get('Color'),
               data.get('Transparency'), data.get('pH'), data.get('Specific Gravity'),
               data.get('WBC'), data.get('RBC'), data.get('Glucose'),
               data.get('Protein'), data.get('Epithelial Cells'),
               data.get('Mucous Threads'), data.get('Amorphous Urates'),
               data.get('Bacteria'), label,
               round(conf*100, 1) if conf else None, 'Not provided']
        row_id = save_row('UTI', row)
        return jsonify({'prediction': label, 'confidence': conf,
                        'confidence_pct': round(conf*100,1) if conf else None,
                        'id': row_id, 'tab': 'UTI'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict/heart', methods=['POST'])
def predict_heart_route():
    print("✅ Predict Heart Route Started")
    try:
        data = request.json
        df   = preprocess_heart(data)
        pred, conf = predict_single('heart', df)
        label = 'POSITIVE' if pred == 1 else 'NEGATIVE'
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        row = [ts, data.get('age'), data.get('sex'), data.get('cp'),
               data.get('trestbps'), data.get('chol'), data.get('fbs'),
               data.get('restecg'), data.get('thalach'), data.get('exang'),
               data.get('oldpeak'), data.get('slope'), label,
               round(conf*100,1) if conf else None, 'Not provided']
        row_id = save_row('Heart', row)
        return jsonify({'prediction': label, 'confidence': conf,
                        'confidence_pct': round(conf*100,1) if conf else None,
                        'id': row_id, 'tab': 'Heart'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict/liver', methods=['POST'])
def predict_liver_route():
    print("✅ Predict Liver Route Started")
    try:
        data = request.json
        df   = preprocess_liver(data)
        pred, conf = predict_single('liver', df)
        label = 'LIVER DISEASE' if pred == 1 else 'NO LIVER DISEASE'
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        row = [ts, data.get('age'), data.get('gender'), data.get('total_bilirubin'),
               data.get('direct_bilirubin'), data.get('alkphos'), data.get('sgpt'),
               data.get('sgot'), data.get('total_proteins'), data.get('albumin'),
               data.get('ag_ratio'), label,
               round(conf*100,1) if conf else None, 'Not provided']
        row_id = save_row('Liver', row)
        return jsonify({'prediction': label, 'confidence': conf,
                        'confidence_pct': round(conf*100,1) if conf else None,
                        'id': row_id, 'tab': 'Liver'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict/copd', methods=['POST'])
def predict_copd_route():
    print("✅ Predict COPD Route Started")
    try:
        data = request.json
        df   = preprocess_copd(data)
        m    = models['copd']['model']
        pred = int(m.predict(df)[0])
        try:
            conf = float(np.max(m.predict_proba(df)[0]))
        except:
            conf = None
        severity = COPD_SEVERITY.get(pred, f'Class {pred}')
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        row = [ts, data.get('age'), data.get('gender'), data.get('pack_history'),
               data.get('mwt1best'), data.get('fev1'), data.get('fev1pred'),
               data.get('fvc'), data.get('fvcpred'), data.get('cat'),
               data.get('had'), data.get('sgrq'), data.get('age_quartiles'),
               data.get('smoking'), data.get('diabetes'), data.get('muscular'),
               data.get('hypertension'), data.get('atrial_fib'), data.get('ihd'),
               severity, round(conf*100,1) if conf else None, 'Not provided']
        row_id = save_row('COPD', row)
        return jsonify({'prediction': severity, 'class': pred, 'confidence': conf,
                        'confidence_pct': round(conf*100,1) if conf else None,
                        'id': row_id, 'tab': 'COPD'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict/pancreatic', methods=['POST'])
def predict_pancreatic_route():
    print("✅ Predict Pancreatic Route Started")
    try:
        data = request.json
        df   = preprocess_pancreatic(data)
        pred, conf = predict_single('pancreatic', df)
        label = 'HIGH RISK' if pred == 1 else 'LOW RISK'
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        row = [ts, data.get('country'), data.get('age'), data.get('gender'),
               data.get('smoking_history'), data.get('obesity'), data.get('diabetes'),
               data.get('chronic_pancreatitis'), data.get('family_history'),
               data.get('hereditary_condition'), data.get('jaundice'),
               data.get('abdominal_discomfort'), data.get('back_pain'),
               data.get('weight_loss'), data.get('stage_at_diagnosis'),
               data.get('survival_time'), data.get('treatment_type'),
               data.get('survival_status'), label,
               round(conf*100,1) if conf else None, 'Not provided']
        row_id = save_row('Pancreatic', row)
        return jsonify({'prediction': label, 'confidence': conf,
                        'confidence_pct': round(conf*100,1) if conf else None,
                        'id': row_id, 'tab': 'Pancreatic'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE – FEEDBACK (shared, uses tab from frontend)
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/feedback', methods=['POST'])
def feedback():
    print("✅ Feedback Route Started")
    try:
        data = request.json
        row_id       = data.get('id')
        feedback_val = data.get('feedback')   # 'Yes' | 'No'
        tab          = data.get('tab')        # 'UTI' | 'Heart' | 'Liver' | 'COPD' | 'Pancreatic'
        update_feedback(tab, row_id, feedback_val)
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────
load_models()
init_sheets()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
