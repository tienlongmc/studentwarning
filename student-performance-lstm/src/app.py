
# --- REFACTORED APP.PY ---
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from src.data_loader import load_csv, prepare_sequences_for_prediction
from src.utils import evaluate_binary, plot_roc, plot_precision_recall, plot_confusion, plot_training_history

st.set_page_config(page_title='Student Performance Predictor', layout='wide')
st.title('Student Performance Prediction - LSTM + Baselines')

# --- Model selection ---
models_dir = 'models'
available_models = []
exclude_files = {'encoders.joblib', 'scaler.joblib', 'metadata.joblib', 'metadata.json'}
if os.path.exists(models_dir):
    raw = [f for f in os.listdir(models_dir) if not f.startswith('.')]
    available_models = [
        f for f in raw
        if f.endswith(('.keras', '.h5', '.hdf5', '.joblib')) and f not in exclude_files
    ]
if not available_models:
    st.warning('No models found in models/. Please train or add model files.')
model_choice = st.selectbox('Choose model', available_models) if available_models else None
model_path = os.path.join(models_dir, model_choice) if model_choice else None
scaler_path = os.path.join(models_dir, 'scaler.joblib')
encoders_path = os.path.join(models_dir, 'encoders.joblib')

# --- Metadata loading ---
metadata = None
meta_path_joblib = os.path.join(models_dir, 'metadata.joblib')
meta_path_json = os.path.join(models_dir, 'metadata.json')
if os.path.exists(meta_path_joblib):
    try:
        metadata = joblib.load(meta_path_joblib)
    except Exception:
        metadata = None
elif os.path.exists(meta_path_json):
    try:
        metadata = pd.read_json(meta_path_json).to_dict()
    except Exception:
        metadata = None

# --- CSV column detection ---
detected_csv_columns = None
csv_found = None
try:
    cwd = os.getcwd()
    possible = [os.path.join(cwd, 'data'), os.path.join(cwd, '..', 'data'), os.path.join(cwd, '..', '..', 'data')]
    for d in possible:
        if os.path.exists(d) and os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith('.csv'):
                    csv_found = os.path.join(d, f)
                    break
        if csv_found:
            break
    if csv_found:
        detected_csv_columns = list(pd.read_csv(csv_found, nrows=0).columns)
except Exception:
    detected_csv_columns = None

st.markdown('---')
st.header('1. Upload CSV for batch prediction')
uploaded_file = st.file_uploader('Upload CSV', type=['csv'])

if uploaded_file is not None:
    df = load_csv(uploaded_file)
    st.write('Preview:')
    st.dataframe(df.head())

    # Optional long format settings
    student_id = st.text_input('Student ID column (optional)', value='')
    semester_col = st.text_input('Semester column (optional)', value='')
    feature_cols = st.text_input('Comma-separated per-semester feature columns when using long format (optional)', value='')
    label_col = st.text_input('Label column (optional)', value='Cảnh báo học tập (0/1)')

    if st.button('Predict'):
        try:
            if student_id and semester_col and feature_cols:
                fcols = [c.strip() for c in feature_cols.split(',')]
                X, scaler, encoders = prepare_sequences_for_prediction(df, scaler=joblib.load(scaler_path) if os.path.exists(scaler_path) else None, encoders=joblib.load(encoders_path) if os.path.exists(encoders_path) else None, metadata=metadata)
            else:
                X, scaler, encoders = prepare_sequences_for_prediction(df, scaler=joblib.load(scaler_path) if os.path.exists(scaler_path) else None, encoders=joblib.load(encoders_path) if os.path.exists(encoders_path) else None, metadata=metadata)
        except Exception as e:
            st.error(f'Failed to prepare input: {e}')
            st.stop()

        if not model_choice:
            st.warning('No model selected. Please train or move a model file into the models directory.')
            st.stop()
        if model_choice.endswith(('.keras', '.h5', '.hdf5')):
            model_type_local = 'lstm'
        elif model_choice.endswith('.joblib'):
            if 'rf' in model_choice.lower():
                model_type_local = 'rf'
            elif 'log' in model_choice.lower():
                model_type_local = 'logistic'
            else:
                model_type_local = 'logistic'
        else:
            model_type_local = 'auto'

        y_scores = None
        model = None
        if not os.path.exists(model_path):
            st.error(f'Model {model_path} not found. Please ensure models are trained and saved in the models directory.')
            st.stop()
        try:
            if model_type_local == 'lstm':
                model = load_model(model_path)
                y_scores = model.predict(X).ravel()
            else:
                model = joblib.load(model_path)
                Xf = X.reshape(X.shape[0], -1)
                if hasattr(model, 'predict_proba'):
                    y_scores = model.predict_proba(Xf)[:, 1]
                else:
                    y_scores = model.predict(Xf).astype(float)
        except Exception as e:
            st.error(f'Failed to load or predict with the selected model: {e}')
            st.stop()

        preds = (y_scores >= 0.5).astype(int)
        res_df = df.copy()
        res_df['prediction_proba'] = y_scores
        res_df['prediction'] = preds
        st.write('Predictions:')
        st.dataframe(res_df.head(50))

        # If label column present, compute metrics
        if label_col and label_col in df.columns:
            y_true = df[label_col].astype(int).values
            metrics = evaluate_binary(y_true, y_scores)
            st.write('Metrics:')
            st.json(metrics)
            st.write('ROC curve:')
            plot_roc(y_true, y_scores, title='ROC', out_path='tmp_roc.png')
            roc_img = plt.imread('tmp_roc.png')
            st.image(roc_img)
            plot_precision_recall(y_true, y_scores, title='PR', out_path='tmp_pr.png')
            pr_img = plt.imread('tmp_pr.png')
            st.image(pr_img)
            plot_confusion(y_true, preds, out_path='tmp_cm.png')
            cm_img = plt.imread('tmp_cm.png')
            st.image(cm_img)

        # Allow download
        csv_buf = res_df.to_csv(index=False).encode('utf-8')
        st.download_button('Download predictions CSV', csv_buf, file_name='predictions.csv')
        # Display training history if available
        try:
            history_log = os.path.join('models', 'training.log')
            if os.path.exists(history_log):
                plots = plot_training_history(history_log, out_dir='outputs')
                for k, p in plots.items():
                    img = plt.imread(p)
                    st.image(img, caption=f'Training {k}')
        except Exception as e:
            st.warning(f'Could not load training history: {e}')
        # Display selected model metadata if present
        if os.path.exists(meta_path_json):
            try:
                mdf = pd.read_json(meta_path_json)
                st.write('Model metadata:')
                st.json(mdf.to_dict())
            except Exception:
                st.write('Could not load metadata JSON')

st.markdown('---')
st.header('2. Manual single-student input')
with st.expander('Enter a single student (manual)'):
    # Build form fields from metadata or CSV columns
    base_vals = {}
    seq_prefix = 'Điểm TB HK'
    seq_input = ''
    seq_cols = []
    # Always add 'Giới tính' selectbox
    gender_options = ['Nam', 'Nữ', 'Khác']
    gender_value = st.selectbox('Giới tính', gender_options)
    base_vals['Giới tính'] = gender_value
    # Always add 'Xếp loại học lực' selectbox
    # Try to get valid options from encoder or metadata
    xeploai_options = None
    # Try encoder first
    try:
        if os.path.exists(encoders_path):
            xeploai_encoders = joblib.load(encoders_path)
            if 'Xếp loại học lực' in xeploai_encoders and hasattr(xeploai_encoders['Xếp loại học lực'], 'classes_'):
                xeploai_options = list(xeploai_encoders['Xếp loại học lực'].classes_)
    except Exception:
        xeploai_options = None
    # Try metadata
    if xeploai_options is None and metadata and 'xeploai_options' in metadata:
        xeploai_options = metadata['xeploai_options']
    # Fallback default
    if xeploai_options is None:
        xeploai_options = ['Xuất sắc', 'Giỏi', 'Khá', 'Trung bình', 'Yếu', 'Kém']
    xeploai_value = st.selectbox('Xếp loại học lực', xeploai_options)
    base_vals['Xếp loại học lực'] = xeploai_value
    if metadata and 'base_cols' in metadata:
        st.write('Detected model metadata. Fill values for these fields:')
        for c in metadata['base_cols']:
            if c not in ['Giới tính', 'Xếp loại học lực']:
                base_vals[c] = st.text_input(f'{c}', value='')
        seq_cols = metadata.get('seq_cols', []) or []
        if seq_cols:
            st.write(f'Nhập điểm từng học kỳ (dấu phẩy):')
            seq_input = st.text_input('Per-semester values (comma-separated)', value='')
    elif detected_csv_columns:
        st.write('Detected CSV columns from data folder. Please fill all fields:')
        for c in detected_csv_columns:
            if c not in ['Giới tính', 'Xếp loại học lực']:
                base_vals[c] = st.text_input(f'{c}', value='')
        seq_prefix = st.text_input('Per-semester feature prefix', value='Điểm TB HK')
        seq_input = st.text_input('Per-semester values (comma-separated)', value='')
    else:
        st.write('No metadata or CSV columns found. Provide column names and values.')
        base_cols_input = st.text_input('Base columns (comma-separated names)', value='')
        if base_cols_input:
            for c in [x.strip() for x in base_cols_input.split(',') if x.strip()]:
                if c not in ['Giới tính', 'Xếp loại học lực']:
                    base_vals[c] = st.text_input(f'{c}', value='')
        seq_prefix = st.text_input('Per-semester feature prefix', value='Điểm TB HK')
        seq_input = st.text_input('Per-semester values (comma-separated)', value='')

    alg_choice = st.selectbox('Algorithm', ['auto', 'lstm', 'rf', 'logistic'])

    submit_manual = st.button('Predict single student')
    if submit_manual:
        # Require all fields
        if (metadata and 'base_cols' in metadata) or detected_csv_columns:
            missing = [k for k, v in base_vals.items() if str(v).strip() == '']
            if missing:
                st.error(f'Missing values for fields: {missing}. Please fill all fields before predicting.')
                st.stop()
        row = {}
        for k, v in base_vals.items():
            try:
                row[k] = float(str(v).replace(',', '.')) if v != '' else np.nan
            except Exception:
                row[k] = v
        seq_vals = [s.strip() for s in seq_input.split(',')] if seq_input else []
        if seq_cols:
            for i, col in enumerate(seq_cols):
                if i < len(seq_vals) and seq_vals[i] != '':
                    try:
                        row[col] = float(seq_vals[i].replace(',', '.'))
                    except Exception:
                        row[col] = seq_vals[i]
                else:
                    row[col] = np.nan
        elif seq_input:
            for i, v in enumerate(seq_vals):
                colname = f"{seq_prefix}_{i+1}"
                try:
                    row[colname] = float(v.replace(',', '.'))
                except Exception:
                    row[colname] = v
        manual_df = pd.DataFrame([row])
        st.write('Constructed input:')
        st.dataframe(manual_df)

        # Predict
        if not model_path:
            st.error('No model selected. Choose a model from the models directory to run predictions.')
        else:
            try:
                X_manual, scaler_used, encoders_used = prepare_sequences_for_prediction(manual_df, scaler=joblib.load(scaler_path) if os.path.exists(scaler_path) else None, encoders=joblib.load(encoders_path) if os.path.exists(encoders_path) else None, metadata=metadata)
            except Exception as e:
                st.error(f'Failed to prepare input for prediction: {e}')
                X_manual = None
            if X_manual is not None:
                try:
                    # Algorithm selection
                    model_type_local = alg_choice
                    if model_type_local == 'auto':
                        if model_choice and model_choice.endswith(('.keras', '.h5', '.hdf5')):
                            model_type_local = 'lstm'
                        elif model_choice and model_choice.endswith('.joblib'):
                            if 'rf' in model_choice.lower():
                                model_type_local = 'rf'
                            elif 'log' in model_choice.lower():
                                model_type_local = 'logistic'
                            else:
                                model_type_local = 'logistic'
                    if model_type_local == 'lstm':
                        model_local = load_model(model_path)
                        y_scores = model_local.predict(X_manual).ravel()
                    else:
                        model_local = joblib.load(model_path)
                        Xf = X_manual.reshape(X_manual.shape[0], -1)
                        if hasattr(model_local, 'predict_proba'):
                            y_scores = model_local.predict_proba(Xf)[:, 1]
                        else:
                            y_scores = model_local.predict(Xf).astype(float)
                    pred = int((y_scores[0] >= 0.5))
                    st.write('Prediction probability:', float(y_scores[0]))
                    st.write('Predicted class:', pred)
                except Exception as e:
                    st.error(f'Prediction failed: {e}')
