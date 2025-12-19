import argparse
import os
import joblib
import pandas as pd
import numpy as np
try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None
from src.data_loader import prepare_sequences_for_prediction, load_csv
from src.utils import plot_roc, plot_precision_recall, plot_confusion, evaluate_binary


def predict(model_path, scaler_path, encoders_path, input_csv, out_csv=None, model_type='lstm', label_col='Cảnh báo học tập (0/1)'):
    print('Loading data...')
    try:
        df = load_csv(input_csv)
    except Exception as e:
        raise RuntimeError(f"Failed to load {input_csv}: {e}")

    scaler = None
    encoders = None
    metadata = None
    if scaler_path and encoders_path:
        try:
            scaler = joblib.load(scaler_path)
        except Exception:
            scaler = None
        try:
            encoders = joblib.load(encoders_path)
        except Exception:
            encoders = None
        # Try to load metadata if present
        try:
            metadata = joblib.load(encoders_path.replace('encoders', 'metadata'))
        except Exception:
            metadata = None

    X, scaler, encoders = prepare_sequences_for_prediction(df, scaler=scaler, encoders=encoders, metadata=metadata)

    print('Loading model...')
    y_scores = None
    model = None
    if not model_path or not os.path.exists(model_path):
        raise RuntimeError(f'Model file {model_path} does not exist')
    try:
        # Auto-detect model type based on extension
        if model_type == 'auto':
            if model_path.endswith(('.keras', '.h5', '.hdf5')):
                model_type = 'lstm'
            elif model_path.endswith('.joblib'):
                model_type = 'rf' if 'rf' in os.path.basename(model_path).lower() else 'logistic'
        if model_type == 'lstm':
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
        raise RuntimeError(f'Failed to load or predict with model {model_path}: {e}')
    # no else: we've already handled prediction above
    if y_scores is None:
        raise RuntimeError('No prediction scores were produced by the model')
    y_pred = (y_scores >= 0.5).astype(int)

    out_df = df.copy()
    out_df['prediction_proba'] = y_scores
    out_df['prediction'] = y_pred

    if out_csv:
        out_df.to_csv(out_csv, index=False)
        print('Predictions saved to', out_csv)
    else:
        print(out_df[['prediction', 'prediction_proba']].head())

    # If label is present, compute metrics and save plots
    if label_col in df.columns:
        y_true = df[label_col].astype(int).values
        metrics = evaluate_binary(y_true, y_scores)
        print('Metrics:', metrics)
        plot_roc(y_true, y_scores, out_path=out_csv.replace('.csv', '_roc.png') if out_csv else None)
        plot_precision_recall(y_true, y_scores, out_path=out_csv.replace('.csv', '_pr.png') if out_csv else None)
        plot_confusion(y_true, (y_scores >= 0.5).astype(int), out_path=out_csv.replace('.csv', '_cm.png') if out_csv else None)
    return out_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to model file (LSTM .h5/.keras or joblib)')
    parser.add_argument('--scaler', required=False, default=None, help='Path to scaler.joblib')
    parser.add_argument('--encoders', required=False, default=None, help='Path to encoders.joblib')
    parser.add_argument('--input', required=True, help='CSV file to predict')
    parser.add_argument('--label', dest='label', default='Cảnh báo học tập (0/1)', help='Label column name if present in data')
    parser.add_argument('--out', default=None, help='Output CSV to save predictions')
    parser.add_argument('--model-type', dest='model_type', choices=['lstm', 'rf', 'logistic', 'auto'], default='auto', help='Type of model to load (auto will detect from extension)')
    args = parser.parse_args()
    predict(args.model, args.scaler, args.encoders, args.input, args.out, model_type=args.model_type, label_col=args.label)
