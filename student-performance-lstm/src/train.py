import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

from src.data_loader import load_csv, prepare_sequences, prepare_sequences_from_long
from src.utils import evaluate_binary, plot_roc, flatten_sequences
from src.evaluate import evaluate_and_save


def build_lstm(input_shape, hidden_units=32, dropout=0.2):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(hidden_units))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_all(data_path, out_dir='models', outputs='outputs', label_col='Cảnh báo học tập (0/1)', test_size=0.2, random_state=42, epochs=30, batch_size=32, feature_prefixes=['Điểm TB HK'], student_id_col=None, semester_col=None, feature_cols=None, max_timesteps=None, model_name=None, train_lstm=True, train_lr=True, train_rf=True, hidden_units=32):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(outputs, exist_ok=True)

    print('Loading data...')
    df = load_csv(data_path, label_col=label_col)
    if student_id_col and semester_col and feature_cols:
        print('Preparing sequences from long format...')
        X, y, scaler, encoders, base_cols, cat_cols, seq_cols = prepare_sequences_from_long(df, student_id_col, semester_col, feature_cols, label_col=label_col, max_timesteps=max_timesteps, sort_col=semester_col)
    else:
        print('Preparing sequences from wide format...')
        X, y, scaler, encoders, base_cols, cat_cols, seq_cols = prepare_sequences(df, label_col=label_col, feature_prefixes=feature_prefixes)

    X_flat = flatten_sequences(X)

    # train/test split
    # Use a single split so indices match across X and flattened X
    X_train, X_test, Xf_train, Xf_test, y_train, y_test = train_test_split(
        X, X_flat, y, test_size=test_size, random_state=random_state, stratify=y)

    # LSTM model
    lstm = None
    if train_lstm:
        print('Training LSTM...')
        input_shape = (X_train.shape[1], X_train.shape[2])
        lstm = build_lstm(input_shape, hidden_units=hidden_units)
        from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
        early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ckpt = ModelCheckpoint(os.path.join(out_dir, f"{model_name or 'lstm_model'}.keras"), save_best_only=True)
        csv_logger = CSVLogger(os.path.join(outputs, 'training.log'))
        # Compute class weights to help with imbalance
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weight_dict = {c: w for c, w in zip(classes, class_weights)}
        history = lstm.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, callbacks=[early, ckpt, csv_logger], class_weight=class_weight_dict, verbose=2)
        lstm_path = os.path.join(out_dir, f"{model_name or 'lstm_model'}.keras")
        lstm.save(lstm_path)
    # Save training metadata and history
    try:
        # save model metadata (already done) and training log
        import shutil
        if os.path.exists(os.path.join(outputs, 'training.log')):
            shutil.copy(os.path.join(outputs, 'training.log'), os.path.join(out_dir, 'training.log'))
    except Exception:
        pass

    # Logistic Regression
    if train_lr:
        print('Training Logistic Regression...')
        lr = LogisticRegression(max_iter=1000)
        lr.fit(Xf_train, y_train)
        lr_path = os.path.join(out_dir, f"{model_name or 'logistic_model'}.joblib")
        joblib.dump(lr, lr_path)
    else:
        lr = None

    # Random Forest
    if train_rf:
        print('Training Random Forest...')
        rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
        rf.fit(Xf_train, y_train)
        rf_path = os.path.join(out_dir, f"{model_name or 'rf_model'}.joblib")
        joblib.dump(rf, rf_path)
    else:
        rf = None

    # Evaluate
    print('Evaluating models...')
    y_lstm_scores = lstm.predict(X_test).ravel() if lstm is not None else np.zeros(len(y_test))
    y_lr_scores = lr.predict_proba(Xf_test)[:, 1] if lr is not None else np.zeros(len(y_test))
    y_rf_scores = rf.predict_proba(Xf_test)[:, 1] if rf is not None else np.zeros(len(y_test))

    metrics = {}
    metrics['lstm'] = evaluate_binary(y_test, y_lstm_scores)
    metrics['logistic'] = evaluate_binary(y_test, y_lr_scores)
    metrics['random_forest'] = evaluate_binary(y_test, y_rf_scores)

    evaluate_and_save(y_test, y_lstm_scores, outputs, model_name='lstm')
    evaluate_and_save(y_test, y_lr_scores, outputs, model_name='logistic')
    evaluate_and_save(y_test, y_rf_scores, outputs, model_name='random_forest')

    # Save metrics
    import json
    metrics_path = os.path.join(outputs, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # ROC plots
    plot_roc(y_test, y_lstm_scores, title='LSTM ROC', out_path=os.path.join(outputs, 'roc_lstm.png'))
    plot_roc(y_test, y_lr_scores, title='Logistic ROC', out_path=os.path.join(outputs, 'roc_lr.png'))
    plot_roc(y_test, y_rf_scores, title='Random Forest ROC', out_path=os.path.join(outputs, 'roc_rf.png'))

    # Save scaler and encoders and metadata
    joblib.dump(scaler, os.path.join(out_dir, 'scaler.joblib'))
    joblib.dump(encoders, os.path.join(out_dir, 'encoders.joblib'))
    # Save metadata (column lists) to ensure consistent inference
    metadata = {
        'base_cols': base_cols,
        'cat_cols': cat_cols,
        'seq_cols': seq_cols,
        'feature_prefixes': feature_prefixes
    }
    joblib.dump(metadata, os.path.join(out_dir, 'metadata.joblib'))
    # Save human readable metadata as JSON
    try:
        import json as _json
        with open(os.path.join(out_dir, 'metadata.json'), 'w', encoding='utf-8') as mf:
            _json.dump(metadata, mf, indent=2, ensure_ascii=False)
    except Exception:
        pass

    print('Training finished. Models and outputs are saved in', out_dir, outputs)
    print('Metrics:\n', metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--out', type=str, default='models', help='Directory to save models')
    parser.add_argument('--outputs', type=str, default='outputs', help='Directory to save metrics/plots')
    parser.add_argument('--label', type=str, default='Cảnh báo học tập (0/1)', help='Label column name')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs for LSTM')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for LSTM training')
    parser.add_argument('--prefix', type=str, default='Điểm TB HK', help='Feature prefix for semester scores')
    parser.add_argument('--student-id', dest='student_id', type=str, default=None, help='Student ID column for long-format input')
    parser.add_argument('--semester-col', dest='semester_col', type=str, default=None, help='Semester/time column for long-format input')
    parser.add_argument('--feature-cols', dest='feature_cols', type=str, default=None, help='Comma-separated list of features to use across semesters (long-format). Example: "Điểm TB"')
    parser.add_argument('--max-timesteps', dest='max_timesteps', type=int, default=None, help='Max timesteps to include from long-format data')
    parser.add_argument('--model-name', dest='model_name', type=str, default=None, help='Optional model name prefix for saved models')
    parser.add_argument('--no-lstm', dest='no_lstm', action='store_true', help='Do not train LSTM')
    parser.add_argument('--no-lr', dest='no_lr', action='store_true', help='Do not train Logistic Regression')
    parser.add_argument('--no-rf', dest='no_rf', action='store_true', help='Do not train Random Forest')
    parser.add_argument('--hidden-units', dest='hidden_units', type=int, default=32, help='Hidden units for LSTM')
    args = parser.parse_args()

    feature_cols = args.feature_cols.split(',') if args.feature_cols else None
    train_all(
        data_path=args.data, out_dir=args.out, outputs=args.outputs, label_col=args.label, epochs=args.epochs, batch_size=args.batch_size,
        feature_prefixes=[args.prefix], student_id_col=args.student_id, semester_col=args.semester_col, feature_cols=feature_cols, max_timesteps=args.max_timesteps,
        model_name=args.model_name, train_lstm=not args.no_lstm, train_lr=not args.no_lr, train_rf=not args.no_rf, hidden_units=args.hidden_units
    )
