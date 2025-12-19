# Student Performance Prediction - LSTM + Baselines

Project: Build an LSTM model (per semesters) to predict student academic warning, compare with Logistic Regression and Random Forest using ROC-AUC, precision, recall, F1-score, and accuracy. Provides a CLI to import a CSV and perform predictions with trained LSTM.

## Structure

-   `data/` - input CSV (place your dataset here; already contains `dataset_sinhvien - dataset_sinhvien_1HK.csv`)
-   `src/` - source code for loader, models, training, and prediction
-   `models/` - saved models
-   `outputs/` - metrics and plots

## Quickstart

1. Create Python environment and install requirements:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate; pip install -r requirements.txt
```

2. Train models

```powershell
python src\train.py --data "data\dataset_sinhvien - dataset_sinhvien_1HK.csv" --out models --outputs outputs --epochs 20 --batch_size 64
```

3. Predict using trained models

```powershell
python src\predict_cli.py --model models\lstm_model.keras --scaler models\scaler.joblib --encoders models\encoders.joblib --input data\your_new.csv --out predictions.csv --model-type lstm
```

If you want to use logistic or random forest, change `--model-type` and pass the corresponding joblib model path (e.g., `models/rf_model.joblib`).

### Training with long format data

If your dataset is LONG format (one row per student per semester), use the following flags:

```powershell
python src\train.py --data "data\longformat.csv" --student-id student_id --semester-col semester --feature-cols "Điểm TB" --max-timesteps 6 --out models --outputs outputs
```

This will pivot the data into per-student sequences and train the LSTM accordingly.

### Streamlit UI

Run the UI to upload CSVs and make predictions interactively:

```powershell
streamlit run src\app.py
```

## Notes

-   The loader detects columns named like `Điểm TB HK1`, `Điểm TB HK2` etc. If your data contains multiple semesters, the LSTM will be trained on temporal sequences per student. If only single semester data is present, the LSTM will use a simple time-step of length 1.
-   The label is expected to be `Cảnh báo học tập (0/1)` column.
