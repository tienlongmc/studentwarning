# Run demo: setup virtualenv, install deps, train models, and predict using LSTM.
# Run in PowerShell

python -m venv .venv; .\.venv\Scripts\Activate; pip install -r requirements.txt

# Train models (edit path if needed to your dataset)
python src\train.py --data "..\data\dataset_sinhvien - dataset_sinhvien_1HK.csv" --out models --outputs outputs --epochs 5 --batch_size 32

# Predict on some rows (the training dataset for demo purposes)
python src\predict_cli.py --model models\lstm_model.keras --scaler models\scaler.joblib --encoders models\encoders.joblib --input "..\data\dataset_sinhvien - dataset_sinhvien_1HK.csv" --out outputs\predictions.csv --model-type lstm

Write-Host "Demo complete. Models in models/ outputs in outputs/"
