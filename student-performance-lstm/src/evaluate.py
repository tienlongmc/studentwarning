import json
from sklearn.metrics import classification_report
from src.utils import evaluate_binary, plot_roc, plot_precision_recall, plot_confusion
import pandas as pd


def evaluate_and_save(y_true, scores, out_dir, model_name='model'):
    metrics = evaluate_binary(y_true, scores)
    report = classification_report(y_true, (scores >= 0.5).astype(int), output_dict=True)
    with open(f'{out_dir}/metrics_{model_name}.json', 'w', encoding='utf-8') as f:
        json.dump({'metrics': metrics, 'report': report}, f, ensure_ascii=False, indent=2)
    plot_roc(y_true, scores, title=f'{model_name} ROC', out_path=f'{out_dir}/roc_{model_name}.png')
    plot_precision_recall(y_true, scores, title=f'{model_name} PR', out_path=f'{out_dir}/pr_{model_name}.png')
    plot_confusion(y_true, (scores >= 0.5).astype(int), out_path=f'{out_dir}/cm_{model_name}.png')