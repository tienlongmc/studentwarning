import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve, confusion_matrix


def evaluate_binary(y_true, y_scores, threshold=0.5):
    y_pred = (y_scores >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_scores)
    except Exception:
        roc_auc = float('nan')
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }


def plot_roc(y_true, y_scores, title='ROC Curve', out_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    if out_path:
        plt.savefig(out_path)
    plt.close()


def plot_precision_recall(y_true, y_scores, title='Precision-Recall Curve', out_path=None):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, lw=2, label='PR curve (area = %0.2f)' % pr_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    if out_path:
        plt.savefig(out_path)
    plt.close()


def plot_confusion(y_true, y_pred, out_path=None, labels=['0', '1']):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='white')
    if out_path:
        plt.savefig(out_path)
    plt.close()


def plot_training_history(csv_log_path, out_dir=None):
    """Reads CSV logged by CSVLogger and plots loss/val_loss and accuracy/val_accuracy.

    Returns paths to generated PNGs.
    """
    import os
    df = pd.read_csv(csv_log_path)
    plots = {}
    if 'loss' in df.columns:
        plt.figure(); plt.plot(df['loss'], label='loss');
        if 'val_loss' in df.columns:
            plt.plot(df['val_loss'], label='val_loss');
        plt.title('Training Loss'); plt.xlabel('epoch'); plt.legend();
        p = os.path.join(out_dir or '.', 'training_loss.png')
        plt.savefig(p); plt.close(); plots['loss'] = p
    if 'accuracy' in df.columns:
        plt.figure(); plt.plot(df['accuracy'], label='accuracy');
        if 'val_accuracy' in df.columns:
            plt.plot(df['val_accuracy'], label='val_accuracy');
        plt.title('Training Accuracy'); plt.xlabel('epoch'); plt.legend();
        p = os.path.join(out_dir or '.', 'training_accuracy.png')
        plt.savefig(p); plt.close(); plots['accuracy'] = p
    return plots


def flatten_sequences(X):
    # flatten over time dimension to feed logistic and RF
    n, t, f = X.shape
    X_flat = X.reshape(n, t * f)
    return X_flat
