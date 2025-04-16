import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)
from matplotlib import pyplot as plt
import seaborn as sns

def eval_perform(y_pred, y_test):
    y_test_np = y_test.values if isinstance(y_test, pd.DataFrame) else np.array(y_test)
    y_pred_np = np.array(y_pred)
    metrics = {}

    metrics['y_test'] = y_test
    metrics['y_pred'] = y_pred

    if y_pred_np.ndim > 1:
        y_pred_np = np.argmax(y_pred_np, axis=1)
    if y_test_np.ndim > 1:
        y_test_np = np.argmax(y_test_np, axis=1)


    try:
        metrics['accuracy'] = accuracy_score(y_test_np, y_pred_np)
        metrics['precision'] = precision_score(y_test_np, y_pred_np, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_test_np, y_pred_np, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_test_np, y_pred_np, average='weighted', zero_division=0)

        conf_matrix = confusion_matrix(y_test_np, y_pred_np)
        metrics['confusion_matrix'] = conf_matrix

        metrics['true_positives'] = conf_matrix.diagonal()
        metrics['false_positives'] = conf_matrix.sum(axis=0) - conf_matrix.diagonal()
        metrics['false_negatives'] = conf_matrix.sum(axis=1) - conf_matrix.diagonal()

    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return {}
    
    return metrics

def print_perform(metrics) -> None:
    print("\nPerformance Report")
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            if isinstance(value, (np.ndarray, list)):
                print(f"{metric}: {value.tolist()}")
            else:
                print(f"{metric}: {value}")
    
    if 'confusion_matrix' in metrics:
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])

def plot_cm(metrics, labels=None):
    if 'confusion_matrix' not in metrics:
        print("Confusion matrix not found in metrics.")
        return

    cm = metrics['confusion_matrix']
    
    if labels is None:
        labels = list(map(str, range(len(cm))))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

def plot_metric(metrics):
    bar_metrics = ['accuracy', 'precision', 'recall', 'f1']
    values = [metrics.get(m, 0) for m in bar_metrics]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=bar_metrics, y=values, palette="viridis")
    plt.ylim(0, 1)
    plt.title("Evaluation Metrics")
    plt.ylabel("Score")
    plt.show()

def plot_true_false(metrics):
    tp = metrics.get('true_positives', [])
    fp = metrics.get('false_positives', [])
    fn = metrics.get('false_negatives', [])

    x = np.arange(len(tp))
    width = 0.3

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, tp, width, label='True Positives', color='green')
    plt.bar(x, fp, width, label='False Positives', color='red')
    plt.bar(x + width, fn, width, label='False Negatives', color='orange')

    plt.xlabel("Class Index")
    plt.ylabel("Count")
    plt.title("True Positives vs False Positives vs False Negatives")
    plt.legend()
    plt.xticks(x)
    plt.tight_layout()
    plt.show()
    
def plot_regression(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor='k')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Regression: Predicted vs Actual")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

