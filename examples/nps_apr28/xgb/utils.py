import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def calc_recall_precision_auc_and_save(save_path, y_prob, y_test):
    print("Precision at Different Recall Levels: ")
    fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=0)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC (target=0): {roc_auc:.4f}")

    precision, recall, thresholds = precision_recall_curve(y_test, y_prob, pos_label=0)

    target_recalls = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    results = []
    for target_recall in target_recalls:
        idx = np.argmin(np.abs(recall - target_recall))
        if recall[idx] < target_recall and idx < len(recall) - 1:
            idx += 1
        prec_at_rec = precision[idx]
        thresh = thresholds[idx] if idx < len(thresholds) else 0.0
        results.append((target_recall, prec_at_rec, thresh))
        print(f"Recall: {target_recall:4.2f} | Precision: {prec_at_rec:.4f} | Threshold: {thresh:.4f}")

    recall_precision_path = os.path.join(save_path, "_recall_precision.csv")
    results_df = pd.DataFrame(results, columns=['Recall', 'Precision', 'Threshold'])
    results_df.to_csv(recall_precision_path, index=False)
    print(f"Recall-Precision data saved to {recall_precision_path}")
    return precision, recall, results


def save_precision_recall_curve(save_path, precision, recall, results):
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, 'b-', label='PR curve')
    plt.scatter([r[0] for r in results], [r[1] for r in results], c='red', s=50, zorder=10,
                label='Target Recall Points')
    for i, (r, p, t) in enumerate(results):
        plt.annotate(f'recall={r:.1f}\nprec={p:.2f}\nthresh={t:.2f}', (r, p), xytext=(10, -30),
                     textcoords='offset points',
                     arrowprops=dict(arrowstyle="->", color='gray'))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plot_path = os.path.join(save_path, "_pr_curve.png")
    plt.savefig(plot_path)
    print(f"Precision-Recall curve saved to {plot_path}")
    plt.close()


def show_important_feature(X, model, model_train_time, top_n_features):
    feature_importances = model.feature_importances_
    sorted_idx = np.argsort(feature_importances)[::-1]
    print(f"Top {top_n_features} features")
    for i in range(min(top_n_features, len(X.columns))):
        print(f"{i + 1}. {X.columns[sorted_idx[i]]}: {feature_importances[sorted_idx[i]]:.4f}")
    print(f"model train time is :{time.time() - model_train_time}")