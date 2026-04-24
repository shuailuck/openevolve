import os
import matplotlib.pyplot as plt


def save_precision_recall_curve(model_save_path, precision, recall, results):
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
    plot_path = os.path.splitext(model_save_path)[0] + "_pr_curve.png"
    plt.savefig(plot_path)
    print(f"Precision-Recall curve saved to {plot_path}")
    plt.close()
