import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Plot validation and worker metrics")
parser.add_argument(
    "date",
    nargs="?",
    help="Log directory name to use (default: latest in logs/)"
)
parser.add_argument(
    "--x",
    choices=["epoch", "index"],
    default="index",
    help="X-axis for validation plots: 'epoch' or 'index' (default: index)"
)
args = parser.parse_args()

logs_dir = "logs"

if args.date:
    log_dir = os.path.join(logs_dir, args.date)
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"Specified log directory not found: {log_dir}")
else:
    subdirs = [
        os.path.join(logs_dir, d)
        for d in os.listdir(logs_dir)
        if os.path.isdir(os.path.join(logs_dir, d))
    ]
    if not subdirs:
        raise FileNotFoundError("No subdirectories found in logs/")
    log_dir = max(subdirs, key=os.path.getmtime)

print(f"Using log directory: {log_dir}")

val_csv = os.path.join(log_dir, "validation_metrics.csv")
has_validation = os.path.isfile(val_csv)

if has_validation:
    df = pd.read_csv(val_csv)

    if df['loss'].dtype == 'object':
        df['loss'] = df['loss'].str.extract(r'(\d+\.\d+)').astype(float)

    if args.x == "epoch":
        x = df['epoch']
        x_label = "Epoch"
    else:
        df['entry_idx'] = range(1, len(df) + 1)
        x = df['entry_idx']
        x_label = "Entry Index"

    best_loss_idx = df['loss'].idxmin()
    best_acc_idx = df['accuracy'].idxmax()
    best_f1_idx = df['f1'].idxmax()

    print(
        f"Best Loss : {df.loc[best_loss_idx, 'loss']:.4f} "
        f"(epoch {df.loc[best_loss_idx, 'epoch']})"
    )
    print(
        f"Best Acc  : {df.loc[best_acc_idx, 'accuracy']:.4f} "
        f"(epoch {df.loc[best_acc_idx, 'epoch']})"
    )
    print(
        f"Best F1   : {df.loc[best_f1_idx, 'f1']:.4f} "
        f"(epoch {df.loc[best_f1_idx, 'epoch']})"
    )
else:
    print("No validation_metrics.csv found — skipping validation plots")

if has_validation:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax1.plot(x, df['loss'], linewidth=2, label='Loss')
    ax1.set_title('Validation Loss')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(x, df['accuracy'], linewidth=2, label='Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3.plot(x, df['f1'], linewidth=2, label='F1 Score')
    ax3.set_title('F1 Score')
    ax3.set_xlabel(x_label)
    ax3.set_ylabel('F1 Score')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.suptitle('Validation Metrics', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

worker_files = sorted(
    glob.glob(os.path.join(log_dir, "metrics_worker_*.csv"))
)

if worker_files:
    fig_w, (ax_w1, ax_w2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for wf in worker_files:
        name = os.path.splitext(os.path.basename(wf))[0]
        wdf = pd.read_csv(wf)

        ax_w1.plot(wdf['epoch'], wdf['accuracy'], linewidth=1.5, label=name)
        ax_w2.plot(wdf['epoch'], wdf['train_loss_epoch'], linewidth=1.5, label=name)

    ax_w1.set_title('Worker Accuracy vs Epoch')
    ax_w1.set_ylabel('Accuracy')
    ax_w1.grid(True, alpha=0.3)
    ax_w1.legend(ncol=2, fontsize=9)

    ax_w2.set_title('Worker Training Loss vs Epoch')
    ax_w2.set_xlabel('Epoch')
    ax_w2.set_ylabel('Train Loss')
    ax_w2.grid(True, alpha=0.3)

    plt.suptitle('Worker Metrics', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
else:
    print("No metrics_worker_*.csv files found")

plt.show()
