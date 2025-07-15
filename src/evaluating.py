import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, make_scorer, root_mean_squared_error
import pandas as pd

def evaluate_metrics(y_true, y_pred):
    return {
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,
        'Bias': (y_pred - y_true).mean()
    }

def compute_bucket_metrics(df, bucket_col='bucket'):
    return (
        df.groupby(bucket_col, observed=True)
          .agg(
              count=('y_true', 'size'),
              RMSE=('y_true', lambda x: root_mean_squared_error(x, df.loc[x.index, 'y_pred'])),
              MAPE=('y_true', lambda x: mean_absolute_percentage_error(x, df.loc[x.index, 'y_pred']) * 100),
              Bias=('y_pred', lambda x: (x - df.loc[x.index, 'y_true']).mean())
          )
    )

def evaluate_buckets(y_true, y_pred, bins, labels):
    bucket = pd.cut(y_true, bins=bins, labels=labels)

    df_metrics = pd.DataFrame({
        'y_true': y_true.values,
        'y_pred': y_pred,
        'bucket': bucket
    })

    results = compute_bucket_metrics(df_metrics, bucket_col='bucket')
    print(results)

def plot_results(y_true, y_pred, label=''):
    residuals = y_true - y_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolor='k')
    plt.plot([0, y_true.max()], [0, y_true.max()], color='red', lw=2, label='Perfect prediction')
    plt.xlabel('True Price')
    plt.ylabel('Predicted Price')
    plt.title(f'Predicted vs True Prices {label}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/predicted_vs_true_{label}.png')
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, residuals, alpha=0.5, edgecolor='k')
    plt.axhline(0, color='red')
    plt.xlabel('True Price')
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs True Price {label}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/residuals_vs_true_{label}.png')
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.5, edgecolor='k')
    plt.axhline(0, color='red')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs Predicted Price {label}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/residuals_vs_predicted_{label}.png')
    plt.close()