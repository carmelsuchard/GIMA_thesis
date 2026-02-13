import matplotlib.pyplot as plt
import pandas as pd

def boxplot_metrics(metrics_dict, title, save_path: str):
    
    print(metrics_dict)
    metrics_df = pd.DataFrame(metrics_dict)
    medians = metrics_df.median().round(2).tolist()

    plt.figure(figsize=(10, 6))
    metrics_df.boxplot()
    plt.title(title)
    plt.ylabel('Value')
    plt.ylim(0, 1)
    plt.grid(axis='y')

    for i, median_value in enumerate(medians, start=1):
        plt.text(
            i,
            median_value - 0.015,
            f"{median_value:.2f}",
            ha='center',
            va='top',
            fontsize=9
        )

    
    plt.savefig(save_path)
    plt.close()

    
if __name__ == "__main__":
    # Sample df
    data = {
        'title': [0.9, 0.85, 0.88, 0.92, 0.87],
        'author': [0.8, 0.75, 0.78, 0.82, 0.77],
        'issued': [0.95, 0.9, 0.93, 0.96, 0.91],
        'spatial': [0.7, 0.65, 0.68, 0.72, 0.67],
        'inGroup': [0.85, 0.8, 0.83, 0.87, 0.82],
        'subject': [0.9, 0.88, 0.89, 0.91, 0.87]
    }
