import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_learning_curve(df, metric_name):
    print("Here is the df to plot")
    print(df)
    df.rename({
        'spatial': f'{metric_name}-Spatial',
        'author': f'{metric_name}-Author',
        'issued': f'{metric_name}-Issued',
        'inGroup': f'{metric_name}-In Group',
        'subject': f'{metric_name}-Subject',
        'title': f'{metric_name}-Title',
        'overall': f'{metric_name}-Overall weighted',
        'AUTHOR': f'{metric_name}-Author', # For development
        'SPATIAL': f'{metric_name}-Spatial' # For development
    }, axis=1, inplace=True)

    sns.set_theme(style='darkgrid', font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)
    colorset = ["#663300", "#004C99", "#FF8000", "#009999", "#FF33FF"]

    x = range(1, len(df) + 1)
    for col in df.columns:
        if col.startswith(metric_name):
            # Make the markers be a small circle for all of them
            plt.plot(x, df[col], "-o", linewidth=3, markersize=13, label=col)
        
    for col in df.columns:
        if not col.startswith(metric_name):
            plt.plot(x, df[col], "-", linewidth=3, markersize=13, label=col)   
    
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.xticks(x)
    
    # Add legend, but not on top of the plot, on the side
    plt.legend(bbox_to_anchor=(10.02, 1), borderaxespad=0)  # legend in upper right corner outside plot
    plt.title("Learning Curve")
    
    return plt
    
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    df = pd.DataFrame([[0.13, 0.40, 0.1], [0.432, 0.43, 0.2], [0.3, 0.3, 0.3]], columns = ["Spatial", "Author", "Spatial"])
    plot_learning_curve(df)
    print(df)
    