import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_learning_curve(df, metric_name):
    df.rename({
        'author': f'{metric_name}-Author',
        'issued': f'{metric_name}-Issued',
        'title': f'{metric_name}-Title',
        'overall': f'{metric_name}-Overall',
        "training_loss": "Training loss",
        "validation_loss": "Validation loss"
        
    }, axis=1, inplace=True)

    sns.set_theme(style='darkgrid', font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    x = range(1, len(df) + 1)
    for col in df.columns:
        if col.startswith(metric_name):
            # Make the markers be a small circle for all of them. Make it smaller.
            
            plt.plot(x, df[col], "--o", linewidth=2.5, markersize=5, label=col)
        
    for col in df.columns:
        if not col.startswith(metric_name):
            plt.plot(x, df[col], "-", linewidth=2.5, markersize=9, label=col)   
    
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.xticks(x)
    
    # Add legend, but not on top of the plot, on the side
    plt.legend(bbox_to_anchor=(1.03, 1), borderaxespad=0)  # legend in upper right corner outside plot
    plt.title("Learning curve")
    
    # plt.show()
    plt.savefig("C:\\Users\\carme\\Downloads\\Figure_1.png", bbox_inches='tight')
    return plt
    



if __name__ == "__main__":
    df = pd.DataFrame([[0.13, 0.40, 0.1, 0.4, 0.4, 0.43], [0.432, 0.43, 0.2, 0.9, 0.9, 0.1], [0.3, 0.3, 0.3, 0.12, 0.43, 0.4]], columns=["title", "author", "issued", "overall", "training_loss", "validation_loss"])
    plot_learning_curve(df, "Precision")
    print(df)
    