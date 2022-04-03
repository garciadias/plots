from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def normalize_data(df):
    x = df.values
    min_max_scaler = StandardScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, columns=df.columns, index=df.index)
    return df


def make_fake_data(n_sample=200, n_features=9, n_classes=3, class_names=None):
    if class_names is None:
        class_names = {i: f"class_{i}" for i in range(n_classes)}
    X, y = make_classification(n_sample, n_features=n_features, n_classes=n_classes, n_clusters_per_class=1)
    for i in np.random.randint(0, n_features, 5):
        X[:, i] = X[:, i] * 10
    dataset = pd.DataFrame(X, columns=[f"{i}_var" for i in range(n_features)])
    dataset.loc[:, "class"] = y
    dataset.loc[:, "class"] = dataset.loc[:, "class"].astype("category")
    dir(dataset["class"].cat)
    dataset["class"].cat.categories = [class_names[c] for c in dataset["class"].cat.categories]
    dataset["class"] = dataset["class"].cat.as_ordered()
    return dataset


def parallel_plot(df, class_column="class", normalize=True):
    if normalize:
        y = df[class_column]
        df = normalize_data(df[[var for var in df.columns if var != class_column]].copy())
        df.loc[:, class_column] = y
    n_classes = df[class_column].unique().shape[0]
    MEANS = df.groupby(class_column).mean().T
    STD = df.groupby(class_column).std().T
    n_features = MEANS.shape[0]
    colors = [plt.cm.Set1(i) for i in range(n_classes)]
    fig, ax = plt.subplots(figsize=(16*0.7, 9*0.7))
    for i, ((class_name, mean), (class_name, std)) in enumerate(zip(MEANS.iteritems(), STD.iteritems())):
        plt.errorbar(range(n_features), mean, yerr=std, alpha=0.6, linewidth=4,
                     color=colors[i], ecolor=colors[i], label=class_name
                     )
    ax.set_xticks(range(n_features), MEANS.index)
    ax.grid(True)
    plt.legend(bbox_to_anchor=(1.025, 1), loc=2, borderaxespad=0., fontsize=14)
    plt.tight_layout()
    return fig


def pair_plot(df, class_column="class"):
    ax = sns.pairplot(df, hue=class_column, palette="Set1")
    n_rows, n_cols = ax.axes.shape
    # Making the y lables even and legible
    max_len = max([len(label.get_text()) for i in range(n_rows) for label in ax.axes[i, 0].get_yticklabels()])
    for i in range(n_rows):
        ax.axes[i, 0].set_ylabel(ax.axes[i, 0].get_ylabel(), fontsize='32')
        ax.axes[i, 0].tick_params(labelsize=20)
        labels = [" "*(max_len - len(label.get_text())) + label.get_text() for label in ax.axes[i, 0].get_yticklabels()]
        ax.axes[i, 0].set_yticklabels(labels)
    # Making the x lables even and legible
    for i in range(n_cols):
        ax.axes[n_rows-1, i].set_xlabel(ax.axes[n_rows-1, i].get_xlabel(), fontsize='32')
        ax.axes[n_rows-1, i].tick_params(labelsize=20)
    # Making the legend even and legible
    plt.setp(ax._legend.get_texts(), fontsize='32')  # for legend text
    plt.setp(ax._legend.get_title(), fontsize='34')  # for legend title
    plt.setp(ax._legend, bbox_to_anchor=(1.00, 0.5))  # legend outside
    # Adjust postion of the plots in the figure
    plt.subplots_adjust(left=0.06, right=0.90, bottom=0.04)
    return ax


def box_plot(df, class_column="class"):
    n_classes = df[class_column].unique().shape[0]
    columns = [c for c in df.columns if c != class_column]
    colors = [plt.cm.Set1(i) for i in range(n_classes)]
    fig, axes = plt.subplots(1, len(columns), figsize=(16*0.7, 9*0.7))
    for i, (col, ax) in enumerate(zip(columns, axes.flatten())):
        sns.boxplot(data=df, x=class_column, y=col, ax=ax, palette=colors)
        ax.set_ylabel(None)
        plt.setp(ax.get_xticklabels(), rotation=90)
        ax.set_title(col)
    plt.subplots_adjust(left=0.03, right=0.99, bottom=0.13, top=0.96, wspace=0.8)
    return fig


if __name__ == '__main__':
    # To reuse the code you just need to replace this line with loading your DataFrame
    DATASET = make_fake_data()
    # Don't forget to update with your class column
    class_column = "class"
    box_plot(DATASET, class_column=class_column)
    plt.savefig("figs/boxplot_grid.png", dpi=100)
    plt.close()

    if DATASET.shape[1] <= 10:
        pair_plot(DATASET, class_column=class_column)
        plt.savefig("figs/pair_plot.png", dpi=100)
        plt.close()
    else:
        print("There are too many columns for a pair plot.")

    parallel_plot(DATASET, class_column=class_column)
    plt.savefig("figs/parallel_plot.png", dpi=100)
