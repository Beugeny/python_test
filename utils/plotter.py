import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def plot_predict_accuracity(predicted, true_values, title=""):
    fig, ax = plt.subplots()
    plt.title(title)
    ax.scatter(true_values, predicted)
    ax.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()],
            'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()


def plot_hist(X, bins=20):
    plt.hist(X, bins=bins)
    plt.show()


def show_two_hist(x, y, bins=20):
    plt.hist(x, bins, alpha=0.5, label='x')
    plt.hist(y, bins, alpha=0.5, label='y')
    plt.show()


def show_two_plot(x1, y1, x2, y2):
    plt.plot(x1, y1, 'bs', x2, y2, 'ro', alpha=0.5)
    plt.show()


def corr_plot(cor):
    colormap = plt.cm.viridis
    plt.figure(figsize=(12, 12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(cor, linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
    sns.plt.show()


def corr_plot2(cor):
    # correlation matrix
    corrmat = cor
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)


def corr_zoom(cor, data, targetName, k=10):
    cols = cor.nlargest(k, targetName)[targetName].index
    cm = np.corrcoef(data[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': k}, yticklabels=cols.values,
                     xticklabels=cols.values)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()


def relation_plot(target, column, target_name):
    data = pd.concat([target, column], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=var, y=target_name, data=data)
    fig.axis(ymin=0, ymax=pd.DataFrame(target).max())
