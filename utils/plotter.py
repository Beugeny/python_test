import matplotlib.pyplot as plt
import numpy as np


def plot_predict_accuracity(predicted, true_values):
    fig, ax = plt.subplots()
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
    plt.plot(x1, y1, 'bs', x2, y2, 'ro',alpha=0.5)
    plt.show()
