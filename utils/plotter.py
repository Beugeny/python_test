import matplotlib.pyplot as plt


def plot_predict_accuracity(predicted, true_values):
    fig, ax = plt.subplots()
    ax.scatter(true_values, predicted)
    ax.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()],
            'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()
