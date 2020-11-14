import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dualstudent.models import sigmoid_rampup, sinusoidal_cycling, linear_cycling


def plot_schedules(save_path=None):
    x = np.linspace(0, 20, num=100000)

    y = sigmoid_rampup(x, 5)
    plt.plot(x, y, 'royalblue', label='rampup')

    y = sinusoidal_cycling(x, 5)
    plt.plot(x, y, 'orange', label='sinusoidal cycling')

    y = linear_cycling(x, 5)
    plt.plot(x, y, 'red', label='triangular cycling')

    plt.xlim([0, 20])
    plt.xlabel('epoch')
    plt.xticks(np.arange(0, 21, 5))
    plt.legend()

    if save_path is not None:
        save_path = Path(save_path)
        plt.savefig(save_path / 'schedules.pdf')

    plt.show()  # important: after saving! otherwise it saves white image


def get_command_line_arguments():
    parser = argparse.ArgumentParser(
        description='Train Dual Student for Automatic Speech Recognition on TIMIT dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('output', type=str, help='path where to save the image')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_command_line_arguments()
    plot_schedules(args.output)
