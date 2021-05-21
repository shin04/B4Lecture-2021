import os
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

from linear_regression import LinearRegression

TIME_TEMPLATE = '%Y%m%d%H%M%S'


def main(args):
    result_path = Path(args.save_path)
    timestamp = datetime.now().strftime(TIME_TEMPLATE)
    result_path = result_path/timestamp
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    assert result_path.exists(), '{} is not exist'.format(result_path)

    # load data
    df1 = pd.read_csv('../data1.csv')
    df2 = pd.read_csv('../data2.csv')
    df3 = pd.read_csv('../data3.csv')

    # data1
    model_1 = LinearRegression(digree=1)
    X = df1['x1'].values.reshape(-1, 1)
    y = df1['x2'].values
    model_1.fit(X, y)
    reg_eq_1 = model_1.get_equation()

    w = model_1.coef
    linex = np.linspace(-5, 5)
    y_ = w[1]*linex + w[0]

    plt.figure(figsize=(5, 5))
    plt.scatter(X, y, label='obserbed')
    plt.plot(linex, y_, c='indianred', label=reg_eq_1)
    plt.legend()
    plt.savefig(result_path/'data1.png')

    # data2
    model_2 = LinearRegression(digree=3)
    X = df2['x1'].values.reshape(-1, 1)
    y = df2['x2'].values
    model_2.fit(X, y)
    reg_eq_2 = model_2.get_equation()

    w = model_2.coef
    linex = np.linspace(0, 10)
    y_ = w[3]*linex**3 + w[2]*linex**2 + w[1]*linex + w[0]

    plt.figure(figsize=(5, 5))
    plt.scatter(X, y, label='obserbed')
    plt.plot(linex, y_, c='indianred', label=reg_eq_2)
    plt.legend()
    plt.savefig(result_path/'data2.png')

    # data3
    model_3 = LinearRegression(digree=2)
    X1 = df3['x1'].values
    X2 = df3['x2'].values
    y = df3['x3'].values
    X = np.array([X1, X2]).T
    model_3.fit(X, y)
    reg_eq_3 = model_3.get_equation()

    w = model_3.coef
    linex1 = np.linspace(-4, 4)
    linex2 = np.linspace(0, 10)

    ax_1, ax_2 = np.meshgrid(linex1, linex2)
    y_ = w[4]*ax_2 + w[5]*ax_1 + w[1]*ax_2**2 + w[2]*ax_1*ax_2 + w[3]*ax_1**2 + w[0]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], y, label='Observed')
    ax.plot_wireframe(ax_1, ax_2, y_, color='indianred', label=reg_eq_3)
    plt.legend()
    plt.savefig(result_path/'data3.png')


if __name__ == "__main__":
    description = 'Example: python main.py -s ./result'
    parser = argparse.ArgumentParser(description=description)
    # parser.add_argument('data_path', help='path of data')
    parser.add_argument('-s', '--save_path', default='./',
                        help='path to save the result')

    args = parser.parse_args()

    main(args)
