# src/tribo_analyzer/plot_config.py

import matplotlib.pyplot as plt

def apply_plot_style():
    """共通の matplotlib スタイルを設定する"""
    plt.rcParams["font.size"] = 12

    # fontをTeX化
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['cmr10']
    plt.rcParams['font.sans-serif'] = ['cmss10']
    plt.rcParams['font.monospace'] = ['cmtt10']
    plt.rcParams["axes.formatter.use_mathtext"] = True

    # 軸設定
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["ytick.minor.visible"] = True
