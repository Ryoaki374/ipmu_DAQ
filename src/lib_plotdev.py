import pandas as pd
import numpy as np
from nptdms import TdmsFile
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import os
from matplotlib import gridspec


def csv2df():
    path = "./KT2000/20240426192006KT_MULTIMETER.csv"
    df_data = pd.read_csv(
        path,
        names=[
            "Abstime",
            "Reltime",
            "Voltage",
        ],
        skiprows=1,
        # dtype="float64",
        parse_dates=[0],
    )
    df_data


def tdms_lookup(file_name):
    tdms_file = TdmsFile(file_name)
    group = tdms_file.groups()[0]
    for channel in group.channels():
        channel_name = channel.name
        print("channel:", channel_name)


def tdms_load(file_name, data_lab, samplerate=1000, print_=True):
    tdms_file = TdmsFile(file_name)
    group = tdms_file.groups()[0]
    for i in tqdm(range(0, len(group)), desc="Data loading"):
        channel = group.channels()[i]
        data = channel[:]
        if i == 0:
            data_arr = np.empty([len(group) + 1, len(data)])
            data_arr[0] = np.linspace(
                0, len(data) / samplerate, len(data), endpoint=False
            )
            data_arr[i + 1] = data
        else:
            data_arr[i + 1] = data
    df = pd.DataFrame(data_arr, index=data_lab)
    return df.T


def multiple_plot_xaxis(
    ax, time_data, data_arrays, labels=None, plot_col=None, linestyle=None, fmt=None
):
    """
    Plot multiple data arrays with a common time axis on the same axes.

    Parameters:
    - ax (matplotlib.axes._axes.Axes): Axes object to plot on.
    - time_data: Common time data for the x-axis.
    - data_arrays (list): List of data arrays to be plotted on the y-axis.
    - labels (list): List of labels for each data array. (Optional)
    - plot_col (list): List of colors for each data array. If None, default colors are used. (Optional)
    """
    fmt = fmt or "-"
    linestyle = linestyle or "-"
    labels = labels or [f"Data {i+1}" for i in range(len(data_arrays))]
    plot_col = plot_col or [
        "#ff4b00",
        "#fff100",
        "#03af7a",
        "#005aff",
        "#4dc4ff",
        "#ff8082",
        "#f6aa00",
        "#990099",
        "#804000",
    ]

    for data, label, color in zip(data_arrays, labels, plot_col):
        ax.plot(time_data, data, fmt, label=label, color=color, ls=linestyle)
        ax.autoscale(enable=True, axis="x", tight=True)
        ax.autoscale(enable=True, axis="y", tight=False)

def plot_overplot(
    ax, time_data, data_arrays, labels=None, plot_col=None, linestyle=None, fmt=None
):
    """
    Plot multiple data arrays with a common time axis on the same axes.

    Parameters:
    - ax (matplotlib.axes._axes.Axes): Axes object to plot on.
    - time_data: Common time data for the x-axis.
    - data_arrays (list): List of data arrays to be plotted on the y-axis.
    - labels (list): List of labels for each data array. (Optional)
    - plot_col (list): List of colors for each data array. If None, default colors are used. (Optional)
    """
    fmt = fmt or "-"
    linestyle = linestyle or "-"
    labels = labels or [f"Data {i+1}" for i in range(len(data_arrays))]
    plot_col = plot_col or [
        "#ff4b00",
        "#005aff",
        "#03af7a",
        "#fff100",
        "#4dc4ff",
        "#ff8082",
        "#f6aa00",
        "#990099",
        "#804000",
    ]

    for data, label, color in zip(data_arrays, labels, plot_col):
        ax.plot(time_data, data, fmt, label=label, color=color, ls=linestyle)
        ax.autoscale(enable=True, axis="x", tight=True)
        ax.autoscale(enable=True, axis="y", tight=False)

def scatter_overplot(
    ax, time_data, data_arrays, labels=None, plot_col=None, s=20, fmt=None
):
    """
    Scatter plot multiple data arrays with a common time axis on the same axes.

    Parameters:
    - ax (matplotlib.axes._axes.Axes): Axes object to plot on.
    - time_data: Common time data for the x-axis.
    - data_arrays (list): List of data arrays to be plotted on the y-axis.
    - labels (list): List of labels for each data array. (Optional)
    - plot_col (list): List of colors for each data array. If None, default colors are used. (Optional)
    - s (int or list): Marker size(s) for the scatter plot. Default is 20. (Optional)
    """
    labels = labels or [f"Data {i+1}" for i in range(len(data_arrays))]
    plot_col = plot_col or [
        "#ff4b00",
        "#005aff",
        "#03af7a",
        "#fff100",
        "#4dc4ff",
        "#ff8082",
        "#f6aa00",
        "#990099",
        "#804000",
    ]

    for data, label, color in zip(data_arrays, labels, plot_col):
        ax.scatter(time_data, data, label=label, color=color, s=s)
        ax.autoscale(enable=True, axis="x", tight=True)
        ax.autoscale(enable=True, axis="y", tight=False)


def multiple_row_plot(rows, cols, row_ratios=None, col_ratios=None):
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(
        rows,
        cols,
        height_ratios=row_ratios,
        width_ratios=col_ratios,
        hspace=0.0,
        wspace=0.0,
    )
    ax_grid = [plt.subplot(gs[i]) for i in range(rows)]
    for ax in ax_grid[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    return ax_grid


def color_palette():
    # return {
    #    "red": "#ff4b00",
    #    "yellow": "#fff100",
    #    "green": "#03af7a",
    #    "blue": "#005aff",
    #    "sky_blue": "#4dc4ff",
    #    "pink": "#ff8082",
    #    "orange": "#f6aa00",
    #    "purple": "#990099",
    #    "brown": "#804000"
    # }
    return ["#FF4B00", "#005AFF", "#03AF7A", "#4DC4FF", "#F6AA00", "#FFF100", "#000000"]


def color_palette_v2():
    return {
        "red": "#ff4b00",
        "blue": "#005aff",
        "green": "#03af7a",
        "yellow": "#fff100",
        "sky_blue": "#4dc4ff",
        "pink": "#ff8082",
        "orange": "#f6aa00",
        "purple": "#990099",
        "brown": "#804000",
    }


def create_directory(filename):
    # Get the current date
    current_datetime = datetime.now().strftime("%Y%m%d%H%M")

    # Create a new directory name
    directory_name = f"{current_datetime}_{filename}"

    # Create the directory if it doesn't exist
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        print(f"Directory '{directory_name}' has been created.")
    else:
        print(f"Directory '{directory_name}' already exists.")

    return directory_name


def Series_to_label(arr):
    label_arr = []
    for ind in arr:
        label_arr.append(ind.name)
    return label_arr


def set_xylim(ax, xrange, yrange):
    xstr, xfin, xnum = xrange
    ystr, yfin, ynum = yrange
    ax.set_xlim(xstr - xstr * 0.02, xfin)
    ax.set_xticks(np.arange(xstr, xfin, xnum))
    ax.set_ylim(ystr - ystr * 0.05, yfin)
    ax.set_yticks(np.arange(ystr, yfin, ynum))


def gen_ylim(ax, ylim, yrange):
    ystr, yfin, ynum = yrange
    ax.set_yticks(np.arange(ystr, yfin, ynum))
    ystr, yfin = ylim
    ax.set_ylim(ystr, yfin)

def gen_xlim(ax, xlim, xrange):
    """
    Set x-ticks and x-limits for the axis.

    Parameters:
    ax : matplotlib axis
        The axis to apply the limits and ticks to.
    xlim : tuple
        A tuple specifying the lower and upper limits of the x-axis.
    xrange : tuple
        A tuple with the start, end, and step for the x-ticks.
    """
    xstr, xfin, xnum = xrange
    ax.set_xticks(np.arange(xstr, xfin, xnum))
    xstr, xfin = xlim
    ax.set_xlim(xstr, xfin)



def enhanced_set_xylim(ax, xrange, yrange, logscale_x=False, logscale_y=False):
    xstr, xfin, xnum = xrange
    ystr, yfin, ynum = yrange

    # X軸の設定
    if logscale_x:
        ax.set_xscale("log", base=10)  # x軸を底10の対数スケールに設定
        ax.set_xlim(xstr / 1.02, xfin)
        ax.set_xticks(np.logspace(np.log10(xstr), np.log10(xfin), xnum))
    else:
        ax.set_xlim(xstr - xstr * 0.02, xfin)
        ax.set_xticks(np.arange(xstr, xfin, xnum))

    # Y軸の設定
    if logscale_y:
        ax.set_yscale("log", base=10)  # y軸を底10の対数スケールに設定
        ax.set_ylim(ystr / 1.05, yfin)
        ax.set_yticks(np.logspace(np.log10(ystr), np.log10(yfin), ynum))
    else:
        ax.set_ylim(ystr - ystr * 0.05, yfin)
        ax.set_yticks(np.arange(ystr, yfin, ynum))

    # グリッドと目盛りのフォーマット更新
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())


def gen_plotlabel(ydata_arr):
    plot_label = []

    for i in range(len(ydata_arr)):
        plot_label.append(ydata_arr[i].name)

    return plot_label

def gen_axislabel(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)