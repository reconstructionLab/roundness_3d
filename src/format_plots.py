"""
format_plots.py
Author: Henry Yuan

Description: A function to format matplotlib plots
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

def format_plots():
    mpl.rcParams['figure.dpi'] = 300
    plt.rcParams['axes.titlelocation'] = 'left'
    plt.rcParams['axes.titlepad']      = 8
    plt.rcParams['legend.frameon']     = False
    plt.rcParams['xtick.direction']    = 'in'
    plt.rcParams['ytick.direction']    = 'in'
    plt.rcParams['axes.grid']          = False # True; False custom for JSR
    plt.rcParams['grid.color']         = 'black'
    plt.rcParams['grid.linestyle']     = '-'
    plt.rcParams['grid.linewidth']     = 0.125
    plt.rcParams['grid.alpha']         = 0.5
    plt.rcParams['axes.axisbelow'] = True
    plt.rcParams['svg.fonttype']       = 'none'
    plt.rcParams['pdf.fonttype']       = 42
    plt.rcParams['font.family']        = 'sans-serif'
    plt.rcParams['text.usetex']        = False
    plt.rcParams['savefig.dpi']=300

    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.titlesize'] = 8

    # remove plot margins (gap between left side and first data)
    mpl.rcParams['axes.xmargin'] = 0

    # *** custom for JSR
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    # *** end custom

    # # Changes font for ALL labels (x and y)
    # plt.rcParams['axes.labelsize'] = 8

    # # Changes font for ALL tick numbers
    # plt.rcParams['xtick.labelsize'] = 8
    # plt.rcParams['ytick.labelsize'] = 8
    # plt.rcParams['legend.fontsize'] = 8

    # def global variables
    CR = '#0085a9' # round color
    CS = '#004e08' # semi-round color
    CE = '#f3a257' # euhedral color
    CL = 'black' # outline of histogram bars color
    CRD = '#005ca9' # dark round color
    CED = '#f36d2f' # dark euhedral color
    CWAD = '#f99d1b' # color for Wadell roundness
    CSTD = "#205073" # color for std of curvature metric

    return (CR, CS, CE, CL, CRD, CED, CWAD, CSTD)
