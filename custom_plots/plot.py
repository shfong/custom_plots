"""Plotting functions"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pandas.api.types import is_float_dtype
import scipy as sp
from scipy.cluster import hierarchy
from scipy import ndimage
import seaborn as sns

import colors
import utils

def plotContourFromScatter(
    x, y, 
    densityLevels = [1,5,10,25,50,75,100], 
    xlabel = '', 
    ylabel = '', 
    ax=None
):
    """Make Contour plot

    Code was taken from M. Horlbeck's github page with minor modifications

    Paramters
    ---------
    x : Numpy Array
        x-coordinates
    y : Numpy Array
        y-coordinates
    densityLevels : List of Ints
        Contour levels to show
    xlabel : str
        Name of the x-axis on the figure
    ylabel : str
        Name of the y-axis on the figure

    Outputs
    -------
    fig : matplotlib Figure
        Figure object
    ax : matplotlib Axis
        Axis object
    """

    phenGrid, phenExtents, phenDensities = fast_kde(x, y, sample=True)

    if ax is None: 
        fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_aspect('equal')

    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.scatter(x, y, s=5, c=colors.ALMOSTBLACK, alpha=.5, rasterized=True)
    cf = ax.contour(
        phenGrid, 
        extent=phenExtents, 
        origin='lower', 
        levels = np.sort(np.percentile(phenDensities, densityLevels)), 
        colors=colors.ALMOSTBLACK, 
        linewidths=1
    )
    
    ax.contourf(
        phenGrid, 
        extent=phenExtents, 
        origin='lower', 
        levels = np.percentile(phenDensities, densityLevels), 
        colors='w'
    )

    ax.plot((0,0), (-25,10), color='#BFBFBF', lw=.5)
    ax.plot((-25,10), (0,0), color='#BFBFBF', lw=.5)

    ax.set_xlim((np.floor(min(phenExtents)),np.ceil(max(phenExtents))))
    ax.set_ylim((np.floor(min(phenExtents)),np.ceil(max(phenExtents))))
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)

    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)

    try: 
        return fig, ax
    except NameError: 
        return ax


def fast_kde(x, y, gridsize=(400, 400), extents=None, weights=None,
             sample=False):
    """
    Performs a gaussian kernel density estimate over a regular grid using a
    convolution of the gaussian kernel with a 2D histogram of the data.
    This function is typically several orders of magnitude faster than
    scipy.stats.kde.gaussian_kde for large (>1e7) numbers of points and
    produces an essentially identical result.
    
    Paramters
    ---------
    x: array-like
        The x-coords of the input data points
    y: array-like
        The y-coords of the input data points
    gridsize: tuple, optional
        An (nx,ny) tuple of the size of the output
        grid. Defaults to (400, 400).
    extents: tuple, optional
        A (xmin, xmax, ymin, ymax) tuple of the extents of output grid.
        Defaults to min/max of x & y input.
    weights: array-like or None, optional
        An array of the same shape as x & y that weighs each sample (x_i,
        y_i) by each value in weights (w_i).  Defaults to an array of ones
        the same size as x & y.
    sample: boolean
        Whether or not to return the estimated density at each location.
        Defaults to False
    
    Outputs
    -------
    density : 2D array of shape *gridsize*
        The estimated probability distribution function on a regular grid
    extents : tuple
        xmin, xmax, ymin, ymax
    sampled_density : 1D array of len(*x*)
        Only returned if *sample* is True.  The estimated density at each
        point.
    """
    #---- Setup --------------------------------------------------------------
    x, y = np.atleast_1d([x, y])
    x, y = x.reshape(-1), y.reshape(-1)

    if x.size != y.size:
        raise ValueError('Input x & y arrays must be the same size!')

    nx, ny = gridsize
    n = x.size

    if weights is None:
        # Default: Weight all points equally
        weights = np.ones(n)
    else:
        weights = np.squeeze(np.asarray(weights))
        if weights.size != x.size:
            raise ValueError('Input weights must be an array of the same size'
                    ' as input x & y arrays!')

    # Default extents are the extent of the data
    if extents is None:
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
    else:
        xmin, xmax, ymin, ymax = map(float, extents)
    extents = xmin, xmax, ymin, ymax
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    #---- Preliminary Calculations -------------------------------------------

    # Most of this is a hack to re-implment np.histogram2d using `coo_matrix`
    # for better memory/speed performance with huge numbers of points.

    # First convert x & y over to pixel coordinates
    # (Avoiding np.digitize due to excessive memory usage!)
    ij = np.column_stack((y, x))
    ij -= [ymin, xmin]
    ij /= [dy, dx]
    ij = np.floor(ij, ij).T

    # Next, make a 2D histogram of x & y
    # Avoiding np.histogram2d due to excessive memory usage with many points
    grid = sp.sparse.coo_matrix((weights, ij), shape=(ny, nx)).toarray()

    # Calculate the covariance matrix (in pixel coords)
    cov = image_cov(grid)

    # Scaling factor for bandwidth
    scotts_factor = np.power(n, -1.0 / 6) # For 2D

    #---- Make the gaussian kernel -------------------------------------------

    # First, determine how big the kernel needs to be
    std_devs = np.diag(np.sqrt(cov))
    kern_nx, kern_ny = np.round(scotts_factor * 2 * np.pi * std_devs)

    # Determine the bandwidth to use for the gaussian kernel
    inv_cov = np.linalg.inv(cov * scotts_factor**2)

    # x & y (pixel) coords of the kernel grid, with <x,y> = <0,0> in center
    xx = np.arange(kern_nx, dtype=np.float) - kern_nx / 2.0
    yy = np.arange(kern_ny, dtype=np.float) - kern_ny / 2.0
    xx, yy = np.meshgrid(xx, yy)

    # Then evaluate the gaussian function on the kernel grid
    kernel = np.vstack((xx.flatten(), yy.flatten()))
    kernel = np.dot(inv_cov, kernel) * kernel
    kernel = np.sum(kernel, axis=0) / 2.0
    kernel = np.exp(-kernel)
    kernel = kernel.reshape((int(kern_ny), int(kern_nx)))

    #---- Produce the kernel density estimate --------------------------------

    # Convolve the gaussian kernel with the 2D histogram, producing a gaussian
    # kernel density estimate on a regular grid

    # Big kernel, use fft...
    if kern_nx * kern_ny > np.product(gridsize) / 4.0:
        grid = sp.signal.fftconvolve(grid, kernel, mode='same')
    # Small kernel, use ndimage
    else:
        grid = ndimage.convolve(grid, kernel, mode='constant', cval=0)

    # Normalization factor to divide result by so that units are in the same
    # units as scipy.stats.kde.gaussian_kde's output.
    norm_factor = 2 * np.pi * cov * scotts_factor**2
    norm_factor = np.linalg.det(norm_factor)
    norm_factor = n * dx * dy * np.sqrt(norm_factor)

    # Normalize the result
    grid /= norm_factor

    if sample:
        i, j = ij.astype(int)
        return grid, extents, grid[i, j]
    else:
        return grid, extents


def image_cov(data):
    """Efficiently calculate the cov matrix of an image."""
    def raw_moment(data, ix, iy, iord, jord):
        data = data * ix**iord * iy**jord
        return data.sum()

    ni, nj = data.shape
    iy, ix = np.mgrid[:ni, :nj]
    data_sum = data.sum()

    m10 = raw_moment(data, ix, iy, 1, 0)
    m01 = raw_moment(data, ix, iy, 0, 1)
    x_bar = m10 / data_sum
    y_bar = m01 / data_sum

    u11 = (raw_moment(data, ix, iy, 1, 1) - x_bar * m01) / data_sum
    u20 = (raw_moment(data, ix, iy, 2, 0) - x_bar * m10) / data_sum
    u02 = (raw_moment(data, ix, iy, 0, 2) - y_bar * m01) / data_sum

    cov = np.array([[u20, u11], [u11, u02]])
    
    return cov


def clustermap(
    data, 
    linkage_method="average", 
    linkage_metric="correlation", 
    optimal_ordering=True,
    cmap=colors.BLUE_YELLOW_CMAP,
    **clustermap_kwargs
): 
    """Convenience function to draw seaborn clustermap with custom linkage"""

    linkage_params = {
        "method": linkage_method, 
        "metric": linkage_metric, 
        "optimal_ordering": optimal_ordering
    }

    row_linkage = hierarchy.linkage(data, **linkage_params)
    col_linkage = hierarchy.linkage(data.T, **linkage_params)

    cm = sns.clustermap(
        data, 
        row_linkage=row_linkage, col_linkage=col_linkage, 
        cmap=cmap, **clustermap_kwargs)

    return cm


def change_clustermap_aspect(cm, x=1, y=1, w=1, h=1):
    """Change seaborn clustermap aspect ratio to an arbitrary ratio"""

    hm = cm.ax_heatmap.get_position()
    col = cm.ax_col_dendrogram.get_position()
    row = cm.ax_row_dendrogram.get_position()

    aspect = {"x":x, "y": y, "w":w, "h":h}

    cm.ax_heatmap.set_position(change_aspect(hm, **aspect))
    cm.ax_col_dendrogram.set_position(change_aspect(col, **aspect))
    cm.ax_row_dendrogram.set_position(change_aspect(row, **aspect))


def change_aspect(pos, x=1, y=1, w=1, h=1):
    """Multiplies the current position by the input ratio""" 

    return [pos.x0*x, pos.y0*y, pos.width*w, pos.height*h]


def remove_axis_spines_and_ticks(ax):
    """Removes the spines in a Matplotlib Axis"""
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])


def shaded_lines(x, y, colors='grey', linecolors='black', edgecolor='black', ylim=None, 
    x_offset=0.05, scatter_size=50, linewidth=1.5, ylabel=None, xlabel=None, 
    labelsize=14, 
    ax=None): 

    """Plot shaded lines
    
    All color parameters are required. xlim is taken to be the min and max of
    x with some offset

    Parameters
    ----------
    x : numpy ndarray
        x-coordinates 
    y : numpy ndarray
        y-coordinates; can be an array of 1 dimension or 2 dimensions
        If 2 dimensions, expect the number of rows to match the number of colors, 
        linecolors, etc
    colors : list
        An iterable of fill-colors
    linecolors : list
        An iterable of line colors (for the lines connecting the scatters)
    edgecolors : list
        An iterable of edge colors (for the circles of the scatters)
    ylim : list
        A 2-element list (or tuple) that defines the lower and upper bound of 
        the y-axis. If None, this will be left to matplotlib to infer
    x_offset : float
        The extra space left on the x-axis (purely for aesthetics)

    Return
    ------
    fig, ax : matplotlib figure and axis
        If an axis was provided, the function will return None. Otherwise, the
        function will return the figure and axis that was created
    """
    if y.ndim == 1:
        y = y.reshape(1, -1)
        
    if ax is None: 
        fig, ax = plt.subplots()

    number_of_lines = y.shape[0]
    colors, linecolors, edgecolor = [
        utils.expand_string_arguments(arg, number_of_lines) for arg in [colors, linecolors, edgecolor]
    ]

    for b, c, lc, e in zip(y, colors, linecolors, edgecolor):
        ax.fill_between(x, 0, b, color=c, zorder=1)
        ax.plot(x, b, color=lc, zorder=2)
        ax.scatter(x, b, color=c, edgecolors=e, linewidths=linewidth, 
            marker='o', zorder=3, s=scatter_size, clip_on=False)

    sns.despine()

    if ylim is not None: 
        ax.set_ylim(ylim)

    if x_offset is not None: 
        ax.set_xlim([np.min(x) - x_offset, np.max(x) + x_offset])
    
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.set_ylabel(ylabel, fontsize=labelsize)

    if ax is not None: 
        return 

    return fig, ax


def reverse_bars(bins, heights, width=None, x_offset=0.05, color=colors.GREYS[-1], 
    edgecolor="", xlabel="", ylabel="", labelsize=12, ax=None):

    """Plot an upside-down histogram

    In most cases, you would just use the built-in histogram function. This is 
    just a wrapper for the `shaded_lines_and_histgoram` function.

    Parameters
    ----------
    bins : numpy array 
        x axis that define the bins for the histogram; expect bins to be equally
        spaced (used in the assumption for width)
    heights : numpy array 
        y axis that define the heights of the bars
    width : float
        Width of the bars
    x_offset : float
        The extra space left on the x-axis (purely for aesthetics)
    color : str
        Face color of the bars
    edgecolor : str
        Edge color of the bars
    xlabel : str
        X-label
    ylabel : str
        Y-label
    labelsize : int
        font-size of the label
    ax : matplotlib axis
        If provided, the plot will be drawn on that axis

    Returns
    -------
    fig, ax : matplotlib figure and axis
        If an axis was provided, the function will return None. Otherwise, the
        function will return the figure and axis that was created
    """

    if ax is None: 
        fig, ax = plt.subplots()

    if width is None:
        width = bins[1] - bins[0]

    ax.bar(bins, heights, width=width, color=color, edgecolor=edgecolor)
    ax.set_ylim(ax.get_ylim()[::-1]) # reverse the y-axis
    ax.xaxis.tick_top()
    ax.spines['bottom'].set_visible(False)

    ax.set_xlim([np.min(bins) - x_offset, np.max(bins) + x_offset])
    ax.set_ylabel(ylabel, fontsize=labelsize)
    ax.set_xlabel(xlabel,  fontsize=labelsize)

    # ax.set_ticks(axis="x", )


def shaded_lines_and_histogram(bins, heights, total, title = '', figsize=(8,5), 
    titlesize=16, height_ratio=(2,1), reverse_bars_kwargs=None, 
    shaded_lines_kwargs=None):

    """Draw shaded line and histogram combination figure

    Parameters
    ----------
    bins : numpy array
        x-axis (bar and scatter center)
    heights : numpy array
        y for the scatter and line plots
    total : numpy array
        y for the histogram 
    title : str
        Title of the figure
    figsize : tuple
        2-tuple that defines the x,y size of the figure
    titlesize : int
        Fontsize of the figure title
    height_ratio : tuple
        The ratio of the top figure (line + scatter plot) and the bottom figure 
        (histogram)
    reverse_bars_kwargs : dict or None
        Keyword arguments for `reverse_bars` function
    shaded_lines_kwargs : dict or None
        Keyword arguments for `shaded_lines`
    
    Returns
    -------
    fig, ax : matplotlib figure and axis
    """

    if reverse_bars_kwargs is None: 
        reverse_bars_kwargs = {}

    if shaded_lines_kwargs is None: 
        shaded_lines_kwargs  = {}
        
    fig, (ax1,ax2) = plt.subplots(2, 1, figsize=figsize,
        sharex=True, gridspec_kw={'height_ratios': height_ratio})

    shaded_lines_kwargs['ax'] = ax1
    shaded_lines(bins, heights, **shaded_lines_kwargs)

    reverse_bars_kwargs['ax'] = ax2
    reverse_bars(bins, total, **reverse_bars_kwargs)

    if title: 
        fig.suptitle(title, fontsize=titlesize)

    return fig, (ax1, ax2)


def boxplots(data, x=None, y=None, hue=None,
    baseline=None, baseline_color=None, baseline_linestyle='--',
    title='', titlesize=18, xlabel='', ylabel='', labelsize=16, 
    ticklabelsize=12, legend_title=None, legend_text_size=None, 
    legend_title_size=None, figsize=(12,6)
): 

    """Utility to make cleaning up a Seaborn boxplot easier"""
   
    fig, ax = plt.subplots(figsize=figsize)

    if baseline is not None: 
        ax.axhline(
            baseline, 
            color=colors.GREYS[3] if baseline_color is None else baseline_color, 
            linestyle=baseline_linestyle,
        )
    
    if is_float_dtype(data[x]):
        order = np.sort(np.unique(data[x]))
        xticklabels = ['{:5.3f}'.format(i) for i in order]

    boxplot = sns.boxplot(data=data, x=x, y=y, hue=hue, ax=ax, order=order)

    try:     
        ax.set_xticklabels(xticklabels)
    except NameError: 
        pass

    ax.set_title(title, fontsize=titlesize)
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    ax.tick_params(labelsize=ticklabelsize)

    remove_legend_box(ax)

    change_legend_text(ax, title=legend_title, titlesize=legend_title_size, 
        textsize=legend_text_size
    )

    sns.despine()

    return fig, ax

def remove_legend_box(ax): 
    ax.get_legend().get_frame().set_facecolor('none')
    ax.get_legend().get_frame().set_edgecolor('none')


def change_legend_text(ax, textsize=None, titlesize=None, title=None):
    """Change legend title and text size and set title"""

    if textsize is not None: 
        plt.setp(ax.get_legend().get_texts(), fontsize=textsize)
    
    if titlesize is not None: 
        plt.setp(ax.get_legend().get_title(), fontsize=titlesize)

    if title is not None: 
        ax.get_legend().set_title(title)

def make_annotation_legend(x=1, top=1, height=0.13, title="", titlesize=16,
    texts=(), textsize=14, textcolor="black", facecolor='white', 
    edgecolor='black', ax=None
): 
    textcolor, facecolor, edgecolor = [
        utils.expand_string_arguments(arg, len(texts)) for arg in [textcolor, facecolor, edgecolor]
    ]

    ax.annotate(title, xy=(x, top), fontsize=titlesize, 
        xycoords='axes fraction', horizontalalignment='center')

    for ind, (text, fc, ec, c) in enumerate(zip(texts, facecolor, edgecolor, textcolor)): 
        ax.annotate(
            text, 
            xy=(x, top-(ind+1)*height), 
            fontsize=textsize, 
            xycoords='axes fraction', 
            horizontalalignment='center', 
            bbox=dict(facecolor=fc, edgecolor=ec), 
            color=c,
        )


def shaded_lines_with_stems(x, y, ax=None, title="", xlabel="x", ylabel="y"):
    return_fig = False
    if ax is None: 
        return_fig = True
        fig, ax = plt.subplots(figsize=(8,8))
        
    ax.axhline(0, color=colors.GREYS[3], linewidth=1)
        
    ax.fill_between(x, y, 0, color=colors.GREYS[0])
    _, stemlines, _ = ax.stem(x, y, linefmt=colors.GREYS[2], markerfmt=" ", basefmt=" ")
    plt.setp(stemlines, 'linewidth', 1)

    ax.plot(x,y, color=colors.GREYS[4], linewidth=1)

    ax.scatter(x, y, color=colors.GREYS[2], edgecolor=colors.GREYS[-1], s=80, zorder=3, clip_on=False)

#     ax.set_ylabel(xlabel, fontsize=20)
#     ax.set_xlabel(ylabel, fontsize=20)
    ax.set_title(title, fontsize=18)
    ax.tick_params(labelsize=14)

#     sns.despine()
    
    if return_fig:
        return fig, ax
    
    return ax