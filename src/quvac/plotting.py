"""
Useful plotting functions.
"""

import os
from pathlib import Path

import numpy as np
from scipy.constants import pi

try:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ModuleNotFoundError:
    MATPLOTLIB_AVAILABLE = False


def _check_matplotlib():
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required for this function.")

def save_fig(save_path, fig_name):
    """
    Save figure to file as png and pdf.

    Parameters
    ----------
    save_path : str
        Path to the directory where the figure will be saved.
    fig_name : str
        Name of the figure file.
    """
    _check_matplotlib()
    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        name = os.path.join(save_path, fig_name)
        plt.savefig(f"{name}.png", bbox_inches='tight')
        plt.savefig(f"{name}.pdf", bbox_inches='tight')


def pi_formatter(x, pos):
    """
    Format axis ticks as multiples of pi.

    Parameters
    ----------
    x : float
        Tick value.
    pos : int
        Tick position.

    Returns
    -------
    str
        Formatted tick label.
    """
    fractions = {0: "0", pi/8: r"$\frac{\pi}{8}$", pi/4: r"$\frac{\pi}{4}$", 
                 3*pi/8: r"$\frac{3\pi}{8}$", pi/2: r"$\frac{\pi}{2}$",
                 5*pi/8: r"$\frac{5\pi}{8}$", 3*pi/4: r"$\frac{3\pi}{4}$",
                 7*pi/8: r"$\frac{7\pi}{8}$", pi: r"$\pi$",
                 9*pi/8: r"$\frac{9\pi}{8}$", 5*pi/4: r"$\frac{5*\pi}{4}$",
                 11*pi/8: r"$\frac{11\pi}{8}$", 3*pi/2: r"$\frac{3\pi}{2}$",
                 13*pi/8: r"$\frac{13\pi}{8}$", 7*pi/4: r"$\frac{7\pi}{4}$",
                 15*pi/8: r"$\frac{15\pi}{8}$", 2*pi: r"$2\pi$"}
    return fractions.get(x, f"${x/np.pi:.2g}\\pi$")


def plot_roi(ax, x0, y0, dx, dy, line_kwargs):
    """
    Plot a rectangular region of interest (ROI) on a given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the ROI.
    x0 : float
        The x-coordinate of the center of the ROI.
    y0 : float
        The y-coordinate of the center of the ROI.
    dx : float
        Half the width of the ROI.
    dy : float
        Half the height of the ROI.
    line_kwargs : dict
        Keyword arguments to customize the appearance of the ROI lines 
        (e.g., color, linestyle).

    Returns
    -------
    matplotlib.axes.Axes
        The axis with the plotted ROI.

    Notes
    -----
    The ROI is represented as a rectangle centered at (x0, y0) with width 2*dx and 
    height 2*dy.
    """
    x_left, x_right = x0-dx, x0+dx
    y_top, y_bottom = y0-dy, y0+dy
    pts = [(x_right,y_top),(x_left,y_top),(x_left,y_bottom),
           (x_right,y_bottom),(x_right,y_top)]
    for pt1,pt2 in zip(pts[:-1],pts[1:]):
        ax.plot([pt1[0],pt2[0]], [pt1[1],pt2[1]], **line_kwargs)
    return ax


def plot_mollweide(fig, ax, phi, theta, data, cmap='coolwarm', norm=None):
    """
    Plot data on a Mollweide projection.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    phi : numpy.ndarray
        Array of azimuthal angles.
    theta : numpy.ndarray
        Array of polar angles.
    data : numpy.ndarray
        Data to be plotted.
    cmap : str, optional
        Colormap, by default 'coolwarm'.
    norm : matplotlib.colors.Normalize, optional
        Normalization for the colormap, by default None.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object with the plot.
    """
    _check_matplotlib()
    theta_ = theta - np.pi/2
    # flip theta axis so mollweide plot shows usual sphere surface
    theta_ = theta_[::-1]
    phi_ = phi - np.pi
    phi_mesh, theta_mesh = np.meshgrid(phi_, theta_)
    
    im = ax.pcolormesh(phi_mesh, theta_mesh, data, cmap=cmap,
                       shading='gouraud', rasterized=True, norm=norm)
    cbar = fig.colorbar(im, ax=ax, shrink=0.5)

    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    xtick_labels = np.linspace(60, 360, 5, endpoint=False, dtype=int)
    ax.xaxis.set_ticklabels(r'$%s^{\circ}$' %num for num in xtick_labels)
    ytick_labels = np.linspace(0, 180, 7, endpoint=True, dtype=int)[::-1]
    ax.yaxis.set_ticklabels(r'$%s^{\circ}$' %num for num in ytick_labels)
    for item in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
        item.set_fontsize(18)
    ax.grid()
    return ax, cbar


def plot_fields(field, t, plot_keys=None, cmap='coolwarm',
                cnorm='log', norm_lim=1e-20, save_path=None):
    """
    Plot specified field components.

    Plotted figures:
    1. (x,y) and (x,z) profiles at the focus.
    2. x, y, z slices through the focus.

    Parameters
    ----------
    field : quvac.field.Field
        Field object containing the field data.
    t : float
        Time at which to plot the field.
    plot_keys : list of str, optional
        List of field components to plot, by default None.
    cmap : str, optional
        Colormap, by default 'coolwarm'.
    cnorm : str, optional
        Normalization type, by default 'log'.
    norm_lim : float, optional
        Normalization limit, by default 1e-20.
    save_path : str, optional
        Path to save the plots, by default None.

    Returns
    -------
    field_comps : dict
        Dictionary containing the field components.
    """
    _check_matplotlib()
    Nxyz = field.grid_xyz.grid_shape
    E_out, B_out = [[np.zeros(Nxyz, dtype=np.complex128) for _ in range(3)]
                    for _ in range(2)]
    E, B = field.calculate_field(t=t, E_out=E_out, B_out=B_out)
    E, B = [np.real(Ex) for Ex in E], [np.real(Bx) for Bx in B]
    nx, ny, nz = Nxyz
    x, y, z = [ax*1e6 for ax in field.grid_xyz.grid]

    intensity = (E[0]**2 + E[1]**2 + E[2]**2 + B[0]**2 + B[1]**2 + B[2]**2)/2
    field_comps = {
        "Ex": E[0],
        "Ey": E[1],
        "Ez": E[2],
        "Bx": B[0],
        "By": B[1],
        "Bz": B[2],
        "Intensity": intensity,
    }

    plot_keys = plot_keys if plot_keys is not None else field_comps.keys()
    # 1st plot: xy and xz profiles at focus for given components
    n_rows = len(plot_keys)
    n_cols = 2

    plt.figure(figsize=(12, 5*n_rows), layout="constrained")
    for i,key in enumerate(plot_keys):
        if key == "Intensity":
            cmap = "inferno"

        ax_bottom, ax_top = [y, z], [x, x]
        x_labels = "y z".split()
        comps = [field_comps[key][:, :, nz//2], field_comps[key][:, ny//2, :]]

        for j,comp in enumerate(comps):
            norm = None
            if key == "Intensity" and cnorm == "log":
                norm = mcolors.LogNorm(vmin=comp.max()*norm_lim, vmax=comp.max())
            ax = plt.subplot(n_rows, n_cols, i*n_cols+j+1)
            im = plt.pcolormesh(ax_bottom[j], ax_top[j], comp, shading=None,
                                rasterized=True, cmap=cmap, norm=norm)
            plt.xlabel(f"{x_labels[j]} [$\\mu$m]")
            if j == 0:
                plt.ylabel("x [$\\mu$m]")
            plt.title(f"{key} at t={t*1e15:.1f} fs")

            plt.colorbar(im, shrink=0.6)

            ax.set_aspect('equal')
    save_fig(save_path, "field_profiles_focus")
    plt.show()

    # 2nd plot: x, y, z slices through focus for given components
    n_rows = len(plot_keys)
    n_cols = 3
    axs = [x, y, z]
    axs_names = ["x", "y", "z"]

    plt.figure(figsize=(18, 5*n_rows), layout="constrained")
    for i,key in enumerate(plot_keys):
        comp = field_comps[key]
        slices = [comp[:, ny//2, nz//2],
                  comp[nx//2, :, nz//2],
                  comp[nx//2, ny//2, :]]
        for j,slc in enumerate(slices):
            plt.subplot(n_rows, n_cols, i*n_cols+j+1)
            plt.plot(axs[j], np.abs(slc))
            I0 = np.abs(slc).max()
            if norm_lim:
                plt.ylim(I0*norm_lim, 3*I0)
            plt.yscale("log")
            plt.xlabel(f"{axs_names[j]} [$\\mu$m]")
            if j == 0:
                plt.ylabel(key)
            plt.title(f"{axs_names[j]} slice")
    save_fig(save_path, "field_slices_focus")
    plt.show()

    return field_comps

