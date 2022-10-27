import warnings
import numpy as np
from scipy.io import savemat, loadmat
from scipy.stats import levy_stable, uniform
import matplotlib.pyplot as plt
from matplotlib import cm
import copy

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def simulate_GD(lpath, T, lr, xystart, add_noise=False, levy_vars=(2, 0, 0, 1), BCs='reflecting', save_results=True, save_path=None):
    """Simulates gradient descent on a landscape with reflecting boundary conditions. 
    Can save dictionary of results in the same folder as the landscape under fname with gradient, trajectory, and plotting segments.

    Args:
        landscape (str or ndarray): Path to file for landscape or ndarray of landscape.
        T (int): Total number of iterations
        lr (float): Learning rate. 
        xystart (tup or list): Initial (x,y) position.
        add_noise (bool): If True, add symmetric Levy alpha stable noise in random direction for imitating SGD. Defaults to False.
        levy_vars (tup): (alpha, beta, location, scale) of Levy alpha-stable distribution for the noise. 
            Defaults to Gaussian noise with sigma = 1. 
        BCs (str): Boundary conditions, either 'periodic' or 'reflecting'
            Default is 'reflecting'.
        save_results (bool): If True, dictionary of results are saved to fpath under fname. If False, dictionary of results are returned.
            Defaults to True.
        save_path (str): Path for saving. Defaults to None.

    Returns:
        dict:   "landscape" contains landscape
                "trajectory" contains [x,y] position of optimiser at each iteration
                "segments" contains segments for plotting purposes only
                "trajectory_continuous" contains [x,y] position accounting for reflecting boundary conditions.
    """
    if isinstance(lpath, str):
        # Load landscape
        landscape = loadmat(lpath)["landscape"]
    else:
        landscape = lpath
    assert (type(landscape) is np.ndarray), "landscape is not a numpy array"

    M = landscape.shape[0]
    grad_landscape = np.gradient(landscape)
    xystart = np.array(xystart, dtype='float64')

    # Variables to keep track of, particularly for plotting
    coords = np.zeros((T, 2))
    coords[0, :] = xystart
    coords_cont = np.zeros((T, 2))
    coords_cont[0, :] = xystart
    niters = 1
    segments = []  # segments for plotting because of periodic BCs
    x = [xystart[0]]
    y = [xystart[1]]
    xyold = xystart
    xynew = np.zeros(2)
    xy_cont = xystart
    rbcs = np.array([1, 1])

    grad = []

    # Simulation
    while niters < T:
        xygrad = _interpolate_grad(xyold, grad_landscape)
        grad.append(xygrad)
        if add_noise:
            noise_mag = np.abs(levy_stable.rvs(
                alpha=levy_vars[0], beta=levy_vars[1], loc=levy_vars[2], scale=levy_vars[3]))
            theta = uniform.rvs(loc=0, scale=2*np.pi)
            noise_dir = np.array([np.cos(theta), np.sin(theta)])
            xygrad += noise_mag * noise_dir
        xynew = xyold - xygrad * lr
        xy_cont = xy_cont - np.multiply(xygrad, rbcs) * lr
        if any(xynew // np.array([M-1, M-1])):
            if len(x) != 0:
                segments.append([x, y, False])
            crosspts = _find_crossing(xyold, xynew, M)
            if BCs == 'periodic':
                xynew = xynew % np.array([M-1, M-1])
                if len(crosspts) == 2:
                    x = [xyold[0], crosspts[0][0], crosspts[1][0], xynew[0]]
                    y = [xyold[1], crosspts[0][1], crosspts[1][1], xynew[1]]
                else:
                    x = [xyold[0], crosspts[0][0], crosspts[1][0],
                         crosspts[2][0], crosspts[3][0], xynew[0]]
                    y = [xyold[1], crosspts[0][1], crosspts[1][1],
                         crosspts[2][1], crosspts[3][1], xynew[1]]
                segments.append([x, y, True])
                x = []
                y = []
            elif BCs == 'reflecting':
                for i in range(len(xynew)):
                    x_or_y = copy.deepcopy(xynew[i])
                    while x_or_y // (M-1) != 0:
                        if x_or_y < 0:
                            x_or_y *= -1
                        elif x_or_y > 0:
                            excess = x_or_y - (M-1)
                            x_or_y = (M-1) - excess
                    xynew[i] = x_or_y
        x.append(xynew[0])
        y.append(xynew[1])
        coords[niters, :] = xynew
        coords_cont[niters, :] = xy_cont
        xyold = xynew
        niters += 1

    if len(x) != 0:
        segments.append([x, y, False])

    mdic = {"landscape": landscape, "trajectory": coords,
            "segments": segments, "trajectory_continuous": coords_cont}
    if save_results:
        assert isinstance(save_path, str)
        savemat(save_path, mdic)
    return mdic


def plot_trajectory(results, linecolour='k', cmap=cm.terrain, sfcolour='r', mincolour='m', legend_loc='best', include_min=True, return_figax=False, savepath=None):
    """Plots trajectory on fractal landscape. If return_figax is True, then figure and axes are returned.

    Args:
        results (str or dict): Path to file containing dictionary results of simulation or dictionary itself.
        linecolour (str, list, optional): Colour of trajectory line. Defaults to 'k' (black). Alternatively,
            if linecolour is a list then colours are used corresponding to the regimes. E.g., ['r', 'g', 'b', ...]
        cmap (cmap, optional): Colour map from matplotlib. Defaults to cm.terrain.
        sfcolour (str, optional): Colour of markers for start and finish positions. Defaults to 'r'.
        mincolour (str, optional): Colour of marker for global minimum position. Defaults to 'm'.
        include_min (bool, optional): Whether to plot the global minimum with a marker. Defaults to True.
        return_figax (bool, optional): Whether to return the figure and axis. Defaults to False.
        savepath (str, optional): Path for saving. Defaults to None.
    """
    if isinstance(results, str):
        results = loadmat(results)
    assert isinstance(results, dict)

    landscape = results["landscape"]
    traj = results["trajectory"]
    segments = results["segments"]

    fig, ax = _plot_trajectory(
        landscape=landscape, segments=segments, cmap=cmap, linecolour=linecolour)

    # Plot markers
    ax.scatter(traj[0, 0], traj[0, 1], s=50, c=sfcolour,
               marker='^', label='Start', zorder=2, rasterized=True)
    ax.scatter(traj[-1, 0], traj[-1, 1], s=50, c=sfcolour,
               marker='v', label='Finish', zorder=2, rasterized=True)
    if include_min:
        ind = np.where(landscape == np.min(landscape))
        ax.scatter(ind[1], ind[0], s=50, c=mincolour,
                   marker='*', label='Global minimum', zorder=2, rasterized=True)

    ax.legend(loc=legend_loc)
    if type(savepath) == str:
        fig.savefig(savepath, dpi=600, bbox_inches='tight')
    else:
        print("Figure not saved as savepath was not given.")
    plt.show()
    if return_figax:
        return fig, ax


def _find_crossing(xyold, xynew, M):
    """Calculates the crossing point across the periodic boundary (for plotting purposes).
    Returns the crossing points if crossed, otherwise returns False.

    Args:
        xyold (ndarray): Old (x,y) position
        xynew (ndarray): New (x,y) position
        M (int): Size of landscape
    """
    fdiv = xynew // np.array([M-1, M-1])
    crosspt1 = np.zeros(2)
    i = np.nonzero(fdiv)[0]
    if len(i) == 1:
        if fdiv[i] > 0:
            crosspt1[i] = M-1
        elif fdiv[i] < 0:
            crosspt1[i] = 0
        m = (xynew[1] - xyold[1])/(xynew[0] - xyold[0])
        if i == 0:
            crosspt1[1] = xyold[1] + m * (crosspt1[i] - xyold[0])
        else:
            crosspt1[0] = xyold[0] + (crosspt1[i] - xyold[1])/m
        crosspt2 = _find_opposite_crosspt(crosspt1, M)
        return [crosspt1, crosspt2]
    else:
        crosspt3 = np.zeros(2)
        m1 = (xynew[1] - xyold[1])/(xynew[0] - xyold[0])
        if fdiv[0] == 1 and fdiv[1] == 1:
            m2 = (M-1 - xyold[1])/(M-1 - xyold[0])
            if m1 > m2:
                crosspt1[1] = M-1
                crosspt1[0] = xyold[0] + (crosspt1[1] - xyold[1])/m1
                crosspt3[0] = M-1
                crosspt3[1] = m1 * (M-1 - crosspt1[0])
            else:
                crosspt1[0] = M-1
                crosspt1[1] = xyold[1] + m1 * (crosspt1[0] - xyold[0])
                crosspt3[1] = M-1
                crosspt3[0] = (crosspt3[1] - crosspt1[1])/m1
        elif fdiv[0] == 1 and fdiv[1] == -1:
            m2 = (0 - xyold[1])/(M-1 - xyold[0])
            if m1 > m2:
                crosspt1[0] = M-1
                crosspt1[1] = xyold[1] + m1 * (crosspt1[0] - xyold[0])
                crosspt3[1] = 0
                crosspt3[0] = (crosspt3[1] - crosspt1[1])/m1
            else:
                crosspt1[1] = 0
                crosspt1[0] = xyold[0] + (crosspt1[1] - xyold[1])/m1
                crosspt3[0] = M-1
                crosspt3[1] = M-1 + m1 * (crosspt3[0] - crosspt1[0])
        elif fdiv[0] == -1 and fdiv[1] == 1:
            m2 = (M-1 - xyold[1])/(0 - xyold[0])
            if m1 > m2:
                crosspt1[0] = 0
                crosspt1[1] = xyold[1] + m1 * (crosspt1[0] - xyold[0])
                crosspt3[1] = M-1
                crosspt3[0] = M-1 + (crosspt3[1] - crosspt1[1])/m1
            else:
                crosspt1[1] = M-1
                crosspt1[0] = xyold[0] + (crosspt1[1] - xyold[1])/m1
                crosspt3[0] = 0
                crosspt3[1] = m1 * (crosspt3[0] - crosspt1[0])
        elif fdiv[0] == -1 and fdiv[1] == -1:
            m2 = (0 - xyold[1])/(0 - xyold[0])
            if m1 > m2:
                crosspt1[1] = 0
                crosspt1[0] = xyold[0] + (crosspt1[1] - xyold[1])/m1
                crosspt3[0] = 0
                crosspt3[1] = M-1 + m1 * (crosspt3[0] - crosspt1[0])
            else:
                crosspt1[0] = 0
                crosspt1[1] = xyold[1] + m1 * (crosspt1[0] - xyold[0])
                crosspt3[1] = 0
                crosspt3[0] = M-1 + (crosspt3[1] - crosspt1[1])/m1
        crosspt2 = _find_opposite_crosspt(crosspt1, M)
        crosspt4 = _find_opposite_crosspt(crosspt3, M)
        return [crosspt1, crosspt2, crosspt3, crosspt4]


def _find_opposite_crosspt(crosspt1, M):
    """Finds the opposite crossing point in periodic BCs given one crossing point.

    Args:
        crosspt1 (ndarray): (x,y) position of crossing point.
        M (int): Length of grid.
    """
    crosspt2 = []
    for val in crosspt1:
        if val == 0:
            crosspt2.append(M-1)
        elif val == M-1:
            crosspt2.append(0)
        else:
            crosspt2.append(val)
    return np.array(crosspt2)


def _interpolate_grad(coord, grad):
    """Calculates 2D linear interpolation from a surface.

    Args:
        coord (array): (x,y) position
        grad (list of ndarrays): output from np.gradient of fractal landscape
    """
    x = coord[0]
    y = coord[1]
    xval = grad[1][int(np.floor(y)), int(np.floor(x))] + (grad[1][int(np.floor(y)), int(
        np.ceil(x))] - grad[1][int(np.floor(y)), int(np.floor(x))]) * (x - np.floor(x))
    yval = grad[0][int(np.floor(y)), int(np.floor(x))] + (grad[0][int(np.ceil(y)), int(
        np.floor(x))] - grad[0][int(np.floor(y)), int(np.floor(x))]) * (y - np.floor(y))
    return np.array([xval, yval])


def _plot_trajectory(landscape, segments, cmap, linecolour):
    fig, ax = plt.subplots()
    ax.imshow(landscape, cmap=cmap, origin='lower', rasterized=True)
    # Plot trajectory with bold line
    for elem in segments:
        if not elem[-1]:
            ax.plot(elem[0].flatten(), elem[1].flatten(),
                    color=linecolour, linewidth=1, zorder=1)
        else:
            ax.plot(elem[0][0][:2].flatten(), elem[1][0]
                    [:2].flatten(), color=linecolour, linewidth=1, zorder=1, rasterized=True)
            ax.plot(elem[0][0][2:4].flatten(), elem[1][0]
                    [2:4].flatten(), color=linecolour, linewidth=1, zorder=1, rasterized=True)
        if len(elem[0][0]) == 6:
            ax.plot(elem[0][0][4:6].flatten(), elem[1][0]
                    [4:6].flatten(), color=linecolour, linewidth=1, zorder=1, rasterized=True)
    return fig, ax
