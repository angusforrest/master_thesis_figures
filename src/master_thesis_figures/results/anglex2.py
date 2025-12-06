import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
import matplotlib as mpl
import pickle


def contour(data):
    # with open("unheated_anglesData.pk", "rb") as file:
    #     data = pickle.load(file)
    theta = np.array(data[0])
    y = np.array(data[1])
    print(theta)
    print(y)

    # Compute weights to make theta (zeta) distribution uniform
    n_bins_weight = 50  # Number of bins for computing weights
    hist_theta, bin_edges = np.histogram(
        theta, bins=n_bins_weight, range=(-np.pi, np.pi)
    )

    # Assign each point to a bin
    bin_indices = np.digitize(theta, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins_weight - 1)

    # Weight is inverse of bin count (with smoothing to avoid division by zero)
    bin_counts = hist_theta[bin_indices]
    weights = np.ones_like(theta, dtype=float)
    mask = bin_counts > 0
    weights[mask] = 1.0 / bin_counts[mask]
    # Don't normalize yet - let histogram2d handle the total

    # Create 2D histogram with fewer bins
    nx, ny = 100, 75  # Reduced from 400, 300

    H, xedges, yedges = np.histogram2d(
        theta,
        y,
        bins=[nx, ny],
        range=[[-np.pi, np.pi], [-np.pi / 2, np.pi / 2]],
        weights=weights,
    )

    # Get bin centers for plotting
    ti_centers = (xedges[:-1] + xedges[1:]) / 2
    yi_centers = (yedges[:-1] + yedges[1:]) / 2
    TI, YI = np.meshgrid(ti_centers, yi_centers, indexing="xy")
    ZI = H.T  # Transpose to match meshgrid orientation

    # Smooth the histogram (optional, for better contours)
    from scipy.ndimage import gaussian_filter

    ZI = gaussian_filter(ZI, sigma=1.5)

    # Compute credible region levels
    Zflat = ZI.ravel().astype(float)
    Zsum = Zflat.sum()

    print(f"Data points: {len(theta)}")
    print(f"Histogram sum: {Zsum}")
    print(f"Histogram max: {Zflat.max()}")
    print(f"Histogram min: {Zflat.min()}")

    if not np.isfinite(Zsum) or Zsum <= 0:
        raise ValueError(f"Z has nonpositive or nonfinite total mass: {Zsum}")

    Znorm = Zflat / Zsum

    # Sort by descending density and compute cumulative probability
    order = np.argsort(Zflat)[::-1]
    Zsorted = Zflat[order]
    Pcum = np.cumsum(Znorm[order])

    # Find density thresholds for credible regions
    levels = []
    for p in [0.95, 0.5]:
        idx = np.searchsorted(Pcum, p)
        idx = min(idx, len(Zsorted) - 1)
        levels.append(Zsorted[idx])

    print(f"Levels: {levels}")

    return TI, YI, ZI, levels


FONTSIZE = 13
mpl.rcParams.update(
    {
        "font.size": FONTSIZE,
        "axes.titlesize": FONTSIZE,
        "axes.labelsize": FONTSIZE,
        "xtick.labelsize": FONTSIZE,
        "ytick.labelsize": FONTSIZE,
        "legend.fontsize": FONTSIZE,
    }
)
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Gentium Book Plus",
            "DejaVu Sans",
            "Helvetica",
            "Arial",
            "Liberation Sans",
        ],
        "mathtext.fontset": "stixsans",
    }
)
mpl.rcParams.update(
    {
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
        "mathtext.it": "serif:italic",
        "mathtext.bf": "serif:bold",
    }
)


def template_anglex2():
    fig, axs = plt.subplots(2, 1, figsize=(5, 7), sharex=True)
    axs[1].set_box_aspect(3 / 5)

    with open("unheated_anglesData.pk", "rb") as file:
        data = pickle.load(file)
    x, y, z, levels = contour(data)
    axs[0].contour(
        x, y, z, levels=levels, linestyles=["dotted", "dashed"], colors="k"
    )
    with open("anglesData.pk", "rb") as file:
        data = pickle.load(file)
    x, y, z, levels = contour(data)
    axs[1].contour(
        x, y, z, levels=levels, linestyles=["dotted", "dashed"], colors="k"
    )

    fig.supylabel(r"$\beta$ (rad)")
    fig.supxlabel(r"$\alpha$ (rad)")

    axs[1].tick_params(
        axis="both", which="both", direction="in", top=True, right=True
    )
    xlocs = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    ylocs = [-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2]
    xlabels = [r"-$\pi$", r"-$\pi/2$", r"0", r"$\pi/2$", r"$\pi$"]
    ylabels = [r"-$\pi/2$", r"-$\pi/4$", r"0", r"$\pi/4$", r"$\pi/2$"]
    for ax in axs:
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi / 2, np.pi / 2)
        ax.grid(True, alpha=0.25)
        ax.yaxis.set_major_locator(FixedLocator(ylocs))
        ax.yaxis.set_major_formatter(FixedFormatter(ylabels))

    axs[1].xaxis.set_major_locator(FixedLocator(xlocs))
    axs[1].xaxis.set_major_formatter(FixedFormatter(xlabels))

    plt.tight_layout()
    # plt.savefig("unheated_anglex2.png", dpi=300)
    plt.savefig("anglex2_2.png", dpi=300)
    plt.close()


def hist():
    with open("anglesData.pk", "rb") as file:
        data = pickle.load(file)
    theta = np.array(data[0])
    ax = plt.axes()
    ax.hist(theta, bins=50)
    plt.savefig("hist.png")
    plt.close()


if __name__ == "__main__":
    # hist()
    template_anglex2()

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import (
#     FixedLocator,
#     FixedFormatter,
# )
# import matplotlib as mpl
# from scipy.stats import gaussian_kde
# import pickle


# def contour():
#     period = 2 * np.pi
#     with open("anglesData.pk", "rb") as file:
#         data = pickle.load(file)
#     theta = np.array(data[0])
#     y = np.array(data[1])
#     # theta = np.array([1.0, 0.0, -1.0, 0.2, -0.2])
#     # y = np.array([1.0, 0.0, 1.2, 0.1, -0.1])
#     theta_tiled = np.concatenate([theta - period, theta, theta + period])
#     y_tiled = np.concatenate([y, y, y])
#     vals = np.vstack([theta_tiled, y_tiled])
#     kde = gaussian_kde(vals, bw_method=0.20)
#     nx, ny = 400, 300
#     ti = np.linspace(-np.pi, np.pi, nx)
#     yi = np.linspace(-np.pi / 2, np.pi / 2, ny)
#     TI, YI = np.meshgrid(ti, yi, indexing="xy")
#     ZI = kde(np.vstack([TI.ravel(), YI.ravel()])).reshape(TI.shape)
#     Zflat = ZI.ravel().astype(float)
#     Zsum = Zflat.sum()
#     if not np.isfinite(Zsum) or Zsum <= 0:
#         raise ValueError("Z has nonpositive or nonfinite total mass.")
#     Znorm = Zflat / Zsum

#     # Sort by descending density and compute cumulative probability
#     order = np.argsort(Zflat)[::-1]
#     Zsorted = Zflat[order]
#     Pcum = np.cumsum(Znorm[order])

#     # For each target probability p, find the smallest density threshold
#     # such that mass above it is >= p
#     levels = []
#     for p in [0.95, 0.5]:
#         idx = np.searchsorted(Pcum, p)
#         idx = min(idx, len(Zsorted) - 1)
#         levels.append(Zsorted[idx])
#     return TI, YI, ZI, levels


# FONTSIZE = 13
# mpl.rcParams.update(
#     {
#         "font.size": FONTSIZE,  # base font size (axes labels, ticks, etc.)
#         "axes.titlesize": FONTSIZE,
#         "axes.labelsize": FONTSIZE,
#         "xtick.labelsize": FONTSIZE,
#         "ytick.labelsize": FONTSIZE,
#         "legend.fontsize": FONTSIZE,
#     }
# )
# mpl.rcParams.update(
#     {
#         "font.family": "sans-serif",
#         "font.sans-serif": [
#             "Gentium Book Plus",
#             "DejaVu Sans",
#             "Helvetica",
#             "Arial",
#             "Liberation Sans",
#         ],
#         "mathtext.fontset": "stixsans",  # makes math digits/symbols match sans
#     }
# )
# mpl.rcParams.update(
#     {
#         "mathtext.fontset": "cm",  # Computer Modern
#         "mathtext.rm": "serif",
#         "mathtext.it": "serif:italic",
#         "mathtext.bf": "serif:bold",
#     }
# )


# def template_anglex2():
#     plt.subplots_adjust(
#         top=0.98
#     )  # increase toward 1.0 to reduce top whitespace
#     plt.figure(figsize=(5, 3.5))  # width, height in inches
#     plt.tight_layout()  # or fig.tight_layout()
#     ax = plt.axes()
#     ax.set_box_aspect(3 / 5)
#     # ax.scatter(DATA[:, 0], DATA[:, 1], color="k")
#     x, y, z, levels = contour()
#     ax.contour(
#         x, y, z, levels=levels, linestyles=["dotted", "dashed"], colors="k"
#     )
#     ax.set_xlim(-np.pi, np.pi)
#     ax.set_ylim(-np.pi / 2, np.pi / 2)
#     ax.set_xlabel(r"$\zeta$ (rad)")
#     ax.set_ylabel(r"$\xi$ (rad)")
#     # ax.xaxis.set_minor_locator(
#     #     MultipleLocator(1)
#     # )
#     ax.tick_params(
#         axis="both", which="both", direction="in", top=True, right=True
#     )
#     ax.grid(True, alpha=0.25)
#     xlocs = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
#     ylocs = [-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2]

#     # Matching labels
#     xlabels = [
#         r"-$\pi$",
#         r"-$\pi/2$",
#         r"0",
#         r"$\pi/2$",
#         r"$\pi$",
#     ]
#     ylabels = [r"-$\pi/2$", r"-$\pi/4$", r"0", r"$\pi/4$", r"$\pi/2$"]

#     ax.xaxis.set_major_locator(FixedLocator(xlocs))
#     ax.xaxis.set_major_formatter(FixedFormatter(xlabels))

#     ax.yaxis.set_major_locator(FixedLocator(ylocs))
#     ax.yaxis.set_major_formatter(FixedFormatter(ylabels))
#     # ax.yaxis.set_minor_locator(MultipleLocator(1))
#     # ax.yaxis.set_minor_formatter(NullFormatter())
#     # ax.text(2, 1e1, "template", fontsize=24)
#     # ax.text(2, 1e0, "NORMAL", fontsize=24)
#     # ax.text(2, 1e-1, "EXP", fontsize=24)

#     plt.savefig(
#         "anglex2.png",
#         dpi=300,
#     )
#     plt.close()


# def hist():
#     with open("anglesData.pk", "rb") as file:
#         data = pickle.load(file)
#     theta = np.array(data[0])
#     # y = np.array(data[1])
#     ax = plt.axes()
#     ax.hist(theta, bins=50)
#     plt.savefig("hist.png")
#     plt.close()
#     return


# if __name__ == "__main__":
#     hist()
# template_anglex2()
