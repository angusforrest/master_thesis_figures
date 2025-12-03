import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (
    MultipleLocator,
    FixedLocator,
    LogLocator,
    NullFormatter,
)
import matplotlib as mpl

FONTSIZE = 13
mpl.rcParams.update(
    {
        "font.size": FONTSIZE,  # base font size (axes labels, ticks, etc.)
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
        "mathtext.fontset": "stixsans",  # makes math digits/symbols match sans
    }
)
mpl.rcParams.update(
    {
        "mathtext.fontset": "cm",  # Computer Modern
        "mathtext.rm": "serif",
        "mathtext.it": "serif:italic",
        "mathtext.bf": "serif:bold",
    }
)


def gaussian(mu, sigma, x):
    # Works for scalar or 1D array x
    coef = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    if np.isscalar(x):
        z = (x - mu) / sigma
        return coef * np.exp(-0.5 * z * z)
    else:
        out = np.empty_like(x, dtype=np.float64)
        for i in range(x.size):
            z = (x[i] - mu) / sigma
            out[i] = coef * np.exp(-0.5 * z * z)
        return out


EXP_CUTOFF = 6.97
GAUSSIAN_COEFF = 21.172
SIGMA = 1.877
MU = 4.85
EXP_COEFF = 0.1123 * GAUSSIAN_COEFF
EXP_SCALE_HEIGHT = 2.89
MAX_RADIUS = 20
MIN_RADIUS = 3


def surface_density_equation(r):
    out = np.empty_like(r)
    mask_gauss = (r >= MIN_RADIUS) & (r < EXP_CUTOFF)
    mask_exp = (r >= EXP_CUTOFF) & (r <= MAX_RADIUS)
    out[mask_gauss] = GAUSSIAN_COEFF * gaussian(MU, SIGMA, r[mask_gauss])
    out[mask_exp] = EXP_COEFF * np.exp(
        -(r[mask_exp] - EXP_CUTOFF) / EXP_SCALE_HEIGHT
    )
    return out


def template_surface_density_molecular():
    plt.subplots_adjust(
        top=0.98
    )  # increase toward 1.0 to reduce top whitespace
    plt.figure(figsize=(5, 3.5))  # width, height in inches
    plt.tight_layout()  # or fig.tight_layout()
    ax = plt.axes()
    xs = np.linspace(3, 18, 100)
    ax.set_box_aspect(3 / 5)
    ax.plot(xs, surface_density_equation(xs), linestyle="dotted", color="k")
    ax.set_xlim(0, 20)
    ax.set_ylim(1e-2, 30)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\mathrm{R}$ (kpc)")
    ax.set_ylabel(r"$\Sigma$ ($\mathrm{M}_\odot$  pc${}^{-2}$)")
    xticks = np.arange(0, 21, 5)
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_minor_locator(
        MultipleLocator(1)
    )  # optional minor ticks every 1
    ax.tick_params(
        axis="both", which="both", direction="in", top=True, right=True
    )
    ax.grid(True, alpha=0.25)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
    )
    ax.yaxis.set_minor_formatter(NullFormatter())
    # ax.text(2, 1e1, "TEMPLATE", fontsize=24)
    # ax.text(2, 1e0, "NORMAL", fontsize=24)
    # ax.text(2, 1e-1, "EXP", fontsize=24)

    plt.savefig(
        "surface_density_molecular.png",
        dpi=300,
    )
    plt.close()
    return


if __name__ == "__main__":
    template_surface_density_molecular()
