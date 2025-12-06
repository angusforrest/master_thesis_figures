import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (
    FixedLocator,
    FixedFormatter,
)
import matplotlib as mpl
from scipy.stats import gaussian_kde
import pickle

from matplotlib.patches import Arc, Ellipse
from matplotlib.transforms import Bbox, IdentityTransform, TransformedBbox


class AngleAnnotation(Arc):
    """
    Draws an arc between two vectors which appears circular in display space.
    """

    def __init__(
        self,
        xy,
        p1,
        p2,
        size=75,
        unit="points",
        ax=None,
        text="",
        textposition="inside",
        text_kw=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        xy, p1, p2 : tuple or array of two floats
            Center position and two points. Angle annotation is drawn between
            the two vectors connecting *p1* and *p2* with *xy*, respectively.
            Units are data coordinates.

        size : float
            Diameter of the angle annotation in units specified by *unit*.

        unit : str
            One of the following strings to specify the unit of *size*:

            * "pixels": pixels
            * "points": points, use points instead of pixels to not have a
              dependence on the DPI
            * "axes width", "axes height": relative units of Axes width, height
            * "axes min", "axes max": minimum or maximum of relative Axes
              width, height

        ax : `matplotlib.axes.Axes`
            The Axes to add the angle annotation to.

        text : str
            The text to mark the angle with.

        textposition : {"inside", "outside", "edge"}
            Whether to show the text in- or outside the arc. "edge" can be used
            for custom positions anchored at the arc's edge.

        text_kw : dict
            Dictionary of arguments passed to the Annotation.

        **kwargs
            Further parameters are passed to `matplotlib.patches.Arc`. Use this
            to specify, color, linewidth etc. of the arc.

        """
        self.ax = ax or plt.gca()
        self._xydata = xy  # in data coordinates
        self.vec1 = p1
        self.vec2 = p2
        self.size = size
        self.unit = unit
        self.textposition = textposition

        super().__init__(
            self._xydata,
            size,
            size,
            angle=0.0,
            theta1=self.theta1,
            theta2=self.theta2,
            **kwargs,
        )

        self.set_transform(IdentityTransform())
        self.ax.add_patch(self)

        self.kw = dict(
            ha="center",
            va="center",
            xycoords=IdentityTransform(),
            xytext=(0, 0),
            textcoords="offset points",
            annotation_clip=True,
        )
        self.kw.update(text_kw or {})
        self.text = ax.annotate(text, xy=self._center, **self.kw)

    def get_size(self):
        factor = 1.0
        if self.unit == "points":
            factor = self.ax.figure.dpi / 72.0
        elif self.unit[:4] == "axes":
            b = TransformedBbox(Bbox.unit(), self.ax.transAxes)
            dic = {
                "max": max(b.width, b.height),
                "min": min(b.width, b.height),
                "width": b.width,
                "height": b.height,
            }
            factor = dic[self.unit[5:]]
        return self.size * factor

    def set_size(self, size):
        self.size = size

    def get_center_in_pixels(self):
        """return center in pixels"""
        return self.ax.transData.transform(self._xydata)

    def set_center(self, xy):
        """set center in data coordinates"""
        self._xydata = xy

    def get_theta(self, vec):
        vec_in_pixels = self.ax.transData.transform(vec) - self._center
        return np.rad2deg(np.arctan2(vec_in_pixels[1], vec_in_pixels[0]))

    def get_theta1(self):
        return self.get_theta(self.vec1)

    def get_theta2(self):
        return self.get_theta(self.vec2)

    def set_theta(self, angle):
        pass

    # Redefine attributes of the Arc to always give values in pixel space
    _center = property(get_center_in_pixels, set_center)
    theta1 = property(get_theta1, set_theta)
    theta2 = property(get_theta2, set_theta)
    width = property(get_size, set_size)
    height = property(get_size, set_size)

    # The following two methods are needed to update the text position.
    def draw(self, renderer):
        self.update_text()
        super().draw(renderer)

    def update_text(self):
        c = self._center
        s = self.get_size()
        angle_span = (self.theta2 - self.theta1) % 360
        angle = np.deg2rad(self.theta1 + angle_span / 2)
        r = s / 2
        if self.textposition == "inside":
            r = s / np.interp(
                angle_span, [60, 90, 135, 180], [3.3, 3.5, 3.8, 4]
            )
        self.text.xy = c + r * np.array([np.cos(angle), np.sin(angle)])
        if self.textposition == "outside":

            def R90(a, r, w, h):
                if a < np.arctan(h / 2 / (r + w / 2)):
                    return np.sqrt(
                        (r + w / 2) ** 2 + (np.tan(a) * (r + w / 2)) ** 2
                    )
                else:
                    c = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
                    T = np.arcsin(
                        c * np.cos(np.pi / 2 - a + np.arcsin(h / 2 / c)) / r
                    )
                    xy = r * np.array([np.cos(a + T), np.sin(a + T)])
                    xy += np.array([w / 2, h / 2])
                    return np.sqrt(np.sum(xy**2))

            def R(a, r, w, h):
                aa = (a % (np.pi / 4)) * ((a % (np.pi / 2)) <= np.pi / 4) + (
                    np.pi / 4 - (a % (np.pi / 4))
                ) * ((a % (np.pi / 2)) >= np.pi / 4)
                return R90(aa, r, *[w, h][:: int(np.sign(np.cos(2 * a)))])

            bbox = self.text.get_window_extent()
            X = R(angle, r, bbox.width, bbox.height)
            trans = self.ax.figure.dpi_scale_trans.inverted()
            offs = trans.transform(((X - s / 2), 0))[0] * 72
            self.text.set_position([offs * np.cos(angle), offs * np.sin(angle)])


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


def template_anglex2():
    plt.subplots_adjust(
        top=0.98
    )  # increase toward 1.0 to reduce top whitespace
    plt.figure(figsize=(5, 3.5))  # width, height in inches
    plt.tight_layout()  # or fig.tight_layout()
    ax = plt.axes()
    ax.set_box_aspect(3 / 5)
    ax
    # ax.scatter(DATA[:, 0], DATA[:, 1], color="k")
    ax.plot(0, 0, c="k")
    # ax.text(0, 0, "Guiding Centre", va="top")
    ax.arrow(0, 0, 10, 10)
    ax.arrow(0, 0, 5, 0)
    ax.arrow(10, 10, 5, 0)
    ax.arrow(10, 10, 6 * np.cos(-np.pi / 4), 6 * np.sin(-np.pi / 4))

    AngleAnnotation([0, 0], [1, 0], [1, 1], ax=ax, size=130, text=r"$\alpha$")
    AngleAnnotation(
        (10, 10),
        (10 + 6 * np.cos(-np.pi / 4), 10 + 6 * np.sin(-np.pi / 4)),
        (12, 10),
        ax=ax,
        size=130,
        text=r"$\beta$",
    )

    ax.set_xlim(-25, 25)
    ax.set_ylim(-15, 15)
    ax.set_xlabel(r"$U$ (km/s)")
    ax.set_ylabel(r"$V-\mathcal{G}(x)$ (km/s)")
    ax.add_patch(
        Ellipse((10, 10), 10, 5, angle=-45, fc="None", edgecolor="k", lw=1)
    )
    ax.add_patch(
        Ellipse(
            (0, 0),
            40,
            22,
            angle=0,
            fc="None",
            edgecolor="k",
            lw=1,
            linestyle="dashed",
            alpha=0.2,
        )
    )
    # ax.xaxis.set_minor_locator(
    #     MultipleLocator(1)
    # )
    ax.tick_params(
        axis="both", which="both", direction="in", top=True, right=True
    )
    ax.grid(True, alpha=0.25)
    xlocs = [-35, 0, 35]
    ylocs = [-25, 0, 25]

    # Matching labels
    # xlabels = [
    #     r"-$\pi$",
    #     r"-$\pi/2$",
    #     r"0",
    #     r"$\pi/2$",
    #     r"$\pi$",
    # ]
    # ylabels = [r"-$\pi/2$", r"-$\pi/4$", r"0", r"$\pi/4$", r"$\pi/2$"]

    ax.xaxis.set_major_locator(FixedLocator(xlocs))
    # ax.xaxis.set_major_formatter(FixedFormatter(xlabels))

    ax.yaxis.set_major_locator(FixedLocator(ylocs))
    # ax.yaxis.set_major_formatter(FixedFormatter(ylabels))
    # ax.yaxis.set_minor_locator(MultipleLocator(1))
    # ax.yaxis.set_minor_formatter(NullFormatter())
    # ax.text(2, 1e1, "template", fontsize=24)
    # ax.text(2, 1e0, "NORMAL", fontsize=24)
    # ax.text(2, 1e-1, "EXP", fontsize=24)

    plt.savefig(
        "diagram.png",
        dpi=300,
    )
    plt.close()


def hist():
    with open("anglesData.pk", "rb") as file:
        data = pickle.load(file)
    theta = np.array(data[0])
    # y = np.array(data[1])
    ax = plt.axes()
    ax.hist(theta, bins=50)
    plt.savefig("hist.png")
    plt.close()
    return


if __name__ == "__main__":
    # hist()
    template_anglex2()
