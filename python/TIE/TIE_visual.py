# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 12:20:00 2021
@author: Anna Rauch

This script contains a set of functions/methods that are useful to
display and visualize the TIE-results.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from TIE import TIE_general as TIEgen
from TIE import TIE_classes as TIEclass
from mayavi import mlab
import matplotlib.patches as mpatches


def bed2cmap(BEDrst: np.ndarray, legendfile: str, leg_labels: bool = False) -> tuple:
    """
    Color Information For Bedrock.

    Transforms the raw-bedrock raster in a scalar matrix and extracts based
    on a prepared textfile the color (rgb-values) of the litho-stratigraphic
    units present and re-structures it in a cmap-matrix. If the legendfile
    contains label information, you can also extract the legend string values.

    Parameters
    ----------
    BEDrst : numpy.ndarray
        Bedrock raster matrix.
    legendfile : str
        Tab-separated textfile [bedrock_ID, R, G, B, label].
    leg_labels : bool, optional
        Boolean indicating whether to extract labels (e.g., 'Schrattenkalk') or not.
        Default is False.

    Returns
    -------
    tuple
        cm : numpy.ndarray
            Scalar matrix.
        cmap : numpy.ndarray
            Colormap for bedrock information.
        formations : list
            List of labels for legend. If leg_labels==False, then formations corresponds to the original Bedrock ID.
    """
    kind = np.unique(BEDrst.flatten())
    kind = np.extract(np.isnan(kind) == False, kind)
    cm = np.zeros(np.size(BEDrst.flatten()))
    leg = np.loadtxt(legendfile, usecols=(0, 1, 2, 3))
    k_col = np.zeros(np.shape(kind))

    for k in range(len(kind)):
        i = (BEDrst.flatten() == kind[k]).nonzero()
        cm[i] = k + 1
        k_col[k] = (leg[:, 0] == kind[k]).nonzero()[0]

    cm = cm.reshape(np.shape(BEDrst))
    cm = np.fliplr(cm)
    cmap = np.zeros((255, 4))

    if any(np.isnan(BEDrst.flatten())):
        c_step = np.linspace(0, 255, np.size(k_col) + 2)
        c_step = np.round(c_step, 0).astype(int)
        cmap[0 : int(c_step[1]), 0:3] = np.ones((int(c_step[1]), 3)) * [230, 230, 230]

        for ci in range(1, len(c_step) - 1):
            ind1 = int(c_step[ci])
            ind2 = int(c_step[ci + 1])
            rgb_c = leg[int(k_col[ci - 1]), 1:4]
            cmap[ind1:ind2, 0:3] = np.ones((ind2 - ind1, 3)) * rgb_c
    else:
        c_step = np.linspace(0, 255, np.size(k_col) + 1)
        c_step = np.round(c_step, 0).astype(int)

        for ci in range(len(c_step) - 1):
            ind1 = int(c_step[ci])
            ind2 = int(c_step[ci + 1])
            rgb_c = leg[int(k_col[ci]), 1:4]
            cmap[ind1:ind2, 0:3] = np.ones((ind2 - ind1, 3)) * rgb_c
    cmap[:, 3] = (np.ones((255, 1)) * 255).flatten()

    if leg_labels == True:
        f = open(legendfile)
        lines = f.readlines()
        lines = [lines[int(ki)] for ki in k_col]
        tokens_column_number = 4
        formations = []
        if any(np.isnan(BEDrst.flatten())):
            formations.append("Unconsolidated Deposit")
        for x in lines:
            formations.append(x.split("\t")[tokens_column_number][:-1])
        f.close()
    else:
        formations = [str(ki) for ki in kind]

    return cm, cmap, formations


def colorClassID(segment_OBJ: TIEclass.segment_OBJ) -> list:
    """
    Color Information Trace Classification.

    Extracts color code for TIE-classification.

    Parameters
    ----------
    segment_OBJ : TIEclass.segment_OBJ
        Segment object.

    Returns
    -------
    list
        Normalized RGB-value corresponding to TIE-class.
    """

    cmapblue = np.flipud(
        [
            [
                0,
                1,
                1,
            ],
            [0, 0.7, 1],
            [0, 0.4, 1],
            [0.3, 0, 0.8],
        ]
    )
    cmapred = np.flipud(
        [
            [0.98, 0.7, 0.9],
            [255 / 255, 100 / 255, 180 / 255],
            [216 / 255, 0, 113 / 255],
            [0.616, 0.106, 0.286],
        ]
    )
    classID = segment_OBJ.classID
    if len(segment_OBJ.ind_normal) < 50:
        c_code = [0.4, 0.4, 0.4]
    else:
        if classID > 0:
            c_code = cmapblue[classID - 1]
        else:
            c_code = cmapred[classID * -1 - 1]
    return c_code


def createCmap(rgbv: np.ndarray, n: int) -> np.ndarray:
    """
    Create Cmap.

    Creates a continuous color range (n values) from one rgb value (rgbv) to the other(s).

    Parameters
    ----------
    rgbv : numpy.ndarray
        Vector containing rgb-values of endmembers.
    n : int
        Resolution of cmap (number of values in cmap).

    Returns
    -------
    numpy.ndarray
        Newly created cmap.
    """

    cmap = np.zeros([n, 3])
    nint = int(np.round(n / (len(rgbv) - 1), 0))
    cint = np.concatenate((np.arange(0, n, nint), np.array(n)), axis=None)
    if len(rgbv) == 1:
        TIEgen.Mbox(
            "SORRY", "at least two rgb values are necessary to crate a colormap", 1
        )
    else:
        for j in range(len(rgbv) - 1):
            cint1 = cint[j]
            cint2 = cint[j + 1]
            dif = cint2 - cint1
            cstp = np.array(
                [
                    (rgbv[j + 1, 0] - rgbv[j, 0]) / (dif - 1),
                    (rgbv[j + 1, 1] - rgbv[j, 1]) / (dif - 1),
                    (rgbv[j + 1, 2] - rgbv[j, 2]) / (dif - 1),
                ]
            )
            if cstp[0] != 0:
                cmap1 = np.linspace(rgbv[j, 0], rgbv[j + 1, 0], dif).reshape((dif, 1))
            else:
                cmap1 = np.ones((dif, 1)) * rgbv[j, 0]

            if cstp[1] != 0:
                cmap2 = np.linspace(rgbv[j, 1], rgbv[j + 1, 1], dif).reshape((dif, 1))
            else:
                cmap2 = np.ones((dif, 1)) * rgbv[j, 1]

            if cstp[2] != 0:
                cmap3 = np.linspace(rgbv[j, 2], rgbv[j + 1, 2], dif).reshape((dif, 1))
            else:
                cmap3 = np.ones((dif, 1)) * rgbv[j, 2]
            cmap[cint[j] : cint[j + 1], :] = np.concatenate(
                (cmap1, cmap2, cmap3), axis=1
            )
    return cmap


def showOrientBars(traces, vx, vy, vz, stereo=False):
    """
    Show orientation bars along traces in 3D or stereo.

    Parameters
    ----------
    traces : list of trace_OBJ (TIE_classes.trace_OBJ)
        List of trace objects.
    vx, vy, vz : numpy.ndarray
        Flattened x-, y-, and z-coordinate matrices.
    stereo : bool, optional
        Boolean indicating whether to display the bars in stereo.
        If True, the bar length is stereographically projected on the map.
        Default is False.

    Returns
    -------
    bar : mayavi.modules.surface.Surface
        Handle to the plot.
    """

    for T in traces:
        index = T.index
        or_bar = T.orientbar
        cs = vx[0] - vx[1]
        barlength = 200 / cs

        for m in range(0, len(or_bar), int(20 / cs)):
            if stereo == True:
                tr, pl = TIEgen.vect2angle(or_bar[m])
                x1, y1 = TIEgen.stereoLine(tr, pl)
                bars = mlab.plot3d(
                    [vx[int(index[m])], vx[int(index[m])] + x1 * barlength],
                    [vy[int(index[m])], vy[int(index[m])] + y1 * barlength],
                    [vz[int(index[m])], vz[int(index[m])]],
                    tube_radius=3,
                    color=(0, 0, 0),
                )
            else:
                bars = mlab.plot3d(
                    [vx[int(index[m])], vx[int(index[m])] + or_bar[m][0] * barlength],
                    [vy[int(index[m])], vy[int(index[m])] + or_bar[m][1] * barlength],
                    [vz[int(index[m])], vz[int(index[m])] + or_bar[m][2] * barlength],
                    tube_radius=3,
                    color=(0, 0, 0),
                )
    return bars


def showOverview3D(
    mx,
    my,
    mz,
    mc,
    cmap=[],
    BTraces=[],
    FTraces=[],
    ATraces=[],
    WithlabelsB=True,
    WithlabelsF=True,
    WithlabelsA=True,
    Textsize=30,
):
    """Show Geological Map In 3d (No Tie)

    Displays the geological map in 3D, as well as the different trace sets and
    their ID (yet, no TIE classification).

    Parameters
    ----------
    mx, my, mz : array-like
        Coordinate matrices of the analyzed extent.
    mc : array-like
        Scalar field of bedrock id's.
    cmap : list, optional
        Bedrock colormap. Default is Mayavi's default cmap.
    BTraces : list, optional
        List of trace_OBJ of bedrock interface traces. Default is an empty list.
    FTraces : list, optional
        List of trace_OBJ of faults (tectonic boundaries). Default is an empty list.
    ATraces : list, optional
        List of trace_OBJ of axial traces. Default is an empty list.
    Textsize : int, optional
        Text size of trace labels. Default is 30.
    WithlabelsB : bool, optional
        Boolean indicating whether to add trace ID on the figure for bedrock traces.
        Default is True.
    WithlabelsF : bool, optional
        Boolean indicating whether to add trace ID on the figure for fault traces.
        Default is True.
    WithlabelsA : bool, optional
        Boolean indicating whether to add trace ID on the figure for axial traces.
        Default is True.

    Returns
    -------
    fig : mayavi.mlab.Figure
        Handle to the figure.
    """

    fig = mlab.figure(1, fgcolor=(1, 1, 1), bgcolor=(0.9, 0.9, 0.9))
    mesh = mlab.mesh(mx, my, mz, scalars=mc)
    if len(cmap) > 0:
        mesh.module_manager.scalar_lut_manager.lut.table = cmap

    vx = np.fliplr(mx).flatten()
    vy = my.flatten()
    vz = np.fliplr(mz).flatten()

    for tr in BTraces:
        mlab.plot3d(
            vx[tr.index.astype(int)],
            vy[tr.index.astype(int)],
            vz[tr.index.astype(int)],
            color=(0, 0, 0.8),
            tube_radius=2,
            name="bedrock trace " + str(tr.id),
        )
        if WithlabelsB:
            mlab.text3d(
                vx[tr.index.astype(int)[4]],
                vy[tr.index.astype(int)[4]],
                vz[tr.index.astype(int)[4]] + 70,
                str(tr.id),
                color=(0, 0, 0.8),
                scale=Textsize,
            )

    for fl in FTraces:
        mlab.plot3d(
            vx[fl.index.astype(int)],
            vy[fl.index.astype(int)],
            vz[fl.index.astype(int)],
            color=(1, 0, 0),
            tube_radius=2,
            name="fault trace " + str(fl.id),
        )
        if WithlabelsF:
            mlab.text3d(
                vx[fl.index.astype(int)[4]],
                vy[fl.index.astype(int)[4]],
                vz[fl.index.astype(int)[4]] + 70,
                str(fl.id),
                color=(1, 0, 0),
                scale=Textsize,
            )

    for ai in ATraces:
        mlab.plot3d(
            vx[ai.index.astype(int)],
            vy[ai.index.astype(int)],
            vz[ai.index.astype(int)],
            color=(1, 0, 0),
            tube_radius=1,
            name="axial trace " + str(ai.id),
        )
        if WithlabelsA:
            mlab.text3d(
                vx[ai.index.astype(int)[4]],
                vy[ai.index.astype(int)[4]],
                vz[ai.index.astype(int)[4]] + 70,
                str(ai.id),
                color=(1, 1, 0),
                scale=Textsize,
            )

    mlab.draw()
    mlab.show()
    return fig


def showOverviewMap(
    mx,
    my,
    mc,
    cmap=[],
    mz=[],
    BTraces=[],
    FTraces=[],
    ATraces=[],
    leg_labels=[],
    WithSeg=True,
):
    """Show Geological Map In 2d As An Overview

    Displays the geological map in 2D (with legend), as well as different
    trace sets and their ID (yet, no TIE classification).

    Parameters
    ----------
    mx, my, mz : array-like
        Coordinate matrices of the analyzed extent.
    mc : array-like
        Scalar field of bedrock id's.
    cmap : list, optional
        Bedrock colormap. Default is Matplotlib's default cmap.
    BTraces : list, optional
        List of trace_OBJ of bedrock interface traces. Default is an empty list.
    FTraces : list, optional
        List of trace_OBJ of faults (tectonic boundaries). Default is an empty list.
    ATraces : list, optional
        List of trace_OBJ of axial traces. Default is an empty list.
    leg_labels : list, optional
        List of labels for bedrock units. Default is an empty list.
    WithSeg : bool, optional
        Boolean - show segmentation. Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Handle to the figure.
    ax : matplotlib.axes._axes.Axes
        Handle to the axes.
    """
    fig, ax = plt.subplots()
    ax.tick_params(labelsize=6)

    if len(cmap) > 0:
        from matplotlib.colors import ListedColormap

        cmap_plt = ListedColormap(cmap / 255)
    else:
        cmap_plt = "Greens"

    vx = np.fliplr(mx).flatten()
    vy = my.flatten()
    ax.imshow(np.flipud(mc), cmap=cmap_plt, extent=(vx[-1], vx[0], vy[0], vy[-1]))

    if len(BTraces) > 0:
        for tr in BTraces:
            tx = vx[tr.index.astype(int)]
            ty = vy[tr.index.astype(int)]

            ax.plot(tx, ty, c="b")
            txt_name = str(tr.id)
            ax.text(
                tx[2],
                ty[2],
                txt_name,
                fontsize=6,
                fontweight="bold",
                c="b",
                bbox=dict(boxstyle="circle,pad=0", fc=(1, 1, 1, 0.7), lw=0),
            )
            if WithSeg == True:
                for s in tr.Segment:
                    si1 = int(s.ind_normal[0])
                    si2 = int(s.ind_normal[-1])
                    ax.scatter(tx[si1], ty[si1], s=1, c="k", zorder=3)
                    ax.scatter(tx[si2], ty[si2], s=1, c="k", zorder=3)
                    if len(tr.Segment) > 1:
                        simid = int(
                            s.ind_normal[0] + np.round(np.size(s.ind_normal) / 2)
                        )
                        seg_name = "(" + str(s.id) + ")"
                        ax.text(
                            tx[simid],
                            ty[simid],
                            seg_name,
                            fontsize=4,
                            fontweight="normal",
                            c="b",
                            bbox=dict(boxstyle="circle,pad=0", fc=(1, 1, 1, 0.7), lw=0),
                        )
    if len(FTraces) > 0:
        for fl in FTraces:
            fx = vx[fl.index.astype(int)]
            fy = vy[fl.index.astype(int)]
            ax.plot(fx, fy, c="r")
            txt_name = str(fl.id)
            ax.text(
                fx[2],
                fy[2],
                txt_name,
                fontsize=6,
                fontweight="bold",
                c="r",
                bbox=dict(boxstyle="circle,pad=0", fc=(1, 1, 1, 0.7), lw=0),
            )
            if WithSeg == True:
                for f in fl.Segment:
                    si1 = int(f.ind_normal[0])
                    si2 = int(f.ind_normal[-1])
                    simid = int(f.ind_normal[0] + np.round(np.size(f.ind_normal) / 2))
                    ax.scatter(fx[si1], fy[si1], s=1, c="k", zorder=3)
                    ax.scatter(fx[si2], fy[si2], s=1, c="k", zorder=3)
                    if len(fl.Segment) > 1:
                        seg_name = "(" + str(f.id) + ")"
                        ax.text(
                            fx[simid],
                            fy[simid],
                            seg_name,
                            fontsize=4,
                            fontweight="normal",
                            c="r",
                            bbox=dict(boxstyle="circle,pad=0", fc=(1, 1, 1, 0.7), lw=0),
                        )
    if len(ATraces) > 0:
        for atr in ATraces:
            atx = vx[atr.index.astype(int)]
            aty = vy[atr.index.astype(int)]
            ax.plot(atx, aty, c="y", linewidth=0.5)
            txt_name = str(atr.id)
            ax.text(
                atx[2],
                aty[2],
                txt_name,
                fontsize=6,
                fontweight="bold",
                c="y",
                bbox=dict(boxstyle="circle,pad=0", fc=(1, 1, 1, 0.7), lw=0),
            )
            if WithSeg == True:
                for s in atr.Segment:
                    si1 = int(s.ind_normal[0])
                    si2 = int(s.ind_normal[-1])
                    simid = int(s.ind_normal[0] + np.round(np.size(s.ind_normal) / 2))
                    ax.scatter(atx[si1], aty[si1], s=1, c="k", zorder=3)
                    ax.scatter(atx[si2], aty[si2], s=1, c="k", zorder=3)
                    if len(atr.Segment) > 1:
                        seg_name = "(" + str(s.id) + ")"
                        ax.text(
                            atx[simid],
                            aty[simid],
                            seg_name,
                            fontsize=4,
                            fontweight="normal",
                            c="y",
                            bbox=dict(boxstyle="circle,pad=0", fc=(1, 1, 1, 0.7), lw=0),
                        )
    if len(mz) > 0:
        levmin = (np.min(mz) - np.min(mz) % 10) + 10
        levmax = np.max(mz) - np.min(mz) % 10
        ax.contour(
            mz,
            extent=(vx[-1], vx[0], vy[0], vy[-1]),
            levels=np.arange(levmin, levmax, 20),
            colors="k",
            linewidths=0.1,
        )

    if len(leg_labels) > 0:
        indexes = np.unique(cmap, return_index=True, axis=0)
        cmap_leg = [cmap[index] / 255 for index in sorted(indexes[1])]
        patches = [
            mpatches.Patch(color=cmap_leg[i], label=leg_labels[i])
            for i in range(len(leg_labels))
        ]
        plt.legend(
            handles=patches,
            bbox_to_anchor=(1.05, 1),
            loc=2,
            borderaxespad=0.0,
            fontsize=6,
        )

    return fig, ax


def showSignal(trace_OBJ):
    """Show Trace Signals

    Displays signals of a specific trace. First subplot: major signals -
    alpha (normal and reverse) and beta (normal and reverse) and their
    signal heights. Second subplot: signal of the orthogonal distance
    between chords forming a chord plane (normalized to the total length of
    the trace).

    Parameters
    ----------
    trace_OBJ : trace object (TIE_classes.trace_OBJ)
        The trace object for which signals are to be displayed.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Handle to the figure.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    for s in trace_OBJ.Segment:
        Alpha = [chord.alpha[0] for chord in s.Chords]
        Beta = [chdplane.beta[0] for chdplane in s.Chordplanes]
        Dist = [chdplane.dist_ratio[0] for chdplane in s.Chordplanes]
        AlphaR = [chord.alpha[1] for chord in s.Chords]
        BetaR = [chdplane.beta[1] for chdplane in s.Chordplanes]
        ha = s.sigheight[0]
        hb = s.sigheight[1]

        l = len(s.ind_normal)
        la = len(Alpha)
        lb = len(Beta)

        lbeg = 100 * (s.id - 1)
        lend = 100 * s.id

        if s.id == 1:
            ax1.plot(
                np.linspace(lbeg, lend, la),
                Alpha,
                "g-",
                label="alpha normal",
                linewidth=1.5,
            )
            ax1.plot(
                np.linspace(lbeg, lend, la),
                AlphaR,
                "g-.",
                label="alpha reverse",
                linewidth=1.5,
            )
            ax1.plot([lbeg, lend], [ha, ha], "g:", linewidth=0.5, label="ha")
            ax1.plot(
                np.linspace(lbeg, lend, lb),
                Beta,
                "y-",
                label="beta normal",
                linewidth=1.5,
            )
            ax1.plot(
                np.linspace(lbeg, lend, lb),
                BetaR,
                "y-.",
                label="beta reverse",
                linewidth=1.5,
            )
            ax1.plot([lbeg, lend], [hb, hb], "y:", linewidth=0.5, label="hb")
        else:
            ax1.plot(np.linspace(lbeg, lend, la), Alpha, "g-", linewidth=1.5)
            ax1.plot(np.linspace(lbeg, lend, la), AlphaR, "g-.", linewidth=1.5)
            ax1.plot([lbeg, lend], [ha, ha], "g:", linewidth=0.5)
            ax1.plot(np.linspace(lbeg, lend, lb), Beta, "y-", linewidth=1.5)
            ax1.plot(np.linspace(lbeg, lend, lb), BetaR, "y-.", linewidth=1.5)
            ax1.plot([lbeg, lend], [hb, hb], "y:", linewidth=0.5)

        ax1.plot([lbeg, lbeg], [0, 180], "k:", linewidth=0.5)
        ax1.plot([lend, lend], [0, 180], "k:", linewidth=0.5)

        tit = "signal trace n° " + str(trace_OBJ.id)
        ax1.set_title(tit)
        ax1.set_xlabel("j / k [% of pixel length (m / l)]")
        ax1.set_ylabel("angle degrees [°]")
        ax1.text(
            lend + 0.4,
            30,
            "Seg. " + str(s.id) + " (" + str(l) + "pix)",
            rotation="vertical",
            fontsize=8,
            bbox=dict(boxstyle="round4,pad=0.1", fc=(1, 1, 1, 1), lw=0),
        )
        ax1.axis([-3, lend + 3, -1, 182])

        if s.id == 1:
            ax2.plot(
                np.linspace(lbeg, lend, lb),
                np.array(Dist) * 100,
                "y-",
                linewidth=1.5,
                label="orthogonal distance btw chords",
            )
        else:
            ax2.plot(
                np.linspace(lbeg, lend, lb), np.array(Dist) * 100, "y-", linewidth=1.5
            )
        ax2.set_xlabel("k in % of trace length (l)")
        ax2.set_ylabel("d in % of trace length")

        ax2.plot([lbeg, lbeg], [0, 10], "k:", linewidth=0.5)
        ax2.plot([lend, lend], [0, 10], "k:", linewidth=0.5)
        ax2.plot([lbeg, lend], [2, 2], "k", linewidth=0.8)
        ax2.plot([lbeg, lend], [5, 5], "k", linewidth=0.8)
        ax2.plot([lbeg, lend], [10, 10], "k", linewidth=0.8)
        ax2.text(
            lend + 0.4,
            2,
            "Seg. " + str(s.id) + " (" + str(l) + "pix)",
            rotation="vertical",
            fontsize=8,
            bbox=dict(boxstyle="round4,pad=0.1", fc=(1, 1, 1, 1), lw=0),
        )

    ax1.legend(loc="upper left", fontsize="xx-small")
    ax2.legend(loc="upper left", fontsize="xx-small")

    ax2.text(0.4, 1.2, "chord planes have physical meaning", fontsize=6)
    ax2.text(0.4, 4.2, "only vague physical meaning", fontsize=6)
    ax2.text(0.4 + 0.2, 8.2, "no physical meaning", fontsize=6)
    plt.xticks(
        np.arange(0, len(trace_OBJ.Segment) * 100 + 100, 100),
        [0] + [100] * (len(trace_OBJ.Segment)),
    )
    ax2.axis([-3, lend + 3, 0, 11])

    plt.show
    return fig


def showSigStereo(trace_OBJ):
    """Show Trace Signals As Stereographic Projection

    Displays the evolution of the chords (1st subplot) and chord planes
    (2nd plot) of a specific trace by means of a stereographic projection.

    Parameters
    ----------
    trace_OBJ : trace object (TIE_classes.trace_OBJ)
        The trace object for which signals are to be displayed.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Handle to the figure.
    """
    ls = len(trace_OBJ.Segment)
    fig, axs = plt.subplots(ls, 2)
    fig.suptitle("Stereographic projection: trace n° " + str(trace_OBJ.id), fontsize=12)
    for s in trace_OBJ.Segment:
        Chords = s.Chords
        l = len(Chords)
        axChd = [chord.vec_trend[0] for chord in Chords]
        axPl = [chord.vec_plunge[0] for chord in Chords]
        ai = s.id - 1

        vec = np.array([TIEgen.stereoLine(axChd[i], axPl[i]) for i in range(l)])
        x_Ax = vec[:, 0]
        y_Ax = vec[:, 1]

        if l % 2 == 0:
            half = int(l / 2)
            halfpoint = int(half)
        else:
            half = int((l - 1) / 2)
            halfpoint = int(half + 1)

        rgbv = np.array([[1, 1, 0], [0.13, 0.54, 0.13], [0, 0, 0.8]])
        cmap = createCmap(rgbv, halfpoint)

        if ls == 1:
            ai_chd = 0
            ai_chdp = 1
        else:
            ai_chd = (ai, 0)
            ai_chdp = (ai, 1)

        axs[ai_chd] = stereoPlot(axs[ai_chd])
        for i in range(halfpoint):
            axs[ai_chd].scatter(x_Ax[i], y_Ax[i], s=25, c=[cmap[i, :]])
            axs[ai_chd].scatter(x_Ax[i + half], y_Ax[i + half], s=25, c=[cmap[i, :]])
        axs[ai_chd].text(-1, 1, "seg. " + str(s.id), fontsize=6)

        ChdPl = s.Chordplanes
        N = len(ChdPl)
        axs[ai_chdp] = stereoPlot(axs[ai_chdp])
        for i in range(N):
            azim = ChdPl[i].plane_orient[0]
            dip = ChdPl[i].plane_orient[1]
            [x, y] = TIEgen.greatCircle(azim, dip)

            axs[ai_chdp].plot(x, y, c=tuple(cmap[i, :]), linewidth=0.3)
            x_stereo, y_stereo = TIEgen.stereoLine(
                ChdPl[i].pole_orient[0], ChdPl[i].pole_orient[1]
            )
            axs[ai_chdp].scatter(x_stereo, y_stereo, s=25, c=[cmap[i, :]])
        axs[ai_chdp].text(-1, 1, "seg. " + str(s.id), fontsize=6)

    axs[ai_chd].set_xlabel("chord vector orientations", fontsize=8)
    axs[ai_chdp].set_xlabel("chord plane (gc) & pole orientations", fontsize=8)
    return fig


def showTIEmap(
    mx: np.ndarray,
    my: np.ndarray,
    mz: np.ndarray,
    mc: list = [],
    cmap: str = "",
    MainTrace_Set: list = [],
    AuxTrace_Set: list = [],
    ShowBars: bool = True,
) -> mlab.figure:
    """
    Show Tie-Results In 3d.

    Displays the TIE-results in 3D on top of the slightly transparent geological map.
    TIE-results: (1) trace segments are colored according to TIE-class, (2) orientation
    bars indicate chord plane orientations along the trace.

    Parameters
    ----------
    mx : numpy.ndarray
        Coordinate matrixes of analyzed extent.
    my : numpy.ndarray
        Coordinate matrixes of analyzed extent.
    mz : numpy.ndarray
        Coordinate matrixes of analyzed extent.
    mc : list, optional
        Scalar field of bedrock id's.
    cmap : str, optional
        Bedrock colormap. Default is mayavi's default cmap.
    MainTrace_Set : list, optional
        List of tie-analyzed trace objects.
    AuxTrace_Set : list, optional
        List of auxiliary trace objects that are to be neutrally displayed (e.g., fault traces).
    ShowBars : bool, optional
        Boolean indicating whether to show orientation bars or not. Default is True.

    Returns
    -------
    mayavi.modules.figure.Figure
        Handle to the figure.
    """

    if len(cmap) == 0:
        cmap = "Greens"

    fig = mlab.figure(2, fgcolor=(1, 1, 1), bgcolor=(0.9, 0.9, 0.9))
    mesh = mlab.mesh(mx, my, mz, scalars=mc, opacity=0.6)
    mesh.module_manager.scalar_lut_manager.lut.table = cmap

    vx = np.fliplr(mx).flatten()
    vy = my.flatten()
    vz = np.fliplr(mz).flatten()

    # analyzed trace set
    for tr in MainTrace_Set:
        tx = vx[tr.index.astype(int)]
        ty = vy[tr.index.astype(int)]
        tz = vz[tr.index.astype(int)]
        for s in tr.Segment:
            mlab.plot3d(
                tx[s.ind_normal],
                ty[s.ind_normal],
                tz[s.ind_normal],
                color=tuple(colorClassID(s)),
                tube_radius=5,
                name="analyzed trace " + str(tr.id),
            )
            mlab.text3d(
                tx[s.ind_normal[4]],
                ty[s.ind_normal[4]],
                tz[s.ind_normal[4]] + 70,
                str(tr.id),
                color=(0, 0, 0),
                scale=25,
            )

    # auxiliary trace set
    for fl in AuxTrace_Set:
        mlab.plot3d(
            vx[fl.index.astype(int)],
            vy[fl.index.astype(int)],
            vz[fl.index.astype(int)],
            color=(0, 0, 0),
            tube_radius=1.5,
            name="auxiliary trace " + str(fl.id),
        )

    if ShowBars == True:
        showOrientBars(MainTrace_Set, vx, vy, vz)

    mlab.draw()
    mlab.show()

    return fig


def sigHdiagram(trace_set, pth=[3, 9, 18, 90], txt=True, scale="lin"):
    """
    Show Signal Height Diagram.

    Displays a scatter plot with the signal height h_alpha as a function of the
    signal height h_beta for each trace in a trace set. The classification
    boundaries (pth) are displayed as background information. The dots are
    colored according to their TIE-classification.

    Parameters
    ----------
    trace_set : list of trace_OBJ (TIE_classes.trace_OBJ)
        List of trace objects.
    pth : list of int, optional
        Planarity threshold used to classify the traces.
        Default is [3, 9, 18, 90].
    txt : bool, optional
        Boolean indicating whether to display the trace's id or not.
        Default is True.
    scale : str, optional
        Scale for displaying the SH-diagram. "log" for logarithmic scale,
        "lin" for linear scale. Default is "lin".

    Returns
    -------
    fig : matplotlib.figure.Figure
        Handle to the figure.
    """

    x1 = [
        0.0001,
        0.0005,
        0.001,
        0.005,
        0.01,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1,
    ]
    x2 = [
        1.2,
        1.2,
        1.5,
        1.8,
        2,
        2.5,
        3,
        3.8,
        5,
        6,
        8,
        10,
        15,
        20,
        30,
        40,
        50,
        60,
        70,
        90,
        180,
    ]
    x0 = x1 + x2
    y0 = [180 / x0i for x0i in x0]

    fig, ax = plt.subplots()
    if scale == "log":
        ax.axis([3, 180, 3, 180])
        ax.set_yscale("log")
        ax.set_xscale("log")
    else:
        ax.axis([0, 180, 0, 180])
    ax.set_aspect("equal", adjustable="box")

    ax.plot(x0, y0, "k:", linewidth=0.7)
    ax.text(
        11,
        11,
        "p = 0°",
        fontsize=8,
        rotation=-45,
        bbox=dict(boxstyle="round4,pad=0.5", fc=(1, 1, 1, 1), lw=0),
    )
    ax.plot([0, 180], [0, 180], "k--", linewidth=1)
    ax.text(
        120,
        120,
        "s = 1",
        fontsize=8,
        rotation=45,
        bbox=dict(boxstyle="round4,pad=0.5", fc=(1, 1, 1, 1), lw=0),
    )

    ax.set_title("Signal-Height-Diagram")
    ax.set_xlabel("h_alpha")
    ax.set_ylabel("h_beta")

    for j in pth:
        x = np.array(x0) + j
        y = 180 / (x - j) + j
        ax.plot(x, y, "k", linewidth=1)
        ax.text(
            170,
            y[-1],
            str(j) + "°",
            fontsize=6,
            bbox=dict(boxstyle="round4,pad=0.1", fc=(1, 1, 1, 1), lw=0),
        )

    for tr in trace_set:
        sl = len(tr.Segment)
        for s in tr.Segment:
            ax.scatter(s.sigheight[0], s.sigheight[1], s=10, c=[colorClassID(s)])
            if txt == True:
                if sl == 1:
                    txt_name = str(tr.id)
                else:
                    txt_name = str(tr.id) + "(" + str(s.id) + ")"
                if scale == "log":
                    dec = s.sigheight[0] / 30
                else:
                    dec = 2

                ax.text(
                    s.sigheight[0] + dec,
                    s.sigheight[1],
                    txt_name,
                    fontsize=6,
                    bbox=dict(boxstyle="round4,pad=0", fc=(1, 1, 1, 0.6), lw=0),
                )
    plt.show
    return fig


def stereoPlot(ax, WithCircles="no"):
    """
    Stereoplot.

    Displays a stereonet ("Wulff - preservation of angles").

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Handle to the axes in which the stereonet should be displayed.
    with_circles : bool, optional
        Boolean indicating whether to show the great circles or only show major axes.
        Default is False.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Handle to the axes.
    """

    N = 50
    v = np.linspace(0, 2 * np.pi, N)
    cx = [math.cos(vi) for vi in v]
    cy = [math.sin(vi) for vi in v]
    xh = [-1, 1]
    yh = [0, 0]
    xv = [0, 0]
    yv = [-1, 1]
    lines = ax.plot(xh, yh, xv, yv, cx, cy, [0, 0], [1, 1.05])
    plt.setp(lines, color="k", linewidth=0.7)

    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis="y", which="both", right=False, left=False, labelleft=False)
    ax.text(-0.04, 1.07, "N")

    if WithCircles == "yes":
        vhf = np.linspace(0, np.pi, N)
        for i in range(1, 9):
            rdip = i * (np.pi / 18)
            rint = np.array(math.tan(rdip) * np.array([math.sin(vi) for vi in vhf]))
            radip = np.array([math.atan(rt) for rt in rint])
            rproj = np.array([math.tan((np.pi / 2 - rdp) / 2) for rdp in radip])
            x1 = rproj * np.array([math.sin(vi) for vi in vhf])
            x2 = rproj * np.array([(-math.sin(vi)) for vi in vhf])
            y = rproj * np.array([math.cos(vi) for vi in vhf])

            ax.plot(x1, y, ":k", x2, y, ":k", linewidth=0.4)
        for i in range(1, 9):
            alp = i * (np.pi / 18)
            xlim = math.sin(alp)
            x = np.arange(-xlim, xlim, 0.01)
            d = 1 / math.cos(alp)
            rd = d * math.sin(alp)
            y00 = rd**2 - np.array([xi**2 for xi in x])
            y0 = np.array([math.sqrt(yi) for yi in y00])
            y1 = np.array(d - np.array([yi for yi in y0]))
            y2 = np.array(-d + np.array([yi for yi in y0]))

            ax.plot(x, y1, ":k", x, y2, ":k", linewidth=0.4)
    ax.axis([-1, 1, -1, 1.15])
    ax.set_aspect("equal", "box")
    plt.draw()
    return ax
