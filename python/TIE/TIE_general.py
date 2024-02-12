# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 16:07:59 2021
@author: Anna Rauch

This script contains a set of general functions/methods usually related to
linear algebra and connectivty issues. These functions are independent from
the TIE-method, yet are used by it.
"""
import math
import numpy as np
import platform
import tkinter as tk
from tkinter import messagebox


def angle2normal(azim, dip):
    """
    Calculate the normal vector of a plane (or set of planes) defined with orientational angles in degrees.

    Parameters
    ----------
    azim : float or array_like
        Angles of azimuth (dip azimuth) in degrees.
    dip : float or array_like
        Dip angles of the plane(s) in degrees.

    Returns
    -------
    numpy.ndarray
        Normal vector of the plane(s) in the form [normalx, normaly, normalz].
    """

    dip = np.array(dip) * np.pi / 180
    azim = np.array(azim) * np.pi / 180

    if np.size(dip) == 1:
        dip = [dip]
        azim = [azim]

    normalz = [np.cos(dp) for dp in dip]
    normalx = [np.sin(dip[ind]) * np.sin(azim[ind]) for ind in range(np.size(dip))]
    normaly = [np.sin(dip[ind]) * np.cos(azim[ind]) for ind in range(np.size(dip))]

    return np.array([normalx, normaly, normalz]).T


def angle2vect(trend, plunge):
    """
    Calculate the directional vector (length = 1) from a line defined by angles - trend and plunge.

    Parameters
    ----------
    trend : float or array_like
        Angles of trend (plunge azimuth) in degrees.
    plunge : float or array_like
        Plunge angles of the directional vector in degrees.

    Returns
    -------
    tuple of numpy.ndarray
        Coordinates of the oriented vector in the form (vx, vy, vz).
    """

    # Convert angles to radians
    trend = np.array(trend) * np.pi / 180
    plunge = np.array(plunge) * np.pi / 180

    # Handle single values or arrays
    if np.size(trend) == 1:
        trend = [trend]
        plunge = [plunge]

    # Calculate coordinates of the oriented vector
    vz = [-np.sin(pl) for pl in plunge]
    vx = [np.sin(trend[k]) * np.cos(plunge[k]) for k in range(np.size(trend))]
    vy = [np.cos(trend[k]) * np.cos(plunge[k]) for k in range(np.size(trend))]

    # Return the coordinates as a tuple of numpy arrays
    return np.array(vx).T, np.array(vy).T, np.array(vz).T


def angleBtwVec(v1, v2):
    """
    Calculate the small angle between two directional oriented vectors.

    Parameters
    ----------
    v1 : array_like
        First vector in the form (x, y, z).
    v2 : array_like
        Second vector in the form (x, y, z).

    Returns
    -------
    float
        Angle between the two vectors in degrees.
    """

    if len(np.shape(v1)) > 1:
        v1n = np.array([j / np.linalg.norm(j) for j in v1])
        v2n = np.array([k / np.linalg.norm(k) for k in v2])
        dotv = np.array(
            np.round([np.dot(v1n[l], v2n[l].T) for l in range(len(np.shape(v1)))], 5)
        )
        angle = np.array([math.acos(dt) for dt in dotv])
    else:
        v1n = v1 / np.linalg.norm(v1)
        v2n = v2 / np.linalg.norm(v2)
        angle = math.acos(np.round(np.dot(v1n, v2n.T), 5))

    return angle / np.pi * 180


def azimRot(dip, trend, plunge):
    """Azimuth according to rotation axis and dip.

    Calculates the azimuth of a plane with a certain dip
    that goes through a rotation axis defined with trend and plunge angles.

    Parameters
    ----------
    dip : float
        Angles of plane dip (angles in degrees).
    trend : float
        Angles of trend (plunge azimuth) of the rotational axis.
    plunge : float
        Plunge of the rotational axis (angles in degrees).

    Returns
    -------
    float
        Angles of plane dip azimuths (angles in degrees) that pass through the axis.
    """

    trend = trend / 180 * np.pi
    dip = dip / 180 * np.pi
    plunge = plunge / 180 * np.pi
    azim = trend - np.pi / 2 + math.asin(math.tan(plunge) / math.tan(dip))

    return azim / np.pi * 180


def dipRot(azim, trend, plunge):
    """Dip According To Rotation Axis And Azimuth.

    Calculates the dip of a plane with a certain dip azimuth
    that passes through a rotation axis defined with trend and plunge angles.

    Parameters
    ----------
    azim : float
        Angles of plane dip azimuth (angles in degrees) (no strike).
    trend : float
        Angles of trend (plunge azimuth) of the rotational axis.
    plunge : float
        Plunge of the rotational axis (angles in degrees).

    Returns
    -------
    float
        Angles of plane dips (angles in degrees) that pass through the axis.
    """

    trend = trend / 180 * np.pi
    azim = azim / 180 * np.pi
    plunge = plunge / 180 * np.pi
    dip = math.atan(math.tan(plunge) / math.cos(azim - trend))

    return dip / np.pi * 180


def distance(P1, P2):
    """Calculate the shortest distance between two points in space.

    Parameters
    ----------
    P1 : list
        Coordinates of the first point [x, y, z].
    P2 : list
        Coordinates of the second point [x, y, z].

    Returns
    -------
    float
        Distance between two points.
    """

    return ((P1[0] - P2[0]) ** 2 + (P1[1] - P2[1]) ** 2) ** 0.5


def greatCircle(azim, dip):
    """Extract coordinates of the great circle of a given plane orientation.

    Parameters
    ----------
    azim : float
        Angle of azimuth (dip azimuth) of the plane.
    dip : float
        Dip angle of the plane.

    Returns
    -------
    list
        Coordinates of great circle points [x_stereo, y_stereo].
    """

    dip = np.array(dip) * np.pi / 180
    azim = -np.array(azim) * np.pi / 180 + np.pi / 2

    N = 51
    psi = np.linspace(0, np.pi, N)

    radint = math.tan(dip) * np.array([math.sin(ps) for ps in psi])
    radip = np.array([math.atan(radi) for radi in radint])
    rproj = np.array([math.tan((np.pi / 2 - radp) / 2) for radp in radip])
    x1 = rproj * np.array([math.sin(ps) for ps in psi])
    y1 = rproj * np.array([math.cos(ps) for ps in psi])

    x_stereo = np.array(
        [x1[k] * math.cos(azim) + y1[k] * math.sin(azim) for k in range(np.size(x1))]
    )
    y_stereo = np.array(
        [x1[k] * math.sin(azim) - y1[k] * math.cos(azim) for k in range(np.size(x1))]
    )

    return [x_stereo, y_stereo]


def neighborPoints(j, rows, columns, connectivity):
    """Define array containing all neighbor indexes of a certain point in a matrix.

    Parameters
    ----------
    j : int
        Index in the matrix of the point being analyzed (flattened matrix).
    rows : int
        Number of rows in the matrix.
    columns : int
        Number of columns in the matrix.
    connectivity : str
        Type of neighbor connectivity (8-connectivity or 4-connectivity).

    Returns
    -------
    numpy.ndarray
        Array with indexes of neighbors.
    """

    lowerleft = (columns * rows) - columns
    lowerright = columns * rows - 1

    if connectivity == 4:
        neigh = np.array([j + 1, j - 1, j - columns, j + columns])

        if np.remainder(j + 1, columns) == 0:
            neigh = np.array([j - 1, j - columns, j + columns])
        if np.remainder(j, columns) == 0:
            neigh = np.array([j + 1, j - columns, j + columns])
        if j < columns:
            neigh = np.array([j - 1, j + 1, j + columns])
        if j > lowerleft:
            neigh = np.array([j - 1, j + 1, j - columns])
        if j == 0:
            neigh = np.array([j + 1, j + columns])
        if j == columns - 1:
            neigh = np.array([j - 1, j + columns])
        if j == lowerleft:
            neigh = np.array([j + 1, j - columns])
        if j == lowerright:
            neigh = np.array([j - 1, j - columns])

    if connectivity == 8:
        neigh = np.array(
            [
                j + 1,
                j - 1,
                j - columns,
                j + columns,
                j - columns + 1,
                j - columns - 1,
                j + columns + 1,
                j + columns - 1,
            ]
        )

        if np.remainder(j + 1, columns) == 0:
            neigh = np.array(
                [j - 1, j - columns, j + columns, j - columns - 1, j + columns - 1]
            )
        if np.remainder(j, columns) == 0:
            neigh = np.array(
                [j + 1, j - columns, j + columns, j - columns + 1, j + columns + 1]
            )
        if j < columns:
            neigh = np.array(
                [j - 1, j + 1, j + columns, j + columns + 1, j + columns - 1]
            )
        if j > lowerleft:
            neigh = np.array(
                [j - 1, j + 1, j - columns, j - columns + 1, j - columns - 1]
            )
        if j == 0:
            neigh = np.array([j + 1, j + columns, j + columns + 1])
        if j == columns - 1:
            neigh = np.array([j - 1, j + columns, j + columns - 1])
        if j == lowerleft:
            neigh = np.array([j + 1, j - columns, j - columns + 1])
        if j == lowerright:
            neigh = np.array([j - 1, j - columns, j - columns - 1])

    return neigh


def neighborPointsD(j, rows, columns, dp):
    """Define array containing the indexes surrounding the index j at a given
       distance dp in a matrix [rows,columns] (8-connectivity).

    Parameters
    ----------
    rows : int
        Number of rows in the matrix.
    columns : int
        Number of columns in the matrix.
    j : int
        Index in matrix of the point being analyzed (flattened matrix).
    dp : int
        Pixel number at the wanted distance (1 being the closest neighbor possible).

    Returns
    -------
    numpy.ndarray
        Array with indexes of neighbors.
    """

    neigh = np.array([j + dp, j - dp, j - dp * columns, j + dp * columns])
    for i in range(1, dp + 1):
        neigh2 = np.array(
            [
                j - dp * columns + i,
                j - dp * columns - i,
                j + dp * columns + i,
                j + dp * columns - i,
            ]
        )
        neigh = np.concatenate((neigh, neigh2))
    for i in range(1, dp + 1):
        neigh2 = np.array(
            [
                j - (dp - i) * columns + dp,
                j - (dp - i) * columns - dp,
                j + (dp - i) * columns + dp,
                j + (dp - i) * columns - dp,
            ]
        )
        neigh = np.concatenate((neigh, neigh2))

    remj = np.remainder(j, columns)
    remn = [np.remainder(n, columns) for n in neigh]
    neigh = neigh.compress((remn <= remj + dp).flat)
    remn = [np.remainder(n, columns) for n in neigh]
    neigh = neigh.compress((remn >= remj - dp).flat)
    neigh = neigh.compress((neigh < rows * columns).flat)
    neigh = neigh.compress((neigh >= 0).flat)
    neigh = np.unique(neigh)

    return neigh


def nmbNeighbors(binaryimage):
    """Extract a matrix (of size of the binary image) that contains the
       number of positive neighbors for each pixel of the binary image.

    Parameters
    ----------
    binaryimage : numpy.ndarray
        Matrix of zeros and ones (or TRUE's and FALSE's).

    Returns
    -------
    numpy.ndarray
        Matrix with the number of positive neighbors.
    """

    bi = binaryimage.flatten()
    i_true = (bi == 1).nonzero()[0]
    nmbN = np.zeros(np.shape(bi))

    for i in i_true:
        neigh = neighborPoints(i, np.shape(binaryimage)[0], np.shape(binaryimage)[1], 8)
        nmbN[i] = np.size((bi[neigh] == 1).nonzero())

    nmbN_mat = np.reshape(nmbN, np.shape(binaryimage))
    return nmbN_mat


def normal2angle(normal):
    """Calculates the orientation (in dip azimuth and dip) of the plane
    according to its normal (in x, y, z).

    Parameters
    ----------
    normal : numpy.ndarray
        Normal vector: n = [x, y, z].

    Returns
    -------
    Tuple[float, float]
        Azimuth (dip azimuth) and dip of the plane expressed in angles.
    """

    if np.ndim(normal) > 1:
        normal = normal[0]
    if normal[2] < 0:
        normal = normal * np.array([-1, -1, -1])
    if normal[1] == 0:
        normal[1] = 0.000001

    zvector = np.array([0, 0, 1])
    dip = math.atan2(np.linalg.norm(np.cross(normal, zvector)), np.dot(normal, zvector))
    dip = dip / np.pi * 180
    azim = math.atan(abs(normal[0]) / abs(normal[1]))
    azim = azim / np.pi * 180

    if normal[0] > 0 and normal[1] <= 0:
        azim = 180 - azim

    if normal[0] <= 0 and normal[1] < 0:
        azim = 180 + azim

    if normal[0] <= 0 and normal[1] > 0:
        azim = 360 - azim

    return [azim, dip]


def plungeRot(azim, dip, trend):
    """Calculates the plunge of an axis with a given trend and through which a
    plane with a given orientation goes through.

    Parameters
    ----------
    azim : float
        Angles of plane orientation (angles degree °).
    dip : float
        Angles of plane dip (angles degree °).
    trend : float
        Angles of trend (plunge azimuth) (angles degree °).

    Returns
    -------
    float
        Angles of axis plunge (angles degree °) that hosts the given plane.
    """

    trend = trend / 180 * np.pi
    azim = azim / 180 * np.pi
    dip = dip / 180 * np.pi
    plunge = math.atan(math.tan(dip) * math.cos(azim - trend))

    return plunge / np.pi * 180


def sortLine(ind, matSize):
    """Sorts a vector of points to connect them into a line and attribute
    an order (2D only).

    Parameters
    ----------
    ind : array_like
        Indexes of points in a matrix.
    matSize : tuple
        Size/shape of matrix.

    Returns
    -------
    newi : ndarray
        Newly sorted/ordered indexes.
    """

    matx = np.arange(0, matSize[1])
    maty = np.arange(0, matSize[0])
    [x, y] = np.meshgrid(matx, maty)
    x = x.flatten()
    y = y.flatten()

    mat = np.zeros(np.shape(x))
    mat[ind] = 1
    mat = np.reshape(mat, (matSize[0], matSize[1]))

    newi = np.zeros(np.shape(ind))
    start = []
    for i in ind:
        neigh = neighborPoints(i, matSize[0], matSize[1], 8)
        npos = np.intersect1d(neigh, ind)

        if np.size(npos) == 1:
            start = i
            break

    if np.size(start) == 0:
        start = ind[0]
        neigh = neighborPoints(start, matSize[0], matSize[1], 8)
        npos = np.intersect1d(neigh, ind)
        newi[-1] = npos[0]
        ind = np.array(ind).compress((np.array(ind) != npos[0]).flat)

    count = 0
    newi[count] = start
    count = count + 1
    ind = np.array(ind).compress((np.array(ind) != start).flat)

    while np.size(ind) > 0:
        d = [distance([x[start], y[start]], [x[ii], y[ii]]) for ii in ind]
        dimin = [di for di in range(np.size(d)) if d[di] == np.min(d)]
        if len(dimin) > 1:
            for di in dimin:
                neigh = neighborPointsD(ind[di], matSize[0], matSize[1], 8)
                npos = np.intersect1d(neigh, ind)
                di_neigh = np.array(dimin).compress((np.array(dimin) != di).flat)
                npos = np.intersect1d(neigh, ind[di_neigh])
                if len(npos) == 0:
                    dimin = di
                    break

        newi[count] = ind[dimin]
        count = count + 1
        start = ind[dimin]
        ind = np.delete(ind, dimin)
    return newi


def stereoLine(trend, plunge):
    """Extracts coordinates of a projected line on a stereonet.

    Parameters
    ----------
    trend : float
        Angle of trend (plunge azimuth) of the line.
    plunge : float
        Plunge of the line.

    Returns
    -------
    [x_stereo, y_stereo] : list
        Coordinates of point(s) in a stereonet.
    """

    trend = np.array(trend / 180 * np.pi)
    plunge = np.array(plunge / 180 * np.pi)

    radius = np.tan(np.pi * (np.pi / 2 - plunge) / (2 * np.pi))
    x_stereo = np.sin(trend) * radius
    y_stereo = np.cos(trend) * radius

    if trend.compress((trend > np.pi / 2).flat) and trend.compress(
        (trend <= np.pi).flat
    ):
        trend = np.pi - trend
        x_stereo = np.sin(trend) * radius
        y_stereo = -np.cos(trend) * radius
    if trend.compress((trend > np.pi).flat) and trend.compress(
        (trend <= np.pi / 2 * 3).flat
    ):
        trend = trend - np.pi
        x_stereo = -np.sin(trend) * radius
        y_stereo = -np.cos(trend) * radius
    if trend.compress((trend > np.pi / 2 * 3).flat):
        trend = 2 * np.pi - trend
        x_stereo = -np.sin(trend) * radius
        y_stereo = np.cos(trend) * radius

    return [x_stereo, y_stereo]


def vect2angle(v):
    """Extracts from vectors (vx, vy, vz) its axis directions in angles (geology style).

    Parameters
    ----------
    v : list
        Vector: v = [vx, vy, vz].

    Returns
    -------
    trend, plunge : tuple
        Angles of trend (plunge azimuth) and plunge of the vector.
    """

    if np.ndim(v) > 1:
        v = v[0]

    if v[2] > 0:
        v = v * np.array([-1, -1, -1])

    if v[1] == 0:
        trend = 90
    else:
        trend = abs(math.atan(v[0] / v[1]))
        trend = trend / np.pi * 180

    plunge = math.atan(abs(v[2]) / np.sqrt(v[0] ** 2 + v[1] ** 2))
    plunge = plunge / np.pi * 180

    if v[0] > 0 and v[1] <= 0:
        trend = 180 - trend

    if v[0] <= 0 and v[1] <= 0:
        trend = 180 + trend

    if v[0] <= 0 and v[1] > 0:
        trend = 360 - trend

    return trend, plunge


def Mbox(title, text, style):
    """Creates a message box.

    Parameters
    ----------
    title : str
        Title/name of the box.
    text : str
        Message text.
    style : int
        Style of text (bold, fontsize, ...).

    Returns
    -------
    None
        Message box appears.
    """
    if platform.system() == "Windows":
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, text, title, style)
    else:  # Assuming other platforms support tkinter
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        messagebox.showinfo(title, text)

        root.destroy()
