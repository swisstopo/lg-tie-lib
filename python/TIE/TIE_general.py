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


def angle2normal(azim, dip):
    """ ANGLE to NORMAL
    calculates the normal of a plane (or a set of planes) defined with 
    orientational angles in degrees - dip azimuth and dip 
    (no specific location). 
     ----------
    INPUT
    azim, dip -> angles of azimuth (dip azimuth) and dip of plane.  
     ----------
    OUTPUT
    normal    -> normal vector: normal = [normalx,normaly,normalz] """

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
    """ ANGLE to VECTOR
    calculates the directional vector (length = 1) from a line defined by 
    angles - trend and plunge
     ----------
    INPUT:
    trend, plunge -> angles of trend (plunge azimuth) and plunge of 
    directional vector.  
     ----------
    OUTPUT
    vx, vy, vz  -> coordinates of oriented vector """

    trend = np.array(trend) * np.pi / 180
    plunge = np.array(plunge) * np.pi / 180

    if np.size(trend) == 1:
        trend = [trend]
        plunge = [plunge]

    vz = [-np.sin(pl) for pl in plunge]
    vx = [np.sin(trend[k]) * np.cos(plunge[k]) for k in range(np.size(trend))]
    vy = [np.cos(trend[k]) * np.cos(plunge[k]) for k in range(np.size(trend))]

    return [np.array(vx).T, np.array(vy).T, np.array(vz).T]


def angleBtwVec(v1, v2):
    """ ANGLE BETWEEN 2 VECTORS
    small angle between two directional oriented vectors
     ----------
    INPUT
      -> v1: first vector (x,y,z)
      -> v2: second vector (x,y,z)
     ----------
    OUTPUT
      -> angle: angle between the two vectors """

    if len(np.shape(v1)) > 1:
        v1n = np.array([j / np.linalg.norm(j) for j in v1])
        v2n = np.array([k / np.linalg.norm(k) for k in v2])
        dotv = np.array(np.round([np.dot(v1n[l], v2n[l].T) for l in range(len(np.shape(v1)))], 5))
        angle = np.array([math.acos(dt) for dt in dotv])
    else:
        v1n = v1 / np.linalg.norm(v1)
        v2n = v2 / np.linalg.norm(v2)
        angle = math.acos(np.round(np.dot(v1n, v2n.T), 5))

    return angle / np.pi * 180


def azimRot(dip, trend, plunge):
    """ AZIMUTH ACCORDING TO ROTATION AXIS AND DIP
    calculates the azimth of a plane with a certain dip
    that goes through a rotation axis defined with trend and plunge angles.
     ----------
    INPUT
    dip           -> angles of plane dip (angles degree °).  
    trend, plunge -> angles of trend (plunge azimuth) and 
                     plunge of rotational axis.       
     ----------
    OUTPUT
    azim -> angles of plane dip azimuths (angles degree °) 
            that pass through axis. """

    trend = trend / 180 * np.pi
    dip = dip / 180 * np.pi
    plunge = plunge / 180 * np.pi
    azim = trend - np.pi / 2 + math.asin(math.tan(plunge) / math.tan(dip))

    return azim / np.pi * 180


def dipRot(azim, trend, plunge):
    """ DIP ACCORDING TO ROTATION AXIS AND AZIMUTH
    calculates the dip of a plane with a certain dip azimuth
    that passes through a rotation axis defined with trend and plunge angles.
     ----------
    INPUT
    azimuth       -> angles of plane dip azimuth (angles degree °) (no strike).  
    trend, plunge -> angles of trend (plunge azimuth) and 
                     plunge of rotational axis.       
     ----------
    OUTPUT
    dip -> angles of plane dips (angles degree °) 
            that pass through axis. """

    trend = trend / 180 * np.pi
    azim = azim / 180 * np.pi
    plunge = plunge / 180 * np.pi
    dip = math.atan(math.tan(plunge) / math.cos(azim - trend))

    return dip / np.pi * 180


def distance(P1, P2):
    """ DISTANCE BETWEEN TWO POINTS
    calculates the shortest distance between two points in space.
     ----------
    INPUT
    P1,P2   -> two points [x,y,z]  
     ----------
    OUTPUT
    distance -> distance between two points """

    return ((P1[0] - P2[0]) ** 2 + (P1[1] - P2[1]) ** 2) ** 0.5


def greatCircle(azim, dip):
    """ GREAT CIRCLES
    extracts coordinates of great circle of a given plane orientation 
    ----------
    INPUT
    azim, dip     -> angles of azimuth (dip azimuth) and dip of plane  
    ----------
    OUTPUT
    [x_stereo, y_stereo]  -> coordinates of great circle points """

    dip = np.array(dip) * np.pi / 180
    azim = -np.array(azim) * np.pi / 180 + np.pi / 2

    N = 51
    psi = np.linspace(0, np.pi, N)

    radint = math.tan(dip) * np.array([math.sin(ps) for ps in psi])
    radip = np.array([math.atan(radi) for radi in radint])
    rproj = np.array([math.tan((np.pi / 2 - radp) / 2) for radp in radip])
    x1 = rproj * np.array([math.sin(ps) for ps in psi])
    y1 = rproj * np.array([math.cos(ps) for ps in psi])

    x_stereo = np.array([x1[k] * math.cos(azim) + y1[k] * math.sin(azim) for k in range(np.size(x1))])
    y_stereo = np.array([x1[k] * math.sin(azim) - y1[k] * math.cos(azim) for k in range(np.size(x1))])

    return [x_stereo, y_stereo]


def neighborPoints(j, rows, columns, connectivity):
    """ NEIGHBOUR INDEXES
    define array containing all neighbour indexes of a certain point in a
    matrix
     ----------
    INPUT
    rows, columns  -> size/shape of matrix in X and Y
    j              -> index in matrix of point analysed (flattend matrix)
    connectivty    -> type of neighbour-connectivity 
                    (8-connectivty or 4-connectivty)
     ----------
    OUTPUT
    neigh             -> array with indexes of neighbors """

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
        neigh = np.array([j + 1, j - 1, j - columns, j + columns,
                          j - columns + 1, j - columns - 1, j + columns + 1, j + columns - 1])

        if np.remainder(j + 1, columns) == 0:
            neigh = np.array([j - 1, j - columns, j + columns,
                              j - columns - 1, j + columns - 1])
        if np.remainder(j, columns) == 0:
            neigh = np.array([j + 1, j - columns, j + columns,
                              j - columns + 1, j + columns + 1])
        if j < columns:
            neigh = np.array([j - 1, j + 1, j + columns, j + columns + 1, j + columns - 1])
        if j > lowerleft:
            neigh = np.array([j - 1, j + 1, j - columns, j - columns + 1, j - columns - 1])
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
    """ NEIGHBOUR INDEXES at defined DISTANCE
    defines array containing the indexes surrounding the index j at a given
    distance dp in a matrix [rows,columns] (8-connectivty)
     ----------
    INPUT
    rows, columns  -> size/shape of matrix in X and Y
    j              -> index in matrix of point analysed (flattend matrix)
    dp             -> pixel number at wanted distance (1 being the closest
                      neighbour possible)
     ----------
    OUTPUT
    neigh             -> array with indexes of neighbors """

    neigh = np.array([j + dp, j - dp, j - dp * columns, j + dp * columns])
    for i in range(1, dp + 1):
        neigh2 = np.array([j - dp * columns + i, j - dp * columns - i, j + dp * columns + i, j + dp * columns - i])
        neigh = np.concatenate((neigh, neigh2))
    for i in range(1, dp + 1):
        neigh2 = np.array([j - (dp - i) * columns + dp, j - (dp - i) * columns - dp, j + (dp - i) * columns + dp,
                           j + (dp - i) * columns - dp])
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
    """ Matrix with NUMBER OF NEIGHBORS
    extracts a matrix (of size of the binaryimage) that contains 
    the number of positive neighbors for each pixel of the binary image. This
    is useful in order to find for instance branching points as they always
    contain more that just two neighbours.
     ----------
    INPUT
    binaryimage  -> matrix of zeros and ones (or TRUE's and FALSE's)
     ----------
    OUTPUT
    nmbN_mat -> matrix with number of positive neighbors """

    bi = binaryimage.flatten()
    i_true = (bi == 1).nonzero()[0]
    nmbN = np.zeros(np.shape(bi))

    for i in i_true:
        neigh = neighborPoints(i, np.shape(binaryimage)[0], np.shape(binaryimage)[1], 8)
        nmbN[i] = np.size((bi[neigh] == 1).nonzero())

    nmbN_mat = np.reshape(nmbN, np.shape(binaryimage))
    return nmbN_mat


def normal2angle(normal):
    """ NORMAL to ANGLE (PLANE ORIENTATION)
    calculates the orientation (in dip azimuth and dip) of the plane
    according to its normal (in x,y,z)
    ----------
    INPUT
    normal        -> normal vector: n = [x,y,z]
    ----------
    OUTPUT
    azim, dip     -> azimuth (dip azimuth) and dip of plane 
                     expressed in angles. """

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
    """ PLUNGE ACCORDING TO GIVEN PLANE AND AXIS TREND
    calculates the plunge of an axis with a given trend and through which a 
    plane with a given orientation goes through.
     ----------
    INPUT
    azim, dip   -> angles of plane orientation (angles degree °).  
    trend       -> angles of trend (plunge azimuth) (angles degree °)
     ----------
    OUTPUT
    plunge      -> angles of axis plunge (angles degree °) 
                that hosts given plane """

    trend = trend / 180 * np.pi
    azim = azim / 180 * np.pi
    dip = dip / 180 * np.pi
    plunge = math.atan(math.tan(dip) * math.cos(azim - trend))

    return plunge / np.pi * 180


def sortLine(ind, matSize):
    """ SORT A LINE
    sorts a vector of points so as to connect them into a line and attribute
    an order (2D only).
     ----------
    INPUT
    ind     -> indexes of points in a matrix
    matSize -> size/shape of matrix
     ----------
    OUTPUT
    newi -> newly sorted/ordered indexes """

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
    """ LINE ON STEREONET
    extracts coordinates of a projected line
    ----------
    INPUT
    trend, plunge -> angles of trend (plunge azimuth) and plunge of line  
    ----------
    OUTPUT
    [x_stereo, y_stereo]  -> coordinates of point(s) in a stereonet """

    trend = np.array(trend / 180 * np.pi)
    plunge = np.array(plunge / 180 * np.pi)

    radius = np.tan(np.pi * (np.pi / 2 - plunge) / (2 * np.pi))
    x_stereo = np.sin(trend) * radius
    y_stereo = np.cos(trend) * radius

    if trend.compress((trend > np.pi / 2).flat) and trend.compress((trend <= np.pi).flat):
        trend = np.pi - trend
        x_stereo = np.sin(trend) * radius
        y_stereo = -np.cos(trend) * radius
    if trend.compress((trend > np.pi).flat) and trend.compress((trend <= np.pi / 2 * 3).flat):
        trend = trend - np.pi
        x_stereo = -np.sin(trend) * radius
        y_stereo = -np.cos(trend) * radius
    if trend.compress((trend > np.pi / 2 * 3).flat):
        trend = 2 * np.pi - trend
        x_stereo = -np.sin(trend) * radius
        y_stereo = np.cos(trend) * radius

    return [x_stereo, y_stereo]


def vect2angle(v):
    """ VECTOR (AXIS) to ANGLE (AXIS ORIENTATION)
    extracts from vectors (vx,vy,vz) its axis directions in angles (geology style) 
    ----------
    INPUT
    v   -> vector: v = [vx,vy,vz]
    ----------
    OUTPUT
    trend, plunge -> angles of trend (plunge azimuth) and plunge of vector. """

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
    """ MESSAGE BOX
    creates message box 
    ----------
    INPUT
    title   -> title/name of the box
    text    -> message text
    style   -> style of text (bold,fontsize,...)
    ----------
    OUTPUT
    message box appears """
    import ctypes
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)
