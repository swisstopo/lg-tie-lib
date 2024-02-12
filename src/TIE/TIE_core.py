# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 09:19:29 2021
@author: Anna Rauch

This script contains a set of general functions/methods that perform the
core-steps of the TIE-method (i.e. the actual TIE-steps)
"""
import copy
from typing import Dict, List, Tuple, Union

import numpy as np
from TIE import TIE_classes as TIEclass
from TIE import TIE_general as TIEgen


def classifyTRACE(TRACES, pth=None):
    """Classify traces based on TIE parameters.

    Parameters
    ----------
    TRACES : list of trace objects (TIE_classes.trace_OBJ)
        List of trace objects to be classified.
    pth : list of float or None, optional
        Planarity thresholds for classification. If None, default thresholds
        [3, 9, 18] will be used. Must be of size 3.

    Returns
    -------
    list of trace objects (TIE_classes.trace_OBJ)
        The input list of trace objects with added classification parameters.
    """
    if pth is None:
        pth = [3, 9, 18]

    p1 = pth[0]
    p2 = pth[1]
    p3 = pth[2]

    for T in TRACES:
        Seg = T.Segment
        for S in Seg:
            aN = np.array(
                [
                    S.Chords[ci].alpha[0]
                    for ci in range(np.size(S.Chords))
                    if np.isnan(S.Chords[ci].alpha[0]) == False
                ]
            ).squeeze()
            aR = np.array(
                [
                    S.Chords[ci].alpha[1]
                    for ci in range(np.size(S.Chords))
                    if np.isnan(S.Chords[ci].alpha[1]) == False
                ]
            ).squeeze()
            bN = np.array(
                [
                    S.Chordplanes[cpi].beta[0]
                    for cpi in range(np.size(S.Chordplanes))
                    if np.isnan(S.Chordplanes[cpi].beta[0]) == False
                ]
            ).squeeze()
            bR = np.array(
                [
                    S.Chordplanes[cpi].beta[1]
                    for cpi in range(np.size(S.Chordplanes))
                    if np.isnan(S.Chordplanes[cpi].beta[1]) == False
                ]
            ).squeeze()

            alpS = np.sum(aN + aR)
            alpD = np.abs(np.sum(aN) - np.sum(aR))
            alp = alpS - alpD
            meana = alp / np.size(S.Chords)

            betS = np.sum(bN + bR)
            betD = np.abs(np.sum(bN) - np.sum(bR))
            bet = betS - betD
            meanb = bet / np.size(S.Chordplanes)
            if meanb == 0:
                meanb = 0.1

            meanr = meana / meanb
            fcurb1b = 180 / (meana - p1) + p1
            fcurb2b = 180 / (meana - p2) + p2
            fcurb3b = 180 / (meana - p3) + p3

            if meana < p1:
                g = 1
            elif meanb <= fcurb1b:
                g = 1
            elif meana < p2:
                g = 2
            elif meanb <= fcurb2b:
                g = 2
            elif meana < p3:
                g = 3
            elif meanb <= fcurb3b:
                g = 3
            elif meanb > fcurb3b:
                g = 4

            if meanr < 1:
                S.classID = -g
            else:
                S.classID = g

            S.sigheight = [meana, meanb]
        T.Segment = Seg
    return TRACES


def changeTHseg(TRACES, X, Y, Z, n, amp=-10, newTH=None):
    """
    Change Threshold For Segmentation.

    Changes the threshold that defines the convexity size to segment the traces
    for a specific trace n and performs the TIE on new segmentation.

    Parameters
    ----------
    TRACES : list of trace objects (TIE_classes.trace_OBJ)
        List of trace objects.
    X, Y, Z : array_like
        Coordinate vectors.
    n : int
        Trace ID of the trace that is going to be analyzed with changed threshold.
    amp : float, optional
        Amplitude expressed in % by which the defined threshold should be
        lowered (negative) or increased (positive). Default is -10.
    newTH : list or None, optional
        Newly defined convexity size threshold [peak prominence, peak width].
        If newTH is defined, the amplitude is ignored. Default is None.

    Returns
    -------
    TRACES : list of trace objects (TIE_classes.trace_OBJ)
        List with adapted segmentation for the specific trace object.
    """
    if newTH is None:
        newTH = []  # Initialize the list here if it's None

    formthresh = TRACES[n - 1].convexityTH
    amp = (100 + amp) / 100
    if np.size(newTH) > 0:
        pthresh = newTH
    else:
        pthresh = [formthresh[0] * amp, formthresh[1] / amp]

    TRACES = clearTIE(TRACES, [n])
    TRACESn = tie(
        [TRACES[n - 1]], X, Y, Z, pth=[3, 9, 18], seg=True, peakthresh=pthresh
    )
    TRACES[n - 1] = TRACESn[0]
    TRACES[n - 1].convexityTH = pthresh

    return TRACES


def clearTIE(TRACES, n=None):
    """
    Clear TIE information from trace objects.

    Parameters
    ----------
    TRACES : list of trace objects (TIE_classes.trace_OBJ)
        List of trace objects.
    n : list or None, optional
        List of trace IDs (integers) that designate the traces to be cleared.
        If n is None (default), all traces will be cleared.

    Returns
    -------
    TRACES : list of trace objects (TIE_classes.trace_OBJ)
        List with suppressed TIE information.
    """
    if n is None:
        n = []  # Initialize the list here if it's None

    if np.size(n) == 0:
        n = np.arange(0, int(np.size(TRACES))).astype(int)
    else:
        n = [ni - 1 for ni in n]

    for T in [TRACES[ni] for ni in n]:
        ltr = np.size(T.index)
        i_ind = np.arange(0, ltr).astype(int)
        seg = TIEclass.segment_OBJ(1, [], i_ind, i_ind[::-1], [], [], [], [])
        T.Segment = [seg]
        T.orientbar = []
        T.convexityTH = float("nan")

    return TRACES


def extractAlpha(TRACES):
    """
    Extracts alpha signal from traces.

    Parameters
    ----------
    TRACES : list of trace objects (TIE_classes.trace_OBJ)
        List of trace objects.

    Returns
    -------
    TRACES : list of trace objects (TIE_classes.trace_OBJ)
        List with added alpha values for each trace object (respectively
        each chord object).
    """

    for T in TRACES:
        Segment = T.Segment
        for S in Segment:
            Chords = S.Chords
            n2 = np.size(Chords)
            vIniN = Chords[0].vector[0]  # initial vect for normal signal
            vIniR = Chords[0].vector[1]  # initial vect for reverse signal

            for i in range(n2):
                vAnaN = Chords[i].vector[0]  # analized vect for normal signal
                vAnaR = Chords[i].vector[1]  # analized vect for reverse signal
                angleN = TIEgen.angleBtwVec(vAnaN, vIniN)
                angleR = TIEgen.angleBtwVec(vAnaR, vIniR)
                Chords[i].alpha = [angleN, angleR]
            S.Chords = Chords
        T.Segment = Segment

    return TRACES


def extractBeta(TRACES):
    """
    Extracts beta signal from traces.

    Parameters
    ----------
    TRACES : list of trace objects (TIE_classes.trace_OBJ)
        List of trace objects.

    Returns
    -------
    TRACES : list of trace objects (TIE_classes.trace_OBJ)
        List with added beta values for each trace object (respectively
        each chordPlane object).
    """

    for T in TRACES:
        Segment = T.Segment
        for S in Segment:
            ChdPlane = S.Chordplanes

            refpN = ChdPlane[0].normal[0]
            refpN = refpN / np.linalg.norm(refpN)
            refpR = ChdPlane[0].normal[1]
            refpR = refpR / np.linalg.norm(refpR)

            betN = [TIEgen.angleBtwVec(CP.normal[0], refpN) for CP in ChdPlane]
            betR = [TIEgen.angleBtwVec(CP.normal[1], refpR) for CP in ChdPlane]

            for j in range(1, np.size(betN)):
                if np.isnan(betN[j]) == False:
                    if betN[j] - betN[j - 1] > 90:
                        betN[j] = np.abs(betN[j] - 180)
                        betN[j] = betN[j]
            for j in range(1, np.size(betR)):
                if np.isnan(betR[j]) == False:
                    if betR[j] - betR[j - 1] > 90:
                        betR[j] = np.abs(betR[j] - 180)
                        betR[j] = betR[j]

            for i in range(np.size(betN)):
                ChdPlane[i].beta = [betN[i], betR[i]]

            S.Chordplanes = ChdPlane
        T.Segment = Segment
    return TRACES


def extractChdPlanes(TRACES, mX, mY, mZ):
    """
    Extracts chord planes from traces.

    Parameters
    ----------
    TRACES : list of trace objects (TIE_classes.trace_OBJ)
        List of trace objects.
    mX, mY, mZ : array-like
        Matrices of coordinates of the analyzed extent.

    Returns
    -------
    TRACES : list of trace objects (TIE_classes.trace_OBJ)
        List with added list of chord plane objects (TIE_classes.chordPlane_OBJ)
        to each trace object (respectively each segment object).
    """

    mXfl = mX.flatten()
    mYfl = mY.flatten()
    mZfl = mZ.flatten()

    for T in TRACES:
        index = (T.index).astype(int)
        Segment = T.Segment

        tracel = np.zeros(np.size(index) - 1)
        for i in range(np.size(tracel)):
            PP1 = [mXfl[index[i]], mYfl[index[i]], mZfl[index[i]]]
            PP2 = [mXfl[index[i + 1]], mYfl[index[i + 1]], mZfl[index[i + 1]]]

            tracel[i] = TIEgen.distance(PP1, PP2)
        tr_length_sum = np.sum(tracel)

        for S in Segment:
            Chords = S.Chords
            N = np.size(Chords)

            if N % 2 == 0:
                step = int(N / 2)
                steppoint = step
            else:
                step = int((N - 1) / 2)
                steppoint = step + 1

            indexsegN = T.index[S.ind_normal].astype(int)  # Normal signal
            indexsegR = T.index[S.ind_reverse].astype(int)  # Reverse signal

            ChdPlane = []
            for i in range(steppoint):
                v1N = Chords[i].vector[0]
                v2N = Chords[i + step].vector[0]
                v1R = Chords[i].vector[1]
                v2R = Chords[i + step].vector[1]

                normalN = np.cross(v1N, v2N)
                azimN, dipN = TIEgen.normal2angle(normalN)
                normalR = np.cross(v1R, v2R)
                azimR, dipR = TIEgen.normal2angle(normalR)

                poleplN = 90 - dipN
                poletrN = azimN - 180
                if poletrN < 0:
                    poletrN = 360 + poletrN

                poleplR = 90 - dipR
                poletrR = azimR - 180
                if poletrR < 0:
                    poletrR = 360 + poletrR

                P1N = np.array(
                    [mXfl[indexsegN[i]], mYfl[indexsegN[i]], mZfl[indexsegN[i]]]
                )
                P2N = np.array(
                    [
                        mXfl[indexsegN[i + step]],
                        mYfl[indexsegN[i + step]],
                        mZfl[indexsegN[i + step]],
                    ]
                )

                P1P2N = P2N - P1N
                distN = np.abs(np.dot(np.cross(v1N, v2N), P1P2N)) / np.linalg.norm(
                    np.cross(v1N, v2N)
                )
                distratioN = distN / tr_length_sum

                P1R = np.array(
                    [mXfl[indexsegR[i]], mYfl[indexsegR[i]], mZfl[indexsegR[i]]]
                )
                P2R = np.array(
                    [
                        mXfl[indexsegR[i + step]],
                        mYfl[indexsegR[i + step]],
                        mZfl[indexsegR[i + step]],
                    ]
                )

                P1P2R = P2R - P1R
                distR = np.abs(np.dot(np.cross(v1R, v2R), P1P2R)) / np.linalg.norm(
                    np.cross(v1R, v2R)
                )
                distratioR = distR / tr_length_sum

                ChdPlane.append(
                    TIEclass.chordPlane_OBJ(
                        i + 1,
                        [normalN, normalR],
                        [azimN, dipN, azimR, dipR],
                        [poletrN, poleplN, poletrR, poleplR],
                        [],
                        [distN, distR],
                        [distratioN, distratioR],
                    )
                )
            S.Chordplanes = ChdPlane
        T.Segment = Segment

    return TRACES


def extractChords(TRACES, mX, mY, mZ):
    """
    Extracts connecting chords of a trace set (1st step of TIE).

    Parameters
    ----------
    TRACES : list of trace objects (TIE_classes.trace_OBJ)
        List of trace objects.
    mX, mY, mZ : array-like
        Matrices of coordinates of the analyzed extent.

    Returns
    -------
    TRACES : list of trace objects (TIE_classes.trace_OBJ)
        List with added list of chords (TIE_classes.chords_OBJ)
        to each trace object (respectively each segment object).
    """

    mXfl = mX.flatten()
    mYfl = mY.flatten()
    mZfl = mZ.flatten()

    for T in TRACES:
        indtr = (T.index).astype(int)
        Segment = T.Segment
        for S in Segment:
            l = np.size(S.ind_normal)
            if l % 2 > 0:
                DELTA = int((l - 1) / 2)
                lstr = DELTA + 1
            else:
                DELTA = int(l / 2)
                lstr = DELTA

            if lstr % 2 > 0:
                DD = int((lstr - 1) / 2)
            else:
                DD = int(lstr / 2)

            S.delta = [DELTA, DD]

            # normal trace
            chordsN = np.zeros([lstr, 3])
            trendN = np.zeros([lstr, 1])
            plungeN = np.zeros([lstr, 1])

            i = np.arange(0, lstr)
            pti = S.ind_normal[i]
            ptf = S.ind_normal[i + DELTA]
            chordsN[i, 0] = mXfl[indtr[pti]] - mXfl[indtr[ptf]]
            chordsN[i, 1] = mYfl[indtr[pti]] - mYfl[indtr[ptf]]
            chordsN[i, 2] = mZfl[indtr[pti]] - mZfl[indtr[ptf]]

            # Reverse trace
            chordsR = np.zeros([lstr, 3])
            trendR = np.zeros([lstr, 1])
            plungeR = np.zeros([lstr, 1])

            i = np.arange(0, lstr)
            pti = S.ind_reverse[i]
            ptf = S.ind_reverse[i + DELTA]
            chordsR[i, 0] = mXfl[indtr[pti]] - mXfl[indtr[ptf]]
            chordsR[i, 1] = mYfl[indtr[pti]] - mYfl[indtr[ptf]]
            chordsR[i, 2] = mZfl[indtr[pti]] - mZfl[indtr[ptf]]

            Chords = []
            for chd in range(lstr):
                trendN[chd], plungeN[chd] = TIEgen.vect2angle(chordsN[chd, :])
                trendR[chd], plungeR[chd] = TIEgen.vect2angle(chordsR[chd, :])
                Chords.append(
                    TIEclass.chord_OBJ(
                        chd + 1,
                        [chordsN[chd, :], chordsR[chd, :]],
                        [trendN[chd], trendR[chd]],
                        [plungeN[chd], plungeR[chd]],
                        [],
                    )
                )
            S.Chords = Chords
        T.Segment = Segment

    return TRACES


def extractOrientBars(TRACES):
    """
    Extraction of orientation bars (TIE).

    Extracts orientation bars from chord planes. Chord plane values are
    distributed along the trace.

    Parameters
    ----------
    TRACES : list of trace_OBJ
        List of trace objects containing basic TRACE information
        (any TRACE set, could also be FAULTS).

    Returns
    -------
    TRACES : list of trace_OBJ
        List with added orientation bar values to each trace object.
    """

    for T in TRACES:
        Segment = T.Segment
        or_new = np.zeros([np.size(T.index), 3])

        for S in Segment:
            bet = [S.Chordplanes[bi].beta[0] for bi in range(np.size(S.Chordplanes))]
            betI = [S.Chordplanes[bi].beta[1] for bi in range(np.size(S.Chordplanes))]

            if np.sum(bet) > np.sum(betI):
                orp = [
                    S.Chordplanes[bi].plane_orient[2:]
                    for bi in range(np.size(S.Chordplanes))
                ]
                si = S.ind_reverse
            else:
                orp = [
                    S.Chordplanes[bi].plane_orient[:2]
                    for bi in range(np.size(S.Chordplanes))
                ]
                si = S.ind_normal

            d = S.delta[0]
            dd = S.delta[1]

            for k in range(len(orp)):
                v1 = np.array(TIEgen.angle2vect(orp[k][0], orp[k][1])).squeeze()
                anchor_a = d + k + 1
                anchor_b = dd + d + k + 1
                vanch_a = slice(k, anchor_a)
                vanch_b = slice(dd + k, anchor_b)

                for a in si[vanch_a]:
                    or_new[a, :] = or_new[a, :] + v1
                for b in si[vanch_b]:
                    or_new[b, :] = or_new[b, :] + v1

        or_new = [m / np.linalg.norm(m) for m in or_new]
        for m in range(len(or_new)):
            if or_new[m][2] > 0:
                or_new[m] = or_new[m] * -1
        T.orientbar = or_new

    return TRACES


def mergeSeg(TRACES, n, s, X, Y, Z):
    """
    Merges two or more segments of a certain trace (trace_OBJ) into one single
    segment and reclassifies it according to TIE.

    Parameters
    ----------
    TRACES : list of trace objects (TIE_classes.trace_OBJ)
        List of trace objects.
    n : int
        ID of specific trace_OBJ (only one trace_OBJ at a time).
    s : list
        List of segments that will be merged (e.g., [2, 3]).
    X, Y, Z : array-like
        Coordinate vectors.

    Returns
    -------
    TRACES : list of trace objects (TIE_classes.trace_OBJ)
        List with merged segments in the specific trace_OBJ.
    """

    s = [i - 1 for i in s]
    s.sort()
    for i in range(len(s) - 1):
        if abs(s[i] - s[i + 1]) > 1:
            TIEgen.Mbox("ERROR", "trace segments are not adjacent!!", 1)

    si = []
    # FIXME: list index out of range exception
    seglist = [TRACES[n - 1].Segment[ii] for ii in s]
    for k in seglist:
        si.extend(k.ind_normal.tolist())
    si = np.unique(si)

    SEG = TRACES[n - 1].Segment

    SEG[s[0]].ind_normal = si
    SEG[s[0]].ind_reverse = si[::-1]
    for i in range(1, len(s)):
        SEG.pop(int(s[i]))

    for sii in range(len(SEG)):
        SEG[sii].id = sii + 1

    TRACES[n - 1].Segment = SEG
    trace = tie([TRACES[n - 1]], X, Y, Z)
    TRACES[n - 1] = trace[0]
    TRACES[n - 1].convexityTH.append("merged " + str(s[0]) + " to " + str(s[-1]))

    return TRACES


def mergeTraces(TRACES, tn, X, Y, Z, seg=False):
    """
    Merges two or more traces into one single trace object
    and reclassifies it according to TIE.

    Parameters
    ----------
    TRACES : list of trace objects (TIE_classes.trace_OBJ)
        List of trace objects.
    tn : list
        List of trace IDs that will be merged (e.g., [2, 3]).
    X, Y, Z : array-like
        Coordinate vectors.

    Returns
    -------
    TRACES : list of trace objects (TIE_classes.trace_OBJ)
        List with merged traces (new IDs).
    """

    tni = [ti - 1 for ti in tn]
    ni = []
    trlist = [TRACES[ii] for ii in tni]
    for k in trlist:
        ni.extend(k.index.tolist())
    ni = np.unique(ni).astype(int)
    ind_sort = TIEgen.sortLine(ni, np.shape(Z))

    TRACES[tni[0]].index = ind_sort.astype(int)
    TRACES = clearTIE(TRACES, [tn[0]])
    trace = tie([TRACES[tni[0]]], X, Y, Z, seg=seg)
    TRACES[tni[0]] = trace[0]

    for tri in range(len(TRACES)):
        if TRACES[tri].id in tn[1:]:
            TRACES[tri] = 1
    while 1 in TRACES:
        TRACES.remove(1)

    for tri in range(len(TRACES)):
        TRACES[tri].id = tri + 1

    return TRACES


def segSeg(TRACES, n, s, X, Y, Z):
    """
    Analyses a certain trace segment for potential segmentation at a
    more detailed scale. Important for long irregular traces that were not
    segmented during the first segmentation process.

    Parameters
    ----------
    TRACES : list of trace objects (TIE_classes.trace_OBJ)
        List of trace objects.
    n : int
        ID of the trace object that contains the segment.
    s : int
        ID of the segment that will be re-analyzed for segmentation.
    X, Y, Z : array-like
        Coordinate vectors.

    Returns
    -------
    TRACES : list of trace objects (TIE_classes.trace_OBJ)
        List with potentially added segments.
    """

    n = n - 1
    s = s - 1
    si = TRACES[n].Segment[s].ind_normal
    ni = TRACES[n].index[si].astype(int)
    TRACEX = copy.deepcopy(TRACES[n])
    TRACEX.index = copy.copy(ni)
    TRACEX = clearTIE([TRACEX])
    TRACEX = tie(TRACEX, X, Y, Z, seg=True)

    ls = len(TRACES[n].Segment)
    lsx = len(TRACEX[0].Segment)
    new_ls = ls + lsx - 1
    rest_i = np.arange(s + lsx, new_ls)
    new_i = np.arange(s, new_ls - np.size(rest_i))

    Seg = TRACES[n].Segment
    SegX = copy.deepcopy(TRACEX[0].Segment)

    Seg.pop(s)
    for k in range(np.size(new_i)):
        si_old = SegX[k].ind_normal
        si_new = si[SegX[k].ind_normal]
        Seg.insert(new_i[k], copy.deepcopy(SegX[k]))
        Seg[new_i[k]].ind_normal = si_new
        Seg[new_i[k]].ind_reverse = si_new[::-1]
        Seg[new_i[k]].id = new_i[k] + 1
        ind_slice = slice(si_new[0], si_new[-1] + 1, 1)
        ind_sliceX = slice(si_old[0], si_old[-1] + 1, 1)
        TRACES[n].orientbar[ind_slice] = TRACEX[0].orientbar[ind_sliceX]
    for k in rest_i:
        Seg[k].id = k + 1

    return TRACES


def segmentTRACE(TRACES, mX, mY, mZ, peakthresh=[100, 15]):
    """
    Segments the traces at their inflection points defined based on
    a peak threshold.

    Parameters
    ----------
    TRACES : list of trace objects (TIE_classes.trace_OBJ)
        List of trace objects.
    mX, mY, mZ : array-like
        Matrices of coordinates of the analyzed extent.
    peakthresh : list of float, optional
        Peak thresholds for convexity [peak prominence, peak width].
        Default is [100, 15].

    Returns
    -------
    TRACES : list of trace objects (TIE_classes.trace_OBJ)
        List with added segmentation where needed: list of
        segment objects (TIE_classes.segment_OBJ). Where no segmentation occurred,
        only one segment for one trace exists.
    """

    import scipy

    mXfl = mX.flatten()
    mYfl = mY.flatten()
    mZfl = mZ.flatten()

    for T in TRACES:
        ind = (T.index).astype(int)
        n = np.size(ind)
        d = 2  # steps between connecting points to create a vector v
        l = n - d  # final vector length of connected paths

        v = np.zeros([l, 3])
        angle = np.zeros([l, 1])

        if n > 5:
            k = np.arange(0, l)
            v[k, 0] = mXfl[ind[k + d]] - mXfl[ind[k]]
            v[k, 1] = mYfl[ind[k + d]] - mYfl[ind[k]]
            v[k, 2] = mZfl[ind[k + d]] - mZfl[ind[k]]

            angle = np.array([TIEgen.angleBtwVec(v[m, :], v[0, :]) for m in range(l)])
            angle = angle * np.pi / 180

            anmean1 = angle.copy()
            anmean2 = angle.copy()
            m = slice(1, l)
            p = slice(0, l - 1)
            anmean1[m] = (angle[m] + angle[p]) / 2
            anmean2[p] = (angle[m] + angle[p]) / 2

            # signal smoothing
            if n % 4 > 0:
                smo = int((n - (n % 4)) / 2)
            else:
                smo = int(n / 2)
            P = 1
            while P < smo:
                anmean1[m] = (anmean1[m] + anmean1[p]) / 2
                anmean2[p] = (anmean2[m] + anmean2[p]) / 2
                P = P + 1

            anmean = (
                np.cos(
                    np.concatenate(
                        (anmean1[int(smo / 2) :], anmean2[-smo : int(-smo / 2)])
                    )
                )
                * n
            )
            prom = peakthresh[0]  # minimal peak prominence
            width = n / peakthresh[1]  # minimal peak width

            # find signal peaks (positive and negative)
            ppi = scipy.signal.find_peaks(anmean, prominence=prom, width=width)
            npi = scipy.signal.find_peaks(
                -anmean + np.max(anmean), prominence=prom, width=width
            )
            pind = np.sort(np.concatenate(([0], ppi[0], npi[0], [l - 1])))
            pval = anmean[pind]

            # with each positive peak must follow a negative one -> peaks within peaks
            if np.size(pval) > 2:
                ploc = scipy.signal.find_peaks(pval)[0]
                nloc = scipy.signal.find_peaks(-pval)[0]
            else:
                ploc = []
                nloc = []

            seg = np.sort(np.concatenate((pind[ploc], pind[nloc])))

            # peak width also at the beginning and the end of the signal
            if np.size(seg) > 0:
                if seg[0] < 50:
                    if np.size(seg) > 1:
                        seg = seg[1:]
                    else:
                        seg = []

            if np.size(seg) > 0:
                if l - seg[-1] < 50:
                    if np.size(seg) > 1:
                        seg = seg[:-1]
                    else:
                        seg = []

            if np.size(seg) > 0:
                ind_normal = np.arange(0, seg[0] + d)
                T.Segment[0].ind_normal = ind_normal
                T.Segment[0].ind_reverse = ind_normal[::-1]

                if np.size(seg) > 1:
                    for s in range(1, np.size(seg)):
                        New_S = TIEclass.segment_OBJ(s + 1, [], [], [], [], [], [], [])
                        ind_normal = np.arange(seg[s - 1], seg[s] + d)
                        New_S.ind_normal = ind_normal
                        New_S.ind_reverse = ind_normal[::-1]
                        T.Segment.append(New_S)

                Last_S = TIEclass.segment_OBJ(
                    np.size(seg) + 1, [], [], [], [], [], [], []
                )
                ind_normal = np.arange(seg[-1], n)
                Last_S.ind_normal = ind_normal
                Last_S.ind_reverse = ind_normal[::-1]
                T.Segment.append(Last_S)

        T.convexityTH = peakthresh
    return TRACES


def tie(
    TRACES: List[TIEclass.trace_OBJ],
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    pth: List[int] = [3, 9, 18],
    seg: bool = False,
    peakthresh: List[int] = [100, 15],
) -> List[TIEclass.trace_OBJ]:
    """
    Performs the entire TIE analysis, step by step.

    Parameters
    ----------
    TRACES : list of trace_OBJ (TIE_classes.trace_OBJ)
        List of trace objects for TIE analysis.
    X, Y : numpy.ndarray
        Vectors of X and Y coordinates of the analyzed extent.
    Z : numpy.ndarray
        Matrix of Z values of the analyzed extent (size [len(X), len(Y)]).
    pth : list of int, optional
        Planarity thresholds for TIE classification. Default is [3, 9, 18].
    seg : bool, optional
        Boolean indicating if the trace should first undergo a segmentation
        process before extracting the trace information. Default is False.
    peakthresh : list of int, optional
        Peak thresholds for convexity [peak prominence, peak width]. Default
        is [90, 15], and it is not used if seg=False.

    Returns
    -------
    TRACES : list of trace_OBJ
        The same list with added TIE information.
    """

    [mX, mY] = np.meshgrid(X, Y)
    mX = np.fliplr(mX)
    mZ = np.fliplr(Z)
    if seg == True:
        TRACES = segmentTRACE(TRACES, mX, mY, mZ, peakthresh)

    TRACES = extractChords(TRACES, mX, mY, mZ)
    TRACES = extractAlpha(TRACES)
    TRACES = extractChdPlanes(TRACES, mX, mY, mZ)
    TRACES = extractBeta(TRACES)
    TRACES = classifyTRACE(TRACES, pth)
    TRACES = extractOrientBars(TRACES)

    return TRACES
