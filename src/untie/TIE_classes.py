# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 11:43:27 2021
@author: Anna Rauch

This script contains a set of object definitions (classes), which are the main
structure forms, in which the trace information is stored. It is a tree-based
structure, so that list of objects can be stored within a different object.
As an example: a trace_OBJ stores a list of segment_OBJ, each segment_OBJ
itself contains a list of chordPlane_OBJ...
"""


class chord_OBJ(object):
    def __init__(self, ID, vector, vec_trend, vec_plunge, alpha):
        self.id = ID
        self.vector = vector
        self.vec_trend = vec_trend
        self.vec_plunge = vec_plunge
        self.alpha = alpha


class chordPlane_OBJ(object):
    def __init__(
        self, ID, normal, plane_orient, pole_orient, beta, dist_meters, dist_ratio
    ):
        self.id = ID
        self.normal = normal
        self.plane_orient = plane_orient
        self.pole_orient = pole_orient
        self.beta = beta
        self.dist_meters = dist_meters
        self.dist_ratio = dist_ratio


class geolMap_OBJ(object):
    def __init__(self, name, mx, my, mz, BEDrst, BTraces, FTraces, ATraces, ORmeas):
        self.name = name
        self.mx = mx
        self.my = my
        self.mz = mz
        self.BEDrst = BEDrst
        self.BTraces = BTraces
        self.FTraces = FTraces
        self.ATraces = ATraces
        self.ORmeas = ORmeas


class segment_OBJ(object):
    def __init__(
        self, ID, delta, index_N, index_R, chords, chordplanes, classID, signalheight
    ):
        self.id = ID
        self.delta = delta
        self.ind_normal = index_N
        self.ind_reverse = index_R
        self.Chords = chords
        self.Chordplanes = chordplanes
        self.classID = classID
        self.sigheight = signalheight


class trace_OBJ(object):
    def __init__(self, ID, index, Type, Segment, convexityTH, orientbar):
        self.id = ID
        self.index = index
        self.type = Type
        self.Segment = Segment
        self.convexityTH = convexityTH
        self.orientbar = orientbar
