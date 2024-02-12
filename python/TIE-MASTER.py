#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 13:33:55 2021
@author: Anna Rauch

TIE - MASTER
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from TIE import TIE_core as TIEcr
from TIE import TIE_load as TIEld
from TIE import TIE_visual as TIEvis


def path_to_str(obj):
    if isinstance(obj, Path):
        return str(obj)

# %% DEFINE DATA

sheet = 1226
GAsheet = 143

cur_dir = os.path.dirname(os.path.realpath(__file__))

path_leg =  path_to_str(Path('./BedrockColors/').resolve())

# DEM folder
DemFolderPath = path_to_str(Path('./data/swissALTI3d').resolve())




# GEOCOVER V2 folder (SHP)
GeocPath = path_to_str(Path('./data/geocover/').resolve())
# ANALYZED ZONE: Widdergalm (Boltigen, 1226)
x = (2592150.0, 2593850.0)
y = (1165850.0, 1167500.0)
zone_name = "Widdergalm"  # optional
sheet_name = "Boltigen"  # optional

# ATTRIBUTE in SHP representing LITSTRAT and TECTONIC BOUNDARY TYPE  
# (must be a numerical value!)
attr_BED = 'SYMBOL'
attr_TEC = 'SYMBOL'

# %% LOAD AND READ DATA & EXTRACT TRACES

BEDbig, TECbig, OMbig = TIEld.loadGeocover(GAsheet, GeocPath)
DEMbig = TIEld.loadLKdem(sheet, DemFolderPath)

extent = TIEld.createExtentPLG(x, y, BEDbig.crs)
DEM = TIEld.cropDEMextent(DEMbig, extent)
del DEMbig

BEDshp = TIEld.adaptSHAPE2DEM(BEDbig, DEM)
TECshp = TIEld.adaptSHAPE2DEM(TECbig, DEM)
OMshp = TIEld.adaptSHAPE2DEM(OMbig, DEM)
del TECbig
del BEDbig
del OMbig

BEDrst = TIEld.rasterizeSHP(BEDshp, attr_BED, DEM)
TECrst = TIEld.rasterizeSHP(TECshp, attr_TEC, DEM)

faults = TIEld.extractTraces(TECrst, 'L')
BTRrst = TIEld.identifyTRACE(BEDrst, faults)
traces = TIEld.extractTraces(BTRrst, 'PLG')
traces = TIEld.findNeighType(traces, BEDrst)

# %% TIE-ANALYSIS

traces = TIEcr.tie(traces, DEM['x'], DEM['y'], np.flipud(DEM['z']), seg=True)
faults = TIEcr.tie(faults, DEM['x'], DEM['y'], np.flipud(DEM['z']), seg=True)

trace = traces[0]
s = trace.Segment[0]
chord = s.Chords[0]

# # %% MANIPULATE TRACES AND SEGMENTS
#
# # no segmentation
# traces = TIEcr.clearTIE(traces)
# traces = TIEcr.tie(traces, DEM['x'], DEM['y'], np.flipud(DEM['z']), seg=False)
#
# # change segmentation parameters on specific trace
# traces = TIEcr.clearTIE(traces)
# traces = TIEcr.tie(traces, DEM['x'], DEM['y'], np.flipud(DEM['z']), seg=True)
# tr_nb = 3
# traces = TIEcr.changeTHseg(traces, DEM['x'], DEM['y'], np.flipud(DEM['z']),
#                            tr_nb, newTH=[90, 15])
#
# # merge two segments in specific trace
# s_nbs = [2, 3]
# traces = TIEcr.mergeSeg(traces, tr_nb, s_nbs,
#                         DEM['x'], DEM['y'], np.flipud(DEM['z']))
#
# # merge traces
# fn = [10, 7, 17, 8, 18]
# faults = TIEcr.mergeTraces(faults, fn, DEM['x'], DEM['y'], np.flipud(DEM['z']),
#                            seg=True)
#
# # re-segment a segment (as if it was a single trace)
# n = 8
# s = 2
# faults = TIEcr.segSeg(faults, n, s, DEM['x'], DEM['y'], np.flipud(DEM['z']))

# %% SHOW RESULTS

# define display variables
mx, my = np.meshgrid(DEM['x'], DEM['y'])
mz = np.flipud(DEM['z'].copy())
mx = mx.astype(float)
my = my.astype(float)

# define legend (based on a txt-file)

legendfile = path_to_str(Path(path_leg, sheet_name + ".txt"))
mc, cmap, formations = TIEvis.bed2cmap(BEDrst, legendfile, leg_labels=True)

# %% SIGNAL HEIGHT DIAGRAM

pth = [3, 9, 18, 90]
fig_SH = TIEvis.sigHdiagram(traces, pth, scale="log")

# %% INDIVIDUAL SIGNALS

nt = 7
fig_Signal = TIEvis.showSignal(traces[nt - 1])
fig_Stereo = TIEvis.showSigStereo(traces[nt - 1])

# %% OVERVIEW MAP (3D)
fig_OV3D = TIEvis.showOverview3D(mx, my, mz, mc, cmap=cmap,
                                 BTraces=traces, FTraces=faults,
                                 WithlabelsF=False)

# %% OVERVIEW MAP (2D)

fig_sidemap, ax = TIEvis.showOverviewMap(mx, my, mc, mz=mz, cmap=cmap,
                                         BTraces=traces, FTraces=faults,
                                         leg_labels=formations, WithSeg=True)
ax.set_title(zone_name)
plt.show()

# %% 3D PLOT with TIE ANALYSIS
fig_3dTIE = TIEvis.showTIEmap(mx, my, mz, mc=mc, cmap=cmap,
                              MainTrace_Set=traces,
                              AuxTrace_Set=faults, ShowBars=True)
#
