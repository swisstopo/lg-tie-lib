# -*à-- coding: utf-8 -*-
"""
Created on Tue Apr 13 11:04:48 2021
@author: Anna Rauch

This script contains a set of general functions/methods that allow you 
to load the data and extract the traces in a way needed to perform a
TIE-analysis.
"""

import glob
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio as rst
from TIE import TIE_classes as TIEclass
from TIE import TIE_general as TIEgen
from rasterio.mask import mask
from rasterio.merge import merge
from shapely.geometry import Polygon
from shapely.geometry import box


def adaptSHAPE2DEM(shapefile, DEM):
    """ ADAPTS SHAPEFILE EXTENT TO DEM
    loads shapefile and crops its extent according to the extent given by
    the DEM.
    
    INPUT:
    ---------
    shapefile:  geopandas shapefile object 
    DEM:        dicitonary of the DEM and coordinate data obtained with 
                TIE_load.cropDEMextent
    RETURNS:
    ---------
    clipped shapefile """

    DEMmeta = DEM['meta']
    DEMtrans = DEMmeta['transform']
    DEMz = DEM['z']

    x = (DEMtrans[2], DEMtrans[2] + DEMtrans[0] * DEMz.shape[1])
    y = (DEMtrans[5], DEMtrans[5] + DEMtrans[4] * DEMz.shape[0])
    crs_shp = shapefile.crs
    df_ext = createExtentPLG(x, y, crs_shp)

    shapefile_clipped = gpd.overlay(shapefile, df_ext, how='intersection')

    return shapefile_clipped


def createExtentPLG(x, y, crs_shp):
    """ CREATES EXTENT POLYGON
    creates en georeferenced polygon (geopandas) covering the extent defined
    with x- and y-coordinates.
    
    INPUT:
    ---------
    x,y:        coordinates of wanted polygon
    crs_shp:    coordinate system of shape (geopandas object
                                            )
    RETURNS:
    ---------
    geopandas object (shapefile) with polygon extent """

    extent = gpd.GeoSeries(Polygon([(x[0], y[0]), (x[0], y[1]), (x[1], y[1]), (x[1], y[0])]))
    extent = extent.set_crs(crs_shp)
    df_ext = gpd.GeoDataFrame({'geometry': extent})

    return df_ext


def cropDEMextent(geotif, shapefile):
    """ CROP DEM WITH SHAPEFILE
    crops DEM extent according to a PLG shapefile (here often extent shapefile).
    
    INPUT:
    ---------
    geotif:     handle to opened geotif (rasterio)
    shapefile:  handle to opened shapefile (geopandas object)

    RETURNS:
    ---------
    dictionary containing coordinate data """

    bbox = box(shapefile.total_bounds[0],
               shapefile.total_bounds[1],
               shapefile.total_bounds[2],
               shapefile.total_bounds[3])
    gbbox = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=shapefile.crs)

    def getFeatures(gdf):
        import json
        return [json.loads(gdf.to_json())['features'][0]['geometry']]

    coords = getFeatures(gbbox)

    DEM, out_trans = mask(geotif, coords, crop=True)
    z = DEM[0]

    xend = out_trans[2] + out_trans[0] * z.shape[1] + 1
    x1 = out_trans[2] + 1
    y1 = out_trans[5] + out_trans[4] * z.shape[0] + 1
    yend = out_trans[5] + 1

    x = np.arange(x1, xend, out_trans[0])
    y = np.arange(y1, yend, out_trans[0])

    out_meta = geotif.meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": DEM.shape[0],
                     "width": DEM.shape[1],
                     "transform": out_trans})

    DEM = {'z': z, 'x': x, 'y': y, 'meta': out_meta}
    return DEM


def extractTraces(TRmatrix, shape):
    """ EXTRACT TRACES as TRACE OBJECTS
    extracts individual, sorted traces (as a trace_OBJ) based on a trace matrix
    
    INPUT:
    ---------
    TRmatrix:   matrix containing the traces (binaryimage), in form of
                it's trace matrix
    shape:      'L' for polylines (e.g. faults), 'PLG' for traces extracted
                from polygones (e.g. bedrock interface traces)
    RETURNS:
    ---------
    traces:     list of trace objects (trace_OBJ) defined in TIE_classes. """

    kind = np.unique(np.extract(np.isnan(TRmatrix) == False, TRmatrix))
    TRACE = []

    if np.size(kind) > 0:
        n = 1
        for r in kind:
            tr = np.zeros(np.shape(TRmatrix.flatten()))
            tr[(TRmatrix.flatten() == r)] = 1
            tr_mat = np.reshape(tr, np.shape(TRmatrix))

            if shape == 'L':
                nmbN = TIEgen.nmbNeighbors(tr_mat).flatten()
                BP = (nmbN > 2).nonzero()
                tr[BP] = 0
                tr_mat = np.reshape(tr, np.shape(TRmatrix))

            from skimage import measure
            CC = measure.label(tr_mat)
            CCmat = CC.flatten()
            ntr = np.unique(CC)[1:]

            for k in ntr:
                index = (CCmat == k).nonzero()[0]
                if np.size(index) > 5:
                    index = TIEgen.sortLine(index, np.shape(tr_mat))
                    i_ind = np.arange(0, np.size(index)).astype(int)
                    seg = TIEclass.segment_OBJ(1, [], i_ind, i_ind[::-1],
                                               [], [], [], [])
                    TRACE.append(TIEclass.trace_OBJ(n, index, r, [seg], None, []))
                    n = n + 1
    return TRACE


def findNeighType(TRACES, BEDrst):
    """ FIND NEIGHBOR BEDROCK TYPES
    extracts the TWO bedrock types stored in BEDrst, which form a bedrock 
    inteface trace

    INPUT
    ----------
    BEDrst:     bedrock matrix (rasterized version of bedrock shapefile) 
    TRACES:     list of trace_OBJ storing trace information of bedrock 
                interface traces
    OUTPUT
    ----------
    TRACES (same list) with added SECOND bedrock type defining a boundary
    type """

    for T in TRACES:
        mid = int(round(np.size(T.index) / 2))
        ind = int(T.index[mid])
        neigh = TIEgen.neighborPoints(ind, np.shape(BEDrst)[0],
                                      np.shape(BEDrst)[1], 4)
        BEDfl = BEDrst.flatten()
        rtype = BEDfl[ind]

        for k in neigh:
            if BEDfl[k] != rtype and np.isnan(BEDfl[k]) == False:
                ntype = BEDfl[k]
                break
            else:
                ntype = float("NaN")
        T.type = [rtype, ntype]

    return TRACES


def identifyTRACE(BEDrst, FAULTS):
    """ IDENTIFIES TRACES IN A BEDROCK MATRIX   
    identifies traces in a bedrock matrix based on the bedrock distinction
    and the fault traces already extracted and creates a new matrix (same
    resolution) containing the traces only
    
    INPUT:
    ----------
    BEDrst:     matrix with Bedrock data (see loadBedrock and rasterizeBedrock)  
    FAULTS:     list of trace_OBJ containing FAULT information
    
    RETURNS:
    ----------
    TRmatrix:   matrix (size BEDrst) with trace information
    """

    TRmatrix = np.zeros(np.shape(BEDrst)).flatten()
    BEDwork = BEDrst.flatten()

    if FAULTS:
        faultind = np.concatenate([FAULTS[fi].index for fi in range(np.shape(FAULTS)[0])])
        faultind = [int(np.unique(fi)) for fi in faultind]
        fneighind = []
        for i in faultind:
            neigh = TIEgen.neighborPoints(i, np.shape(BEDrst)[0],
                                          np.shape(BEDrst)[1], 8)
            fneighind = np.concatenate((fneighind, neigh))
        fneighind = np.unique(fneighind)
        fneighind = [int(np.unique(fni)) for fni in fneighind]
    else:
        faultind = []
        fneighind = []

    BEDwork[faultind + fneighind] = float('NaN')
    indpos = (np.isnan(BEDwork) == False).nonzero()
    for i in indpos[0]:
        neigh = TIEgen.neighborPoints(i, np.shape(BEDrst)[0],
                                      np.shape(BEDrst)[1], 4)
        neigh = neigh.compress((np.isnan(BEDwork[neigh]) == False).flat)
        if any(BEDwork[neigh] > BEDwork[i]):
            TRmatrix[i] = BEDwork[i]

    TRmatrix[(TRmatrix == 0).nonzero()] = float('NaN')
    TRmatrix = np.reshape(TRmatrix, np.shape(BEDrst))

    return TRmatrix


def loadGeocover(sheet, geoc_path, language="de"):
    """ LOADS GEOCOVER DATA   
    loads Geocover Data (all geology data) according to the Geocover structure
    
    INPUT:
    ----------
    sheet:      Identification number of LK sheet (Landeskarte 1.25'000)  
    geoc_path:  Path to Geocover folder
    language:   "de" for the german version (default), or "fr" for the French
                version (optional parameter)
    
    RETURNS:
    ----------
    geopandas handle to shapefile of Bedrock_PLG.shp, Tectonic_Boundaries_L.shp
    and Planar_Structures_PT.shp
    """

    path_whole = geoc_path + "/GC-V-V2-" + str(sheet) + "/Data/SHP/" + language
    BED_path = Path(path_whole + "/Bedrock_PLG.shp")
    TEC_path = Path(path_whole + "/Tectonic_Boundaries_L.shp")
    OM_path = Path(path_whole + "/Planar_structures_PT.shp")

    BED = gpd.read_file(BED_path)
    TEC = gpd.read_file(TEC_path)
    OM = gpd.read_file(OM_path)

    return BED, TEC, OM


def loadLKdem(sheet, path_alti3D_folder):
    """ LOADS GEOCOVER DATA   
    loads DEM (swissALTI3D), glues the different patches together
    
    INPUT:
    ----------
    sheet:          Identification number of LK sheet (Landeskarte 1:25'000)  
    path_alti3d:    Path to swissALTI3D folder
    
    RETURNS:
    ----------
    rasterio handle to DEM of whole LK sheet (Landeskarte 1:25'000)
    """

    search_criteria = "/swiss_" + str(sheet) + "_*.tif"
    dem_fps = glob.glob(path_alti3D_folder + search_criteria)

    dem_mosaic = []
    for d in dem_fps:
        dem1 = rst.open(d)
        dem_mosaic.append(dem1)

    mosaic, out_trans = merge(dem_mosaic)
    out_meta = dem1.meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans})

    # FIXME: No static paths inside script body
    out_fp = Path("D:/TIE/python/temp/swiss_" + str(sheet) + "_merged.tif")
    with rst.open(out_fp, "w", **out_meta) as dest:
        dest.write(mosaic)

    merged = rst.open(out_fp)

    return merged


def rasterizeSHP(shp, attribute, DEM):
    """ RASTERIZES SHAPEFILE  
    rasterizes a shapefile according to a specific attribute field value
    
    INPUT:
    ----------
    shp:        opened geopandes handle to a shapefile  
    attribute:  attribute in shapefile that will be needed to
                distinguish between different types of tectonic boundaries or
                litho-stratigraphic units. The attribute value must be a number.
    DEM:        dictionary containing coordinates (x,y and z) of analyzed zone
                (see crop2DEMextent)
    
    RETURNS:
    ----------
    raster matrix
    """

    KIND = getattr(shp, attribute)
    shp["KIND_NUM"] = KIND.astype(int)

    res = DEM['meta']['transform'][0]
    mat = np.zeros(np.shape(DEM['z']))
    matx, maty = np.meshgrid(DEM['x'], DEM['y'])

    from geocube.api.core import make_geocube
    cube = make_geocube(shp,
                        measurements=["KIND_NUM"],
                        resolution=(res, -res))

    xc = cube.KIND_NUM['x'].values
    yc = cube.KIND_NUM['y'].values

    xci = np.intersect1d(DEM['x'][::-1], xc, return_indices=True)[1]
    yci = np.intersect1d(DEM['y'], yc, return_indices=True)[1]
    matxi, matyi = np.meshgrid(xci, yci)
    matxi = np.fliplr(matxi)
    matv = cube.KIND_NUM.values.flatten()
    mat[matyi.flatten(), matxi.flatten()] = matv

    return mat
