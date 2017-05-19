# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 15:09:48 2012

@author: Alex Stum
Created May 18, 2017
Collaborations with Emily Carbone and Zach Gustafson
Calculates multipath flow accumulation by slicing down through a DEM

"""

import numpy as np
from skimage import measure
from scipy import ndimage as nd
from osgeo import gdal


demP = r"A:\...\fill"
outN = r"A:\...\multi.tif"
    

def labelFlats(source):
    """Identifies and labels regions of the same elevation (flats).
    
    This function identifies flats (pixels with no lower outlet) in digital
    elevation surfaces. These flats, contiguous regions of pixels of the 
    same elevaiton without an outlet, are uniquely labeled. 
    
    - **Parameters**::
        
        Name:       Type:        Description:
        source      numpy array  filled DEM
    
    - **Returns**::
        
        numpy array      labeld flats
        integer          number of classes (the highest class ID value)
        
    .. note:: Intended for sink filled surfaces. 
    .. note:: For every flat there will be at least one pixel, of the same\ 
    elevation, along the perimeter which will be excluded. This is by design,\
    these will be identified as pour points in the :func:`flatThing` function.
    """
    #3x3 kernel excluding the central pixel
    cc = (np.array([0,0,0,1,1,2,2,2]),np.array([0,1,2,0,2,0,1,2]))
    RxC = source.shape
    out=np.zeros(RxC,dtype=np.int8)     #initialize output array
    for i,j in zip(*cc):
        #convolve to sum the number neighboring pixels > or = to central pixel
        out[1:-1,1:-1] += source[i:RxC[0]-2+i,j:RxC[1]-2+j]>=source[1:-1,1:-1]
        #if all 8 neartest neighbors are >=, then pixel is part of a flat
        flat = out==8
    return measure.label(flat,background=False,return_num=True)
        

    
def flatThing(e_c,flats,accum,elev):
    """Distributes flow accumulation for a flat region.
    
        Finds the zonal sum of accumulation (accumulation incident to all pixels
        within flat region. Evenly distributes accumulation to all pour points.

    - **Parameters**::
    
        Name:       Type:           Description:
        e_c         tuple           two numeric values: elvation and flat ID
        flats       numpy array     labeld flat regions
        accum       numpy array     current flow accumulation
        elev        numpy array     DEM (sinks filled)
    
    - **Returns**::
        
        numpy array     updated multipath flow accumulation
        
    .. note:: The presence of sinks will result in division by zero and no distributed\
        accumulation from that flat region.
    .. seealso:: :func:`labelFlats`

    """
    F = flats==e_c[1]   #current flat index
    cSum = accum[F].sum()   #zonal sum
    accum[F] = cSum          #All flat pixels receive flow incident to flat
    expand = nd.binary_dilation(F,structure=np.ones((3,3)))
    #Discover pour points: pixels not in flat but along perimeter <= flat elevation
    accum[~F&expand] = np.where(elev[~F&expand]<=e_c[0],\
        accum[~F&expand]+cSum/float((elev[~F&expand]<=e_c[0]).sum()),accum[~F&expand])
    return accum

    
    
def multipath(elev):
    """Calculates multipath flow accumulation.
    
    To calculate the multipath flow accumulation by proportionally distributing accumulated
    flow (upslope contributing pixels) to all adjacent downslope pixels proportional
    to the focal elvation difference. Pixels in flats receive the zonal sum of 
    all pixels within associated flat region. The zonal accumulation is distributed 
    evenly to pour points.

    - **Parameters**::
    
        Name:      Type:           Description:
        elev       numpy array     DEM (sinks filled)
    
    - **Returns**::
        
        numpy array     multipath flow accumulation
        
    .. note:: Inteneded for sink filled DEM's.
    .. note:: Accummulation incident to outside rows and columns is not distributed; \
        recommend removing outside rows and columns.
        
    .. seealso:: :func:`labelFlats` 
    :func:`flatThing`
    """
    flats,top = labelFlats(elev)
    flatArray = elev[1:-1,1:-1].flatten('F')
    flatFlats = flats[1:-1,1:-1].flatten('F')
    combo = np.array(zip(flatArray,flatFlats),dtype=[('e','<f8'),('c','<i4')])
    indexArray = np.argsort(combo,axis=0,order=['e','c'],kind='mergesort')
    nrows=elev.shape[0]-2
    del flatArray, flatFlats
    
    accum = np.ones_like(elev).astype(float)
    curC = None     #stores current flat ID and elevation
    for i in indexArray[::-1]:
        if combo[i][1]:     #if current pixel is part of a flat region
            #is there a current flat and same region as current pixel
            if curC and (curC[1]!=combo[i][1]):
                accum = flatThing(curC,flats,accum,elev)
                curC = combo[i]
                continue
            #Whether current pixel is associated with current flat region or is the start of a new one
            else:   
                curC = combo[i]
                continue
        else:
            if curC:  #if current pixel is not member of current flat
                accum = flatThing(curC,flats,accum,elev)
                curC = None
                        
            y=i%nrows
            x=i/nrows
        
            win = elev[y:y+3,x:x+3]
            #find difference tween lower cells and center cell
            diff=np.where(win<win[1,1],win[1,1]-win,0)
            #proportionally distribute
            accum[y:y+3,x:x+3] += diff / diff.sum()*accum[y+1,x+1]
    return accum


    
dem = gdal.Open(demP,gdal.GA_ReadOnly)
demA = dem.GetRasterBand(1).ReadAsArray(0,0,dem.RasterXSize, dem.RasterYSize)

multi = multipath(demA)

driver = gdal.GetDriverByName('GTIFF')
driver.Register()
outFile = driver.Create(outN,multi.shape[1],multi.shape[0],1, gdal.GDT_Float32)
outFile.GetRasterBand(1).WriteArray(multi,0,0)
outFile.GetRasterBand(1).ComputeStatistics(False)

geoTran = dem.GetGeoTransform()
geoTran = (geoTran[0]+geoTran[1],geoTran[1],geoTran[2],\
           geoTran[3]+geoTran[5],geoTran[4],geoTran[5])
outFile.SetGeoTransform(geoTran)
outFile.SetProjection(dem.GetProjection())
outFile = None




