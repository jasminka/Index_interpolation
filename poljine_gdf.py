"""
poljine_gdf.py
Problem: returns a model for interpolation of raster pixel values inside GERKs polygons.
Target System: Windows
Requirements: #.
Interface: Command-line
Author: Jasmina Stajdohar

"""
__version__ = 0.1
__mentainer__ = "jasmina.stajdohar@zrc-sazu.si"
__status__ = "Prototype"
__date__ = "20-08-2019"

# Standard imports
import os
import sys
import time

# Related third party imports
from collections import defaultdict
import descartes
import datetime
import geopandas as gpd
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import shapely
import pickle

from constants import KMRS_LIST, GERK_SHAPEFILES_PATH

kmrs_list = KMRS_LIST

data_dump = defaultdict(dict)
for SHAPEFILE in os.listdir(GERK_SHAPEFILES_PATH):
    if SHAPEFILE.endswith(".shp"):
        shp_name = os.path.splitext(SHAPEFILE)[0]
        if SHAPEFILE.startswith("KMRS"):
            year = shp_name[2:6]
        else:
            year = shp_name[2:6]
        print(year)
        data = gpd.read_file(os.path.join(GERK_SHAPEFILES_PATH, SHAPEFILE))
        data_no_intersect = data[data.INTERSECT == 0]
        data_filter_kmrs = data_no_intersect[
            data_no_intersect.SIFRA_KMRS.isin(kmrs_list)
        ]
        data_filter_size = data_filter_kmrs[data_filter_kmrs.geometry.area > 8000]
        data_dump[year] = data_filter_size

with open("filtered_gerk_gdf-test_section.pickle", "wb") as handle:
    pickle.dump(data_dump, handle, protocol=4)

