# Standard imports
import datetime
import os
import sys
import time

# Related third party imports
import descartes
import gdal
import geopandas as gpd
import glob
import itertools
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import os
import pickle
import rasterio
import shapely
from os import listdir
from rasterio.plot import show
from rasterio.mask import mask
from rasterstats import zonal_stats
from tqdm import tqdm
from descartes import PolygonPatch

from constants import KMRS_LIST

PICKLE_PATH = r"E:\work\ARRS susa"

with open(os.path.join(PICKLE_PATH, "affine.pickle"), "rb") as handle:
    affine = pickle.load(handle)

with open(os.path.join(PICKLE_PATH, "crs.pickle"), "rb") as handle:
    crs = pickle.load(handle)

with open(r"E:\work\ARRS susa\filtered_gerk_gdf-test_section.pickle", "rb") as handle:
    poljine_gdf_2017 = pickle.load(handle)["2017"]

poljine_gdf_2017["CENTROID_X"] = poljine_gdf_2017.geometry.centroid.apply(lambda p: p.x)
poljine_gdf_2017["CENTROID_Y"] = poljine_gdf_2017.geometry.centroid.apply(lambda p: p.y)

crop_dict = dict.fromkeys(KMRS_LIST.keys())
poljine_combinations_by_crop_type = {}
crop_distance = {}
poljine_inside_buffer = {key: {} for key in KMRS_LIST}
crop_ids = KMRS_LIST.keys()

for key, __ in crop_dict.items():
    crop_dict[key] = poljine_gdf_2017.loc[poljine_gdf_2017["SIFRA_KMRS"] == key]

for crop, poljine_ids in crop_dict.items():
    poljine_ids = list(set(poljine_ids["POLJINA_ID"]))
    combination_pairs = list(itertools.combinations(poljine_ids, 2))
    poljine_combinations_by_crop_type[crop] = combination_pairs
    for crop, poljine_pairs in poljine_combinations_by_crop_type.items():
        poljine_gdf_by_crop = crop_dict[crop]
        distances = []
        for gerk_1, gerk_2 in poljine_pairs:
            # Distance between gerk 1 and gerk 2
            x1 = float(
                poljine_gdf_by_crop["CENTROID_X"].loc[
                    poljine_gdf_by_crop["POLJINA_ID"] == gerk_1
                ]
            )
            y1 = float(
                poljine_gdf_by_crop["CENTROID_Y"].loc[
                    poljine_gdf_by_crop["POLJINA_ID"] == gerk_1
                ]
            )
            x2 = float(
                poljine_gdf_by_crop["CENTROID_X"].loc[
                    poljine_gdf_by_crop["POLJINA_ID"] == gerk_2
                ]
            )
            y2 = float(
                poljine_gdf_by_crop["CENTROID_Y"].loc[
                    poljine_gdf_by_crop["POLJINA_ID"] == gerk_2
                ]
            )
            distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            distances.append(distance)
        if (
            len(distances) == 0
        ):  # if there is only one gerk, then set distance to 15000 m
            crop_distance[crop] = 15000
        else:
            crop_distance[crop] = sum(distances) / len(distances)

for __, source_poljina in poljine_gdf_2017.iterrows():
    source_crop = source_poljina["SIFRA_KMRS"]
    no_source_crops = [x for x in crop_ids if x != source_crop]
    for crop in no_source_crops:
        buffer_distance_for_crop = crop_distance[crop]
        print(crop_distance)
        source_poljina_buffer = source_poljina["geometry"].centroid.buffer(
            2 * buffer_distance_for_crop
        )
        inside_poljine = []
        for __, poljina in poljine_gdf_2017[
            poljine_gdf_2017.SIFRA_KMRS == crop
        ].iterrows():
            x = poljina["CENTROID_X"]
            y = poljina["CENTROID_Y"]
            poljina_centroid = shapely.geometry.Point(x, y)
            if poljina_centroid.within(source_poljina_buffer):
                inside_poljine.append(poljina["POLJINA_ID"])
        poljine_inside_buffer[crop][source_poljina["POLJINA_ID"]] = inside_poljine


for satellite_pickle in tqdm(
    glob.glob(os.path.join(PICKLE_PATH, "satellite*2017*.pickle"))
):
    with open(satellite_pickle, "rb") as handle:
        satellite_data = pickle.load(handle)

    date = satellite_data["date"]
    image = satellite_data["image"]
    mask = satellite_data["mask"]

    for KMRS, value in poljine_inside_buffer.items():

        for source_poljina, poljine_list in value.items():
            print(poljine_list)
            if not poljine_list:
                continue
            selected_poljine_gdf = poljine_gdf_2017[
                poljine_gdf_2017["POLJINA_ID"].isin(poljine_list)
            ]
            zs_gerks = zonal_stats(
                selected_poljine_gdf.geometry,
                np.where(mask == 100, image, float("nan")),
                affine=affine,
                stats="mean",
                nodata=float("nan"),
            )
            print(
                np.mean(
                    [value["mean"] for value in zs_gerks if value["mean"] is not None]
                )
            )

