"""
gerk_value_assignment_based_on_neigbours.py
Problem: assigns values to gerks from satellite images based on values in its neighbourhood.
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
import datetime
import os
import sys
import time

# Related third party imports
from collections import defaultdict
import descartes
import gdal
import geopandas as gdf
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
from tqdm import tqdm
from os import listdir
from rasterio.plot import show
from rasterio.mask import mask
from rasterstats import zonal_stats
from descartes import PolygonPatch

from constants import SATELLITE_IMGS_PATH

satellite_data = defaultdict(dict)
for f in os.listdir(SATELLITE_IMGS_PATH):
    if f.endswith(".tif") and "mask" in f and "2017" in f:
        mask_date = f[0:8]
        date = datetime.datetime.strptime(mask_date, "%Y%m%d").date()
        satellite_data[date]["mask"] = os.path.join(SATELLITE_IMGS_PATH, f)

    elif f.endswith(".tif") and "mask" not in f and "2017" in f:
        image_date = f[0:8]
        date = datetime.datetime.strptime(image_date, "%Y%m%d").date()
        satellite_data[date]["image"] = os.path.join(SATELLITE_IMGS_PATH, f)

for date in tqdm(sorted(satellite_data.keys())):
    image_path = satellite_data[date]["image"]
    mask_path = satellite_data[date]["mask"]

    image_object = rasterio.open(image_path).read(1)
    mask_object = rasterio.open(mask_path).read(1)

    satellite = {
        "date": date,
        "image": image_object,
        "mask": mask_object,
    }

    satellite_path = "satellite-test_section_{}.pickle".format(
        date.strftime("%Y-%m-%d")
    )
    with open(satellite_path, "wb") as fp:
        pickle.dump(satellite, fp, protocol=4)

#     def zonal_stats(
#         self, yearly_gerks_gdf, gerks_inside_buffer_list, gerk_id, crop, mean_distance
#     ):
#         """
#         Returns mean raster pixel value for GERK id.
#         """
#         geoms = yearly_gerks_gdf[yearly_gerks_gdf["SIFRA_KMRS"] == crop][
#             yearly_gerks_gdf["GERK_PID"].isin(gerks_inside_buffer_list)
#         ].geometry
#         image = self.raster_open()
#         image_read = self.image_read()
#         masked_image = self.apply_mask()
#         mask = self.mask_open()
#         affine = image.transform
#         masked_affine = mask.transform
#         if len(geoms) > 0:
#             gerk_stats = zonal_stats(
#                 geoms, image_read, affine=affine, stats=["count", "mean"]
#             )
#             masked_gerk_stats = zonal_stats(
#                 geoms, masked_image, affine=masked_affine, stats=["count", "mean"]
#             )
#             return gerk_stats, masked_gerk_stats

#     def plot(
#         self, yearly_gerks_gdf, gerks_inside_buffer_list, gerk_id, crop, mean_distance
#     ):
#         gerks_with_crop = yearly_gerks_gdf[
#             yearly_gerks_gdf["SIFRA_KMRS"] == crop
#         ].geometry
#         geoms = yearly_gerks_gdf[yearly_gerks_gdf["SIFRA_KMRS"] == crop][
#             yearly_gerks_gdf["GERK_PID"].isin(gerks_inside_buffer_list)
#         ].geometry
#         gerk = yearly_gerks_gdf[yearly_gerks_gdf["GERK_PID"].isin([gerk_id])].geometry
#         crop_gerk = yearly_gerks_gdf[
#             yearly_gerks_gdf["GERK_PID"] == gerk_id
#         ].SIFRA_KMRS.any()
#         crop_geoms = yearly_gerks_gdf[yearly_gerks_gdf["SIFRA_KMRS"] == crop][
#             yearly_gerks_gdf["GERK_PID"].isin(gerks_inside_buffer_list)
#         ]["SIFRA_KMRS"]
#         image = self.raster_open()
#         image_date = self.raster[12:20]
#         image_year = int(image_date[0:4])
#         image_month = int(image_date[4:6])
#         image_day = int(image_date[6:])
#         date = datetime.date(image_year, image_month, image_day)
#         masked_image = self.apply_mask()
#         annotation = []

#         for feature_id in gerks_inside_buffer_list:
#             centroid_x = yearly_gerks_gdf[
#                 yearly_gerks_gdf["GERK_PID"] == feature_id
#             ].centroid.x.item()
#             centroid_y = yearly_gerks_gdf[
#                 yearly_gerks_gdf["GERK_PID"] == feature_id
#             ].centroid.y.item()
#             fid = yearly_gerks_gdf[yearly_gerks_gdf["GERK_PID"] == feature_id][
#                 "GERK_PID"
#             ].item()
#             annotation.append([fid, centroid_x, centroid_y])
#         annotation.append(
#             [
#                 gerk_id,
#                 yearly_gerks_gdf[
#                     yearly_gerks_gdf["GERK_PID"] == gerk_id
#                 ].centroid.x.item(),
#                 yearly_gerks_gdf[
#                     yearly_gerks_gdf["GERK_PID"] == gerk_id
#                 ].centroid.y.item(),
#             ]
#         )

#         for feature_id in yearly_gerks_gdf[yearly_gerks_gdf["SIFRA_KMRS"] == crop][
#             "GERK_PID"
#         ]:
#             centroid_x = yearly_gerks_gdf[
#                 yearly_gerks_gdf["GERK_PID"] == feature_id
#             ].centroid.x.item()
#             centroid_y = yearly_gerks_gdf[
#                 yearly_gerks_gdf["GERK_PID"] == feature_id
#             ].centroid.y.item()
#             fid = yearly_gerks_gdf[yearly_gerks_gdf["GERK_PID"] == feature_id][
#                 "GERK_PID"
#             ].item()
#             annotation.append([fid, centroid_x, centroid_y])

#         show(masked_image, transform=image.transform)
#         ax = plt.gca()
#         inside_gerk_patches = [
#             PolygonPatch(feature, edgecolor="red", facecolor="none", linewidth=2)
#             for feature in geoms
#         ]
#         gerk_patches = [
#             PolygonPatch(
#                 feature,
#                 edgecolor="cyan",
#                 facecolor="cyan",
#                 linewidth=2,
#                 alpha=0.5,
#                 label="{}".format(crop_geoms.any()),
#             )
#             for feature in gerks_with_crop
#         ]
#         if len(inside_gerk_patches) > 0:
#             ax.add_patch(
#                 PolygonPatch(
#                     gerk.item().buffer(2 * mean_distance),
#                     facecolor="purple",
#                     edgecolor="purple",
#                     alpha=0.3,
#                     zorder=2,
#                 )
#             )
#             ax.add_collection(
#                 mpl.collections.PatchCollection(gerk_patches, match_original=True)
#             )
#             ax.add_patch(
#                 PolygonPatch(
#                     gerk.item(), facecolor="red", edgecolor="red", alpha=0.5, zorder=2
#                 )
#             )
#             ax.legend(
#                 handles=[
#                     gerk_patches[0],
#                     PolygonPatch(
#                         gerk.item(),
#                         facecolor="red",
#                         edgecolor="red",
#                         alpha=0.5,
#                         zorder=2,
#                         label="{}".format(crop_gerk),
#                     ),
#                     PolygonPatch(
#                         gerk.item().buffer(mean_distance),
#                         facecolor="purple",
#                         edgecolor="purple",
#                         alpha=0.3,
#                         zorder=2,
#                         label="Search Area",
#                     ),
#                 ]
#             )

#             for a in annotation:
#                 ax.annotate(a[0], (a[1], a[2]))
#             ax.set_title("EVI2 {}".format(date))
#             plt.show


# # if __name__ == '__main__':
# #    main()
