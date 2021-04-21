"""
evi2_interpolation.py
Problem: assigns evi2 pixel values to gerks based in its neighbourhood.
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
import descartes
import gdal

# import geopandas as gdf
import glob
import itertools
import json
from joblib import Parallel, delayed
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir
import psycopg2
from psycopg2 import sql
import rasterio
from rasterio.mask import mask
from rasterstats import zonal_stats
import shapely
from statistics import mean
import sqlite3
from sqlalchemy import create_engine
from geoalchemy2 import Geometry
import numpy.ma as ma
import pickle
from tqdm import tqdm

# Import from our script
from base import cache
from constants import (
    AGRI_CULTURE_CODE_LIST,
    GERK_SHAPEFILES_PATH,
    N_JOBS,
    SATELLITE_IMGS_PATH,
    CONN_HOST,
    CONN_PORT,
    CONN_DATABASE,
    CONN_USER,
    CONN_PASSWORD,
)
from gerk import Gerks, gerk_shapefiles_list
from raster_image import Raster, evi_images_list, evi_mask_list

EVI_IMAGES = evi_images_list


def db_insert(image, result_dict, yearly_gerks_gdf):
    conn = psycopg2.connect(
        host=CONN_HOST,
        port=CONN_PORT,
        database=CONN_DATABASE,
        user=CONN_USER,
        password=CONN_PASSWORD,
    )

    # INSERT GERKS
    for element in yearly_gerks_gdf.iterrows():
        gerk_pid = element[1]["GERK_PID"]
        cursor = conn.cursor()
        try:
            cursor.execute(
                sql.SQL("insert into \"GERK\" values (%s, '(%s,%s)')"),
                [
                    gerk_pid,
                    element[1]["geometry"].centroid.x,
                    element[1]["geometry"].centroid.y,
                ],
            )
        except psycopg2.errors.UniqueViolation:
            conn.rollback()
        else:
            conn.commit()

        cursor.close()

    # INSERT GERK INDICES
    for gerk, dates in result_dict.items():
        for date, values in dates.items():
            original = values["original_value"]
            cursor = conn.cursor()

            for crop in AGRI_CULTURE_CODE_LIST:
                try:
                    cursor.execute(
                        sql.SQL(
                            'insert into "GERK_INDEX_VALUES" values (%s,%s,%s,%s,%s,%s)'
                        ),
                        [
                            gerk,
                            crop,
                            1,
                            values.get(crop, None),
                            date,
                            crop != original,
                        ],
                    )
                except psycopg2.errors.UniqueViolation:
                    conn.rollback()
                else:
                    conn.commit()

            cursor.close()
    conn.close()


def db_has_gerk_index_date(date):
    conn = psycopg2.connect(
        host=CONN_HOST,
        port=CONN_PORT,
        database=CONN_DATABASE,
        user=CONN_USER,
        password=CONN_PASSWORD,
    )

    cursor = conn.cursor()
    cursor.execute(
        sql.SQL('SELECT COUNT(*) FROM "GERK_INDEX_VALUES" WHERE "image_date"=%s;'),
        [date],
    )
    ngerks = cursor.fetchone()[0]
    cursor.close()
    return ngerks > 0


def process_image(image, shp_years):
    date = image[12:20]
    if db_has_gerk_index_date(date):
        return

    def gerk_distance(gerks_inside_buffer_list, source_gerk, yearly_gerks_gdf):
        distance_sum = 0
        distance_percents = []

        for i in range(len(gerks_inside_buffer_list) - 1):
            # DISTANCES
            gerk_1 = source_gerk
            gerk_2 = gerks_inside_buffer_list[i]

            x1 = float(
                yearly_gerks_gdf["CENTROID_X"].loc[
                    yearly_gerks_gdf["GERK_PID"] == gerk_1
                ]
            )
            y1 = float(
                yearly_gerks_gdf["CENTROID_Y"].loc[
                    yearly_gerks_gdf["GERK_PID"] == gerk_1
                ]
            )
            x2 = float(
                yearly_gerks_gdf["CENTROID_X"].loc[
                    yearly_gerks_gdf["GERK_PID"] == gerk_2
                ]
            )
            y2 = float(
                yearly_gerks_gdf["CENTROID_Y"].loc[
                    yearly_gerks_gdf["GERK_PID"] == gerk_2
                ]
            )
            distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

            if distance:
                distance_sum += distance
                percent = distance * 100 / distance_sum
                distance_percents.append(percent)

        return distance_percents

    result_dict = {}
    year = image[12:16]
    if year not in shp_years:
        print("Shape year {} not found for {}".format(year, image))
        return

    yearly_gerks_gdf, yearly_gerk_model, distances = shp_years[year]

    mask = glob.glob(
        os.path.join(SATELLITE_IMGS_PATH, "*V{}*{}*.tif".format(date, "mask_d48"))
    )[0]

    for crop, model in yearly_gerk_model.items():
        mean_distance = distances[crop]
        for source_gerk, gerks_inside_buffer_list in model.items():
            with rasterio.open(mask) as rsrc:
                masked_affine = rsrc.transform
                mask_array = rsrc.read(1)

            with rasterio.open(os.path.join(SATELLITE_IMGS_PATH, image)) as ssrc:
                # affine = ssrc.transform
                array = ssrc.read(1)

            masked = np.where(mask_array == 100, array, float("nan"))
            masked_gerk_stats = zonal_stats(
                yearly_gerks_gdf[yearly_gerks_gdf["GERK_PID"] == source_gerk].geometry,
                masked,
                affine=masked_affine,
                stats=["mean"],
            )
            # mask_rast = rasterio.open(mask)
            # gdf = yearly_gerks_gdf[yearly_gerks_gdf["GERK_PID"] == source_gerk]
            # gdf.crs = {"init": "epsg:3912"}
            # gdf = gdf.to_crs({"init": "epsg:3912"})

            # fig, ax = plt.subplots()
            # rasterio.plot.show((mask_rast, 1), ax=ax)
            # gdf.plot(ax=ax, color="red")
            # plt.show()

            if len(gerks_inside_buffer_list) > 0:
                raster_image = Raster(image, year, mask)
                mean_pixel_value, masked_mean_stats = raster_image.zonal_stats(
                    yearly_gerks_gdf,
                    gerks_inside_buffer_list,
                    source_gerk,
                    crop,
                    mean_distance,
                )
                mean_value = 0
                distance_percentages = gerk_distance(
                    gerks_inside_buffer_list, source_gerk, yearly_gerks_gdf
                )
                number_pixels_sum = sum(
                    [mean_values["count"] for mean_values in masked_mean_stats]
                )

                for i in range(len(gerks_inside_buffer_list) - 1):
                    if number_pixels_sum:
                        area_weighting_factor = (
                            masked_mean_stats[i]["count"] * 100 / number_pixels_sum
                        )
                    if masked_mean_stats[i]["mean"]:
                        mean_value += (
                            masked_mean_stats[i]["mean"]
                            * area_weighting_factor
                            * 0.0001
                            * (100 - distance_percentages[i])
                        )
                original_crop = (
                    yearly_gerks_gdf["SIFRA_KMRS"]
                    .loc[yearly_gerks_gdf["GERK_PID"] == source_gerk]
                    .any()
                )

                if source_gerk not in result_dict:
                    result_dict[source_gerk] = {}
                    result_dict[source_gerk][date] = {}

                result_dict[source_gerk][date][crop] = round(mean_value, 3)
                result_dict[source_gerk][date]["original_value"] = original_crop
                result_dict[source_gerk][date][original_crop] = (
                    round(masked_gerk_stats[0]["mean"], 3)
                    if masked_gerk_stats[0]["mean"]
                    else masked_gerk_stats[0]["mean"]
                )

    db_insert(image, result_dict, yearly_gerks_gdf)


def process_images(shp_years):
    for image in tqdm(EVI_IMAGES, desc="Loading data"):
        process_image(image, shp_years)

    # Parallel(n_jobs=N_JOBS)(
    #     delayed(process_image)(image, shp_years)
    #     for image in tqdm(EVI_IMAGES, desc="Process images")
    # )


class Data:
    def __init__(self):
        self._image_dict = None
        self._image_dates_dict = None
        self._shp_dict = None
        self._shp_years_dict = None

    @property
    def image_dict(self):
        """
        Returns a dictionary where keys are targeted years, 
        and values are lists of raster filenames for relevant year.
        """
        if self._image_dict is not None:
            return self._image_dict
        years = set()
        for f in evi_images_list:
            year = f[12:16]
            if year not in years:
                years.add(year)
        image_dict = {}
        for year in years:
            filename_list = []
            filename_list = glob.glob(
                os.path.join(SATELLITE_IMGS_PATH, "*V{}*{}*.tif".format(year, "EVI"))
            )
            image_dict[year] = filename_list
        for year, value in image_dict.items():
            print("number of files in {}: {} ".format(year, len(value)))

        self._image_dict = image_dict
        return image_dict

    @property
    def image_dates_dict(self):
        """
        Returns a dictionary where keys are dates of raster images, 
        and values are empty.
        """
        if self._image_dates_dict is not None:
            return self._image_dates_dict

        dates = [f[12:20] for f in evi_images_list]
        available_dates_dict = dict.fromkeys(dates)

        self._image_dates_dict = available_dates_dict
        return available_dates_dict

    @property
    def shp_dict(self):
        """
        Returns a dictionary where keys are years of available 
        shapefiles, and values shapefile names.
        """
        if self._shp_dict is not None:
            return self._shp_dict

        years = set(f[2:6] for f in gerk_shapefiles_list)
        shp_dict = {}
        for year in years:
            filename = glob.glob(
                os.path.join(GERK_SHAPEFILES_PATH, "*{}*.shp".format(year))
            )[0]
            __, file_name = os.path.split(filename)
            shp_dict[year] = file_name

        self._shp_dict = shp_dict
        return shp_dict

    @property
    @cache("shp_years")
    def shp_years(self):
        # TODO separate gerks gdf and gerk model
        """
        Returns a dictionary where keys are targeted years, and values are lists 
        of geopandas dataframe object representing gerks and model of gerk id-s 
        with targeted crop type within a buffer around target gerk-id.
        """
        if self._shp_years_dict is not None:
            return self._shp_years_dict

        shp_years_dict = {}
        for year in sorted(self.shp_dict.keys()):
            filename = self.shp_dict[year]
            gerks_gdf = gdf.read_file(os.path.join(GERK_SHAPEFILES_PATH, filename))

            crops = Gerks(filename, year, gerks_gdf)
            filtered_gdf = crops.filter()
            gerks_gdf_centroid = crops.calculate_centroid(filtered_gdf)
            crop_dict = crops.classify_gerks_by_crop_type(gerks_gdf_centroid)
            distances = crops.calculate_mean_distance(crop_dict)
            gerk_model, gerks_gdf_centroid = crops.find_gerks_inside_buffer(
                crop_dict, gerks_gdf_centroid
            )
            shp_years_dict[year] = [gerks_gdf_centroid, gerk_model, distances]

        self._shp_years_dict = shp_years_dict
        return shp_years_dict


if __name__ == "__main__":
    # DELETE ALL DATA FROM DB
    # conn = psycopg2.connect(
    #     host=CONN_HOST,
    #     port=CONN_PORT,
    #     database=CONN_DATABASE,
    #     user=CONN_USER,
    #     password=CONN_PASSWORD,
    # )
    # cursor = conn.cursor()
    # cursor.execute(sql.SQL('DELETE FROM "GERK"'))
    # cursor.execute(sql.SQL('DELETE FROM "GERK_INDEX_VALUES"'))
    # conn.commit()
    # cursor.close()
    # conn.close()

    data = Data()
    process_images(data.shp_years)
