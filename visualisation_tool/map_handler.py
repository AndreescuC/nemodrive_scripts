import os
import cv2
import utm
import numpy as np
import pandas as pd


def add_wgs84(df, latitude="latitude", longitude="longitude"):
    df = df.assign(**{'easting': -1., 'northing': -1., "zone_no": -1., "zone_letter": ""})

    for idx, row in df.iterrows():
        easting, northing, zone_no, zone_letter = utm.from_latlon(row[latitude],
                                                                  row[longitude])
        df.at[idx, "easting"] = easting
        df.at[idx, "northing"] = northing
        df.at[idx, "zone_no"] = zone_no
        df.at[idx, "zone_letter"] = zone_letter
    return df


class ImageWgsHandler:
    def __init__(self, map_path):
        self.map_image = map_image = cv2.imread(map_path)
        self.img_rows, self.img_cols, _ = map_image.shape

        # Load reference points
        base = os.path.splitext(map_path)[0]
        self.reference_points = reference_points = pd.read_csv(f"{base}.csv")

        self.density = None
        if os.path.isfile(f"{base}.density"):
            with open(f"{base}.density", "r") as f:
                self.density = float(f.read().strip())

        if self.density:
            print(f"Map density: {self.density} m /pixel")

        reference_points = add_wgs84(reference_points)

        (geo_to_row, geo_to_col), (row_f, col_f), (easting_f, northing_f) = self.get_conversion_functions(reference_points)
        self.row_f, self.col_f = row_f, col_f
        self.easting_f, self.northing_f = easting_f, northing_f
        self.geo_to_row, self.geo_to_col = geo_to_row, geo_to_col
        self.reference_points = reference_points

    @staticmethod
    def get_conversion_functions(reference_points):
        # -- Function conversion from WGS to pixel
        x = reference_points.easting.values
        y = reference_points.northing.values

        from sklearn import linear_model

        z = reference_points.pixel_row.values
        row_f = linear_model.LinearRegression()
        row_f.fit(np.column_stack([x, y]), z)

        z = reference_points.pixel_column.values
        col_f = linear_model.LinearRegression()
        col_f.fit(np.column_stack([x, y]), z)

        # -- Function conversion from Pixels to wgs
        x = reference_points.pixel_row.values
        y = reference_points.pixel_column.values

        z = reference_points.easting.values
        easting_f = linear_model.LinearRegression()
        easting_f.fit(np.column_stack([x, y]), z)
        z = reference_points.northing.values
        northing_f = linear_model.LinearRegression()
        northing_f.fit(np.column_stack([x, y]), z)

        x = reference_points.latitude.values
        y = reference_points.longitude.values

        z = reference_points.pixel_row.values
        geo_to_row = linear_model.LinearRegression()
        geo_to_row.fit(np.column_stack([x, y]), z)
        z = reference_points.pixel_column.values
        geo_to_col = linear_model.LinearRegression()
        geo_to_col.fit(np.column_stack([x, y]), z)

        return (geo_to_row, geo_to_col), (row_f, col_f), (easting_f, northing_f)

    def get_image_coord(self, eastings, northings, convert_method=1):

        if self.density is not None and convert_method == 0:
            density = self.density
            ref_points = self.reference_points

            a = np.column_stack([eastings, northings])
            b = ref_points[["easting", "northing"]].values

            dist = np.linalg.norm(a[:, np.newaxis] - b, axis=2)
            ref = ref_points.iloc[dist.argmin(axis=1)]
            cols = (ref.pixel_column + (eastings - ref.easting)/density).values
            rows = (ref.pixel_row - (northings - ref.northing)/density).values
        elif convert_method == 1:
            row_f, col_f = self.row_f, self.col_f
            rows = row_f.predict(np.column_stack([eastings, northings]))
            cols = col_f.predict(np.column_stack([eastings, northings]))
        else:
            geo_to_row, geo_to_col = self.geo_to_row, self.geo_to_col
            #these are actually latitude and longitutde given
            rows = geo_to_row.predict(np.column_stack([eastings, northings]))
            cols = geo_to_col.predict(np.column_stack([eastings, northings]))

        return rows, cols
