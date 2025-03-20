from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xarray as xr

from mhwpype.core import assign_static_attributes


def assign_depth(ds: xr.Dataset, depth: float) -> xr.Dataset:
    """
    This function assigns a depth coordinated to a 1D time-series dataset if that information
    is not available in the dataset. The depth input must be in meters.

    :param ds: The input dataset.
    :param depth: The depth value to be assigned. For example, sea surface temperature could have a depth of 0 meters.
    :return: The dataset with a depth coordinate.
    """
    ds = ds.expand_dims({"depth": [depth]})

    ds["depth"] = assign_static_attributes(ds.depth)
    return ds


def assign_location(ds: xr.Dataset, latitude: float, longitude: float) -> xr.Dataset:
    """
    This function assigns a latitude and longitude coordinate to a 1D time-series dataset if that information
    is not already available in the dataset. Latitude and longitude inputs must be in decimal degrees. It is recommended
    that latitude values fall between -90 and 90 and longitude values fall between -180 and 180.

    :param ds: The input dataset.
    :param latitude: A singular latitude value.
    :param longitude: A singular longitude value.
    :return: The dataset with latitude and longitude coordinates.
    """
    if "latitude" not in ds.coords or "lat" not in ds.coords:
        ds = ds.expand_dims({"latitude": [latitude]})
    if "longitude" not in ds.coords or "lon" not in ds.coords:
        ds = ds.expand_dims({"longitude": [longitude]})
    return ds


def reformat_longitude(da: xr.DataArray) -> xr.DataArray:
    """
    Reformat the longitude values to be between -180 and 180 degrees.
    :param da: The input longitude DataArray.
    :return: Longitude reformatted.
    """
    longitude = ((da + 180) % 360) - 180

    # Update attributes.
    longitude.name = "longitude"
    longitude = assign_static_attributes(longitude)
    longitude.attrs["actual_range"] = [float(longitude.min()), float(longitude.max())]
    return longitude


def update_names(
    ds: xr.Dataset,
    coord_mapper: dict | None = None,
    var_mapper: dict | None = None,
) -> xr.Dataset:
    """
    Rename dataset coordinates and variables based on a mapping dictionary.
    :param ds: The input dataset.
    :param coord_mapper: Coordinates/dimensions to rename in the dictionary format {old_name: new_name}.
    :param var_mapper: Variables to rename in the dictionary format {old_name: new_name}.
    :return:
    """
    # I don't know why I split coords and variables. Could probably be compressed.

    if var_mapper is None:
        var_mapper = {"sst": "sea_water_temperature"}
    if coord_mapper is None:
        coord_mapper = {"lat": "latitude", "lon": "longitude"}
    for old_coord, new_coord in coord_mapper.items():
        if old_coord in ds.coords:
            ds = ds.rename({old_coord: new_coord})
    for old_var, new_var in var_mapper.items():
        if old_var in ds.data_vars:
            ds = ds.rename({old_var: new_var})
    return ds
