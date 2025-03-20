from __future__ import annotations

import warnings
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml


def load_metadata(filepath: str) -> dict:
    """
    This function loads metadata from a yaml file.

    :param filepath: The path to the metadata yaml file.
    :return: A dictionary containing the metadata.
    """
    with Path.open(filepath) as _file:
        return yaml.safe_load(_file)


def assign_static_attributes(da: xr.DataArray, remove_existing_attributes: bool = True) -> xr.DataArray:
    """
    Assign attributes found within metadata.yaml to a DataArray based on the dimension or variable name.
    :param da: The data that needs updated attributes.
    :param remove_existing_attributes: If True remove existing attributes. This removes attributes that may be assigned by the data provider.
    :return: The DataArray with updated attributes.
    """
    # Remove existing attributes.
    if remove_existing_attributes is True:
        existing_attrs = list(da.attrs.keys())
        for key in existing_attrs:
            del da.attrs[key]

    # Load metadata.
    metadata = load_metadata("../mhwpype/metadata.yaml")
    if da.name not in metadata:
        return da
    attrs = metadata[da.name]
    for key, value in attrs.items():
        if key == "dtype":  # dtype is used for assigning datatypes and netCDF encoding in export functions.
            continue
        da.attrs[key] = value
    return da


def circular_rolling_mean(da: xr.DataArray, half_window_width: int) -> xr.DataArray:
    """
    This function allows for the rolling smoothing of the beginning and end of a climatology or threshold dataset,
    which might otherwise be unsmoothed. Higher level wrapper users will rarely call this function directly.

    Pitfalls: This function really only works if the input climatology/threshold data has a full year of data
    and no significant gaps between ends. If only a part of a climatology dataset is provided (e.g. dayofyear 250-350),
    then this function will inappropriately smooth the original dataset with the prepended and appended data.
    If a subset of the climatology is needed, it is recommended that you subset after circular smoothing.

    :param da: A dataset that represents a climatology or threshold dataset with a dayofyear coordinate.
    :param half_window_width: The number of days to include on either side of the central day for smoothing.
    :return: A smoothed dataset with the same dayofyear values as the original dataset.
    """
    # Create a new dataset which can be prepended to the original for "circular" smoothing.
    pre = da.copy(deep=True)
    pre["dayofyear"] = da.dayofyear.min() - da.dayofyear
    pre = pre.sortby("dayofyear")

    # Create a new dataset which can be appended to the original data for "circular" smoothing.
    post = da.copy(deep=True)
    post["dayofyear"] = da.dayofyear.max() + da.dayofyear
    post = post.sortby("dayofyear")

    # Combine the original data with the prepended and appended data.
    circ = xr.combine_by_coords([pre, da, post])

    # Smooth the data with a rolling mean using the half window width.
    rda = circ.rolling({"dayofyear": 2 * half_window_width + 1}, center=True).mean(skipna=True)
    rda = rda.sel(dayofyear=da.dayofyear)  # Select the original dayofyear values for return.
    return rda[da.name]  # Why does it become a dataset?


class ReferencePeriod:
    """
    A class for creating simple daily climatologies from a reference period. Used as the base class for
    Fixed Baseline MHW analysis and implemented in the Shifted Baseline class.
    """

    def __init__(self, reference_begin_datetime: datetime, reference_end_datetime: datetime) -> None:
        self.reference_begin_datetime = reference_begin_datetime
        self.reference_end_datetime = reference_end_datetime

    def build_daily_climatology(
        self,
        temperature_data: xr.DataArray,
        half_window_width: int = 5,
        use_circular: bool = True,
        reset_to_input_time: bool = False,
    ) -> xr.DataArray:
        """
        Build a daily climatology from a reference period. The climatology is calculated on a 366 day year.
        :param temperature: The input temperature DataArray.
        :param half_window_width: The window half width for smoothing the climatology.
        :param use_circular: Setting to True will wrap the climatology during smoothing.
        :param reset_to_input_time: If True, the output climatology will be mapped to the time
            coordinates of the original input dataset. At the moment this is only intended functionality
            for Fixed Baseline analysis.
        :return: An xr.DataArray representing the daily climatology.
        """
        temperature = temperature_data.sel(time=slice(self.reference_begin_datetime, self.reference_end_datetime))
        cda = temperature.groupby("time.dayofyear", restore_coord_dims=True).mean(skipna=True)

        if use_circular is True:
            rcda = circular_rolling_mean(cda, half_window_width)
        else:
            rcda = cda.rolling({"dayofyear": 2 * half_window_width + 1}, center=True).mean(skipna=True)
            rcda = rcda.sel(dayofyear=cda.dayofyear)  # Select the original dayofyear values for return.

        # Reset the time coordinate to the original input time for use in shifting or detrended baselines.
        if reset_to_input_time is True:
            reset_bins = []
            years = np.unique(temperature.time.dt.year)
            for year in years:
                _aligned_rcda = rcda.copy(deep=True)
                _aligned_rcda["time"] = (
                    ["dayofyear"],
                    [datetime.strptime(f"{year}-{int(dt)}", "%Y-%j") for dt in _aligned_rcda.dayofyear.values.tolist()],
                )
                _aligned_rcda = _aligned_rcda.swap_dims({"dayofyear": "time"})
                _aligned_rcda = _aligned_rcda.drop_duplicates(dim="time", keep="last")
                reset_bins.append(_aligned_rcda)
            rcda = xr.combine_by_coords(reset_bins)
            rcda = rcda[temperature.name]  # Why does it become a dataset?

        # Update attributes.
        rcda.name = f"climatology_{rcda.name}"
        rcda = assign_static_attributes(rcda)
        rcda.attrs["climatology_method"] = "mean"
        rcda.attrs["climatology_half_window_smooth"] = half_window_width
        rcda.attrs["climatology_reference_begin"] = temperature.time.min().dt.strftime("%Y-%m-%d").values
        rcda.attrs["climatology_reference_end"] = temperature.time.max().dt.strftime("%Y-%m-%d").values
        rcda.attrs["climatology_reference_period"] = (
            int((temperature.time.max().values - temperature.time.min().values).astype("timedelta64[Y]")) + 1
        )
        return rcda

    def _build_daily_threshold(
        self,
        temperature_data: xr.DataArray,
        threshold_value: float | list[float],
        half_window_width: int,
        threshold_method: str,
        use_circular: bool,
        reset_to_input_time: bool,
    ) -> xr.DataArray:
        """
        Build a daily threshold from a reference period. The threshold is calculated on a 366 day year.

        :param temperature: The input temperature DataArray.
        :param threshold_value: The percentile values representing the threshold. They must be floats between 0 and 1.
            Example: 0.90 is for the 90th percentile.
        :param half_window_width: The window half width for smoothing the threshold.
        :param threshold_method: The method for calculating the percentile threshold. Default is linear.
            See xr.quantile for options.
        :param use_circular: Setting to True will wrap the threshold during smoothing.
        :param reset_to_input_time: If True, the output threshold will be mapped to the time coordinates of the original
            input dataset. At the moment this is only intended functionality for Fixed Baseline analysis.
        :return: An xr.DataArray representing the daily threshold.
        """
        temperature = temperature_data.sel(time=slice(self.reference_begin_datetime, self.reference_end_datetime))
        pda = temperature.groupby("time.dayofyear", restore_coord_dims=True).quantile(
            threshold_value, method=threshold_method, skipna=True
        )

        if use_circular is True:
            rpda = circular_rolling_mean(pda, half_window_width)
        else:
            rpda = pda.rolling({"dayofyear": 2 * half_window_width + 1}, center=True).mean(skipna=True)
            rpda = rpda.sel(dayofyear=pda.dayofyear)  # Select the original dayofyear values for return.

        # Reset the time coordinate to the original input time for use in shifting or detrended baselines.
        if reset_to_input_time is True:
            reset_bins = []
            years = np.unique(temperature.time.dt.year)
            for year in years:
                _aligned_rpda = rpda.copy(deep=True)
                _aligned_rpda["time"] = (
                    ["dayofyear"],
                    [datetime.strptime(f"{year}-{int(dt)}", "%Y-%j") for dt in _aligned_rpda.dayofyear.values.tolist()],
                )
                _aligned_rpda = _aligned_rpda.swap_dims({"dayofyear": "time"})
                _aligned_rpda = _aligned_rpda.drop_duplicates(dim="time", keep="last")
                reset_bins.append(_aligned_rpda)
            rpda = xr.combine_by_coords(reset_bins)
            rpda = rpda[temperature.name]  # Why does it become a dataset?

        # Update attributes.
        rpda.name = f"threshold_{rpda.name}"
        rpda = assign_static_attributes(rpda)
        rpda.attrs["threshold_method"] = threshold_method
        rpda.attrs["threshold_half_window_smooth"] = half_window_width
        rpda.attrs["threshold_reference_begin"] = temperature.time.min().dt.strftime("%Y-%m-%d").values
        rpda.attrs["threshold_reference_end"] = temperature.time.max().dt.strftime("%Y-%m-%d").values
        rpda.attrs["threshold_reference_period"] = (
            int((temperature.time.max().values - temperature.time.min().values).astype("timedelta64[Y]")) + 1
        )
        return rpda

    def build_daily_threshold(
        self,
        temperature_data: xr.DataArray,
        threshold_value: float | list[float] | None = None,
        half_window_width: int = 5,
        threshold_method: str = "linear",
        use_circular: bool = True,
        reset_to_input_time: bool = False,
        suppress_runtime_warnings: bool = True,
    ) -> xr.DataArray:
        """
         Build a daily threshold from a reference period. The threshold is calculated on a 366 day year.
         A wrapper function for _build_daily_threshold that allows for optional suppression of runtime warnings.
        :param temperature: The input temperature DataArray.
        :param threshold_value: The percentile values representing the threshold. They must be floats between 0 and 1.
            Example: 0.90 is for the 90th percentile.
        :param half_window_width: The window half width for smoothing the threshold.
        :param threshold_method: The method for calculating the percentile threshold. Default is linear.
            See xr.quantile for options.
        :param use_circular: Setting to True will wrap the threshold during smoothing.
        :param reset_to_input_time: If True, the output threshold will be mapped to the time coordinates of the original
            input dataset. At the moment this is only intended functionality for Fixed Baseline analysis.
        :param suppress_runtime_warnings: If True, warnings will be suppressed. These warnings typically occur when
            and attempt to calculate the quantile is performed on an all-NaN slice.(e.g. A land cell with no data).

        :return: An xr.DataArray representing the daily threshold.
        """
        if threshold_value is None:
            threshold_value = [0.9, 0.95, 0.99]
        if suppress_runtime_warnings is True:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rpda = self._build_daily_threshold(
                    temperature_data,
                    threshold_value,
                    half_window_width,
                    threshold_method,
                    use_circular,
                    reset_to_input_time,
                )
        else:
            rpda = self._build_daily_threshold(
                temperature_data,
                threshold_value,
                half_window_width,
                threshold_method,
                use_circular,
                reset_to_input_time,
            )
        return rpda


def build_heat_spike_mask(
    temperature_data: xr.DataArray, threshold_data: xr.DataArray, core_dim: str = "dayofyear"
) -> xr.DataArray:
    """
    This function builds a mask for heat spikes based on a threshold value. Data are flagged as a heat spike if the
    value exceeds the threshold. In the case of MHWs, the threshold is typically defined as a climatological percentile.

    :param temperature_data: Temperature data with at least a time dimension.
    :param threshold_data: The threshold, typically a daily percentile value along a dayofyear dimension.
    :param core_dim: The dimension along which the threshold is applied. Default is 'dayofyear'.
    :return: A binary mask indicating if value oriented by the dataset dimensions exceeds the given threshold.
    """
    if core_dim.lower() == "dayofyear":
        mask = xr.where(temperature_data > threshold_data.sel(dayofyear=temperature_data.time.dt.dayofyear), 1, 0)
    else:
        mask = xr.where(temperature_data > threshold_data, 1, 0)

    # Update attributes.
    mask.name = "heat_spike_mask"
    mask.attrs = threshold_data.attrs  # Assign attributes from the threshold DataArray.
    return assign_static_attributes(mask, remove_existing_attributes=False)


def fill_gaps(mask: np.array, maximum_gap_length: int) -> np.array:
    """
    This function fills gaps in a binary mask. A gap is defined as a sequence of zeros surrounded by ones on both sides.
    This function will only work on 1D arrays and is intended to be used with xr.apply_ufunc. It also assumes that the
    data are sequential in time.
    :param mask: The masked timeseries.
    :param maximum_gap_length: The maximum gap length to fill.
    :return: The filled mask.
    """
    _m = (
        mask.copy()
    )  # xr.apply_ufunc with vectorize=True does not work with inplace operations. This copy is necessary.
    for i in range(len(mask)):
        for gap in range(maximum_gap_length, 0, -1):
            gap_condition = np.array([1] + [0] * gap + [1])
            subset = _m[i : i + len(gap_condition)]
            if np.array_equal(subset, gap_condition):
                _m[i : i + len(gap_condition)] = 1
    return _m


def build_mhw_mask(
    temperature_data: xr.DataArray,
    threshold_data: xr.DataArray,
    minimum_event_length: int = 5,
    maximum_gap_length: int = 2,
    core_dim: str = "dayofyear",
):
    """
    This function builds a mask for marine heatwaves (MHWs) based on a heat spike mask, the minimum required days to
        quantify as a MHW and the maximum gap length to consider two events as one.
    :param da:
    :param pda:
    :param minimum_event_length:
    :param maximum_gap_length:
    :param core_dim:
    :return:
    """
    gt_mask = build_heat_spike_mask(temperature_data, threshold_data, core_dim)
    gt_windows = gt_mask.rolling({"time": minimum_event_length}).construct("gt_window")
    raw_mhw_mask = xr.where(gt_windows.sum(dim="gt_window") == minimum_event_length, 1, 0)
    shifted = [raw_mhw_mask] + [raw_mhw_mask.shift(time=-i) for i in range(1, minimum_event_length)]
    patched_mhw_mask = xr.where(sum(shifted) >= 1, 1, 0)
    updated_mhw_mask = xr.apply_ufunc(
        fill_gaps,
        patched_mhw_mask,
        kwargs={"maximum_gap_length": maximum_gap_length},
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
    )
    updated_mhw_mask = updated_mhw_mask.drop_vars("dayofyear", errors="ignore")

    # Update attributes.
    updated_mhw_mask.name = "mhw_mask"
    updated_mhw_mask.attrs = temperature_data.attrs
    return assign_static_attributes(updated_mhw_mask, remove_existing_attributes=False)


def group_cell_events(cell_mask: xr.DataArray) -> xr.Dataset:
    """
    Obtain beginning and end times of MHW events from a mask that is unique in its dimensions.
    :param cell_mask: A binary mask indicating the presence of MHW events. Effectively a timeseries.
    :return: An xr.Dataset containing the start and end times of MHW events.
    """
    t = cell_mask.time.values
    m = cell_mask.values
    dts = np.where(m == 1, t, np.datetime64("NaT"))
    dts = sorted(dts[~np.isnat(dts)])

    if len(dts) == 0:
        return None
    groups = [[dts[0]]]
    for i in dts:
        if i == groups[-1][-1] + np.timedelta64(86400, "s"):
            groups[-1].append(i)
        else:
            groups.append([i])
    groups = [v for v in groups if len(v) > 1]
    if len(groups) == 0:
        return None
    ds_groups = []
    for group in groups:
        ts = pd.to_datetime(min(group)).replace(hour=0, minute=0, second=0)
        te = pd.to_datetime(max(group)).replace(hour=23, minute=59, second=59)

        _ds = xr.Dataset()
        _ds = _ds.expand_dims({"event_id": [ts]})
        _ds["event_start"] = (["event_id"], [ts])
        _ds["event_end"] = (["event_id"], [te])
        _ds["event_duration"] = np.ceil((_ds.event_end - _ds.event_start).astype(float) / (1e9 * 86400)).astype(int)
        ds_groups.append(_ds)
    return xr.combine_by_coords(ds_groups)


def group_from_mask(mhw_mask: xr.DataArray) -> xr.Dataset:
    """
    #TODO: Figure out a more efficient way to do this than a for loop. Multiprocessing?
        with multiprocess.Pool(cpu) as pool:
            pool.map()  #Or starmap.

    Group MHW events from a multi-dimensional mask. This function loops through each unique cell in a dataset..
    :param mhw_mask: A multi-dimensional binary mask.
    :return: An xr.Dataset containing the start and end times of MHW events for each unique combination of dimensions.
    """
    coords = [
        v for v in mhw_mask.coords if v not in ["time", "dayofyear"]
    ]  # Remove time-based coordinates for iterator.
    cell_groups = []
    for c in product(*[mhw_mask[coord] for coord in coords]):
        cell_mhw_mask = mhw_mask.sel(dict(zip(coords, c, strict=False)))
        grouped = group_cell_events(cell_mhw_mask)
        if grouped is not None:
            grouped = grouped.expand_dims({k: [v] for k, v in zip(coords, c, strict=False)})
            grouped = grouped.sortby("event_id")
            cell_groups.append(grouped)
    gds = xr.concat(cell_groups, dim="event_id")
    return gds.sortby("event_id")


def group_stats(
    temperature_data: xr.DataArray,
    climatology_data: xr.DataArray,
    threshold_data: xr.DataArray,
    mhw_groups: xr.Dataset,
    core_dim: str = "dayofyear",
) -> xr.Dataset:
    """

    #TODO: Adjust to include the day before and after an event so that r_onset and r_decline can be calculated.

    Using beginning and end datetimes for MHW events, calculate statistics for each event.
    :param temperature_data: An xr.DataArray containing the temperature data.
    :param climatology_data: An xr.DataArray containing the climatology data.
    :param threshold_data: An xr.DataArray containing the threshold data.
    :param mhw_groups: The grouped MHW events calculated from group_from_mask
    :param core_dim: The core dimension of the climatology and threshold datasets.
    :return: An xr.Dataset containing basic statistics for each MHW event.
    """
    coords = [v for v in mhw_groups.coords if v not in ["time", "dayofyear", "event_id"]]
    new_cell_groups = []
    for c in product(*[mhw_groups[coord] for coord in coords]):
        cell_groups = mhw_groups.sel(dict(zip(coords, c, strict=False)))
        cell_groups = cell_groups.where(~np.isnan(cell_groups.event_duration), drop=True)
        if len(cell_groups.event_id) == 0:
            continue
        for event_id in cell_groups.event_id:
            cell_group = cell_groups.sel(event_id=event_id)

            da_sel = {k: [v] for k, v in zip(coords, c, strict=False)}
            da_sel["time"] = slice(
                pd.to_datetime(cell_group.event_start.values), pd.to_datetime(cell_group.event_end.values)
            )
            del da_sel["quantile"]
            cell_da = temperature_data.sel(da_sel)

            if core_dim == "dayofyear":
                # Subset the daily temperature climatology dataset based on unique selection criteria.
                cda_sel = {k: [v] for k, v in zip(coords, c, strict=False)}
                del cda_sel["quantile"]
                cda_sel["dayofyear"] = cell_da.time.dt.dayofyear
                cell_cda = climatology_data.sel(cda_sel)

                # Subset the daily temperature threshold dataset based on unique selection criteria.
                pda_sel = {k: [v] for k, v in zip(coords, c, strict=False)}
                pda_sel["dayofyear"] = cell_da.time.dt.dayofyear
                cell_pda = threshold_data.sel(pda_sel)

            else:
                cda_sel = {k: [v] for k, v in zip(coords, c, strict=False)}
                del cda_sel["quantile"]
                cda_sel["time"] = cell_da.time
                cell_cda = climatology_data.sel(cda_sel)

                # Subset the daily temperature threshold dataset based on unique selection criteria.
                pda_sel = {k: [v] for k, v in zip(coords, c, strict=False)}
                pda_sel["time"] = cell_da.time
                cell_pda = threshold_data.sel(pda_sel)

            anom = cell_da - cell_cda

            # Calculate MHW time statistics.
            maxmask = xr.where(cell_da == cell_da.max(), 1, 0)
            temp_peak_time = maxmask.where(maxmask == 1, drop=True).time.values[0]
            cell_group[f"event_peak_{temperature_data.name}"] = cell_da.max()
            cell_group[f"event_peak_{temperature_data.name}_time"] = temp_peak_time

            maxmask = xr.where(anom == anom.max(), 1, 0)
            anom_peak_time = maxmask.where(maxmask == 1, drop=True).time.values[0]
            cell_group["event_peak_anomaly"] = anom.max()
            cell_group["event_peak_anomaly_time"] = anom_peak_time

            cell_group = cell_group.expand_dims(coords)  # I don't know why this only works here.

            # Calculate climatology MHW statistics.
            cell_group["event_intensity_max"] = anom.max(dim="time")
            cell_group["event_intensity_mean"] = anom.mean(dim="time")
            cell_group["event_intensity_stdev"] = anom.std(dim="time")
            cell_group["event_intensity_cumulative"] = anom.sum(dim="time")

            # Calculate category statistics.
            cat = np.floor(cell_group.event_intensity_max / (cell_pda - cell_cda))
            cat = cat.where(cat < 4, 4)
            cell_group["event_peak_category"] = cat.max(dim="time")

            cell_group = cell_group.expand_dims("event_id")
            new_cell_groups.append(cell_group)
    return xr.combine_by_coords(new_cell_groups)
