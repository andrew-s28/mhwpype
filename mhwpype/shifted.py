from __future__ import annotations

from datetime import datetime

import xarray as xr

from mhwpype.core import ReferencePeriod


class ShiftedBaseline:
    def __init__(self):
        self.methodology = "ShiftedBaseline"

    def build_daily_climatology(
        self,
        temperature_data: xr.DataArray,
        target_years: list[int],
        reference_period_length: int = 30,
        half_window_width: int = 5,
        use_circular: bool = True,
    ) -> xr.DataArray:
        """
        Build a daily climatology for a given temperature data array using a shifted baseline approach.

        :param temperature_data: An xr.DataArray of temperatue data.
        :param target_years: The years for which the climatology should be calculated.
        :param reference_period_length: The shifted reference period length in years.
        :param half_window_width: The smoothing window width in days.
        :param use_circular: If True, the climatology for the year is smoothed using a circular window.
        :return: The climatology with the primary dimension of 'time'.
        """
        shiftclims = []
        for target_year in target_years:
            ref_data = temperature_data.sel(
                time=slice(
                    datetime(target_year - reference_period_length, 1, 1), datetime(target_year - 1, 12, 31, 23, 59, 59)
                )
            )
            rp = ReferencePeriod(ref_data.time.min(), ref_data.time.max())
            rp_clim = rp.build_daily_climatology(ref_data, half_window_width, use_circular, reset_to_input_time=False)
            rp_clim["time"] = (
                ["dayofyear"],
                [datetime.strptime(f"{target_year}-{int(dt)}", "%Y-%j") for dt in rp_clim.dayofyear.values.tolist()],
            )
            rp_clim = rp_clim.swap_dims({"dayofyear": "time"})
            shiftclims.append(rp_clim)
        shifted_climatology = xr.combine_by_coords(shiftclims, combine_attrs="drop")
        shifted_climatology = shifted_climatology.drop_duplicates("time", keep="last")
        return shifted_climatology[f"climatology_{temperature_data.name}"]

    def build_daily_threshold(
        self,
        temperature_data: xr.DataArray,
        threshold_value: float | list[float] | None = None,
        threshold_method: str = "linear",
        start_year: int = 2015,
        end_year: int = 2024,
        reference_period_length: int = 30,
        half_window_width: int = 5,
        use_circular: bool = True,
    ) -> xr.DataArray:
        """
        Build a daily threshold for a given temperature data array using a shifted baseline approach.

        :param temperature_data: An xr.DataArray of temperature data.
        :param threshold_value: The percentiles to calculate.
        :param threshold_method: The method for calculating the threshold.
        :param target_years: The years for which the threshold should be calculated.
        :param reference_period_length: The shifted reference period length in years.
        :param half_window_width: The smoothing window width in days.
        :param use_circular: Whether to use a circular window for smoothing.
        :return: The threshold with the primary dimension of 'time'.
        """
        if threshold_value is None:
            threshold_value = [0.9, 0.95, 0.99]
        shift_threshs = []
        for target_year in range(start_year, end_year + 1):
            ref_data = temperature_data.sel(
                time=slice(
                    datetime(target_year - reference_period_length, 1, 1), datetime(target_year - 1, 12, 31, 23, 59, 59)
                )
            )
            rp = ReferencePeriod(ref_data.time.min(), ref_data.time.max())
            rp_thresh = rp.build_daily_threshold(
                ref_data, threshold_value, half_window_width, threshold_method, use_circular, reset_to_input_time=False
            )
            rp_thresh["time"] = (
                ["dayofyear"],
                [datetime.strptime(f"{target_year}-{int(dt)}", "%Y-%j") for dt in rp_thresh.dayofyear.values.tolist()],
            )
            rp_thresh = rp_thresh.swap_dims({"dayofyear": "time"})
            shift_threshs.append(rp_thresh)
        shifted_thresholds = xr.combine_by_coords(shift_threshs, combine_attrs="drop")
        shifted_thresholds = shifted_thresholds.drop_duplicates("time", keep="last")
        return shifted_thresholds[f"threshold_{temperature_data.name}"]
