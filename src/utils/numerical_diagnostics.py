from __future__ import annotations

import operator as _operator_mod
import re
from collections.abc import Sequence, Mapping
from typing import Callable, Any
 
import numpy as np
import xarray as xr
from scipy import stats

from src.utils.data_io_and_subset import get_spatial_dims

xr.set_options(keep_attrs=True)

# ===================================================================
# Dimension helpers
# ===================================================================

def _get_dims(data: xr.DataArray, dims: str | Sequence[str] | None):
    if dims is None:
        return None
    if isinstance(dims, str):
        return [dims]
    return list(dims)

def get_dz(data: xr.DataArray, *, depth_dim: str = "depth") -> xr.DataArray:
    """Return layer thickness (m) along *depth_dim*.

    Prefers the 'dz' coordinate attached by load_cube (from lev_bnds),
    and falls back to np.gradient on the depth values when bnds were
    unavailable (e.g. for derived arrays that lost the coord).
    """
    if depth_dim not in data.dims and depth_dim not in data.coords:
        raise KeyError(f"Depth dim '{depth_dim}' not found")
    if "dz" in data.coords:
        return data.coords["dz"]
    z = data[depth_dim]
    dz_vals = np.abs(np.gradient(z.values))
    return xr.DataArray(dz_vals, coords={depth_dim: z}, dims=(depth_dim,))


# ===================================================================
# Basic statistics  (supports range + percentile codes)
# ===================================================================

def compute_basic_statistic(
    data: xr.DataArray,
    stat_operator: str,
    *,
    dims: str | Sequence[str] | None = None,
    skipna: bool = True,
) -> xr.DataArray:
    """Compute a basic statistic over one or more dimensions.
 
    Supported operators
    -------------------
    - ``mean``, ``median``, ``sum``, ``min``, ``max``, ``std``, ``var``
    - ``range``  (max − min)
    - Percentile codes: ``5_%``, ``10_%``, ``25_%``, ``50_%``,
      ``75_%``, ``90_%``, ``95_%``  (pattern ``<int>_%``)
    """
    dims = _get_dims(data, dims)
    op = stat_operator.lower().strip()
 
    # ---- standard reductions ----
    simple = {
        "mean":   lambda: data.mean(dim=dims, skipna=skipna),
        "median": lambda: data.median(dim=dims, skipna=skipna),
        "sum":    lambda: data.sum(dim=dims, skipna=skipna),
        "min":    lambda: data.min(dim=dims, skipna=skipna),
        "max":    lambda: data.max(dim=dims, skipna=skipna),
        "std":    lambda: data.std(dim=dims, skipna=skipna),
        "var":    lambda: data.var(dim=dims, skipna=skipna),
    }
    if op in simple:
        return simple[op]()
 
    # ---- range ----
    if op == "range":
        return (data.max(dim=dims, skipna=skipna) - data.min(dim=dims, skipna=skipna))
 
    # ---- percentile codes  e.g. "5_%", "25_%" ----
    m = re.fullmatch(r"(\d+)_%", op)
    if m:
        q = int(m.group(1)) / 100.0
        return data.quantile(q, dim=dims, skipna=skipna)
 
    raise ValueError(
        f"Unsupported operator '{stat_operator}'. "
        f"Expected one of: {sorted(simple)}; 'range'; or '<int>_%' percentile."
    )

def compute_basic_statistic_with_area(
    data: xr.DataArray,
    stat_operator: str,
    *,
    area: xr.DataArray | None = None,
    spatial_dims: tuple[str, ...] | None = None,
    skipna: bool = True,
) -> xr.DataArray:
    """Reduce all dims to a scalar.

    For 'mean' with non-None area: reduce non-spatial dims unweighted,
    then take area-weighted spatial mean.
    For all other operators: standard unweighted compute_basic_statistic.

    Rationale: weighting percentiles / median / std by area changes their
    meaning, so we keep them unweighted; mean is the only stat where
    cell-area weighting is unambiguously the physically correct choice.
    """
    op = stat_operator.lower().strip()
    if op == "mean" and area is not None:
        if spatial_dims is None:
            try:
                spatial_dims = get_spatial_dims(data)
            except (KeyError, ValueError):
                spatial_dims = ()
        non_spatial = [d for d in data.dims if d not in spatial_dims]
        reduced = data.mean(dim=non_spatial, skipna=skipna) if non_spatial else data
        return area_weighted_mean(reduced, area, spatial_dims=spatial_dims)
    return compute_basic_statistic(data, stat_operator, skipna=skipna)


# ===================================================================
# Threshold parsing
# ===================================================================
 
def parse_threshold(threshold_str: str) -> float:
    """Extract the numeric value from a threshold string.
 
    ``sample_threshold()`` returns strings like ``"28 °C"``, ``"1e-4 m/s"``,
    ``"8.1"`` (no unit for pH).  The first whitespace-delimited token is
    always the number.
 
    Examples
    --------
    >>> parse_threshold("28 °C")
    28.0
    >>> parse_threshold("1e-4 m/s")
    0.0001
    >>> parse_threshold("8.1")
    8.1
    """
    return float(threshold_str.strip().split()[0])


# ===================================================================
# Vertical operations
# ===================================================================

def compute_inventory(
    data: xr.DataArray,
    *,
    depth_dim: str = "depth",
    thickness: xr.DataArray | None = None,
    area: xr.DataArray | None = None,
) -> xr.DataArray:
    """Compute vertically integrated inventory.

    Parameters
    ----------
    data:
        DataArray containing at least a vertical dimension.
    depth_dim:
        Name of the vertical dimension.
    thickness:
        Optional layer thickness with dimension ``depth_dim`` (or broadcastable
        to ``data``). If omitted, thickness is estimated from coordinate
        spacing.
    area:
        Optional horizontal cell area, broadcastable to ``data`` without the
        vertical dimension. If supplied, the result is a total inventory over
        volume. If omitted, the result is a vertically integrated column
        inventory.
    """
    if depth_dim not in data.dims:
        raise KeyError(f"Depth dimension '{depth_dim}' not found")

    if thickness is None:
        thickness = get_dz(data, depth_dim=depth_dim)

    integrated = (data * thickness).sum(dim=depth_dim, skipna=True)
    if area is not None:
        try:
            spatial_dims = set(get_spatial_dims(integrated))
        except (KeyError, ValueError):
            spatial_dims = {"lat", "lon", "nlat", "nlon", "i", "j", "x", "y"}
        sum_dims = [d for d in area.dims if d in integrated.dims and d in spatial_dims]
        integrated = (integrated * area).sum(dim=sum_dims, skipna=True)
    return integrated

def compute_vertical_gradient(
    data: xr.DataArray,
    *,
    depth_dim: str = "depth",
    reduce_dims: list[str] | None = None,
    stat_operator: str = "mean",
) -> xr.DataArray:
    """Vertical gradient  d(data)/d(depth)."""
    if depth_dim not in data.dims:
        raise KeyError(f"Depth dimension '{depth_dim}' not found")
 
    work = data
    if reduce_dims:
        if stat_operator == "mean":
            work = work.mean(dim=reduce_dims, skipna=True)
        elif stat_operator == "median":
            work = work.median(dim=reduce_dims, skipna=True)
        else:
            raise ValueError("stat_operator must be 'mean' or 'median'")
 
    z = work[depth_dim].values
    grad = np.gradient(work.values, z, axis=work.get_axis_num(depth_dim))
    return xr.DataArray(grad, coords=work.coords, dims=work.dims, attrs=work.attrs)


# ===================================================================
# Trend analysis
# ===================================================================

def regrid_to_grid(
    data: xr.DataArray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    *,
    method: str = "linear",
) -> xr.DataArray:
    """Regrid a DataArray onto a regular lat/lon grid via bilinear interpolation.

    Source can be:
      - Rectilinear (1-D lat/lon, e.g. GFDL): xarray's native interp.
      - Curvilinear (2-D lat/lon, e.g. CESM2 / MPI): scipy.griddata
        (linear, i.e. Delaunay triangulation) per non-spatial slice.

    For both paths, source longitudes are first normalized to [0, 360)
    and dateline-adjacent cells are duplicated periodically so that
    interpolation across the date line stays continuous.

    Output dims: (..., 'lat', 'lon').
    """
    target_lat = np.asarray(target_lat, dtype=float)
    target_lon = np.asarray(target_lon, dtype=float) % 360
    n_lat_t, n_lon_t = len(target_lat), len(target_lon)

    lat_coord = data["lat"]

    # ---------------- Rectilinear path ----------------
    if lat_coord.ndim == 1:
        # Normalize source lon to [0, 360) and ensure sorted ascending.
        if (data["lon"] < 0).any():
            data = data.assign_coords(lon=(data["lon"] % 360)).sortby("lon")
        # Periodic padding for ~global coverage (handles dateline cleanly).
        lon_span = float(data["lon"].max() - data["lon"].min())
        if lon_span >= 350:
            n_pad = 5
            pre = data.isel(lon=slice(-n_pad, None)).assign_coords(
                lon=data["lon"].isel(lon=slice(-n_pad, None)) - 360,
            )
            post = data.isel(lon=slice(0, n_pad)).assign_coords(
                lon=data["lon"].isel(lon=slice(0, n_pad)) + 360,
            )
            data = xr.concat([pre, data, post], dim="lon").sortby("lon")
        return data.interp(lat=target_lat, lon=target_lon, method=method)

    # ---------------- Curvilinear path ----------------
    from scipy.interpolate import griddata

    spatial_dims = lat_coord.dims                           # e.g. ('nlat','nlon') or ('j','i')
    non_spatial = [d for d in data.dims if d not in spatial_dims]

    src_lat = lat_coord.values.ravel()
    src_lon = (data["lon"].values % 360).ravel()
    pad_e = src_lon < 5
    pad_w = src_lon > 355
    src_lat_pad = np.concatenate([src_lat, src_lat[pad_e], src_lat[pad_w]])
    src_lon_pad = np.concatenate([src_lon, src_lon[pad_e] + 360, src_lon[pad_w] - 360])
    src_pts = np.column_stack([src_lat_pad, src_lon_pad])

    LAT2, LON2 = np.meshgrid(target_lat, target_lon, indexing="ij")
    target_pts = np.column_stack([LAT2.ravel(), LON2.ravel()])

    def _interp_one(arr_flat):
        arr_pad = np.concatenate([arr_flat, arr_flat[pad_e], arr_flat[pad_w]])
        valid = np.isfinite(arr_pad)
        if valid.sum() < 4:
            return np.full((n_lat_t, n_lon_t), np.nan)
        pts = src_pts[valid]
        # Qhull needs non-collinear input; bail out (→ NaN) if the source
        # points span no extent in lat or lon (e.g. a 1-cell-wide sliver).
        if np.ptp(pts[:, 0]) < 1e-6 or np.ptp(pts[:, 1]) < 1e-6:
            return np.full((n_lat_t, n_lon_t), np.nan)
        return griddata(
            pts, arr_pad[valid], target_pts, method=method,
        ).reshape(n_lat_t, n_lon_t)

    if non_spatial:
        data_t = data.transpose(*non_spatial, *spatial_dims)
        flat = data_t.values.reshape(-1, src_lat.size)
        out = np.array([_interp_one(row) for row in flat])
        out = out.reshape(*[data.sizes[d] for d in non_spatial], n_lat_t, n_lon_t)
        out_dims = [*non_spatial, "lat", "lon"]
        out_coords = {d: data[d] for d in non_spatial if d in data.coords}
    else:
        out = _interp_one(data.values.ravel())
        out_dims = ["lat", "lon"]
        out_coords = {}
    out_coords["lat"] = target_lat
    out_coords["lon"] = target_lon
    return xr.DataArray(out, dims=out_dims, coords=out_coords, name=data.name)

def compute_linear_trend(
    data: xr.DataArray,
    *,
    time_dim: str = "time",
) -> xr.DataArray:
    """Compute linear trend slope along a time dimension.

    The x-axis is the integer index of the time coordinate.
    """
    if time_dim not in data.dims:
        raise KeyError(f"Time dimension '{time_dim}' not found")

    x = np.arange(data.sizes[time_dim], dtype=float)

    def _slope(y: np.ndarray) -> float:
        mask = np.isfinite(y)
        if mask.sum() < 2:
            return np.nan
        x_valid = x[mask]
        y_valid = y[mask]
        x_centered = x_valid - x_valid.mean()
        return np.dot(x_centered, y_valid - y_valid.mean()) / np.dot(x_centered, x_centered)

    return xr.apply_ufunc(
        _slope,
        data,
        input_core_dims=[[time_dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

def compute_trend_significance(
    data: xr.DataArray,
    *,
    alpha: float = 0.05,
    time_dim: str = "time",
) -> xr.Dataset:
    """Estimate linear trend significance using a t-test on the slope.

    Returns
    -------
    xarray.Dataset
        Dataset with variables ``slope``, ``stderr``, ``t_stat``, ``p_value``,
        and ``is_significant``.
    """
    if time_dim not in data.dims:
        raise KeyError(f"Time dimension '{time_dim}' not found")

    x = np.arange(data.sizes[time_dim], dtype=float)

    def _stats(y: np.ndarray) -> np.ndarray:
        mask = np.isfinite(y)
        if mask.sum() < 3:
            return np.array([np.nan, np.nan, np.nan, np.nan, 0.0])
        x_valid = x[mask]
        y_valid = y[mask]
        slope, intercept, r, p_value, stderr = stats.linregress(x_valid, y_valid)
        t_stat = slope / stderr if stderr not in (0.0, np.nan) and np.isfinite(stderr) else np.nan
        significant = float(np.isfinite(p_value) and p_value < alpha)
        return np.array([slope, stderr, t_stat, p_value, significant], dtype=float)

    stacked = xr.apply_ufunc(
        _stats,
        data,
        input_core_dims=[[time_dim]],
        output_core_dims=[["metric"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"metric": 5}},
    ).assign_coords(metric=["slope", "stderr", "t_stat", "p_value", "is_significant"])

    return xr.Dataset({name: stacked.sel(metric=name).drop_vars("metric") for name in stacked.metric.values})


# ===================================================================
# Condition / comparison
# ===================================================================

_OPERATORS: dict[str, Callable[[Any, Any], Any]] = {
    ">":  _operator_mod.gt,
    ">=": _operator_mod.ge,
    "<":  _operator_mod.lt,
    "<=": _operator_mod.le,
    "==": _operator_mod.eq,
    "!=": _operator_mod.ne,
}

def evaluate_condition_or_compare(
    lhs: xr.DataArray | xr.Dataset | float | int,
    comparison_operator: str,
    rhs: xr.DataArray | xr.Dataset | float | int,
):
    """Evaluate a comparison and return the boolean result.

    If ``lhs`` or ``rhs`` is an xarray object, xarray-style broadcasting is
    preserved and the function returns an xarray object of booleans.
    """
    if comparison_operator not in _OPERATORS:
        raise ValueError(
            f"Unsupported operator '{comparison_operator}'. Supported operators: {sorted(_OPERATORS)}"
        )
    return _OPERATORS[comparison_operator](lhs, rhs)


# ===================================================================
# Area-weighted spatial reductions
# ===================================================================
 
def area_weighted_mean(
    data: xr.DataArray,
    area: xr.DataArray,
    spatial_dims: tuple[str, ...] | None = None,
) -> xr.DataArray:
    """Area-weighted mean over *spatial_dims* (NaN-safe)."""
    if spatial_dims is None:
        spatial_dims = get_spatial_dims(data)
    dims_to_reduce = [d for d in spatial_dims if d in data.dims]
    return data.weighted(area.fillna(0)).mean(dim=dims_to_reduce)
 
def annual_area_weighted_mean(
    data: xr.DataArray,
    area: xr.DataArray,
    *,
    spatial_dims: tuple[str, ...] | None = None,
    time_dim: str = "time",
) -> xr.DataArray:
    """Annual-mean of the area-weighted spatial mean → 1-D yearly time series."""
    spatial_mean = area_weighted_mean(data, area, spatial_dims=spatial_dims)
    return spatial_mean.resample({time_dim: "YE"}).mean()
 
 
# ===================================================================
# Pearson correlation
# ===================================================================
 
def compute_pearson_correlation(
    ts1: xr.DataArray, ts2: xr.DataArray,
) -> dict[str, float]:
    """Pearson *r* between two 1-D time series (aligned, NaN-safe)."""
    ts1, ts2 = xr.align(ts1, ts2, join="inner")
    valid = np.isfinite(ts1.values) & np.isfinite(ts2.values)
    if valid.sum() < 3:
        return {"r": float("nan"), "p_value": float("nan")}
    ts1_valid = ts1.values[valid]
    ts2_valid = ts2.values[valid]
    if np.std(ts1_valid) == 0 or np.std(ts2_valid) == 0:
        return {"r": float("nan"), "p_value": float("nan")}
    r, p = stats.pearsonr(ts1_valid, ts2_valid)
    return {"r": float(r), "p_value": float(p)}


# ===================================================================
# Climatology and anomaly
# ===================================================================
 
def compute_climatology(
    data: xr.DataArray,
    *,
    time_dim: str = "time",
) -> xr.DataArray:
    """Time-mean climatology (collapse *time_dim* by averaging)."""
    return data.mean(dim=time_dim, skipna=True)
 
def compute_monthly_climatology(
    data: xr.DataArray,
    *,
    time_dim: str = "time",
) -> xr.DataArray:
    """Monthly climatology → 12-step DataArray indexed by month (1..12)."""
    return data.groupby(f"{time_dim}.month").mean(dim=time_dim, skipna=True)

_SEASON_MONTHS_MAP = {
    "DJF": [12, 1, 2], "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],  "SON": [9, 10, 11],
}

def filter_complete_seasons(
    data: xr.DataArray,
    season: str,
    *,
    time_dim: str = "time",
) -> xr.DataArray:
    """Drop time steps belonging to incomplete seasons, returning the
    same DataArray with an attached `season_year` coordinate.

    For DJF: a season-year is the calendar year of its J/F months; the Dec
    of the prior calendar year belongs to that season-year. Season-years
    missing any of D/J/F are dropped.

    For MAM/JJA/SON: season-year == calendar year. Years missing any of
    the three months are dropped.

    Input is assumed to already be season-subsetted (i.e. only contains
    months belonging to *season*).
    """
    season = season.upper()
    if season not in _SEASON_MONTHS_MAP:
        raise ValueError(f"Unknown season '{season}'")
    expected_len = len(_SEASON_MONTHS_MAP[season])

    t = data[time_dim]
    if season == "DJF":
        season_year = xr.where(t.dt.month == 12, t.dt.year + 1, t.dt.year)
    else:
        season_year = t.dt.year
    data = data.assign_coords(season_year=(time_dim, season_year.values))

    counts = data[time_dim].groupby(data["season_year"]).count()
    valid = counts.where(counts == expected_len, drop=True)["season_year"].values
    return data.where(data["season_year"].isin(valid), drop=True)

def compute_seasonal_climatology(
    data: xr.DataArray,
    season: str,
    *,
    time_dim: str = "time",
) -> xr.DataArray:
    """Seasonal climatology: mean of complete seasonal means across years.

    For DJF: each season is {year Y Dec, year Y+1 Jan, year Y+1 Feb}, indexed
    by Y+1. Incomplete seasons at the endpoints (missing Dec, or missing JF)
    are dropped before averaging across years.

    For MAM/JJA/SON: all three months lie in the same calendar year, so no
    special handling is needed.
    """
    from src.utils.data_io_and_subset import subset_season

    data_season = subset_season(data, season, time_dim=time_dim)
    data_season = filter_complete_seasons(data_season, season, time_dim=time_dim)
    
    if data_season.sizes.get(time_dim, 0) == 0:
        return data_season.mean(dim=time_dim, skipna=True)  # all-NaN sentinel

    annual = data_season.groupby("season_year").mean(dim=time_dim, skipna=True)
    return annual.mean(dim="season_year", skipna=True)
 
def compute_anomaly(
    data: xr.DataArray,
    climatology: xr.DataArray,
) -> xr.DataArray:
    """Anomaly = *data* − *climatology* (xarray broadcasting handles alignment).
 
    If *climatology* has a ``month`` dimension (from ``compute_monthly_climatology``),
    groupby subtraction is used automatically.
    """
    if "month" in climatology.dims:
        return data.groupby("time.month") - climatology
    return data - climatology

 
# ===================================================================
# Volume / area fraction
# ===================================================================
 
def compute_volume_fraction(
    mask: xr.DataArray,
    area: xr.DataArray,
    *,
    valid: xr.DataArray | None = None,
    depth_dim: str = "depth",
    time_dim: str = "time",
) -> xr.DataArray:
    """Fraction of the water column (by thickness) where *mask* is True.
 
    Returns a 2-D (lat, lon) or scalar fraction in [0, 1].
    """
    reduce_dims = [d for d in mask.dims if d != time_dim]

    if depth_dim in mask.dims:
        weight = get_dz(mask, depth_dim=depth_dim) * area
    else:
        weight = area

    valid_arr = (
        xr.ones_like(mask, dtype=float)
        if valid is None
        else valid.astype(float)
    )

    num = (mask.astype(float) * weight).sum(dim=reduce_dims, skipna=True)
    den = (valid_arr * weight).sum(dim=reduce_dims, skipna=True)

    frac = xr.where(den > 0, num / den, np.nan)
    frac.name = "volume_fraction"
    return frac