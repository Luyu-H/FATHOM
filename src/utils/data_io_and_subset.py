from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any, Mapping
import warnings
import xarray as xr
import numpy as np

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*multiple fill values.*",
    category=xr.SerializationWarning,
)


DEFAULT_VAR_CANDIDATES = ("thetao", "areacello", "variable", "data", "value")

TimeLike = str | slice | tuple[str | None, str | None] | dict[str, Any]
RegionSpec = Mapping[str, float]
DepthLike = tuple[float, float] | Mapping[str, float]


# ===================================================================
# Region-spec conversion
# ===================================================================
 
def get_region_spec(region_code) -> dict[str, float]:
    if isinstance(region_code, dict):
        if "lat_min" in region_code:
            return dict(region_code)
        return {
            "lat_min": region_code["lat1"],
            "lat_max": region_code["lat2"],
            "lon_min": region_code["lon1"],
            "lon_max": region_code["lon2"],
        }
    if isinstance(region_code, (tuple, list)):
        lat1, lat2, lon1, lon2 = region_code
        return {"lat_min": lat1, "lat_max": lat2, "lon_min": lon1, "lon_max": lon2}
    raise TypeError(f"Unsupported region_code type: {type(region_code)}")

def get_spatial_dims(data: xr.DataArray) -> tuple[str, ...]:
    """Infer spatial dimension names from lat coordinate.
    
    - GFDL (1D):  lat(lat), lon(lon) -> ("lat", "lon")
    - CESM2 (2D): lat(nlat, nlon) -> ("nlat", "nlon")
    - MPI (2D):   lat(j, i) -> ("j", "i")
    - Other variants: rlat, grid_latitude, Latitude, etc.
    """
    # List of possible latitude coordinate names (case-insensitive)
    lat_names = ["lat", "latitude", "rlat", "grid_latitude", "nlat", "j"]
    lon_names = ["lon", "longitude", "rlon", "grid_longitude", "nlon", "i"]
    
    # Find the latitude coordinate (case-insensitive)
    lat_coord = None
    for name in lat_names:
        if name in data.coords:
            lat_coord = name
            break
        # Also check capitalized versions
        cap_name = name.capitalize()
        if cap_name in data.coords:
            lat_coord = cap_name
            break
    
    if lat_coord is not None:
        lat_dims = data[lat_coord].dims
        if len(lat_dims) == 1 and lat_dims[0] == lat_coord:
            # 1D case: assume lon has same name pattern
            lon_coord = None
            for name in lon_names:
                if name in data.coords:
                    lon_coord = name
                    break
                cap_name = name.capitalize()
                if cap_name in data.coords:
                    lon_coord = cap_name
                    break
            if lon_coord is not None:
                return (lat_coord, lon_coord)
        # 2D case: return the dims of lat coord, excluding time/depth
        return tuple(d for d in lat_dims if d not in ("time", "depth"))
    
    raise ValueError("Cannot determine spatial dims: no recognized lat coordinate found")


# ===================================================================
# Subsetting primitives
# ===================================================================

def _normalize_lon(lon: xr.DataArray, lon_min: float, lon_max: float):
    """Normalize longitudes to the same convention as the data."""
    lon_values = lon.values
    data_min = float(np.nanmin(lon_values))
    data_max = float(np.nanmax(lon_values))

    # data in [0, 360) convention
    if data_min >= 0 and data_max <= 360:
        if lon_min < 0:
            lon_min = lon_min % 360
        if lon_max < 0:
            lon_max = lon_max % 360
        # clamp 360 to data_max to avoid missing the last cell
        if lon_max == 360 and data_max < 360:
            lon_max = data_max

    # data in [-180, 180) convention
    elif data_min < 0:
        if lon_min > 180:
            lon_min = lon_min - 360
        if lon_max > 180:
            lon_max = lon_max - 360

    return lon_min, lon_max

_LAST_DAY = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
def _expand_partial_date(t: str | None, *, is_end: bool) -> str | None:
    """Expand 'YYYY' or 'YYYY-MM' to a full date so slice boundaries are
    unambiguous regardless of calendar / mid-month timestamps.

    - 'YYYY'    -> 'YYYY-01-01' (start) or 'YYYY-12-31' (end)
    - 'YYYY-MM' -> 'YYYY-MM-01' (start) or 'YYYY-MM-31' (end)
                  (xarray clips day=31 to the actual month length)
    - longer strings: returned unchanged
    """
    if t is None:
        return None
    s = t.strip()
    if len(s) == 4 and s.isdigit():
        return f"{s}-12-31" if is_end else f"{s}-01-01"
    if len(s) == 7 and s[4] == "-":
        if is_end:
            month = int(s[5:7])
            return f"{s}-{_LAST_DAY[month]:02d}"
        return f"{s}-01"
    return s

def subset_time(
    data: xr.DataArray | xr.Dataset,
    time_period: TimeLike,
    *,
    time_dim: str = "time",
) -> xr.DataArray | xr.Dataset:
    """Subset data using a time range.

    Accepted ``time_period`` formats
    --------------------------------
    - ``slice("2000-01-01", "2010-12-31")``
    - ``("2000-01-01", "2010-12-31")``
    - ``{"start": "2000-01-01", "end": "2010-12-31"}``
    - ISO-like string ``"2000-01-01/2010-12-31"``
    - single timestamp string, interpreted as exact selection if possible

    Returns
    -------
    xarray.DataArray or xarray.Dataset
    """
    if isinstance(time_period, slice):
        selector = slice(
            _expand_partial_date(time_period.start, is_end=False),
            _expand_partial_date(time_period.stop,  is_end=True),
        )
    elif isinstance(time_period, tuple):
        start, end = time_period
        selector = slice(
            _expand_partial_date(start, is_end=False),
            _expand_partial_date(end,   is_end=True),
        )
    elif isinstance(time_period, dict):
        selector = slice(
            _expand_partial_date(time_period.get("start"), is_end=False),
            _expand_partial_date(time_period.get("end"),   is_end=True),
        )
    elif isinstance(time_period, str) and "/" in time_period:
        start, end = time_period.split("/", 1)
        selector = slice(
            _expand_partial_date(start or None, is_end=False),
            _expand_partial_date(end   or None, is_end=True),
        )
    elif isinstance(time_period, str):
        return data.sel({time_dim: time_period})
    else:
        raise TypeError(f"Unsupported time_period type: {type(time_period)!r}")

    return data.sel({time_dim: selector})

def subset_region(
    data: xr.DataArray | xr.Dataset,
    region_spec: RegionSpec,
    *,
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> xr.DataArray | xr.Dataset:
    """Subset data by a latitude-longitude bounding box.

    Parameters
    ----------
    region_spec:
        Mapping with keys ``lat_min``, ``lat_max``, ``lon_min``, ``lon_max``.
        If ``lon_min > lon_max`` after normalization, the function assumes the
        region crosses the dateline.
    """
    for key in ("lat_min", "lat_max", "lon_min", "lon_max"):
        if key not in region_spec:
            raise KeyError(f"region_spec must contain '{key}'")

    lat = data[lat_name]
    lon = data[lon_name]
    lat_min = float(region_spec["lat_min"])
    lat_max = float(region_spec["lat_max"])
    lon_min = float(region_spec["lon_min"])
    lon_max = float(region_spec["lon_max"])
    lon_min, lon_max = _normalize_lon(lon, lon_min, lon_max)

    lat_mask = (lat >= min(lat_min, lat_max)) & (lat <= max(lat_min, lat_max))
    if lon_min <= lon_max:
        lon_mask = (lon >= lon_min) & (lon <= lon_max)
    else:
        lon_mask = (lon >= lon_min) | (lon <= lon_max)
    mask = lat_mask & lon_mask

    # 1D coords (GFDL): standard drop
    if lat.ndim == 1 and lon.ndim == 1:
        return data.where(mask, drop=False)

    # 2D coords (CESM2, MPI): bbox isel + masked where
    # mask has dims like (nlat, nlon) or (j, i). Find bounding indices
    # along each spatial dim, isel to that sub-rectangle, then mask.
    spatial = mask.dims
    if len(spatial) != 2:
        return data.where(mask, drop=False)

    d0, d1 = spatial
    any0 = mask.any(dim=d1).values  # along d0
    any1 = mask.any(dim=d0).values  # along d1
    idx0 = np.flatnonzero(any0)
    idx1 = np.flatnonzero(any1)
    if idx0.size == 0 or idx1.size == 0:
        # Region doesn't intersect the grid — return all-NaN slice with
        # one cell so downstream skipna-aware code doesn't blow up.
        return data.where(mask, drop=False)

    sl0 = slice(int(idx0[0]), int(idx0[-1]) + 1)
    sl1 = slice(int(idx1[0]), int(idx1[-1]) + 1)
    sliced = data.isel({d0: sl0, d1: sl1})
    mask_sliced = mask.isel({d0: sl0, d1: sl1})
    return sliced.where(mask_sliced, drop=False)

def subset_depth(
    data: xr.DataArray | xr.Dataset,
    depth_range: DepthLike,
    *,
    depth_dim: str = "depth",
    drop: bool = True,
) -> xr.DataArray | xr.Dataset:
    """Subset data using a depth interval.

    Parameters
    ----------
    depth_range:
        Either ``(min_depth, max_depth)`` or a mapping with keys
        ``min_depth`` and ``max_depth``.
    """
    if depth_dim not in data.dims and depth_dim not in data.coords:
        raise KeyError(f"Depth dimension/coordinate '{depth_dim}' not found")

    if isinstance(depth_range, Mapping):
        zmin = float(depth_range["min_depth"])
        zmax = float(depth_range["max_depth"])
    else:
        zmin, zmax = map(float, depth_range)

    zlo, zhi = min(zmin, zmax), max(zmin, zmax)
    return data.where((data[depth_dim] >= zlo) & (data[depth_dim] <= zhi), drop=drop)

def subset_season(
    data: xr.DataArray | xr.Dataset,
    season: str,
    *,
    time_dim: str = "time",
) -> xr.DataArray | xr.Dataset:
    """Select months belonging to a meteorological season.
 
    Parameters
    ----------
    season : {"DJF", "MAM", "JJA", "SON"}
    """
    _SEASON_MONTHS = {
        "DJF": [12, 1, 2],
        "MAM": [3, 4, 5],
        "JJA": [6, 7, 8],
        "SON": [9, 10, 11],
    }
    season = season.upper()
    if season not in _SEASON_MONTHS:
        raise ValueError(f"Unknown season '{season}'. Use DJF/MAM/JJA/SON.")
 
    months = data[time_dim].dt.month
    return data.sel({time_dim: months.isin(_SEASON_MONTHS[season])})

def subset_single_depth(
    data: xr.DataArray | xr.Dataset,
    depth_value: float,
    *,
    depth_dim: str = "depth",
) -> xr.DataArray | xr.Dataset:
    """Select the nearest single depth level."""
    if depth_dim not in data.dims and depth_dim not in data.coords:
        raise KeyError(f"Depth dimension/coordinate '{depth_dim}' not found")
    return data.sel({depth_dim: depth_value}, method="nearest")


# ===================================================================
# Low-level: open a single file
# ===================================================================

def find_data(experiment_id: str, source_id: str, variable_id: str,
              start_time: str, end_time: str, data_root: str) -> list[str]:
    dir_path = Path(data_root) / experiment_id / source_id / variable_id
    if not dir_path.exists():
        raise FileNotFoundError(f"Data directory not found: {dir_path}")

    nc_files = sorted(dir_path.glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No NC files found in {dir_path}")

    # Fixed fields (areacello etc.) or empty time bounds → return all
    if not start_time or not end_time:
        return [str(f) for f in nc_files]

    def _to_yyyymm(t: str) -> int:
        return int(t.replace("-", "")[:6])

    req_start = _to_yyyymm(start_time)
    req_end = _to_yyyymm(end_time)

    matched = []
    for f in nc_files:
        # e.g. "intdoc_Omon_CESM2_historical_r1i1p1f1_gn_185001-201412"
        time_token = f.stem.rsplit("_", 1)[-1]  # last underscore-delimited part
        if "-" in time_token:
            fs, fe = time_token.split("-")
            if int(fs[:6]) <= req_end and int(fe[:6]) >= req_start:
                matched.append(str(f))
        else:
            # no time range in filename → fixed field, include directly
            matched.append(str(f))

    if not matched:
        raise FileNotFoundError(
            f"No files covering {start_time}~{end_time} in {dir_path}")

    return sorted(matched)

def _standardize_coords(da: xr.DataArray) -> xr.DataArray:
    """Unify coord/dim names: depth(m), lat, lon."""
    rename_map = {}

    # depth dimension: lev/olevel -> depth
    for alias in ("lev", "olevel", "deptht", "depth_coord"):
        if alias in da.dims and "depth" not in da.dims:
            rename_map[alias] = "depth"
            break
    if rename_map:
        da = da.rename(rename_map)

    # CESM2 depth cm -> m
    if "depth" in da.coords:
        units = da["depth"].attrs.get("units", "").lower()
        if "cm" in units or "centimeter" in units:
            da = da.assign_coords(depth=da["depth"].values * 0.01)
            da["depth"].attrs["units"] = "m"

    # lat/lon coordinate names (non-dimension coords for curvilinear)
    #   MPI: latitude(j,i) -> lat(j,i),  longitude(j,i) -> lon(j,i)
    #   CESM2: lat(nlat,nlon) already OK
    #   GFDL:  lat(lat) already OK
    coord_rename = {}
    if "latitude" in da.coords and "lat" not in da.coords:
        coord_rename["latitude"] = "lat"
    if "longitude" in da.coords and "lon" not in da.coords:
        coord_rename["longitude"] = "lon"
    if coord_rename:
        da = da.rename(coord_rename)

    return da

def _attach_layer_thickness(da: xr.DataArray, ds: xr.Dataset) -> xr.DataArray:
    """If ds has lev_bnds (or similar), compute dz from bounds and attach
    as a non-dimension coordinate 'dz' on the depth dim.

    Why: np.gradient(depth) only approximates layer thickness; CMIP6 files
    ship the exact bounds, which differ from gradient especially at top/bottom
    levels and in stretched grids (CESM2 60-level, GFDL 35-level, MPI 40-level).
    """
    if "depth" not in da.dims:
        return da

    bnds_var = None
    for cand in ("lev_bnds", "depth_bnds", "olevel_bnds", "deptht_bounds"):
        if cand in ds.variables:
            bnds_var = ds[cand]
            break
    if bnds_var is None or bnds_var.ndim != 2:
        return da

    # Identify the size-2 dim (bounds) vs the size-N dim (depth).
    bnds_dim = next((d for d in bnds_var.dims if bnds_var.sizes[d] == 2), None)
    depth_dim_orig = next((d for d in bnds_var.dims if d != bnds_dim), None)
    if bnds_dim is None or depth_dim_orig is None:
        return da

    upper = bnds_var.isel({bnds_dim: 1}).values
    lower = bnds_var.isel({bnds_dim: 0}).values
    dz_vals = np.abs(upper - lower).astype(float)

    units = bnds_var.attrs.get("units", "").lower()
    if "cm" in units or "centimeter" in units:
        dz_vals = dz_vals * 0.01  # CESM2 lev_bnds is in cm

    if len(dz_vals) != da.sizes["depth"]:
        return da  # dimension mismatch — bail safely

    return da.assign_coords(dz=("depth", dz_vals))

DEFAULT_DROP_VARIABLES = (
    "lat_bnds", "lon_bnds",
    "latitude_bnds", "longitude_bnds",
    "vertices_latitude", "vertices_longitude",
    "lat_vertices", "lon_vertices",
    # NOTE: do NOT drop lev_bnds / depth_bnds / time_bnds — needed for
    # exact layer-thickness integration and robust time bounds handling.
)

def load_cube(
    source: str | Path,
    variable: str | None = None,
    *,
    engine: str | None = None,
    chunks: dict | None = None,
    drop_variables: Iterable[str] | None = None,
) -> xr.DataArray:
    """Load a variable from a NetCDF or Zarr source as an xarray.DataArray."""
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Data file not found: {source}")
    if drop_variables is None:
        drop_variables = DEFAULT_DROP_VARIABLES

    if source.suffix == ".zarr" or source.name.endswith(".zarr"):
        ds = xr.open_zarr(source, chunks=chunks)
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*multiple fill values.*")
            ds = xr.open_dataset(
                source,
                engine=engine,
                chunks=chunks,
                drop_variables=drop_variables,
            )

    if variable is None:
        # Case 1: only one variable
        if len(ds.data_vars) == 1:
            variable = next(iter(ds.data_vars))

        # Case 2: try known candidates
        else:
            for candidate in DEFAULT_VAR_CANDIDATES:
                if candidate in ds.data_vars:
                    variable = candidate
                    break

        # Case 3: exclude common auxiliary/bounds variables
        if variable is None:
            filtered = [
                name for name in ds.data_vars
                if not (
                    name.endswith("_bnds")
                    or name.endswith("_bounds")
                    or name.startswith("vertices_")
                )
            ]
            if len(filtered) == 1:
                variable = filtered[0]

        # Case 4: if still not found, prefer the variable with most dimensions
        if variable is None:
            if len(ds.data_vars) > 0:
                variable = max(ds.data_vars, key=lambda name: ds[name].ndim)

    if variable is None or variable not in ds.data_vars:
        raise KeyError(
            f"Could not determine a data variable from {list(ds.data_vars)}"
        )

    da = ds[variable]
    da.name = variable
    da = _standardize_coords(da)
    da = _attach_layer_thickness(da, ds) 
    return da


# ===================================================================
# High-level: load & concatenate across multiple files
# ===================================================================

def load_variable(
    experiment_id: str, source_id: str,
    variable_id: str, start_time: str, end_time: str, data_root,
    **cube_kwargs,
) -> xr.DataArray:
    """Locate, load and time-concatenate files for one variable / time span."""
    file_list = find_data(experiment_id, source_id, variable_id,
                          start_time, end_time, data_root)
    if not file_list:
        raise FileNotFoundError(
            f"No data files for {variable_id} in {experiment_id}/{source_id}")
    
    arrays = [load_cube(f, variable=variable_id, **cube_kwargs) for f in file_list]
    data = arrays[0] if len(arrays) == 1 else xr.concat(
        arrays, dim="time", coords="minimal", compat="override",
    ).sortby("time")

    if start_time and end_time and "time" in data.dims:
        data = subset_time(data, (start_time, end_time))

    return data

def load_variable_from_meta(
    data_meta: dict,
    variable_id: str | None = None,
    **cube_kwargs,
) -> xr.DataArray:
    """Shorthand: load using a *data_metadata* dict."""
    if isinstance(data_meta["experiment_id"], (list, tuple)):
        eid = data_meta["experiment_id"][0]
    else:
        eid = data_meta["experiment_id"]
    return load_variable(
        experiment_id=eid,
        source_id=data_meta["source_id"],
        variable_id=variable_id or data_meta["variable_id"],
        start_time=data_meta["start_time"],
        end_time=data_meta["end_time"],
        data_root=data_meta["data_root"],
        **cube_kwargs,
    )

def load_variable_auto_experiment(
    source_id: str,
    variable_id: str,
    start_time: str,
    end_time: str,
    data_root: str,
) -> xr.DataArray:
    """Load a variable over a time range, auto-selecting CMIP6 experiment.
    
    If the period lies entirely within historical (≤2014) or ssp245 (≥2015),
    loads from that single experiment. If it spans the boundary, concatenates
    both and subsets to the requested range.
    """
    from src.utils.domain_knowledge import experiment_for_period  # local import to avoid cycle

    exp = experiment_for_period(start_time, end_time) # return "historical", or "ssp245"
    if isinstance(exp, list):
        parts = [
            load_variable(e, source_id, variable_id, start_time, end_time, data_root)
            for e in exp
        ]
        return xr.concat(parts, dim="time").sortby("time")
    return load_variable(exp, source_id, variable_id, start_time, end_time, data_root)

def load_area_for_model(
    experiment_id: str | list[str] | tuple[str, ...],
    source_id: str,
    data_root: str,
    *,
    fallback_var: str = "thetao",
) -> xr.DataArray:
    """Convenience wrapper around load_area_weights for a (model, experiment) pair.

    Caller no longer needs to construct a partial data_meta dict. The
    fallback path inside load_area_weights still needs a sample data file
    to infer cell area when areacello is missing — fallback_var picks
    which variable to load for that fallback.
    """
    if isinstance(experiment_id, (list, tuple)):
        experiment_id = experiment_id[0]

    if experiment_id == "historical":
        start_time = "1850-01"
    elif experiment_id == "ssp245":
        start_time = "2015-01"
    else:
        raise ValueError(f"Unknown experiment id for area lookup: {experiment_id}")

    return load_area_weights({
        "experiment_id": experiment_id,
        "source_id":     source_id,
        "data_root":     data_root,
        "variable_id":   fallback_var,
        "start_time":    start_time,
        "end_time":      start_time,
    })

def load_bases(
    source_id: str,
    base_codes: list[str],
    start_time: str,
    end_time: str,
    data_root: str,
    *,
    region_spec: RegionSpec | None = None,
    depth_range: DepthLike | None = None,
) -> dict[str, xr.DataArray]:
    """Batch-load multiple variables for one model with auto-experiment time
    handling, optionally applying region and/or depth-range subsets to each.

    Returns a dict {code: DataArray}. Caller decides what semantic
    transformations (implicit_depth, derived computation, etc.) to apply
    next.
    """
    bases: dict[str, xr.DataArray] = {}
    for code in base_codes:
        da = load_variable_auto_experiment(source_id, code, start_time, end_time, data_root)
        if region_spec is not None:
            da = subset_region(da, region_spec)
        if depth_range is not None and "depth" in da.dims:
            da = subset_depth(da, depth_range)
        bases[code] = da
    return bases

def load_bases_for_derived(
    source_id: str,
    derived_spec: dict,
    start_time: str,
    end_time: str,
    data_root: str,
    *,
    region_spec=None,
    depth_range=None,
    implicit_depth: str | None = None,
) -> dict | None:
    """Load all base variables for a derived computation, with optional
    region / depth-range / implicit_depth subsetting.

    `implicit_depth` dispatches by derived type via
    apply_implicit_depth_for_derived. Returns None when the
    implicit_depth + derived combination is degenerate (e.g. "sea floor"
    + inventory-type), so the caller can reject the sample.
    """
    from src.utils.domain_knowledge import apply_implicit_depth_for_derived  # local: avoid cycle

    # Branch A: spec has a fallback chain — try primary, then each fallback.
    if derived_spec.get("fallbacks"):
        primary = derived_spec.get("primary_var", derived_spec["base_codes"][0])
        candidates = [primary] + [fb["variable"] for fb in derived_spec["fallbacks"]]
        bases = None
        for code in candidates:
            try:
                bases = load_bases(
                    source_id, [code], start_time, end_time, data_root,
                    region_spec=region_spec, depth_range=depth_range,
                )
                break
            except FileNotFoundError:
                continue
        if bases is None:
            return None

    # Branch B: standard — load all base_codes (current behavior).
    else:
        bases = load_bases(
            source_id, derived_spec["base_codes"],
            start_time, end_time, data_root,
            region_spec=region_spec, depth_range=depth_range,
        )
        
    if implicit_depth is None:
        return bases

    out = {}
    for k, v in bases.items():
        v2 = apply_implicit_depth_for_derived(v, implicit_depth, derived_spec)
        if v2 is None:
            return None
        out[k] = v2
    return out

def _compute_cell_area(sample_da: xr.DataArray) -> xr.DataArray:
    """Compute approximate cell area from a DataArray's lat/lon coordinates."""
    R = 6.371e6
    lat = sample_da["lat"]
    lon = sample_da["lon"]

    if lat.ndim == 1 and lon.ndim == 1:
        # Regular grid (e.g. GFDL-ESM4)
        lat_v = lat.values
        lon_v = lon.values
        lat_rad = np.deg2rad(lat_v)
        lon_rad = np.deg2rad(lon_v)
        dlat = np.abs(np.gradient(lat_rad))
        dlon = np.abs(np.gradient(lon_rad))
        lat_lo = np.clip(lat_rad - dlat / 2, -np.pi / 2, np.pi / 2)
        lat_hi = np.clip(lat_rad + dlat / 2, -np.pi / 2, np.pi / 2)
        area_vals = R**2 * np.abs(np.sin(lat_hi) - np.sin(lat_lo))[:, None] * dlon[None, :]
        return xr.DataArray(area_vals, coords={"lat": lat_v, "lon": lon_v},
                            dims=("lat", "lon"), attrs={"units": "m2"})
    else:
        # Curvilinear grid (CESM2, MPI) — approximate with cos(lat) weighting
        lat_rad = np.deg2rad(lat.values)
        # Estimate local grid spacing from coordinate gradients
        dlat = np.gradient(np.deg2rad(lat.values), axis=0)
        dlon = np.gradient(np.deg2rad(lon.values), axis=1)
        area_vals = R**2 * np.abs(np.cos(lat_rad) * dlat * dlon)
        return xr.DataArray(area_vals, dims=lat.dims,
            coords={
                "lat": (lat.dims, lat.values),
                "lon": (lon.dims, lon.values),
            },
            attrs={"units": "m2", "long_name": "cell area (approx)"},
        )

def load_area_weights(data_meta: dict) -> xr.DataArray:
    """Load ``areacello`` (ocean grid-cell area) for the same model / experiment."""
    try:
        if isinstance(data_meta["experiment_id"], (list, tuple)):
            exp_id = data_meta["experiment_id"][0]
        else:
            exp_id = data_meta["experiment_id"]
        file_list = find_data(
            experiment_id=exp_id,
            source_id=data_meta["source_id"],
            variable_id="areacello",
            start_time="", end_time="",
            data_root=data_meta["data_root"],
        )
        area = load_cube(file_list[0], variable="areacello")
    except FileNotFoundError:
        # Pick a concrete variable to use as the lat/lon sample.
        vid = data_meta.get("variable_id")
        if isinstance(vid, (list, tuple)):
            sample_var = vid[0] if vid else "thetao"
        else:
            sample_var = vid or "thetao"
        sample = load_variable_from_meta(data_meta, variable_id=sample_var)
        area = _compute_cell_area(sample)

    # Some CMIP6 areacello files ship with a degenerate (size-1) time or
    # depth dim. Drop them so downstream broadcasting / sum reductions
    # don't accidentally collapse over time.
    _SPATIAL = {"lat", "lon", "nlat", "nlon", "i", "j", "x", "y"}
    extra = [d for d in area.dims if d not in _SPATIAL and area.sizes[d] == 1]
    if extra:
        area = area.squeeze(extra, drop=True)
    return area

def load_depth_levels(data_root: str, experiment_id: str, source_id: str, variable_id: str) -> np.ndarray | None:
    dir_path = Path(data_root) / experiment_id / source_id / variable_id
    nc_files = sorted(dir_path.glob("*.nc"))
    if not nc_files:
        return None
    ds = xr.open_dataset(nc_files[0])
    for dim_name in ("lev", "depth", "olevel", "z"):
        if dim_name in ds.coords:
            depths = ds[dim_name].values.astype(float)
            units = ds[dim_name].attrs.get("units", "").lower()
            if "cm" in units or "centimeter" in units:
                depths = depths * 0.01
            ds.close()
            return depths
    ds.close()
    return None


# ===================================================================
# Combined subsetting (driven by task_metadata keys)
# ===================================================================

def subset_data(
    data: xr.DataArray,
    task_meta: dict,
    *,
    do_time: bool = True,
    do_region: bool = True,
    do_depth: bool = True,
) -> xr.DataArray:
    """Apply time / region / depth subsetting from *task_metadata* keys.
 
    Key conventions
    ---------------
    - time:   ``start_month``, ``end_month``
    - region: ``region_code``  (``{"lat1","lat2","lon1","lon2"}`` from ``sample_region``)
    - depth:  ``depth_lower``, ``depth_upper``
    """
    if do_time and "start_month" in task_meta:
        data = subset_time(data, (task_meta["start_month"], task_meta["end_month"]))
 
    if do_region and "region_code" in task_meta:
        data = subset_region(data, get_region_spec(task_meta["region_code"]))
 
    if do_depth and "depth_lower" in task_meta and task_meta["depth_lower"] is not None:
        data = subset_depth(data, (task_meta["depth_lower"], task_meta["depth_upper"]))
    
    if any(data.sizes[d] == 0 for d in data.dims):
        raise ValueError(
            f"Subset produced empty data. "
            f"Dims after subset: {dict(data.sizes)}"
        )
    return data
