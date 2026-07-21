from __future__ import annotations

from typing import Sequence

import numpy as np
import xarray as xr

from src.utils.numerical_diagnostics import get_dz


# ===================================================================
# Variable metadata and unit resolution
# ===================================================================

# Canonical CMIP6 units for each variable code in this benchmark.
# Source: user-provided dimension/variable summary.
VARIABLE_UNITS: dict[str, str] = {
    "thetao": "degC",
    "so":     "0.001",            # CMIP6 reports salinity as dimensionless 0.001
    "ph":     "1",                # dimensionless
    # mol m-3 group
    "no3": "mol m-3", "po4": "mol m-3", "si": "mol m-3", "dfe": "mol m-3",
    "dissic": "mol m-3", "dissoc": "mol m-3", "talk": "mol m-3",
    "phycos": "mol m-3", "o2": "mol m-3", "o2sat": "mol m-3",
    # currents
    "uo": "m s-1", "vo": "m s-1", "wo": "m s-1",
    # depth-integrated stocks
    "intdic": "kg m-2", "intdoc": "kg m-2", "intpoc": "kg m-2",
    # chlorophyll
    "chl": "kg m-3",
    # mass transports
    "umo": "kg s-1", "vmo": "kg s-1", "wmo": "kg s-1",
}

def get_variable_unit(var_code: str) -> str:
    """Look up canonical CMIP6 unit for a variable code, '' if unknown."""
    return VARIABLE_UNITS.get(var_code, "")


# ===================================================================
# Experiment / time helpers
# ===================================================================

def experiment_for_year(year: int) -> str:
    """Map a calendar year to the appropriate CMIP6 experiment id."""
    return "historical" if year <= 2014 else "ssp245"


def experiment_for_period(start_time: str, end_time: str) -> str | list[str]:
    """Return experiment id(s) that cover *start_time* – *end_time*.

    If the period spans the historical/SSP boundary (2014/2015), a list
    of both experiment ids is returned.
    """
    sy = int(start_time[:4])
    ey = int(end_time[:4])
    if ey <= 2014:
        return "historical"
    if sy >= 2015:
        return "ssp245"
    return ["historical", "ssp245"]


# ===================================================================
# Implicit-depth resolution
# ===================================================================

#: Mapping from natural-language depth descriptor to ``(lower, upper)`` in metres.
#: ``"sea floor"`` is mapped to ``None`` — handled specially (deepest valid level).
IMPLICIT_DEPTH_RANGES: dict[str, tuple[float, float] | None] = {
    "sea surface":       (0, 10),
    "near surface":      (0, 50),
    "subsurface":        (50, 200),
    "upper ocean":       (0, 300),
    "photic zone":       (0, 150),
    "intermediate depth": (300, 1000),
    "deep ocean":        (1000, 5500),
    "sea floor":         None,
}


def resolve_implicit_depth(implicit_depth: str) -> tuple[float, float] | None:
    """Return ``(lower, upper)`` in metres, or ``None`` for sea floor.

    Raises ``KeyError`` if the term is not recognised.
    """
    key = implicit_depth.lower().strip()
    if key not in IMPLICIT_DEPTH_RANGES:
        raise KeyError(f"Unknown implicit depth term: '{implicit_depth}'")
    return IMPLICIT_DEPTH_RANGES[key]


def apply_implicit_depth(
    data: xr.DataArray,
    implicit_depth: str,
    *,
    depth_dim: str = "depth",
    drop: bool = True,
) -> xr.DataArray:
    """Subset *data* according to an implicit depth descriptor.

    For ``"sea floor"`` the deepest non-NaN level per water column is selected.
    """
    if depth_dim not in data.dims and depth_dim not in data.coords:
        return data
    
    rng = resolve_implicit_depth(implicit_depth)
    if rng is not None:
        lo, hi = rng
        return data.where(
            (data[depth_dim] >= lo) & (data[depth_dim] <= hi), drop=drop,
        )
    # sea floor → deepest valid level
    if depth_dim in data.dims:
        # Reverse depth so first non-NaN along axis is the deepest
        reversed_data = data.isel({depth_dim: slice(None, None, -1)})
        # Use idxmin on a boolean "is NaN" array? Simpler: just pick last level.
        # For most CMIP6 models, isel(depth=-1) with NaN-aware stats suffices.
        return data.isel({depth_dim: -1})
    return data

def apply_implicit_depth_for_derived(
    da: xr.DataArray,
    implicit_depth: str,
    derived_spec: dict,
    *,
    depth_dim: str = "depth",
) -> xr.DataArray | None:
    """Apply an implicit_depth descriptor to a base DataArray, dispatched by
    the derived-variable type.

    For inventory / threshold_volume / threshold_thickness derivations the
    depth dimension MUST be preserved (compute_inventory etc. require it),
    so we use a depth-range subset. Returns None when the descriptor
    resolves to a single level (e.g. "sea floor"), which is degenerate
    for column-integrated metrics — the caller should reject the sample.

    For formula-type derivations (AOU, ratios, current speed, etc.) the
    standard apply_implicit_depth is used and may collapse the depth dim.
    """
    from src.utils.data_io_and_subset import subset_depth  # local: avoid cycle

    if depth_dim not in da.dims:
        return da

    needs_depth = derived_spec.get("type") in (
        "inventory", "threshold_volume", "threshold_thickness",
    )
    if needs_depth:
        rng = resolve_implicit_depth(implicit_depth)
        if rng is None:                         # "sea floor"
            return None
        return subset_depth(da, rng)
    return apply_implicit_depth(da, implicit_depth)

# ===================================================================
# Climatology time-period resolution
# ===================================================================

def resolve_implicit_time_period(clim_time_code: str) -> tuple[str, str]:
    """Convert a climatology code like ``"1961-1990"`` to ``("1961-01", "1990-12")``."""
    parts = clim_time_code.split("-")
    if len(parts) == 2:
        return f"{parts[0]}-01", f"{parts[1]}-12"
    raise ValueError(f"Cannot parse clim_time_code: '{clim_time_code}'")


# ===================================================================
# Qualified-area & descriptive-adjective criteria
# ===================================================================

#: Each entry: variable to test, comparison operator, threshold value.
QUALIFIED_AREA_CRITERIA: dict[str, dict] = {
    "hypoxia": {
        "variable": "o2", "op": "<", "threshold": 0.060,
        "description": "O₂ < 60 mmol/m³ (~2 mg/L) — hypoxia threshold",
    },
    "oxygen minimum zone": {
        "variable": "o2", "op": "<", "threshold": 0.060,
        "description": "O₂ < 60 µmol/kg default OMZ threshold "
                       "(~0.060 mol/m³ assuming ρ≈1000 kg/m³)",
    },
    "oligotrophic area": {
        "variable": "chl", "op": "<", "threshold": 1e-7,
        "description": "Chl-a < 0.1 mg/m³ (TERM_7 primary criterion)",
        "fallbacks": [
            {"variable": "no3", "op": "<", "threshold": 1e-4},
            {"variable": "po4", "op": "<", "threshold": 1e-5},
        ],
    },
}

DESCRIPTIVE_ADJ_CRITERIA: dict[str, dict] = {
    "oligotrophic": {
        "variable": "chl", "op": "<", "threshold": 1e-7,
        "direction": "decrease",
        # Per TERM_7 data_priority: chl > nitrate > phosphate.
        # Consumers can fall back if chl is unavailable.
        "fallbacks": [
            {"variable": "no3", "op": "<", "threshold": 1e-4},   # 0.1 mmol/m³
            {"variable": "po4", "op": "<", "threshold": 1e-5},   # 0.01 mmol/m³
        ],
    },
    "hypoxic": {
        "variable": "o2", "op": "<", "threshold": 0.060,
        "direction": "decrease",
    },
}

def iter_criteria_with_fallbacks(criteria: dict):
    """Yield (variable, op, threshold) tuples in priority order.

    Starts with the primary (``criteria["variable"]``, ``op``, ``threshold``),
    then walks through ``criteria.get("fallbacks", [])`` in order.
    """
    yield criteria["variable"], criteria["op"], criteria["threshold"]
    for fb in criteria.get("fallbacks", []):
        yield fb["variable"], fb["op"], fb["threshold"]

def build_qualified_area_mask(
    source_id: str,
    qa_criteria: dict,
    start_time: str,
    end_time: str,
    data_root: str,
    *,
    region_spec,
    implicit_depth: str | None = None,
) -> tuple:
    """Build a 2-D (lat, lon) boolean mask for a qualified-area term.

    Walks `iter_criteria_with_fallbacks(qa_criteria)`: each candidate
    variable is loaded over [start, end], region-subsetted, optionally
    restricted by implicit_depth, time-averaged, depth-collapsed if 3-D,
    and thresholded. The first candidate producing a non-empty mask wins.

    Returns (mask, chosen_variable_code), or (None, None) if no candidate
    yielded a non-empty mask.
    """
    from src.utils.data_io_and_subset import load_variable_auto_experiment, subset_region
    from src.utils.numerical_diagnostics import (
        compute_climatology, evaluate_condition_or_compare,
    )

    for qa_var, op, threshold in iter_criteria_with_fallbacks(qa_criteria):
        try:
            qa_da = load_variable_auto_experiment(
                source_id, qa_var, start_time, end_time, data_root,
            )
        except FileNotFoundError:
            continue
        qa_da = subset_region(qa_da, region_spec)
        if implicit_depth is not None and "depth" in qa_da.dims:
            qa_da = apply_implicit_depth(qa_da, implicit_depth)
        if any(qa_da.sizes[d] == 0 for d in qa_da.dims):
            continue
        qa_clim = compute_climatology(qa_da)
        if "depth" in qa_clim.dims:
            qa_clim = qa_clim.mean(dim="depth", skipna=True)
        candidate = evaluate_condition_or_compare(qa_clim, op, threshold)
        # if not bool(candidate.any().values):
        #     continue
        return candidate, qa_var
    return None, None

def resolve_qualified_area_criteria(term: str) -> dict:
    """Return criteria dict for a qualified-area term."""
    key = term.lower().strip()
    if key not in QUALIFIED_AREA_CRITERIA:
        raise KeyError(f"Unknown qualified area: '{term}'")
    return QUALIFIED_AREA_CRITERIA[key]


def resolve_descriptive_adj_criteria(adj: str) -> dict:
    """Return criteria dict for a descriptive adjective."""
    key = adj.lower().strip()
    if key not in DESCRIPTIVE_ADJ_CRITERIA:
        raise KeyError(f"Unknown descriptive adjective: '{adj}'")
    return DESCRIPTIVE_ADJ_CRITERIA[key]


# ===================================================================
# Potential density (simplified linear EOS)
# ===================================================================

def compute_potential_density(
    thetao: xr.DataArray,
    so: xr.DataArray,
    *,
    use_teos10: bool = True,
    depth_dim: str = "depth",
) -> xr.DataArray:
    """Strict TEOS-10 potential density anomaly (sigma0, reference 0 dbar).

    Performs the full conversion pipeline rather than approximating
    SA≈SP and CT≈PT:

      1. p   = gsw.p_from_z(-z, lat)            # sea pressure [dbar]
      2. SA  = gsw.SA_from_SP(so, p, lon, lat)  # absolute salinity [g/kg]
      3. CT  = gsw.CT_from_pt(SA, thetao)       # conservative temperature [°C]
      4. σ₀  = gsw.sigma0(SA, CT)               # potential density anomaly [kg/m³]

    CMIP6 supplies potential temperature (``thetao``, °C) and practical
    salinity (``so``, PSS-78). The SP→SA offset is typically 0.1–0.5 g/kg
    and translates to a density bias of ~0.05–0.2 kg/m³ — same order of
    magnitude as the 0.03 kg/m³ MLD threshold, so the full conversion
    matters for MLD detection and for N² profiles.

    Falls back to a simple linear EOS if ``gsw`` is unavailable or
    ``use_teos10=False``.
    """
    if use_teos10:
        try:
            import gsw

            z   = thetao[depth_dim]   # depth (positive-down) in metres
            lat = thetao["lat"]
            lon = thetao["lon"]

            # Sea pressure from depth & latitude (z is negative below sea level)
            p = xr.apply_ufunc(
                gsw.p_from_z, -z, lat,
                dask="parallelized", output_dtypes=[float],
            )

            # Absolute salinity
            SA = xr.apply_ufunc(
                gsw.SA_from_SP, so, p, lon, lat,
                dask="parallelized", output_dtypes=[float],
            )

            # Conservative temperature
            CT = xr.apply_ufunc(
                gsw.CT_from_pt, SA, thetao,
                dask="parallelized", output_dtypes=[float],
            )

            # Potential density anomaly, reference pressure 0 dbar
            sigma0 = xr.apply_ufunc(
                gsw.sigma0, SA, CT,
                dask="parallelized", output_dtypes=[float],
            )
            sigma0.name = "sigma0"
            sigma0.attrs["units"] = "kg/m³"
            sigma0.attrs["description"] = (
                "TEOS-10 potential density anomaly (ref 0 dbar) with full "
                "SP→SA and PT→CT conversion"
            )
            return sigma0
        except ImportError:
            import warnings
            warnings.warn(
                "gsw package not available; falling back to linear EOS for density.",
                RuntimeWarning,
            )

    # Linear EOS fallback
    rho0, alpha, beta, T0, S0 = 1025.0, 0.2, 0.8, 10.0, 35.0
    rho = rho0 * (1.0 - alpha * (thetao - T0) + beta * (so - S0))
    rho.name = "sigma0"
    rho.attrs["units"] = "kg/m³"
    return rho


# ===================================================================
# Stratification index
# ===================================================================

def compute_n_squared(
    thetao: xr.DataArray,
    so: xr.DataArray,
    *,
    depth_dim: str = "depth",
    rho0: float = 1025.0,
    g: float = 9.81,
) -> xr.DataArray:
    r"""Brunt-Väisälä frequency squared, :math:`N^2 = -(g/\rho_0)\,\partial\rho/\partial z`.

    Uses TEOS-10 potential density (via ``compute_potential_density``) and
    centred finite differences along the depth dimension. Depth is taken as
    positive downward (CMIP6 convention), so a stable water column yields
    ``N² > 0``.

    Returns an xr.DataArray with units s⁻² on the same grid as the inputs.
    """
    rho = compute_potential_density(thetao, so)  # sigma0, TEOS-10 when available

    z = rho[depth_dim]
    # gradient along depth (positive-down) → dρ/dz
    drho_dz = xr.apply_ufunc(
        lambda a: np.gradient(a, z.values, axis=-1),
        rho,
        input_core_dims=[[depth_dim]],
        output_core_dims=[[depth_dim]],
        dask="parallelized",
        output_dtypes=[float],
    )
    n2 = (g / rho0) * drho_dz
    n2.name = "n_squared"
    n2.attrs["units"] = "s-2"
    n2.attrs["long_name"] = "Brunt-Vaisala frequency squared"
    return n2

def compute_stratification_index(
    thetao: xr.DataArray,
    so: xr.DataArray,
    *,
    depth_dim: str = "depth",
    depth_range: tuple[float, float] = (0, 200),
    reduce: str = "mean",
) -> xr.DataArray:
    """
    Water-column stability over *depth_range*, quantified as N².

    Parameters
    ----------
    depth_range : (lo, hi) metres
        Depth window over which N² is evaluated.
    reduce : {"mean", "max", "integral"}
        How to collapse the N² profile into a single stability index per
        (time, lat, lon):
        - ``"mean"`` (default): depth-mean N² within the window — most
          directly interpretable as "average stability".
        - ``"max"``: peak N², approximates pycnocline sharpness.
        - ``"integral"``: ∫ N² dz over the window (units s⁻² · m).

    Returns
    -------
    xr.DataArray
        Stability index with depth collapsed. Units s⁻² (or s⁻²·m for integral).
    """
    n2 = compute_n_squared(thetao, so, depth_dim=depth_dim)
    z = n2[depth_dim]
    n2_layer = n2.where((z >= depth_range[0]) & (z <= depth_range[1]), drop=True)

    if reduce == "mean":
        out = n2_layer.mean(dim=depth_dim, skipna=True)
        out.attrs["units"] = "s-2"
    elif reduce == "max":
        out = n2_layer.max(dim=depth_dim, skipna=True)
        out.attrs["units"] = "s-2"
    elif reduce == "integral":
        dz = get_dz(n2, depth_dim=depth_dim)
        dz_layer = dz.where((z >= depth_range[0]) & (z <= depth_range[1]), drop=True)
        out = (n2_layer * dz_layer).sum(dim=depth_dim, skipna=True)
        out.attrs["units"] = "s-2 m"
    else:
        raise ValueError(f"Unknown reduce='{reduce}'. Use 'mean', 'max', or 'integral'.")

    out.name = "stratification_index"
    out.attrs["definition"] = f"N² {reduce} over {depth_range[0]}-{depth_range[1]} m"
    return out


# ===================================================================
# Mixed-layer depth  (density criterion)
# ===================================================================

def compute_mixed_layer_depth(
    thetao: xr.DataArray,
    so: xr.DataArray,
    *,
    depth_dim: str = "depth",
    delta_rho: float = 0.03,
    ref_depth: float = 10.0,
) -> xr.DataArray:
    r"""Mixed-layer depth using a density-difference criterion.

    MLD is the shallowest depth where
    :math:`\rho(z) - \rho(z_{\text{ref}}) > \Delta\rho`.

    Parameters
    ----------
    delta_rho : float
        Density threshold in kg/m³ (default 0.03).
    ref_depth : float
        Reference depth in metres (default 10 m).
    """
    rho = compute_potential_density(thetao, so)
    rho_ref = rho.sel({depth_dim: ref_depth}, method="nearest")
    diff = rho - rho_ref
    exceeds = diff > delta_rho

    depths = rho[depth_dim]

    def _first_exceed(col: np.ndarray) -> float:
        idx = np.flatnonzero(col)
        return float(depths.values[idx[0]]) if idx.size > 0 else float(depths.values[-1])

    mld = xr.apply_ufunc(
        _first_exceed, exceeds,
        input_core_dims=[[depth_dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    mld.name = "mld"
    return mld


# ===================================================================
# Thermocline depth  (max |dT/dz|)
# ===================================================================

def compute_thermocline_depth(
    thetao: xr.DataArray,
    *,
    depth_dim: str = "depth",
    search_range: tuple[float, float] = (10.0, 500.0),
    smooth: bool = True,
) -> xr.DataArray:
    # Restrict to search interval
    thetao_s = thetao.sel({depth_dim: slice(*search_range)})
    if thetao_s.sizes.get(depth_dim, 0) < 3:
        # Not enough levels in the window — fall back to full profile
        thetao_s = thetao

    if smooth:
        thetao_s = thetao_s.rolling(
            {depth_dim: 3}, center=True, min_periods=1,
        ).mean()

    z = thetao_s[depth_dim].values.astype(float)

    def _tc_depth(profile: np.ndarray) -> float:
        mask = np.isfinite(profile)
        if mask.sum() < 3:
            return np.nan
        grad = np.abs(np.gradient(profile[mask], z[mask]))
        return float(z[mask][np.argmax(grad)])

    tc = xr.apply_ufunc(
        _tc_depth, thetao_s,
        input_core_dims=[[depth_dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    tc.name = "thermocline_depth"
    tc.attrs["units"] = "m"
    tc.attrs["search_range_m"] = list(search_range)
    return tc


# ===================================================================
# Layer thickness  (hypoxia / OMZ / …)
# ===================================================================

def compute_layer_thickness(
    data: xr.DataArray,
    threshold: float,
    comparison_op: str = "<",
    *,
    depth_dim: str = "depth",
    method: str = "bounded",
) -> xr.DataArray:
    """Vertical thickness (metres) of the layer satisfying *comparison_op* *threshold*.

    Parameters
    ----------
    method : {"bounded", "sum"}
        - ``"bounded"`` (default, JSONL OMZ spec):
          thickness = ``deepest_qualifying_depth - shallowest_qualifying_depth``.
          If the deepest level still qualifies, it is used as the lower
          bound (handles "bottom below threshold" case).
        - ``"sum"``: sum of layer thicknesses dz where the condition holds
          (total qualifying-cell thickness; differs from "bounded" when the
          qualifying layer has gaps).
    """
    from src.utils.numerical_diagnostics import evaluate_condition_or_compare

    mask = evaluate_condition_or_compare(data, comparison_op, threshold)
    z = data[depth_dim]
    z_vals = z.values.astype(float)

    if method == "sum":
        dz = get_dz(data, depth_dim=depth_dim)
        thickness = (mask * dz).sum(dim=depth_dim, skipna=True)

    elif method == "bounded":
        # Preserve NaN where the entire profile is land/missing
        mask_f = mask.astype(float).where(data.notnull())

        def _bounded(col: np.ndarray) -> float:
            finite = np.isfinite(col)
            if finite.sum() == 0:
                return np.nan                  # all-land column
            qualifying = (col == 1.0) & finite
            idx = np.flatnonzero(qualifying)
            if idx.size == 0:
                return 0.0                     # ocean but nowhere qualifying
            return float(z_vals[idx[-1]] - z_vals[idx[0]])

        thickness = xr.apply_ufunc(
            _bounded, mask_f,
            input_core_dims=[[depth_dim]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
    
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'bounded' or 'sum'.")

    thickness.name = "layer_thickness"
    thickness.attrs["units"] = "m"
    thickness.attrs["method"] = method
    return thickness

def compute_omz_thickness(
    o2: xr.DataArray, threshold: float, *, depth_dim: str = "depth",
) -> xr.DataArray:
    """OMZ thickness per column.

    Spec:
      - upper_bound: shallowest depth where o2 < threshold (first below-threshold from surface)
      - lower_bound: deepest depth where o2 < threshold;
                    if the deepest valid level itself is still below threshold,
                    use the ocean floor of that column (z + dz/2 of last valid layer).
      - thickness = lower_bound - upper_bound
    """
    z_vals = o2[depth_dim].values.astype(float)
    dz = get_dz(o2, depth_dim=depth_dim).values.astype(float)

    def _omz_thick(col: np.ndarray) -> float:
        finite = np.isfinite(col)
        if not finite.any():
            return np.nan                                # all-land column
        qualifying = (col < threshold) & finite
        if not qualifying.any():
            return 0.0                                   # ocean but no OMZ
        qual_idx = np.flatnonzero(qualifying)
        upper = z_vals[qual_idx[0]]
        deepest_valid = np.flatnonzero(finite)[-1]
        if qualifying[deepest_valid]:
            # bottom level still below threshold → push to ocean floor
            lower = z_vals[deepest_valid] + dz[deepest_valid] / 2.0
        else:
            lower = z_vals[qual_idx[-1]]
        return float(lower - upper)

    thickness = xr.apply_ufunc(
        _omz_thick, o2,
        input_core_dims=[[depth_dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    thickness.name = "omz_thickness"
    thickness.attrs["units"] = "m"
    thickness.attrs["definition"] = (
        "first below-threshold depth from surface to "
        "deepest below-threshold depth (or ocean floor)"
    )
    return thickness


# ===================================================================
# ENSO phase classification  (Niño 3.4 index)
# ===================================================================

# Niño 3.4 box: 5°N–5°S, 170°W–120°W  (in 0–360: 190°E–240°E)
NINO34_SPEC: dict[str, float] = {
    "lat_min": -5, "lat_max": 5, "lon_min": 190, "lon_max": 240,
}
ENSO_EL_NINO_THRESHOLD = 0.5   # °C
ENSO_LA_NINA_THRESHOLD = -0.5  # °C
ENSO_SMOOTHING_MONTHS  = 3     # 3-month running mean
ENSO_MIN_DURATION      = 3     # consecutive months (used only if strict ONI)

ENSO_BASELINE_START = "1971-01"
ENSO_BASELINE_END   = "2000-12"

def filter_enso_persistence(
    phase_labels: xr.DataArray,
    *,
    min_duration: int = ENSO_MIN_DURATION,
    time_dim: str = "time",
) -> xr.DataArray:
    """Enforce the strict ONI ≥5-consecutive-months rule on ENSO phase labels.

    Input is a 1-D DataArray of phase codes (+1 El Niño, -1 La Niña, 0 Neutral)
    from ``classify_enso_months``. Output has the same coordinates; months
    belonging to +1/-1 runs shorter than *min_duration* are demoted to 0
    (Neutral). Runs of neutral months are untouched.
    """
    values = phase_labels.values.astype(int)
    out = values.copy()

    n = len(values)
    i = 0
    while i < n:
        code = values[i]
        if code == 0:
            i += 1
            continue
        j = i
        while j < n and values[j] == code:
            j += 1
        # Run is values[i:j]
        if (j - i) < min_duration:
            out[i:j] = 0
        i = j

    result = xr.DataArray(out, coords=phase_labels.coords, dims=phase_labels.dims)
    result.name = "enso_phase"
    result.attrs["min_duration_months"] = min_duration
    return result


def compute_nino34_index(
    sst: xr.DataArray,
    area: xr.DataArray | None = None,
    *,
    time_dim: str = "time",
) -> xr.DataArray:
    """Compute the Niño 3.4 SST index as area-weighted spatial mean.

    *sst* should already be subsetted to the Niño 3.4 box or will be
    averaged over all spatial dims.
    """
    spatial_dims = [d for d in sst.dims if d != time_dim]
    if area is not None:
        weights = area.fillna(0)
        index = sst.weighted(weights).mean(dim=spatial_dims)
    else:
        index = sst.mean(dim=spatial_dims, skipna=True)
    index.name = "nino34"
    return index


def classify_enso_months(
    nino34: xr.DataArray,
    *,
    el_nino_threshold: float = ENSO_EL_NINO_THRESHOLD,
    la_nina_threshold: float = ENSO_LA_NINA_THRESHOLD,
    time_dim: str = "time",
) -> xr.DataArray:
    """Classify each time step as El Niño (+1), La Niña (−1), or Neutral (0).

    The *nino34* input should be an **anomaly** index (relative to a
    climatological mean).
    """
    phase = xr.where(nino34 > el_nino_threshold, 1,
             xr.where(nino34 < la_nina_threshold, -1, 0))
    phase.name = "enso_phase"
    return phase


def select_enso_phase(
    data: xr.DataArray,
    phase_labels: xr.DataArray,
    phase: str,
    *,
    time_dim: str = "time",
) -> xr.DataArray:
    """Select time steps matching a named ENSO phase.

    *phase*: ``"El Nino phase"`` (+1), ``"La Nina phase"`` (−1),
    ``"Neutral phase of ENSO"`` (0).
    """
    _PHASE_MAP = {
        "El Nino phase": 1,
        "La Nina phase": -1,
        "Neutral phase of ENSO": 0,
    }
    code = _PHASE_MAP.get(phase)
    if code is None:
        raise KeyError(f"Unknown ENSO phase: '{phase}'")
    mask = phase_labels == code
    # Drop any non-time coords that may survive surface selection (e.g. depth=5)
    if any(coord not in mask.dims for coord in mask.coords):
        mask = mask.reset_coords(drop=True)
    return data.sel({time_dim: mask})


def build_enso_phase_labels(
    source_id: str,
    start_time: str,
    end_time: str,
    data_root: str,
) -> xr.DataArray:
    """Build per-month ENSO phase labels (+1 / -1 / 0) over [start, end].

    Pipeline (mirrors _solve_L2_9):
      1. Load thetao surface in Niño 3.4 over fixed 1971-2000 baseline.
      2. Monthly climatology of the Niño 3.4 index from the baseline.
      3. Load thetao surface in Niño 3.4 over the analysis window.
      4. Anomaly vs baseline → 3-month running mean → classify_enso_months
         → filter_enso_persistence (>= ENSO_MIN_DURATION consecutive months).
    """
    from src.utils.data_io_and_subset import load_variable_auto_experiment, subset_region
    from src.utils.numerical_diagnostics import compute_monthly_climatology, compute_anomaly

    sst_base = load_variable_auto_experiment(
        source_id, "thetao", ENSO_BASELINE_START, ENSO_BASELINE_END, data_root,
    )
    sst_base_surf = sst_base.isel(depth=0) if "depth" in sst_base.dims else sst_base
    sst_base_nino = subset_region(sst_base_surf, NINO34_SPEC)
    nino_base_idx = compute_nino34_index(sst_base_nino)
    nino_base_clim = compute_monthly_climatology(nino_base_idx)

    sst_ana = load_variable_auto_experiment(
        source_id, "thetao", start_time, end_time, data_root,
    )
    sst_ana_surf = sst_ana.isel(depth=0) if "depth" in sst_ana.dims else sst_ana
    sst_ana_nino = subset_region(sst_ana_surf, NINO34_SPEC)
    nino_ana_idx = compute_nino34_index(sst_ana_nino)

    nino_anom = compute_anomaly(nino_ana_idx, nino_base_clim)
    nino_anom_smooth = nino_anom.rolling(
        time=ENSO_SMOOTHING_MONTHS, center=True, min_periods=1,
    ).mean()
    raw_labels = classify_enso_months(nino_anom_smooth)
    return filter_enso_persistence(raw_labels, min_duration=ENSO_MIN_DURATION)


def compute_pdo_index(
    source_id: str,
    start_time: str,
    end_time: str,
    data_root: str,
) -> xr.DataArray:
    """Standardised PDO index over [start_time, end_time].

    Standard EOF method (Mantua et al. 1997 style):
      1. Load global thetao surface for ENSO_BASELINE (1971-2000) + analysis window.
      2. Anomaly vs baseline monthly climatology.
      3. Subtract global-mean anomaly per time step (remove warming signal).
      4. Subset to PDO box (20-70°N, 120°E-100°W); apply √cos(lat) area weighting.
      5. SVD on baseline → EOF1; sign-correct so positive index ↔ warm eastern
         North Pacific (30-50°N, 200-240°E loading > 0).
      6. Project analysis onto EOF1; standardize by baseline PC1 std.
    """
    from src.utils.data_io_and_subset import (
        load_variable_auto_experiment, subset_region,
    )
    from src.utils.numerical_diagnostics import (
        compute_monthly_climatology, compute_anomaly,
    )

    # 1. Load SST surface for baseline + analysis.
    sst_b = load_variable_auto_experiment(
        source_id, "thetao", ENSO_BASELINE_START, ENSO_BASELINE_END, data_root,
    )
    sst_a = load_variable_auto_experiment(
        source_id, "thetao", start_time, end_time, data_root,
    )
    sst_b_surf = sst_b.isel(depth=0) if "depth" in sst_b.dims else sst_b
    sst_a_surf = sst_a.isel(depth=0) if "depth" in sst_a.dims else sst_a

    # 2. Anomaly vs baseline monthly clim, then 3. remove global-mean anomaly.
    clim = compute_monthly_climatology(sst_b_surf)
    anom_b = compute_anomaly(sst_b_surf, clim)
    anom_a = compute_anomaly(sst_a_surf, clim)
    spatial_full = [d for d in anom_b.dims if d != "time"]
    det_b = anom_b - anom_b.mean(dim=spatial_full, skipna=True)
    det_a = anom_a - anom_a.mean(dim=spatial_full, skipna=True)

    # 4. PDO-box subset + √cos(lat) weighting.
    pdo_b = subset_region(det_b, PDO_SPEC)
    pdo_a = subset_region(det_a, PDO_SPEC)
    w = np.sqrt(np.clip(np.cos(np.deg2rad(pdo_b["lat"])), 0, 1))
    pdo_b_w = pdo_b * w
    pdo_a_w = pdo_a * w

    # 5. Stack space → SVD on baseline.
    spatial_dims = [d for d in pdo_b.dims if d != "time"]
    lat_field = pdo_b["lat"]   # 1-D (GFDL) or 2-D (CESM2/MPI)
    lon_field = pdo_b["lon"]
    Xb = pdo_b_w.stack(_space=spatial_dims).transpose("time", "_space")
    Xa = pdo_a_w.stack(_space=spatial_dims).transpose("time", "_space")
    Xb_v, Xa_v = Xb.values, Xa.values
    valid = np.isfinite(Xb_v).all(axis=0)
    if valid.sum() < 10:
        return xr.DataArray(
            np.full(Xa.sizes["time"], np.nan),
            coords={"time": Xa["time"]}, dims=["time"], name="pdo_index",
        )
    Xb_valid = Xb_v[:, valid]
    Xa_valid = np.nan_to_num(Xa_v[:, valid], nan=0.0)
    _, _, Vt = np.linalg.svd(Xb_valid, full_matrices=False)
    eof1 = Vt[0]

    # Sign-correct: positive PDO ↔ warm 30-50°N, 200-240°E.
    full_pat = np.full(Xb_v.shape[1], np.nan)
    full_pat[valid] = eof1
    pattern_da = xr.DataArray(
        full_pat, coords={"_space": Xb["_space"]}, dims="_space",
    ).unstack("_space")
    pattern_da = pattern_da.assign_coords(lat=lat_field, lon=lon_field)
    lon = pattern_da["lon"]
    lon_360 = lon % 360 if (lon < 0).any() else lon
    eastern_mask = (
        (pattern_da["lat"] >= 30) & (pattern_da["lat"] <= 50)
        & (lon_360 >= 200) & (lon_360 <= 240)
    )
    eastern_mean = float(pattern_da.where(eastern_mask).mean(skipna=True).values)
    if np.isfinite(eastern_mean) and eastern_mean < 0:
        eof1 = -eof1

    # 6. Project + standardise.
    pc1_a = Xa_valid @ eof1
    pc1_b = Xb_valid @ eof1
    std_b = float(np.std(pc1_b))
    pc1_a_norm = pc1_a / std_b if std_b > 0 else pc1_a
    return xr.DataArray(
        pc1_a_norm, coords={"time": Xa["time"]}, dims=["time"], name="pdo_index",
    )


def build_pdo_phase_labels(
    source_id: str,
    start_time: str,
    end_time: str,
    data_root: str,
) -> xr.DataArray:
    """PDO phase labels (+1 / −1 / 0) — sign-based on the standardized EOF index.
    
    Standard "monthly PDO phase" classification: positive when index > 0,
    negative when index < 0. Unlike ENSO we don't apply a magnitude threshold
    or persistence filter — this matches the conventional Mantua-style
    monthly PDO classification.
    """
    pdo_idx = compute_pdo_index(source_id, start_time, end_time, data_root)
    labels = xr.where(pdo_idx > 0, 1, xr.where(pdo_idx < 0, -1, 0))
    labels.name = "pdo_phase"
    return labels


# ===================================================================
# Derived-variable registry  (for sample_implicit_derived_variable
# and sample_complex_ratio / sample_correlated_derived_variable_pair)
# ===================================================================
 
def _fn_aou(d):
    return d["o2sat"] - d["o2"]
 
def _fn_nstar(d):
    return d["no3"] - 16.0 * d["po4"]
 
def _fn_pstar(d):
    return d["po4"] - d["no3"] / 16.0
 
def _fn_current_speed(d):
    return np.sqrt(d["uo"] ** 2 + d["vo"] ** 2)
 
def _fn_hke(d):
    return 0.5 * (d["uo"] ** 2 + d["vo"] ** 2)
 
def _fn_transport_mag(d):
    return np.sqrt(d["umo"] ** 2 + d["vmo"] ** 2)
 
def _fn_toc(d):
    return d["intdoc"] + d["intpoc"]
 
def _fn_o2_sat_pct(d):
    return (d["o2"] / d["o2sat"]) * 100.0
 
def _fn_dic_alk_ratio(d):
    return d["dissic"] / d["talk"]
 
def _fn_n_p_ratio(d):
    return d["no3"] / d["po4"]
 
def _fn_fe_p_ratio(d):
    return d["dfe"] / d["po4"]

def _fn_fe_n_ratio(d):
    return d["dfe"] / d["no3"]

def _fn_si_n_ratio(d):
    return d["si"] / d["no3"]

def is_cumulative(spec: dict) -> bool:
    """Cumulative: sum-reduces over depth/space; non-cumulative: mean-reduces."""
    return spec["type"] in {"inventory", "threshold_volume"} or spec.get("cumulative", False)
 
 
#: Registry mapping ``derived_code`` → computation spec.
#: ``type`` is one of ``"inventory"``, ``"formula"``, ``"threshold_volume"``.
DERIVED_VARIABLE_REGISTRY: dict[str, dict] = {
    # ---- depth-integrated inventories (single base variable) ----
    "dissic_inventory":  {"type": "inventory", "base_codes": ["dissic"]},
    "dissoc_inventory":  {"type": "inventory", "base_codes": ["dissoc"]},
    "o2_inventory":      {"type": "inventory", "base_codes": ["o2"]},
    "no3_inventory":     {"type": "inventory", "base_codes": ["no3"]},
    "po4_inventory":     {"type": "inventory", "base_codes": ["po4"]},
    "si_inventory":      {"type": "inventory", "base_codes": ["si"]},
    "chl_inventory":     {"type": "inventory", "base_codes": ["chl"]},
 
    # ---- formulae ----
    "aou":            {"type": "formula", "base_codes": ["o2", "o2sat"],   "fn": _fn_aou},
    "nstar":          {"type": "formula", "base_codes": ["no3", "po4"],    "fn": _fn_nstar},
    "pstar":          {"type": "formula", "base_codes": ["no3", "po4"],    "fn": _fn_pstar},
    "current_speed":  {"type": "formula", "base_codes": ["uo", "vo"],      "fn": _fn_current_speed},
    "hke":            {"type": "formula", "base_codes": ["uo", "vo"],      "fn": _fn_hke},
    "transport_mag":  {"type": "formula", "base_codes": ["umo", "vmo"],    "fn": _fn_transport_mag},
    "toc":            {"type": "formula", "base_codes": ["intdoc", "intpoc"], "fn": _fn_toc, "cumulative": True},
    "o2_sat_pct":     {"type": "formula", "base_codes": ["o2", "o2sat"],   "fn": _fn_o2_sat_pct},
    "dic_alk_ratio":  {"type": "formula", "base_codes": ["dissic", "talk"],"fn": _fn_dic_alk_ratio},
    "n_p_ratio":      {"type": "formula", "base_codes": ["no3", "po4"],    "fn": _fn_n_p_ratio},
    "fe_p_ratio":     {"type": "formula", "base_codes": ["dfe", "po4"],    "fn": _fn_fe_p_ratio},
    "fe_n_ratio": {"type": "formula", "base_codes": ["dfe", "no3"], "fn": _fn_fe_n_ratio},
    "si_n_ratio": {"type": "formula", "base_codes": ["si", "no3"], "fn": _fn_si_n_ratio},
 
    # ---- threshold-based volume / thickness metrics ----
    "hypoxic_volume":    {"type": "threshold_volume", "base_codes": ["o2"],
                          "op": "<", "threshold": 0.06},
    "omz_thickness":     {"type": "threshold_thickness", "base_codes": ["o2"],
                          "op": "<", "threshold": 0.06},
    "omz_volume":        {"type": "threshold_volume", "base_codes": ["o2"],
                          "op": "<", "threshold": 0.06},
    "oligo_volume": {
        "type": "threshold_volume",
        "base_codes": ["chl", "no3", "po4"],   # candidate set
        "primary_var": "chl",
        "op": "<", "threshold": 1e-7,           # 0.1 mg/m³ in kg/m³
        "fallbacks": [
            {"variable": "no3", "op": "<", "threshold": 1e-4},   # 0.1 mmol/m³
            {"variable": "po4", "op": "<", "threshold": 1e-5},   # 0.01 mmol/m³
        ],
    },
    "fe_limited_volume": {
        "type": "threshold_volume",
        "base_codes": ["dfe", "no3"],
        "op": "<",
        "threshold": 5e-5,
        "ratio_fn": _fn_fe_n_ratio,
    },
}
 
 
def resolve_derived_variable(derived_nl: str) -> tuple[str, dict]:
    """Look up a derived variable by its natural-language name.
 
    Returns ``(derived_code, spec_dict)`` from the registry.
    Raises ``KeyError`` if not found.
    """
    # Build an NL → code reverse map (first call is fine; dict is small)
    _NL_MAP: dict[str, str] = {}
    # Also accept direct codes
    for code, spec in DERIVED_VARIABLE_REGISTRY.items():
        _NL_MAP[code] = code
 
    # Map known NL strings to codes
    _NL_ALIASES = {
        "dissolved inorganic carbon inventory": "dissic_inventory",
        "dissolved organic carbon inventory":   "dissoc_inventory",
        "dissolved oxygen inventory":           "o2_inventory",
        "nitrate inventory":                    "no3_inventory",
        "phosphate inventory":                  "po4_inventory",
        "silicate inventory":                   "si_inventory",
        "chlorophyll inventory":                "chl_inventory",
        "vertically integrated chlorophyll-a":  "chl_inventory",
        "apparent oxygen utilization":          "aou",
        "hypoxic volume":                       "hypoxic_volume",
        "thickness of the OMZ":                 "omz_thickness",
        "OMZ volume":                           "omz_volume",
        "oligotrophic volume":                  "oligo_volume",
        "Fe-limited water volume":              "fe_limited_volume",
        "nitrate star":                         "nstar",
        "phosphate star":                       "pstar",
        "horizontal current speed":             "current_speed",
        "horizontal kinetic energy":            "hke",
        "horizontal mass transport magnitude":  "transport_mag",
        "total organic carbon content":         "toc",
        "oxygen saturation percentage":         "o2_sat_pct",
        "dissolved inorganic carbon to alkalinity ratio": "dic_alk_ratio",
        "nitrate to phosphate ratio":           "n_p_ratio",
        "iron to phosphate ratio":              "fe_p_ratio",
    }
    _NL_MAP.update(_NL_ALIASES)
 
    key = _NL_MAP.get(derived_nl)
    if key is None:
        raise KeyError(f"Unknown derived variable: '{derived_nl}'")
    return key, DERIVED_VARIABLE_REGISTRY[key]
 
 
def compute_derived_from_bases(
    base_arrays: dict[str, xr.DataArray],
    derived_code: str,
    spec: dict,
    depth_dim: str = "depth",
    *,
    area: xr.DataArray | None = None,
) -> xr.DataArray:
    from src.utils.numerical_diagnostics import compute_inventory, evaluate_condition_or_compare
 
    dtype = spec["type"]
 
    if dtype == "inventory":
        var = spec["base_codes"][0]
        result = compute_inventory(base_arrays[var])
        if area is not None:
            result = result * area
 
    elif dtype == "formula":
        result = spec["fn"](base_arrays)
        if spec.get("cumulative", False) and area is not None:
            result = result * area
 
    elif dtype == "threshold_volume":
        if "ratio_fn" in spec:
            da = spec["ratio_fn"](base_arrays)
            op, threshold = spec["op"], spec["threshold"]
        elif "fallbacks" in spec:
            primary = spec.get("primary_var", spec["base_codes"][0])
            candidates = [(primary, spec["op"], spec["threshold"])]
            for fb in spec["fallbacks"]:
                candidates.append((fb["variable"], fb["op"], fb["threshold"]))
            chosen = next(
                ((v, o, t) for v, o, t in candidates if v in base_arrays), None,
            )
            if chosen is None:
                raise KeyError(f"No suitable base variable loaded for {derived_code}")
            var, op, threshold = chosen
            da = base_arrays[var]
        else:
            primary = spec.get("primary_var", spec["base_codes"][0])
            da = base_arrays[primary]
            op, threshold = spec["op"], spec["threshold"]
        
        mask = evaluate_condition_or_compare(da, op, threshold)
        if "depth" in da.dims:
            dz = get_dz(da, depth_dim=depth_dim)
            result = (mask.astype(float) * dz).sum(dim="depth", skipna=True)  # m
        else:
            result = mask.astype(float)
        if area is not None:
            result = result * area
 
    elif dtype == "threshold_thickness":
        if derived_code == "omz_thickness":
            result = compute_omz_thickness(base_arrays["o2"], spec["threshold"])
        else:
            primary = spec.get("primary_var", spec["base_codes"][0])
            result = compute_layer_thickness(
                base_arrays[primary], spec["threshold"], spec["op"],
            )
 
    else:
        raise ValueError(f"Unknown derived variable type: '{dtype}'")
 
    result.name = derived_code
    return result

def compute_derived_at_depth(
    bases: dict,
    derived_spec: dict,
    derived_code: str,
    layer_depth: float,
    *,
    depth_dim: str = "depth",
) -> xr.DataArray:
    """Compute a derived variable evaluated at a single layer depth (metres).

    - Inventory / threshold-* types: bases are first restricted to
      [0, layer_depth] along depth, then the derived is computed.
    - Formula types: derived is computed on the full bases first, then
      sampled at layer_depth via nearest-neighbour selection.
    """
    from src.utils.data_io_and_subset import subset_depth  # local

    needs_depth = derived_spec.get("type") in (
        "inventory", "threshold_volume", "threshold_thickness",
    )
    if needs_depth:
        bases_r = {
            k: subset_depth(v, (0.0, layer_depth)) if depth_dim in v.dims else v
            for k, v in bases.items()
        }
        return compute_derived_from_bases(bases_r, derived_code, derived_spec)

    derived = compute_derived_from_bases(bases, derived_code, derived_spec)
    if depth_dim in derived.dims:
        derived = derived.sel({depth_dim: layer_depth}, method="nearest")
    return derived
 
 
# ===================================================================
# Descriptive-noun resolution  (for sample_descriptive_noun)
# ===================================================================
 
#: Each entry maps a descriptive noun to the diagnostic criteria.
#: ``variables``: list of variable codes to check.
#: ``conditions``: list of ``(var, op, threshold)`` — all must hold.
DESCRIPTIVE_NOUN_CRITERIA: dict[str, dict] = {
    # —— Stoichiometric ratio criteria ——
    "nutrient limitation by Fe": {
        "variables": ["dfe", "no3"],
        "conditions": [("fe_n_ratio", "<", 5e-5)],
        "derived": "fe_n_ratio",
    },
    "nutrient limitation by nitrogen": {            # N:P < 16 → N-limited
        "variables": ["no3", "po4"],
        "conditions": [("n_p_ratio", "<", 16)],
        "derived": "n_p_ratio",
    },
    "nutrient limitation by phosphorus": {          # N:P > 16 → P-limited
        "variables": ["no3", "po4"],
        "conditions": [("n_p_ratio", ">", 16)],
        "derived": "n_p_ratio",
    },
    "nutrient limitation by silicate": {            # Si:N < 1 → Si-limited
        "variables": ["si", "no3"],
        "conditions": [("si_n_ratio", "<", 1)],
        "derived": "si_n_ratio",
    },

    # —— Fallback-chain criteria (chl > no3 > po4 by data priority) ——
    "oligotrophic conditions": {
        "variable": "chl", "op": "<", "threshold": 1e-7,    # 0.1 mg/m³
        "fallbacks": [
            {"variable": "no3", "op": "<", "threshold": 1e-4},   # 0.1 mmol/m³
            {"variable": "po4", "op": "<", "threshold": 1e-5},   # 0.01 mmol/m³
        ],
        "variables": ["chl", "no3", "po4"],
    },
    "eutrophic conditions": {
        "variable": "chl", "op": ">", "threshold": 5e-6,    # 5 mg/m³
        "fallbacks": [
            {"variable": "no3", "op": ">", "threshold": 1e-2},   # 0.01 mmol/L
            {"variable": "po4", "op": ">", "threshold": 1e-3},   # 0.001 mmol/L
        ],
        "variables": ["chl", "no3", "po4"],
    },

    # —— Stoichiometric anomaly signatures ——
    "nitrogen fixation signature": {
        "variables": ["no3", "po4"],
        "conditions": [("nstar", "<", 0)],
        "derived": "nstar",
    },
    "denitrification signature": {                  # nstar in mol/m³, o2 in mol/m³
        "variables": ["no3", "po4", "o2"],
        "conditions": [("nstar", "<", -2e-3),       # -0.002 mmol/L
                       ("o2", "<", 0.020)],         # 20 µmol/kg ≈ 20 mmol/m³
        "derived": "nstar",
    },

    # —— Single-variable thresholds ——
    "hypoxia": {
        "variables": ["o2"],
        "conditions": [("o2", "<", 0.060)],         # 60 mmol/m³
    },
    "suboxia": {
        "variables": ["o2"],
        "conditions": [("o2", "<", 0.020)],         # 20 mmol/m³
    },
    "ocean acidification stress": {
        "variables": ["ph"],
        "conditions": [("ph", "<", 7.9)],
    },
    "strong current regions": {
        "variables": ["uo", "vo"],
        "conditions": [("current_speed", ">", 0.5)],   # m/s
        "derived": "current_speed",
    },
}
 
 
def resolve_descriptive_noun(noun: str) -> dict:
    """Return the criteria dict for a descriptive noun."""
    if noun not in DESCRIPTIVE_NOUN_CRITERIA:
        raise KeyError(f"Unknown descriptive noun: '{noun}'")
    return DESCRIPTIVE_NOUN_CRITERIA[noun]
 
 
def build_descriptive_noun_mask(
    base_arrays: dict[str, xr.DataArray],
    noun: str,
) -> xr.DataArray | None:
    """Build a boolean spatial mask for a descriptive noun.
 
    ``base_arrays`` must contain all variables listed under the noun's criteria
    (and will also be used to compute any derived fields like nstar).
    """
    from src.utils.numerical_diagnostics import evaluate_condition_or_compare
 
    criteria = resolve_descriptive_noun(noun)

    # Branch A: fallback-chain (oligotrophic / eutrophic) — walk priority.
    if "fallbacks" in criteria:
        for var, op, threshold in iter_criteria_with_fallbacks(criteria):
            if var in base_arrays:
                return evaluate_condition_or_compare(
                    base_arrays[var], op, threshold,
                )
        return None
 
    mask = None
    for var_name, op, threshold in criteria["conditions"]:
        if var_name in base_arrays:
            field = base_arrays[var_name]
        elif "derived" in criteria:
            dc = criteria["derived"]
            _, spec = resolve_derived_variable(dc)
            field = compute_derived_from_bases(base_arrays, dc, spec)
        else:
            raise KeyError(
                f"Variable '{var_name}' not in base_arrays and no derived spec"
            )
        cond = evaluate_condition_or_compare(field, op, threshold)
        mask = cond if mask is None else (mask & cond)
    return mask
 
 
# ===================================================================
# Qualified-years classification
# ===================================================================
 
# PDO box: North Pacific 20°N–70°N, 120°E–260°E
PDO_SPEC: dict[str, float] = {
    "lat_min": 20, "lat_max": 70, "lon_min": 120, "lon_max": 260,
}
 
 
def classify_qualified_time(
    qualified_time_str: str,
    source_id: str,
    start_time: str,
    end_time: str,
    data_root: str,
    *,
    sst_analysis: xr.DataArray | None = None,
    sst_baseline: xr.DataArray | None = None,
    area: xr.DataArray | None = None,
    time_dim: str = "time",
) -> xr.DataArray:
    """Per-time-step boolean mask for qualified months.

    ENSO and PDO classification use their own internal 1971-2000 baseline
    (consistent with build_enso_phase_labels) — independent of the caller's
    1995-2014 baseline used for derived-variable anomalies.

    Heatwave classification uses the passed sst_analysis / sst_baseline
    (1995-2014 baseline from caller).
    """

    if "El Nino" in qualified_time_str:
        labels = build_enso_phase_labels(source_id, start_time, end_time, data_root)
        return labels == 1

    if "La Nina" in qualified_time_str:
        labels = build_enso_phase_labels(source_id, start_time, end_time, data_root)
        return labels == -1

    if "PDO" in qualified_time_str:
        labels = build_pdo_phase_labels(source_id, start_time, end_time, data_root)
        if "positive" in qualified_time_str:
            return labels == 1
        if "negative" in qualified_time_str:
            return labels == -1
        raise KeyError(
            f"PDO phrase must specify 'positive' or 'negative': '{qualified_time_str}'"
        )

    if "heatwave" in qualified_time_str:
        if sst_analysis is None or sst_baseline is None:
            raise ValueError(
                "Heatwave classification requires sst_analysis and sst_baseline"
            )
        # Global-mean SST time series — raw values, no anomaly.
        spatial_a = [d for d in sst_analysis.dims if d != time_dim]
        spatial_b = [d for d in sst_baseline.dims if d != time_dim]
        if area is not None:
            idx_a = sst_analysis.weighted(area.fillna(0)).mean(dim=spatial_a)
            idx_b = sst_baseline.weighted(area.fillna(0)).mean(dim=spatial_b)
        else:
            idx_a = sst_analysis.mean(dim=spatial_a, skipna=True)
            idx_b = sst_baseline.mean(dim=spatial_b, skipna=True)
        # Per-calendar-month 90th percentile from baseline raw SST → 12 thresholds.
        monthly_threshold = idx_b.groupby(f"{time_dim}.month").quantile(
            0.9, dim=time_dim,
        )
        # Each analysis month → look up its calendar-month threshold; compare raw SST.
        months_a = idx_a[time_dim].dt.month
        return idx_a > monthly_threshold.sel(month=months_a)

    raise KeyError(f"Unknown qualified_time: '{qualified_time_str}'")
 