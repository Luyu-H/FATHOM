import random


# Basic sampling for level 1
def sample_statistical_operator():
    operator_dict = {
        "mean": "mean",
        "median": "median",
        "std": "standard deviation",
        "var": "variance",
        "min": "minimum",
        "max": "maximum",
        "range": "difference between maximum and minimum",

        "5_%": "5th percentile",
        "10_%": "10th percentile",
        "25_%": "25th percentile",
        "50_%": "50th percentile",
        "75_%": "75th percentile",
        "90_%": "90th percentile",
        "95_%": "95th percentile",
    }

    op_code = random.choice(list(operator_dict.keys()))
    op_nl = operator_dict[op_code]

    return op_code, op_nl

def sample_comparison_operator():
    # Is the {statistical_operator} of {variable_1} over the {region_1} during {time_period_1} {comparison_operator} the {statistical_operator} of {variable_2} over the {region_2} during {time_period_2}?
    comparison_operator_dict = {
        ">": "greater than",
        "<": "less than"
    }
    op_code = random.choice(list(comparison_operator_dict.keys()))
    op_nl = comparison_operator_dict[op_code]
    return op_code, op_nl

def sample_variable():
    # sample a variable from the following list and return its natural language description
    variable_dict = {
        # Temperature and salinity
        "thetao": "sea water potential temperature",
        "so": "sea water salinity",

        # Currents
        "uo": "Sea Water X Velocity",
        "vo": "Sea Water Y Velocity",
        "wo": "Sea Water Vertical Velocity",

        "umo": "Ocean Mass X Transport",
        "vmo": "Ocean Mass Y Transport",
        "wmo": "Upward Ocean Mass Transport",

        # Chemistry
        "dissic": "Dissolved Inorganic Carbon Concentration",
        "intdic": "Dissolved Inorganic Carbon Content",
        "dissoc": "Dissolved Organic Carbon Concentration",
        "intdoc": "Dissolved Organic Carbon Content",
        "intpoc": "Particulate Organic Carbon Content",

        "talk": "total alkalinity",
        "ph": "seawater pH",

        "no3": "Dissolved Nitrate Concentration",
        "po4": "Total Dissolved Inorganic Phosphorus Concentration",
        "si": "Total Dissolved Inorganic Silicon Concentration",
        "dfe": "dissolved iron concentration",

        # Oxygen
        "o2": "dissolved oxygen concentration",
        "o2sat": "Dissolved Oxygen Concentration at Saturation",

        # Biology
        "chl": "Mass Concentration of Total Phytoplankton Expressed as Chlorophyll in Sea Water",
        "phycos": "Sea Surface Phytoplankton Carbon Concentration"
    }

    var_code = random.choice(list(variable_dict.keys()))
    var_nl = variable_dict[var_code]

    return var_code, var_nl

def sample_variable_with_depth():
    """Sample a variable that is guaranteed to have a depth dimension."""
    variables_with_depth = {
        "thetao", "so", "uo", "vo", "wo",
        "dissic", "dissoc", "talk", "ph", 
        "no3", "po4", "si", "dfe", "o2", 
        "o2sat", "umo", "wmo", "vmo",
        
    }
    while True:
        var_code, var_nl = sample_variable()
        if var_code in variables_with_depth:
            return var_code, var_nl

def sample_base_and_derived_variable():
    """
    Sample a (base_variable, derived_variable) pair.
    Returns: (base_var_code, base_var_nl, derived_var_code, derived_var_nl)

    Applied to the question template: "What is the {statistical_operator} of global ocean {derived_variable} from {base_variable} during {time_period}?"
    """
    # Each entry: (base_var_nl, [list of possible derived_var_nl])
    pairs = [
        # Depth-integrated inventories
        {
            "base_code": "o2",
            "base_nl": "dissolved oxygen concentration",
            "derived_code": "o2_inventory",
            "derived_nl": ["dissolved oxygen inventory"],
        },
        {
            "base_code": "dissic",
            "base_nl": "dissolved inorganic carbon concentration",
            "derived_code": "dissic_inventory",
            "derived_nl": ["dissolved inorganic carbon inventory"],
        },
        {
            "base_code": "dissoc",
            "base_nl": "dissolved organic carbon concentration",
            "derived_code": "dissoc_inventory",
            "derived_nl": ["dissolved organic carbon inventory"],
        },
        {
            "base_code": "no3",
            "base_nl": "nitrate concentration",
            "derived_code": "no3_inventory",
            "derived_nl": ["nitrate inventory"],
        },
        {
            "base_code": "po4",
            "base_nl": "phosphate concentration",
            "derived_code": "po4_inventory",
            "derived_nl": ["phosphate inventory"],
        },
        {
            "base_code": "si",
            "base_nl": "silicate concentration",
            "derived_code": "si_inventory",
            "derived_nl": ["silicate inventory"],
        },
        {
            "base_code": "chl",
            "base_nl": "chlorophyll-a concentration",
            "derived_code": "chl_inventory",
            "derived_nl": ["vertically integrated chlorophyll-a", "chlorophyll inventory"],
        }
    ]

    entry = random.choice(pairs)
    derived_nl = random.choice(entry["derived_nl"])

    return {
        "base_var_code": entry["base_code"],
        "base_var_nl": entry["base_nl"],
        "derived_var_code": entry["derived_code"],
        "derived_var_nl": derived_nl,
    }

def sample_two_variables_for_comparison():
    """
    Sample two variables (var1, var2) that are scientifically meaningful to compare.
    Covers both same-variable comparisons (across regions/times) and cross-variable comparisons with physical/biogeochemical relevance.
    Returns dict with var1_code, var1_nl, var2_code, var2_nl.
    """
    # Is the {statistical_operator} of {variable_1} over the {region_1} during {time_period_1} {comparison_operator} the {statistical_operator} of {variable_2} over the {region_2} during {time_period_2}?
    # What is the {statistical_operator} difference between sea surface {variable_1} and sea floor {variable_2} over the {region} during {time_period}?

    comparable_pairs = [
        # Horizontal velocity components (m/s)
        {
            "var1_code": "uo",
            "var1_nl": "eastward ocean velocity",
            "var2_code": "vo",
            "var2_nl": "northward ocean velocity",
        },
        # Carbonate system (mol/m³)
        {
            "var1_code": "dissic",
            "var1_nl": "dissolved inorganic carbon",
            "var2_code": "talk",
            "var2_nl": "total alkalinity",
        },
        # Macro-nutrients (mol/m³)
        {
            "var1_code": "no3",
            "var1_nl": "nitrate concentration",
            "var2_code": "si",
            "var2_nl": "silicate concentration",
        },
        {
            "var1_code": "no3",
            "var1_nl": "nitrate concentration",
            "var2_code": "po4",
            "var2_nl": "phosphate concentration",
        },
        {
            "var1_code": "si",
            "var1_nl": "silicate concentration",
            "var2_code": "po4",
            "var2_nl": "phosphate concentration",
        },
    ]

    use_same = random.random() < 0.4

    if use_same:
        var_code, var_nl = sample_variable_with_depth()
        return {
            "var1_code": var_code,
            "var1_nl": var_nl,
            "var2_code": var_code,
            "var2_nl": var_nl,
        }
    else:
        entry = random.choice(comparable_pairs)
        # randomly swap order so var1/var2 aren't always fixed
        if random.random() < 0.5:
            return entry
        else:
            return {
                "var1_code": entry["var2_code"],
                "var1_nl": entry["var2_nl"],
                "var2_code": entry["var1_code"],
                "var2_nl": entry["var1_nl"],
            }

def sample_two_variables_for_correlation():
    """
    Sample two distinct variables with scientifically established correlation.
    Returns dict with var1_code, var1_nl, var2_code, var2_nl.
    """
    # What is the Pearson correlation coefficient between annual area-weighted mean {variable_1} and annual area-weighted mean {variable_2} over the {region} during {time_period}?

    correlated_pairs = [
        # Temperature-driven
        {"var1_code": "thetao", "var1_nl": "sea water potential temperature",
         "var2_code": "o2",     "var2_nl": "dissolved oxygen concentration"},
        {"var1_code": "thetao", "var1_nl": "sea water potential temperature",
         "var2_code": "ph",     "var2_nl": "seawater pH"},
        {"var1_code": "thetao", "var1_nl": "sea water potential temperature",
         "var2_code": "so",     "var2_nl": "sea water salinity"},
        {"var1_code": "thetao", "var1_nl": "sea water potential temperature",
         "var2_code": "chl",    "var2_nl": "chlorophyll-a concentration"},
        {"var1_code": "thetao", "var1_nl": "sea water potential temperature",
         "var2_code": "dissic", "var2_nl": "dissolved inorganic carbon"},

        # Biological pump productivity
        {"var1_code": "chl",    "var1_nl": "chlorophyll-a concentration",
         "var2_code": "o2",     "var2_nl": "dissolved oxygen concentration"},
        {"var1_code": "chl",    "var1_nl": "chlorophyll-a concentration",
         "var2_code": "no3",    "var2_nl": "nitrate concentration"},
        {"var1_code": "chl",    "var1_nl": "chlorophyll-a concentration",
         "var2_code": "po4",    "var2_nl": "phosphate concentration"},
        {"var1_code": "chl",    "var1_nl": "chlorophyll-a concentration",
         "var2_code": "dfe",    "var2_nl": "dissolved iron concentration"},
        {"var1_code": "chl",    "var1_nl": "chlorophyll-a concentration",
         "var2_code": "phycos", "var2_nl": "phytoplankton carbon concentration"},
        {"var1_code": "chl",    "var1_nl": "chlorophyll-a concentration",
         "var2_code": "dissic", "var2_nl": "dissolved inorganic carbon"},
        {"var1_code": "phycos", "var1_nl": "phytoplankton carbon concentration",
         "var2_code": "o2",     "var2_nl": "dissolved oxygen concentration"},
        {"var1_code": "phycos", "var1_nl": "phytoplankton carbon concentration",
         "var2_code": "no3",    "var2_nl": "nitrate concentration"},
        {"var1_code": "phycos", "var1_nl": "phytoplankton carbon concentration",
         "var2_code": "dissic", "var2_nl": "dissolved inorganic carbon"},

        # Ocean acidification
        {"var1_code": "dissic", "var1_nl": "dissolved inorganic carbon",
         "var2_code": "ph",     "var2_nl": "seawater pH"},
        {"var1_code": "dissic", "var1_nl": "dissolved inorganic carbon",
         "var2_code": "talk",   "var2_nl": "total alkalinity"},
        {"var1_code": "talk",   "var1_nl": "total alkalinity",
         "var2_code": "ph",     "var2_nl": "seawater pH"},
        {"var1_code": "talk",   "var1_nl": "total alkalinity",
         "var2_code": "so",     "var2_nl": "sea water salinity"},

        # Nutrient stoichiometry (Redfield)
        {"var1_code": "no3",    "var1_nl": "nitrate concentration",
         "var2_code": "po4",    "var2_nl": "phosphate concentration"},
        {"var1_code": "no3",    "var1_nl": "nitrate concentration",
         "var2_code": "si",     "var2_nl": "silicate concentration"},
        {"var1_code": "no3",    "var1_nl": "nitrate concentration",
         "var2_code": "o2",     "var2_nl": "dissolved oxygen concentration"},
        {"var1_code": "po4",    "var1_nl": "phosphate concentration",
         "var2_code": "o2",     "var2_nl": "dissolved oxygen concentration"},
        {"var1_code": "po4",    "var1_nl": "phosphate concentration",
         "var2_code": "dissic", "var2_nl": "dissolved inorganic carbon"},
        {"var1_code": "dfe",    "var1_nl": "dissolved iron concentration",
         "var2_code": "no3",    "var2_nl": "nitrate concentration"},
        {"var1_code": "dfe",    "var1_nl": "dissolved iron concentration",
         "var2_code": "chl",    "var2_nl": "chlorophyll-a concentration"},

        # Circulation
        {"var1_code": "uo",     "var1_nl": "eastward ocean velocity",
         "var2_code": "vo",     "var2_nl": "northward ocean velocity"},
        {"var1_code": "wo",     "var1_nl": "upward water velocity",
         "var2_code": "no3",    "var2_nl": "nitrate concentration"},
        {"var1_code": "wo",     "var1_nl": "upward water velocity",
         "var2_code": "o2",     "var2_nl": "dissolved oxygen concentration"},
        {"var1_code": "wo",     "var1_nl": "upward water velocity",
         "var2_code": "chl",    "var2_nl": "chlorophyll-a concentration"},
        {"var1_code": "wo",     "var1_nl": "upward water velocity",
         "var2_code": "thetao", "var2_nl": "sea water potential temperature"},

        # Carbon cycle
        {"var1_code": "dissoc", "var1_nl": "dissolved organic carbon",
         "var2_code": "chl",    "var2_nl": "chlorophyll-a concentration"},
        {"var1_code": "dissoc", "var1_nl": "dissolved organic carbon",
         "var2_code": "dissic", "var2_nl": "dissolved inorganic carbon"},
        {"var1_code": "dissoc", "var1_nl": "dissolved organic carbon",
         "var2_code": "o2",     "var2_nl": "dissolved oxygen concentration"},
    ]

    entry = random.choice(correlated_pairs)
    if random.random() < 0.5:
        return entry
    else:
        return {
            "var1_code": entry["var2_code"],
            "var1_nl": entry["var2_nl"],
            "var2_code": entry["var1_code"],
            "var2_nl": entry["var1_nl"],
        }

def sample_depth_range(dep_bnds=None, min_depth=0, max_depth=5500):
    if dep_bnds is not None:
        valid_depths = sorted([float(d) for d in dep_bnds if min_depth <= d <= max_depth])

        if len(valid_depths) >= 2:
            lower = random.choice(valid_depths[:-1])  # exclude last so upper > lower
            upper = random.choice([d for d in valid_depths if d > lower])
            return lower, upper, f"{lower}-{upper} m"
        # fallback: not enough valid depths, ignore dep_bnds
    
    multiples_of_5 = list(range(min_depth, max_depth + 1, 5))
    lower = random.choice(multiples_of_5[:-1])
    upper = random.choice([d for d in multiples_of_5 if d > lower])
    return lower, upper, f"{lower}-{upper} m"

def sample_depth(min_depth=0, max_depth=5500):
    # Sample a single depth value as a multiple of 5 within the given range
    multiples_of_5 = list(range(min_depth, max_depth + 1, 5))
    depth = random.choice(multiples_of_5)
    return depth, f"{depth} m"

def sample_region():
    """
    Sample an ocean region, returned as either:
        - A plain coordinate string: "lat1°S/N–lat2°S/N, lon1°E/W–lon2°E/W"
        - A named region with coordinates: "{Region Name} ({lat range}, {lon range})"
    """
    def fmt_lat(v):
        return f"{abs(v)}°N" if v >= 0 else f"{abs(v)}°S"

    def fmt_lon(v):
        return f"{abs(v)}°E" if v >= 0 else f"{abs(v)}°W"

    def fmt_box(lat1, lat2, lon1, lon2):
        return f"{fmt_lat(lat1)}–{fmt_lat(lat2)}, {fmt_lon(lon1)}–{fmt_lon(lon2)}"
    
    named_regions = {
        # Global / basin scale
        "Global Ocean": (-90, 90, 0, 360),
        "North Atlantic Ocean": (0, 65, 280, 360),
        "South Atlantic Ocean": (-60, 0, 290, 20),
        "North Pacific Ocean": (0, 65, 120, 240),  # crosses dateline
        "South Pacific Ocean": (-60, 0, 150, 290),
        "Indian Ocean": (-60, 30, 20, 120),
        "Arctic Ocean": (65, 90, 0, 360),
        "Southern Ocean": (-90, -40, 0, 360),

        # Marginal / semi-enclosed seas
        "Mediterranean Sea": (30, 46, 354, 37),
        "Black Sea": (41, 47, 28, 42),
        "Red Sea": (12, 30, 32, 44),
        "Persian Gulf": (23, 30, 48, 57),
        "Bay of Bengal": (5, 23, 80, 100),
        "Arabian Sea": (5, 25, 55, 78),
        "South China Sea": (0, 25, 105, 122),
        "East China Sea": (24, 33, 120, 130),
        "Sea of Japan": (32, 52, 127, 142),
        "Bering Sea": (52, 66, 163, 203),
        "Gulf of Mexico": (18, 31, 262, 280),
        "Caribbean Sea": (10, 23, 274, 300),
        "North Sea": (51, 61, 355, 9),
        "Baltic Sea": (53, 66, 9, 30),

        # Dynamically important sub-regions
        "Equatorial Pacific": (-10, 10, 140, 280),
        "Eastern Tropical Pacific": (-20, 20, 220, 280),
        "Western Pacific Warm Pool": (-10, 10, 120, 165),
        "Gulf Stream Region": (25, 45, 280, 330),
        "Labrador Sea": (50, 65, 295, 320),
        "Nordic Seas": (62, 80, 340, 30),
        "Weddell Sea": (-78, -55, 300, 30),
        "Ross Sea": (-78, -60, 160, 210),
        "Humboldt Current System": (-45, -10, 270, 290),
        "California Current System": ( 25,  50, 225, 245),
        "Equatorial Indian Ocean": (-10, 10, 50, 100),
        "Subtropical North Pacific Gyre": (20, 40, 140, 240),
        "Subtropical South Pacific Gyre": (-40, -20, 210, 280),
        "Atlantic Meridional Overturning": (30, 65, 280, 360),
    }

    use_named = random.random() < 0.7

    if use_named:
        name = random.choice(list(named_regions.keys()))
        lat1, lat2, lon1, lon2 = named_regions[name]
        range = {"lat1": lat1, "lat2": lat2, "lon1": lon1, "lon2": lon2}
        return range, f"{name} ({fmt_box(lat1, lat2, lon1, lon2)})"
    
    else:
        # Draw from the named regions pool and perturb ±5–15° to stay near ocean
        base_name = random.choice(list(named_regions.keys()))
        lat1, lat2, lon1, lon2 = named_regions[base_name]

        def perturb(v, lo, hi):
            delta = random.randint(-10, 10)
            return max(lo, min(hi, v + delta))

        lat1 = perturb(lat1, -90, 88)
        lat2 = perturb(lat2, lat1 + 2, 90)
        # Longitudes live in [0, 360]; perturb each endpoint with
        # wrap-around and always take the SHORTER arc on the lon
        # circle. subset_region handles lon1 > lon2 as a dateline-
        # crossing region, so a small perturbed crossing region stays
        # valid; we just refuse the > 180° "complement" arc that
        # independent perturbation can otherwise produce.
        lon1 = (lon1 + random.randint(-10, 10)) % 360
        lon2 = (lon2 + random.randint(-10, 10)) % 360
        if (lon2 - lon1) % 360 > 180:
            lon1, lon2 = lon2, lon1
        # Guarantee >= 2° width so curvilinear regrid never sees a
        # 1-cell-wide strip (which would collapse Delaunay input).
        if (lon2 - lon1) % 360 < 2:
            lon2 = (lon1 + 2) % 360
        bbox = {"lat1": lat1, "lat2": lat2, "lon1": lon1, "lon2": lon2}

        return bbox, fmt_box(lat1, lat2, lon1, lon2)

def sample_time_period(experiment, min_months=1, max_months=None):
    """
    Sample a time period within CMIP6 historical (1850-01 ~ 2014-12) or SSP scenario (2015-01 ~ 2100-12) bounds. No crossing between the two experiments.
    Returns a string in "YYYY-MM ~ YYYY-MM" format.
    """
    # Choose experiment first, then sample within its bounds

    if experiment == "historical":
        year_start, year_end = 1850, 2014
    elif experiment == "ssp245":
        year_start, year_end = 2015, 2100
    else:
        raise ValueError(f"Please specify the available time period for experiment {experiment}.")

    # Total months in the experiment
    total_months = (year_end - year_start) * 12 + 12   # inclusive of last month

    if min_months >= total_months:
        raise ValueError(f"min_months={min_months} exceeds experiment length ({total_months} months)")

    idx_start = random.randint(0, total_months - 1 - min_months)

    if max_months is None:
        idx_end_max = total_months - 1
    else:
        if max_months < min_months:
            raise ValueError("max_months must be >= min_months")
        idx_end_max = min(total_months - 1, idx_start + max_months - 1)

    idx_end = random.randint(idx_start + min_months, idx_end_max)

    def idx_to_yyyymm(idx):
        y = year_start + idx // 12
        m = idx % 12 + 1
        return f"{y}-{m:02d}"

    start_month = idx_to_yyyymm(idx_start)
    end_month = idx_to_yyyymm(idx_end)
    return start_month, end_month, f"{start_month} ~ {end_month}"

def sample_time_period_yr(experiment, min_years=1, max_years=None):
    """
    Returns a string in "YYYY ~ YYYY" format.
    """
    if experiment == "historical":
        year_start, year_end = 1850, 2014
    elif experiment == "ssp245":
        year_start, year_end = 2015, 2100
    else:
        raise ValueError(f"Please specify the available time period for experiment {experiment}.")

    total_years = year_end - year_start + 1

    if min_years >= total_years:
        raise ValueError(f"min_years={min_years} exceeds experiment length ({total_years} years)")

    # Sample start year index
    idx_start = random.randint(0, total_years - min_years)

    # Determine max end index
    if max_years is None:
        idx_end_max = total_years - 1
    else:
        if max_years < min_years:
            raise ValueError("max_years must be >= min_years")
        idx_end_max = min(total_years - 1, idx_start + max_years - 1)

    # Sample end year index
    idx_end = random.randint(idx_start + min_years - 1, idx_end_max)

    # Convert to actual years
    start_year = year_start + idx_start
    end_year = year_start + idx_end

    return f"{start_year}-01", f"{end_year}-12", f"{start_year} ~ {end_year}"

def sample_significance_level():
    levels = [0.1, 0.05, 0.01]
    return random.choice(levels)

def sample_threshold(variable):
    # Is the {statistical_operator} of {variable} over the {region} during {time_period} {comparison_operator} {threshold}?
    threshold_dict = {
        # Temperature and salinity
        "thetao": [
            "-2 °C", # near freezing / sea ice formation
            "0 °C", # freezing point
            "4 °C", # deep water mass reference
            "10 °C", # subpolar boundary
            "15 °C", # temperate threshold
            "20 °C", # subtropical boundary
            "25 °C", # tropical warm pool
            "28 °C", # coral bleaching risk
            "30 °C", # extreme SST
        ],
        "so": [
            "30 PSU", # brackish/marginal sea
            "33 PSU", # subpolar surface water
            "34 PSU", # North Atlantic Deep Water
            "34.5 PSU",
            "35 PSU", # global ocean mean
        ],

        # Currents
        "uo": [
            "-1.0 m/s", "-0.5 m/s", "-0.1 m/s",
            "0.1 m/s", "0.5 m/s", "1.0 m/s", "1.5 m/s",
        ],
        "vo": [
            "-1.0 m/s", "-0.5 m/s", "-0.1 m/s",
            "0.1 m/s", "0.5 m/s", "1.0 m/s", "1.5 m/s",
        ],
        "wo": [
            "-1e-4 m/s", # downwelling
            "-1e-5 m/s",
            "1e-5 m/s", # weak upwelling
            "1e-4 m/s", # moderate upwelling
            "5e-4 m/s", # strong coastal upwelling
        ],

        # Carbon chemistry
        "dissic": [
            "1.9 mol/m³",
            "2.0 mol/m³", # surface ocean
            "2.1 mol/m³",
            "2.2 mol/m³", # deep water
            "2.3 mol/m³",
            "2.4 mol/m³", # abyssal
        ],
        "dissoc": [
            "0.01 mol/m³",
            "0.05 mol/m³",
            "0.1 mol/m³", # surface productive zone
            "0.2 mol/m³",
        ],
        "talk": [
            "2.1 mol/m³",
            "2.2 mol/m³", # typical surface
            "2.3 mol/m³",
            "2.35 mol/m³", # deep ocean
            "2.4 mol/m³",
        ],
        "ph": [
            "7.8", # severe acidification scenario
            "7.9", # acidification threshold
            "8.0", # near-future projected mean
            "8.1", # current global mean
            "8.2", # pre-industrial reference
            "8.3",
        ],

        # Nutrients
        "no3": [
            "0.001 mol/m³", # oligotrophic surface
            "0.005 mol/m³",
            "0.01 mol/m³", # nutrient-limited boundary
            "0.1 mol/m³", # upwelling zone
            "0.5 mol/m³", # deep water
            "1.0 mol/m³", # abyssal / Antarctic
        ],
        "po4": [
            "1e-4 mol/m³", # oligotrophic surface
            "5e-4 mol/m³",
            "0.001 mol/m³", # Redfield ratio limit
            "0.01 mol/m³", # upwelling zone
            "0.1 mol/m³", # deep Pacific
            "0.15 mol/m³",
        ],
        "si": [
            "0.002 mol/m³", # diatom-limited surface
            "0.01 mol/m³",
            "0.05 mol/m³",
            "0.1 mol/m³", # upwelling / subpolar
            "0.5 mol/m³", # deep Pacific
            "1.0 mol/m³", # abyssal
        ],
        "dfe": [
            "2e-7 mol/m³", # iron-limited (HNLC region)
            "5e-7 mol/m³",
            "1e-6 mol/m³", # typical deep water
            "2e-6 mol/m³",
            "5e-6 mol/m³", # dust-enriched region
        ],

        # Oxygen
        "o2": [
            "0.01 mol/m³", # suboxic threshold
            "0.0625 mol/m³", # hypoxic threshold (~2 mg/L)
            "0.1 mol/m³",
            "0.2 mol/m³", # well-ventilated deep water
            "0.3 mol/m³", # near-surface
        ],
        "o2sat": [
            "0.15 mol/m3",   # warm tropical (low solubility)
            "0.2 mol/m3",
            "0.25 mol/m3",
            "0.3 mol/m3",    # temperate
            "0.35 mol/m3",   # cold polar (high solubility)
        ],

        # Biology
        "chl": [
            "5e-8 kg/m3",    # ultra-oligotrophic (≈0.05 mg/m³)
            "1e-7 kg/m3",    # oligotrophic
            "5e-7 kg/m3",    # mesotrophic
            "1e-6 kg/m3",    # eutrophic boundary
            "2e-6 kg/m3",    # coastal/upwelling
            "5e-6 kg/m3",    # bloom condition
        ],
        "phycos": [
            "0.001 mol/m³",
            "0.005 mol/m³",
            "0.01 mol/m³", # moderate biomass
            "0.05 mol/m³", # bloom
            "0.1 mol/m³", # high biomass event
        ],

        "umo": [
            "-1e9 kg/s", "-1e8 kg/s", "-1e7 kg/s",
            "1e7 kg/s", "1e8 kg/s", "1e9 kg/s",
        ],
        "vmo": [
            "-1e9 kg/s", "-1e8 kg/s", "-1e7 kg/s",
            "1e7 kg/s", "1e8 kg/s", "1e9 kg/s",
        ],
        "wmo": [
            "-1e7 kg/s", "-1e6 kg/s",
            "1e6 kg/s", "1e7 kg/s",
        ],
        "intdic": [
            "5 kg/m2", "10 kg/m2", "20 kg/m2", "50 kg/m2", "100 kg/m2",
        ],
        "intdoc": [
            "0.1 kg/m2", "0.5 kg/m2", "1.0 kg/m2", "2.0 kg/m2", "5.0 kg/m2",
        ],
        "intpoc": [
            "0.01 kg/m2", "0.05 kg/m2", "0.1 kg/m2", "0.5 kg/m2", "1.0 kg/m2",
        ],
    }

    thresholds = threshold_dict.get(variable)
    if thresholds is None:
        raise ValueError(f"Unknown variable: '{variable}'")

    return random.choice(thresholds)

def sample_verb_change_direction():
    return random.choice(["increased", "decreased"])


# Additional sampling for level 2
def sample_implicit_depth(surface=False):
    # vs sample_term_for_depth_layer
    implicit_depth_list = [
        "sea surface",
        "sea floor",
        "near surface",
        "subsurface",
        "upper ocean",
        "intermediate depth",
        "photic zone",
        "deep ocean",
    ]
    if surface:
        return random.choice(["sea surface", "near surface", "subsurface", "upper ocean", "photic zone"])
    else:
        return random.choice(implicit_depth_list)

def sample_clim_time_period():
    # What is the global mean {variable} anomaly in the {implicit_depth} during recent boreal winter for {time_period}, relative to the {clim_time_period} climatology?
    clim_periods = {
        # WMO standard normals
        "1961-1990": "WMO standard normal (1961–1990)",
        "1971-2000": "WMO standard normal (1971–2000)",
        "1981-2010": "WMO standard normal (1981–2010)",
        "1991-2014": "WMO standard normal (1991–2020)",

        # Pre-industrial / historical baselines
        "1850-1900": "pre-industrial baseline",
        "1850-1950": "early historical baseline",

        # Satellite era baselines
        "1979-2000": "satellite era early baseline",
        "1979-2014": "full satellite era (CMIP6 historical end)",
        "1982-2011": "OISST standard climatology",

        # CMIP6-aligned common choices
        "1950-2000": "mid-to-late 20th century baseline",
        "1961-2000": "late 20th century baseline",
        "1985-2014": "late CMIP6 historical period",
    }

    code = random.choice(list(clim_periods.keys()))
    if random.random() < 0.4:
        return code, code
    else:
        return code, clim_periods[code]

def sample_term_for_depth_layer():
    # vs sample_implicit_depth
    terms_for_depth_layer = [
        (["thetao", "so"], "mixed layer"),
        (["thetao"], "thermocline"),
    ]
    return random.choice(terms_for_depth_layer)

def sample_implicit_region():
    implicit_region_dict = {
        "global ocean": (-90, 90, 0, 360),

        "Tropical Pacific": (-20, 20, 220, 280),
        "Equatorial Pacific": (-10, 10, 140, 280),
        "Western Pacific Warm Pool": (-10, 10, 130, 180),
        "South Pacific Convergence Zone": (-25, 0, 150, 240),
        "North Pacific Subtropical Gyres": (10, 40, 120, 260),

        "North Atlantic Subtropical Gyres": (10, 40, 280, 360),
        "North Atlantic Drift region": (40, 60, 280, 360),
        "Subpolar North Atlantic": (45, 70, 290, 360),

        "Indian Ocean Basin": (-30, 30, 20, 120),
        "Indian Ocean Dipole region": (-20, 20, 40, 110),

        "Bay of Bengal": (5, 25, 80, 100),
        "Arabian Sea": (5, 25, 50, 75),
        "Labrador Sea": (50, 65, 280, 320),
        "Gulf of Mexico": (18, 31, 260, 290),
        "Caribbean Sea": (9, 23, 270, 290),
        "Mediterranean Sea": (30, 46, 350, 40),
        "Sea of Japan": (34, 52, 127, 142),
        "South China Sea": (3, 24, 105, 121),
        "Coral Sea": (-25, -8, 145, 165),

        "California Current System": (20, 45, 230, 245),
        "Canary Current System": (15, 35, 340, 360),
        "Benguela Upwelling System": (-35, -10, 5, 20),
        "Peru-Chile Current System": (-45, 0, 285, 300),
        "Kuroshio Extension": (25, 40, 140, 170),
        "Agulhas Current region": (-45, -20, 10, 40),

        "Southern Ocean": (-90, -40, 0, 360),
        "Arctic Ocean": (66, 90, 0, 360),
    }
    region = random.choice(list(implicit_region_dict.keys()))
    range = implicit_region_dict[region]
    return range, region

def sample_ratio():
    """
    Sample a scientifically meaningful ocean biogeochemical ratio.
    Returns dict with ratio_code, ratio_nl, var1_code, var1_nl, var2_code, var2_nl.

    Applied to the question template: "What is the {region} mean {ratio} ratio anomaly in the {implicit_region} during recent annual conditions relative to the {clim_time_period} climatology?"
    """

    ratio_pairs = [
        # Redfield
        {"ratio_code": "C:N", "ratio_nl": "dissolved inorganic carbon to nitrate ratio",
         "var1_code": "dissic", "var1_nl": "dissolved inorganic carbon",
         "var2_code": "no3", "var2_nl": "nitrate"},
        {"ratio_code": "C:P", "ratio_nl": "dissolved inorganic carbon to phosphate ratio",
         "var1_code": "dissic", "var1_nl": "dissolved inorganic carbon",
         "var2_code": "po4", "var2_nl": "phosphate"},
        {"ratio_code": "N:P", "ratio_nl": "nitrate to phosphate ratio",
         "var1_code": "no3", "var1_nl": "nitrate",
         "var2_code": "po4", "var2_nl": "phosphate"},
        {"ratio_code": "Si:N", "ratio_nl": "silicate to nitrate ratio",
         "var1_code": "si", "var1_nl": "silicate",
         "var2_code": "no3", "var2_nl": "nitrate"},
        {"ratio_code": "Si:P", "ratio_nl": "silicate to phosphate ratio",
         "var1_code": "si", "var1_nl": "silicate",
         "var2_code": "po4", "var2_nl": "phosphate"},
        {"ratio_code": "Fe:N", "ratio_nl": "dissolved iron to nitrate ratio",
         "var1_code": "dfe", "var1_nl": "dissolved iron",
         "var2_code": "no3", "var2_nl": "nitrate"},

        # Oxygen-based
        {"ratio_code": "O2:N", "ratio_nl": "dissolved oxygen to nitrate ratio",
         "var1_code": "o2", "var1_nl": "dissolved oxygen",
         "var2_code": "no3", "var2_nl": "nitrate"},
        {"ratio_code": "O2:P", "ratio_nl": "dissolved oxygen to phosphate ratio",
         "var1_code": "o2", "var1_nl": "dissolved oxygen",
         "var2_code": "po4", "var2_nl": "phosphate"},
        {"ratio_code": "O2:C", "ratio_nl": "dissolved oxygen to dissolved inorganic carbon ratio",
         "var1_code": "o2", "var1_nl": "dissolved oxygen",
         "var2_code": "dissic", "var2_nl": "dissolved inorganic carbon"},

        # Carbon system
        {"ratio_code": "DIC:TA", "ratio_nl": "dissolved inorganic carbon to total alkalinity ratio",
         "var1_code": "dissic", "var1_nl": "dissolved inorganic carbon",
         "var2_code": "talk", "var2_nl": "total alkalinity"},
        {"ratio_code": "DOC:DIC", "ratio_nl": "dissolved organic carbon to dissolved inorganic carbon ratio",
         "var1_code": "dissoc", "var1_nl": "dissolved organic carbon",
         "var2_code": "dissic", "var2_nl": "dissolved inorganic carbon"},
    ]

    entry = random.choice(ratio_pairs)
    return entry

def sample_implicit_comparison_operator():
    # Is the {statistical_operator} of {variable_1} over the {region_1} during {time_period_1} {comparison_operator} the {statistical_operator} of {variable_2} over the {region_2} during {time_period_2}?
    i_comparison_operator_dict = {
        ">": "significantly greater than",
        "<": "significantly less than",
        "=": "close to",
    }
    op_code = random.choice(list(i_comparison_operator_dict.keys()))
    op_nl = i_comparison_operator_dict[op_code]
    return op_code, op_nl

def sample_season():
    return random.choice(["DJF", "MAM", "JJA", "SON"])

def sample_term_for_thickness():
    terms_for_thickness = [
        "hypoxia", "oxygen minimum zone",
    ]
    return random.choice(terms_for_thickness)

def sample_implicit_time_period():
    implicit_time_dict = {
        "1995-2014": "recent period",
        "2081-2100": "end of century",
        "1850-1900": "pre-industrial period",

        "1901-1930": "early 20th century",
        "1931-1960": "mid-20th century",
        "1961-2000": "late 20th century",
        "2000-2014": "early 21st century",

        "1981-2010": "modern climatology period",
        "1850-1879": "late Little Ice Age period",
        "1880-1910": "industrial onset period",
        "1960-1989": "pre-modern warming period",

        "1986-2005": "CMIP5 reference period",
        "1991-2010": "WMO climatological normal",
    }
    time_code = random.choice(list(implicit_time_dict.keys()))
    time_nl = implicit_time_dict[time_code]
    return time_code, time_nl

def sample_enso_phase():
    return random.choice(["El Nino phase", "La Nina phase", "Neutral phase of ENSO"])

def sample_qualified_area():
    terms_for_qualified_area = {
        "hypoxia": "o2",
        "oxygen minimum zone": "o2",
        "oligotrophic area": "chl",   # only return the primary variable
    }
    term = random.choice(list(terms_for_qualified_area.keys()))
    vars = terms_for_qualified_area[term]
    return term, vars

def sample_descriptive_adj():
    # Is the {implicit_depth} in the {implicit_region} more {descriptive_adj} in the recent period than in the pre-industrial period?
    descriptive_adj_list = {
        "oligotrophic": "chl",            # only return the primary variable
        "hypoxic":      "o2",
    }
    adj = random.choice(list(descriptive_adj_list.keys()))
    var = descriptive_adj_list[adj]
    return adj, var


# Additional complex sampling for level 3
def sample_implicit_derived_variable():
    implicit_derived_variables = [
        # Depth-integrated inventories (single base variable)
        {"derived_code": "dissic_inventory", "derived_nl": "dissolved inorganic carbon inventory",
         "base_codes": ["dissic"]},
        {"derived_code": "dissoc_inventory", "derived_nl": "dissolved organic carbon inventory",
         "base_codes": ["dissoc"]},
        {"derived_code": "o2_inventory", "derived_nl": "dissolved oxygen inventory",
         "base_codes": ["o2"]},
        {"derived_code": "po4_inventory", "derived_nl": "phosphate inventory",
         "base_codes": ["po4"]},
        {"derived_code": "no3_inventory", "derived_nl": "nitrate inventory",
         "base_codes": ["no3"]},
        {"derived_code": "si_inventory", "derived_nl": "silicate inventory",
         "base_codes": ["si"]},
        {"derived_code": "chl_inventory", "derived_nl": "chlorophyll inventory",
         "base_codes": ["chl"]},

        # Oxygen-derived (two base variables)
        {"derived_code": "aou", "derived_nl": "apparent oxygen utilization",
         "base_codes": ["o2", "o2sat"]},

        # Threshold-based volume/thickness metrics
        {"derived_code": "hypoxic_volume", "derived_nl": "hypoxic volume",
         "base_codes": ["o2"]},
        {"derived_code": "omz_thickness", "derived_nl": "thickness of the OMZ",
         "base_codes": ["o2"]},
        {"derived_code": "omz_volume", "derived_nl": "OMZ volume",
         "base_codes": ["o2"]},
        {"derived_code": "oligo_volume", "derived_nl": "oligotrophic volume",
         "base_codes": ["chl"]},
        {"derived_code": "fe_limited_volume", "derived_nl": "Fe-limited water volume",
         "base_codes": ["dfe", "no3"]},
    ]
    return random.choice(implicit_derived_variables)

def sample_depth_integrated_inventory():
    inventory_list = [
        {"derived_code": "dissic_inventory", "derived_nl": "dissolved inorganic carbon inventory",
         "base_codes": ["dissic"]},
        {"derived_code": "dissoc_inventory", "derived_nl": "dissolved organic carbon inventory",
         "base_codes": ["dissoc"]},
        {"derived_code": "o2_inventory", "derived_nl": "dissolved oxygen inventory",
         "base_codes": ["o2"]},
        {"derived_code": "po4_inventory", "derived_nl": "phosphate inventory",
         "base_codes": ["po4"]},
        {"derived_code": "no3_inventory", "derived_nl": "nitrate inventory",
         "base_codes": ["no3"]},
        {"derived_code": "si_inventory", "derived_nl": "silicate inventory",
         "base_codes": ["si"]},
        {"derived_code": "chl_inventory", "derived_nl": "chlorophyll inventory",
         "base_codes": ["chl"]},
    ]
    return random.choice(inventory_list)

def sample_qualified_time():
    qualified_time_list = [
        "El Nino phase",
        "La Nina phase",
        "positive Pacific Decadal Oscillation (PDO) phase",
        "negative Pacific Decadal Oscillation (PDO) phase",
        "heatwave months",
    ]
    return random.choice(qualified_time_list)

def sample_complex_ratio():
    """
    Sample two derived variables for ratio comparison.
    Returns dict with var1_derived_code, var1_derived_nl, var1_base_codes,
                       var2_derived_code, var2_derived_nl, var2_base_codes.
    """
    derived_vars = {
        "aou": {
            "derived_code": "aou",
            "derived_nl": "apparent oxygen utilization",
            "base_codes": ["o2", "o2sat"],
        },
        "nstar": {
            "derived_code": "nstar",
            "derived_nl": "nitrate star",
            "base_codes": ["no3", "po4"],
        },
        "pstar": {
            "derived_code": "pstar",
            "derived_nl": "phosphate star",
            "base_codes": ["no3", "po4"],
        },
        "current_speed": {
            "derived_code": "current_speed",
            "derived_nl": "horizontal current speed",
            "base_codes": ["uo", "vo"],
        },
        "hke": {
            "derived_code": "hke",
            "derived_nl": "horizontal kinetic energy",
            "base_codes": ["uo", "vo"],
        },
        "transport_mag": {
            "derived_code": "transport_mag",
            "derived_nl": "horizontal mass transport magnitude",
            "base_codes": ["umo", "vmo"],
        },
        "o2_sat_pct": {
            "derived_code": "o2_sat_pct",
            "derived_nl": "oxygen saturation percentage",
            "base_codes": ["o2", "o2sat"],
        },
        "o2_inventory": {
            "derived_code": "o2_inventory",
            "derived_nl": "dissolved oxygen inventory",
            "base_codes": ["o2"],
        },
        "no3_inventory": {
            "derived_code": "no3_inventory",
            "derived_nl": "nitrate inventory",
            "base_codes": ["no3"],
        }
    }

    pairs = [
        ("aou", "nstar"),
        ("aou", "current_speed"),
        ("nstar", "pstar"),
        ("current_speed", "hke"),
        ("transport_mag", "current_speed"),
        ("hke", "aou"),
        ("pstar", "current_speed"),
        ("nstar", "o2_sat_pct"),
    ]

    key1, key2 = random.choice(pairs)
    v1 = derived_vars[key1]
    v2 = derived_vars[key2]

    return {
        "var1_derived_code": v1["derived_code"],
        "var1_derived_nl":   v1["derived_nl"],
        "var1_base_codes":   v1["base_codes"],
        "var2_derived_code": v2["derived_code"],
        "var2_derived_nl":   v2["derived_nl"],
        "var2_base_codes":   v2["base_codes"],
    }

def sample_descriptive_noun():
    desctiptive_noun_list = [
        "nutrient limitation by Fe", # dfe / no3 < 5e-5
        "nutrient limitation by nitrogen", # no3 / po4 < 16 
        "nutrient limitation by phosphorus", # no3 / po4 > 16
        "nutrient limitation by silicate", # si / no3 < 1
        "oligotrophic conditions", # fallback: chl > no3 > po4
        "eutrophic conditions", # fallback: chl (> 5 mg m-3) > no3 (> 0.01 mmol L-1) > po4 (> 0.001 mmol L-1)

        # Nitrate Star (no3 − 16×po4) < 0
        "nitrogen fixation signature",
        # Nitrate Star < -0.002 mmol L-1 and dissolved oxygen concentration < 20 µmol/kg
        "denitrification signature",

        "hypoxia", # < 60 mmol/m3
        "suboxia", # < 20 mmol/m3
        "ocean acidification stress", # ph < 7.9

        # magnitude (sqrt(uo^2 + vo^2)) > 0.5 m/s
        "strong current regions",
    ]
    return random.choice(desctiptive_noun_list)

def sample_correlated_derived_variable_pair():
    """
    Sample two derived variables with lag-correlation relevance.
    Returns dict with var1_derived_code, var1_derived_nl, var1_base_codes,
                       var2_derived_code, var2_derived_nl, var2_base_codes.
    """
    derived_vars = {
        "hke": {
            "derived_code": "hke",
            "derived_nl": "horizontal kinetic energy",
            "base_codes": ["uo", "vo"],
        },
        "current_speed": {
            "derived_code": "current_speed",
            "derived_nl": "horizontal current speed",
            "base_codes": ["uo", "vo"],
        },
        "aou": {
            "derived_code": "aou",
            "derived_nl": "apparent oxygen utilization",
            "base_codes": ["o2", "o2sat"],
        },
        "nstar": {
            "derived_code": "nstar",
            "derived_nl": "nitrate star",
            "base_codes": ["no3", "po4"],
        },
        "dic_alk_ratio": {
            "derived_code": "dic_alk_ratio",
            "derived_nl": "dissolved inorganic carbon to alkalinity ratio",
            "base_codes": ["dissic", "talk"],
        },
        "n_p_ratio": {
            "derived_code": "n_p_ratio",
            "derived_nl": "nitrate to phosphate ratio",
            "base_codes": ["no3", "po4"],
        },
        "o2_sat_pct": {
            "derived_code": "o2_sat_pct",
            "derived_nl": "oxygen saturation percentage",
            "base_codes": ["o2", "o2sat"],
        },
        "fe_p_ratio": {
            "derived_code": "fe_p_ratio",
            "derived_nl": "iron to phosphate ratio",
            "base_codes": ["dfe", "po4"],
        },
    }

    pairs = [
        ("hke", "aou"),
        ("current_speed", "nstar"),
        ("aou", "dic_alk_ratio"),
        ("nstar", "o2_sat_pct"),
        ("hke", "fe_p_ratio"),
        ("aou", "n_p_ratio"),
    ]

    key1, key2 = random.choice(pairs)
    v1 = derived_vars[key1]
    v2 = derived_vars[key2]

    return {
        "var1_derived_code": v1["derived_code"],
        "var1_derived_nl": v1["derived_nl"],
        "var1_base_codes": v1["base_codes"],
        "var2_derived_code": v2["derived_code"],
        "var2_derived_nl": v2["derived_nl"],
        "var2_base_codes": v2["base_codes"],
    }