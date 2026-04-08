"""
Persistent feature identification across time periods.

Provides tools for matching geospatial features (lakes, fields, etc.) detected
at different times and assigning stable, persistent IDs. This is essential for
building time series of individual features from per-timestep detections.

The core challenge: feature boundaries shift between observations due to real
physical changes (lake filling/draining, crop rotation) and detection noise
(cloud gaps, threshold sensitivity). This module provides matching strategies
to determine when features across time periods represent "the same" entity.

Example usage:

    >>> tracker = FeatureTracker(method="centroid", tolerance_m=500)
    >>> tracker.register(lakes_2018_gdf, "2018")
    >>> tracker.register(lakes_2019_gdf, "2019")
    >>> id_table = tracker.get_id_table()
    >>> # {("2018", 42): "FEAT_0001", ("2019", 39): "FEAT_0001", ...}

Known complexities (especially for lakes):
    - Polygons from water masks may split one physical lake into multiple polygons
      (e.g., due to partial cloud cover or threshold sensitivity)
    - Multiple small ponds may merge into one large lake as water levels rise
    - The "identity" of a feature depends on the spatial resolution of the mask
      and the minimum area threshold used during vectorization
    - These are domain-specific decisions — this module provides the matching
      infrastructure, not the domain heuristics
"""

import numpy as np
from scipy.spatial import cKDTree

from .bounds import best_crs_for_point


# ===========================================================================
# Matching functions
# ===========================================================================
# Each function takes two GeoDataFrames and returns a dict of matches:
#   {index_in_gdf_new: index_in_gdf_ref}
# Unmatched features in gdf_new are not included in the dict.

def _match_centroid(gdf_new, gdf_ref, tolerance_m):
    """
    Match features by centroid proximity.

    Uses a KD-tree for efficient nearest-neighbor lookup. For each feature
    in gdf_new, finds the closest feature in gdf_ref by centroid distance.
    Matches are assigned greedily (closest pairs first) so each ref feature
    can only be matched once.

    Parameters
    ----------
    gdf_new : geopandas.GeoDataFrame
        Features to match (e.g., lakes detected in 2020).
    gdf_ref : geopandas.GeoDataFrame
        Reference features to match against (e.g., lakes from 2019).
    tolerance_m : float
        Maximum centroid distance in meters to consider a match.

    Returns
    -------
    dict
        {new_index: ref_index} for matched features.

    Notes
    -----
    Both GeoDataFrames should be in a projected CRS (meters) for accurate
    distance calculations. If they are in lat/lon (EPSG:4326), distances
    will be in degrees — not meters. Use bounds.best_crs_for_point() to
    reproject first, or pass lon/lat columns and set tolerance accordingly.
    """
    # Get centroids as numpy arrays
    new_centroids = np.array([(g.centroid.x, g.centroid.y) for g in gdf_new.geometry])
    ref_centroids = np.array([(g.centroid.x, g.centroid.y) for g in gdf_ref.geometry])

    if len(ref_centroids) == 0 or len(new_centroids) == 0:
        return {}

    # Build KD-tree on reference centroids for fast nearest-neighbor lookup
    tree = cKDTree(ref_centroids)

    # Query: for each new centroid, find the closest ref centroid
    distances, ref_indices = tree.query(new_centroids, k=1)

    # Build candidate pairs sorted by distance (greedy closest-first)
    new_idx_list = list(gdf_new.index)
    ref_idx_list = list(gdf_ref.index)

    candidates = []
    for i, (dist, ref_pos) in enumerate(zip(distances, ref_indices)):
        if dist <= tolerance_m:
            candidates.append((dist, new_idx_list[i], ref_idx_list[ref_pos]))

    # Sort by distance — assign closest pairs first
    candidates.sort(key=lambda x: x[0])

    # Greedy assignment: each ref feature can only be matched once
    matches = {}
    used_ref = set()
    for dist, new_idx, ref_idx in candidates:
        if ref_idx not in used_ref:
            matches[new_idx] = ref_idx
            used_ref.add(ref_idx)

    return matches


def _match_overlap(gdf_new, gdf_ref, min_iou=0.3):
    """
    Match features by polygon overlap (Intersection over Union).

    For each feature in gdf_new, computes IoU with all features in gdf_ref.
    Best IoU match above min_iou threshold is considered a match.

    Parameters
    ----------
    gdf_new : geopandas.GeoDataFrame
        Features to match.
    gdf_ref : geopandas.GeoDataFrame
        Reference features.
    min_iou : float
        Minimum Intersection over Union to consider a match (0–1).

    Returns
    -------
    dict
        {new_index: ref_index} for matched features.

    Notes
    -----
    More robust than centroid matching for features that change shape
    significantly (e.g., lakes that expand/contract). Slower due to
    polygon intersection calculations.
    """
    # TODO: implement — compute pairwise IoU, assign best matches
    raise NotImplementedError("overlap matching not yet implemented")


# Registry of available matching methods
MATCH_METHODS = {
    "centroid": _match_centroid,
    "overlap": _match_overlap,
}


# ===========================================================================
# FeatureTracker
# ===========================================================================

class FeatureTracker:
    """
    Tracks geospatial features across time periods and assigns persistent IDs.

    The tracker maintains an internal registry of known features. Each call to
    register() compares incoming features against the registry:
      - Matched features inherit the existing persistent ID
      - Unmatched features get a new unique ID and are added to the registry

    Parameters
    ----------
    method : str
        Matching strategy. Options:
        - 'centroid': match by centroid proximity (good for stationary features)
        - 'overlap': match by polygon IoU (good for features that change shape)
    tolerance_m : float
        For 'centroid' method: max distance in meters to consider a match.
        Ignored for 'overlap' method (uses min_iou instead).
    min_iou : float
        For 'overlap' method: minimum IoU to consider a match (0–1).
        Ignored for 'centroid' method.
    id_prefix : str
        Prefix for generated persistent IDs (default: 'FEAT').
        Example: 'LAKE' -> 'LAKE_0001', 'LAKE_0002', ...

    Example
    -------
    >>> import geopandas as gpd
    >>> tracker = FeatureTracker(method="centroid", tolerance_m=500, id_prefix="LAKE")
    >>> tracker.register(lakes_2018, "2018")
    >>> tracker.register(lakes_2019, "2019")
    >>> tracker.register(lakes_2020, "2020")
    >>>
    >>> table = tracker.get_id_table()
    >>> # {("2018", 0): "LAKE_0001", ("2019", 3): "LAKE_0001", ...}
    >>>
    >>> tracker.summary()
    >>> # Registered 3 time periods: 2018 (412 features), 2019 (398 features), ...
    >>> # Total persistent features: 450
    >>> # Features appearing in all periods: 320
    >>> # New features (appeared after first period): 38
    >>> # Disappeared features: 52
    """

    def __init__(
        self,
        method="centroid",
        tolerance_m=500,
        min_iou=0.3,
        id_prefix="FEAT",
    ):
        if method not in MATCH_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Available: {list(MATCH_METHODS.keys())}"
            )

        self.method = method
        self.tolerance_m = tolerance_m
        self.min_iou = min_iou
        self.id_prefix = id_prefix

        # Internal state
        self._next_id = 0
        self._registry = {}        # persistent_id -> reference geometry (shapely)
        self._registry_centroids = {}  # persistent_id -> (x, y) centroid
        self._id_table = {}         # (time_label, source_index) -> persistent_id
        self._time_labels = []      # ordered list of registered time periods

    def _generate_id(self):
        """Generate the next persistent ID string."""
        pid = f"{self.id_prefix}_{self._next_id:04d}"
        self._next_id += 1
        return pid

    def register(self, gdf, time_label):
        """
        Register a set of features from one time period.

        Matches incoming features against the existing registry. Matched
        features get the existing persistent ID; unmatched features get
        new IDs and are added to the registry.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Detected features for this time period. Must have a geometry column.
        time_label : str
            Label for this time period (e.g., '2019', '2019-06-08').

        Returns
        -------
        dict
            {source_index: persistent_id} for all features in this time period.
        """
        import geopandas as gpd

        assignments = {}

        if len(self._registry) == 0:
            # First time period — everything gets a new ID
            for idx in gdf.index:
                pid = self._generate_id()
                geom = gdf.loc[idx, "geometry"]
                self._registry[pid] = geom
                self._registry_centroids[pid] = (geom.centroid.x, geom.centroid.y)
                self._id_table[(time_label, idx)] = pid
                assignments[idx] = pid
        else:
            # Build a ref GeoDataFrame from the registry for matching
            ref_pids = list(self._registry.keys())
            ref_geoms = [self._registry[pid] for pid in ref_pids]
            gdf_ref = gpd.GeoDataFrame(
                {"persistent_id": ref_pids},
                geometry=ref_geoms,
                crs=gdf.crs,  # inherit CRS from incoming data
                index=range(len(ref_pids)),
            )

            # Reproject to a projected CRS (meters) for accurate distance matching.
            # Use the mean lon/lat of bounding boxes as a rough center — doesn't
            # need to be precise, just picks the right UTM zone or polar stereo.
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                centroid_lon = gdf.geometry.centroid.x.mean()
                centroid_lat = gdf.geometry.centroid.y.mean()
            proj_crs = best_crs_for_point(centroid_lon, centroid_lat)

            gdf_proj = gdf.to_crs(proj_crs)
            gdf_ref_proj = gdf_ref.to_crs(proj_crs)

            # Run matching in projected coordinates (tolerance is in meters)
            if self.method == "centroid":
                matches = _match_centroid(gdf_proj, gdf_ref_proj, self.tolerance_m)
            elif self.method == "overlap":
                matches = _match_overlap(gdf_proj, gdf_ref_proj, self.min_iou)

            # Assign persistent IDs
            for idx in gdf.index:
                if idx in matches:
                    # Matched to an existing feature
                    ref_row = matches[idx]
                    pid = gdf_ref.loc[ref_row, "persistent_id"]
                    # Update registry geometry to latest observation
                    geom = gdf.loc[idx, "geometry"]
                    self._registry[pid] = geom
                    self._registry_centroids[pid] = (geom.centroid.x, geom.centroid.y)
                else:
                    # New feature — assign new ID
                    pid = self._generate_id()
                    geom = gdf.loc[idx, "geometry"]
                    self._registry[pid] = geom
                    self._registry_centroids[pid] = (geom.centroid.x, geom.centroid.y)

                self._id_table[(time_label, idx)] = pid
                assignments[idx] = pid

        self._time_labels.append(time_label)
        return assignments

    def get_id_table(self):
        """
        Get the full mapping of (time_label, source_index) -> persistent_id.

        Returns
        -------
        dict
            Complete mapping across all registered time periods.
        """
        return dict(self._id_table)

    def get_features_for_id(self, persistent_id):
        """
        Get all (time_label, source_index) entries for a given persistent ID.

        Useful for retrieving the full history of a single feature.

        Parameters
        ----------
        persistent_id : str
            The persistent ID to look up.

        Returns
        -------
        list of tuple
            [(time_label, source_index), ...] for each time period where
            this feature was detected.
        """
        return [
            (tl, si) for (tl, si), pid in self._id_table.items()
            if pid == persistent_id
        ]

    def get_status(self, time_label):
        """
        Get the status of all features for a given time period.

        Returns which features are persistent (matched to existing catalog),
        which are new (first time seen), and which are missing (in catalog
        but not detected this time).

        Parameters
        ----------
        time_label : str
            The time period to check.

        Returns
        -------
        dict with keys:
            'persistent': list of persistent IDs that were matched
            'new': list of persistent IDs that were first seen in this period
            'missing': list of persistent IDs in the catalog but not in this period
        """
        # IDs seen in this time period
        ids_this_period = {
            pid for (tl, _), pid in self._id_table.items() if tl == time_label
        }

        # IDs seen in earlier periods
        period_idx = self._time_labels.index(time_label)
        earlier_labels = set(self._time_labels[:period_idx])
        ids_before = {
            pid for (tl, _), pid in self._id_table.items() if tl in earlier_labels
        }

        persistent = sorted(ids_this_period & ids_before)
        new = sorted(ids_this_period - ids_before)
        missing = sorted(ids_before - ids_this_period)

        return {"persistent": persistent, "new": new, "missing": missing}

    def summary(self):
        """Print a summary of registered features and matching statistics."""
        print(f"FeatureTracker (method={self.method}, tolerance={self.tolerance_m})")
        print(f"  Registered {len(self._time_labels)} time periods: ", end="")

        # Count features per period
        counts = {}
        for (tl, _) in self._id_table:
            counts[tl] = counts.get(tl, 0) + 1
        parts = [f"{tl} ({counts.get(tl, 0)} features)" for tl in self._time_labels]
        print(", ".join(parts))

        # Count unique persistent IDs
        unique_pids = set(self._id_table.values())
        print(f"  Total persistent features: {len(unique_pids)}")

        # Per-period status
        if len(self._time_labels) > 1:
            for tl in self._time_labels[1:]:
                status = self.get_status(tl)
                print(f"  {tl}: {len(status['persistent'])} matched, "
                      f"{len(status['new'])} new, "
                      f"{len(status['missing'])} missing")

            # Features in all periods
            pid_periods = {}
            for (tl, _), pid in self._id_table.items():
                pid_periods.setdefault(pid, set()).add(tl)
            all_periods = set(self._time_labels)
            n_all = sum(1 for periods in pid_periods.values() if periods == all_periods)
            print(f"  Features in ALL periods: {n_all}")
