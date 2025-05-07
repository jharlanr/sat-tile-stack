


## FUNCTION TO DEFINE BOUNDING BOX AROUND A GIVEN CENTROID
def bounds_latlon_around(center_lon, center_lat, side_m=10000):
    """
    center_lon, center_lat : centroid in decimal degrees (EPSG:4326)
    side_m                 : length of box side in meters (default 10 km)
    returns                 : (minx, miny, maxx, maxy) in lon/lat
    """
    
    # FIND BEST CRS FOR CENTROID
    def best_crs_for_point(lon, lat):
        """
        Choose a projected CRS (EPSG) that minimizes distortion
        for a small box around (lon, lat).

        - |lat| ≥ 60° → Polar Stereographic (EPSG:3413 North / 3031 South)
        - else         → UTM zone based on lon

        Returns a pyproj.CRS object.
        """
        if lat >= 60:
            # Arctic Polar Stereographic
            return pyproj.CRS.from_epsg(3413)
        elif lat <= -60:
            # Antarctic Polar Stereographic
            return pyproj.CRS.from_epsg(3031)
        else:
            # UTM
            zone_number = int(math.floor((lon + 180) / 6) + 1)
            is_south   = lat < 0
            # Construct a PROJ string for UTM:
            proj4 = (
                f"+proj=utm +zone={zone_number} "
                f"+{'south' if is_south else 'north'} +datum=WGS84 +units=m +no_defs"
            )
            return pyproj.CRS.from_proj4(proj4)
    
    # SET UP TRANSFORMERS
    centroid = (center_lon, center_lat)
    epsg_code = int(best_crs_for_point(*centroid).to_authority()[1])
    print(f"epsg for bounds_latlon: {epsg_code}")
    to_ps = pyproj.Transformer.from_crs(4326, epsg_code, always_xy=True).transform
    to_ll = pyproj.Transformer.from_crs(epsg_code, 4326, always_xy=True).transform

    # PROJECT CENTROID TO METERS
    x0, y0 = to_ps(center_lon, center_lat)

    # BUILD SQUARE AROUND CENTROID
    half = side_m / 2.0
    sq_m = box(x0 - half, y0 - half, x0 + half, y0 + half)

    # REPROJECT SQUARE BACK TO LAT/LON AND GRAB BOUNDS
    sq_ll = transform(to_ll, sq_m)
    return sq_ll.bounds

## FUNCTIONN TO FIND BEST CRS FOR CENTROID
def best_crs_for_point(lon, lat):
    """
    Choose a projected CRS (EPSG) that minimizes distortion
    for a small box around (lon, lat).

    - |lat| ≥ 60° → Polar Stereographic (EPSG:3413 North / 3031 South)
    - else         → UTM zone based on lon

    Returns a pyproj.CRS object.
    """
    if lat >= 60:
        # Arctic Polar Stereographic
        return pyproj.CRS.from_epsg(3413)
    elif lat <= -60:
        # Antarctic Polar Stereographic
        return pyproj.CRS.from_epsg(3031)
    else:
        # UTM
        zone_number = int(math.floor((lon + 180) / 6) + 1)
        is_south   = lat < 0
        # Construct a PROJ string for UTM:
        proj4 = (
            f"+proj=utm +zone={zone_number} "
            f"+{'south' if is_south else 'north'} +datum=WGS84 +units=m +no_defs"
        )
        return pyproj.CRS.from_proj4(proj4)