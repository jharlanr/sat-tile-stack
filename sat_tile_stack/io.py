import xarray as xr
import numpy as np
import json, re, zarr, warnings

## HELPER FUNCTION to scrub attributes so they JSON-round-trip
def scrub_attrs(xr_obj, drop=()):
    obj = xr_obj.copy()
    for k in list(obj.attrs):
        if k in drop:                       # remove unwanted attrs (e.g. RasterSpec)
            obj.attrs.pop(k)
            continue
        v = obj.attrs[k]
        try:                                # does it JSON-serialise?
            json.dumps(v)
        except (TypeError, OverflowError):
            # numpy scalar → Python scalar ; everything else → string
            if isinstance(v, np.generic):
                obj.attrs[k] = v.item()
            else:
                obj.attrs[k] = str(v)
    return obj

## HELPER FUNCTION to move illegal/object variables → global attributes
_illegal_nc_name = re.compile(r"[^0-9A-Za-z_]")
def sanitise_dataset(ds):
    """
    • Any variable/coord that has ':' or other illegal chars OR has dtype=object
      is converted to a global attribute (so nothing is lost).
    """
    ds = ds.copy()
    for v in list(ds.variables):
        bad_name = (":" in v) or _illegal_nc_name.search(v)
        bad_type = ds[v].dtype == object
        if not (bad_name or bad_type):
            continue

        # extract value(s)
        val = ds[v].values
        if val.size == 1:
            val = val.item()
        elif isinstance(val, np.ndarray):
            # small array → list; huge array → string to keep JSON size down
            val = val.tolist() if val.size < 100 else json.dumps(val.tolist())
        ds = ds.drop_vars(v)
        ds.attrs[v] = val
    return ds

## HELPER FUNCTION to make every attr JSON-safe, including stray sets
def coerce_attrs_to_json_safe(ds):
    for k, v in list(ds.attrs.items()):
        # flatten singleton sets like {10980} → 10980
        if isinstance(v, set):
            if len(v) == 1:
                ds.attrs[k] = next(iter(v))
            else:
                ds.attrs[k] = list(v)

        # NumPy scalars → plain Python scalars
        elif isinstance(v, np.generic):
            ds.attrs[k] = v.item()

        # non-serialisable objects → string
        try:
            json.dumps(ds.attrs[k])
        except (TypeError, OverflowError):
            ds.attrs[k] = str(ds.attrs[k])
    return ds

## FUNCTION TO CLEAN + SAVE
def write_netcdf_from_da(da, outfile, drop_attrs=("spec",)):
    """Clean a DataArray and write it to NetCDF-4 (compressed)."""
    # 3-a  scrub global attrs → JSON-safe
    da_clean  = scrub_attrs(da, drop=drop_attrs)

    # 3-b  promote to dataset (so we control variable name) and keep attrs
    ds        = da_clean.to_dataset(name="reflectance", promote_attrs=True)

    # 3-c  move illegal / object vars → attrs  (no info lost)
    ds        = sanitise_dataset(ds)
    
    ds = coerce_attrs_to_json_safe(ds)

    # 3-d  per-variable compression & dtype
    enc = {"reflectance": dict(zlib=True, complevel=4, dtype="float32")}

    # 3-e  write NetCDF-4
    ds.to_netcdf(
        outfile,
        engine="netcdf4",      # or "h5netcdf"
        format="NETCDF4",
        mode="w",
        encoding=enc
    )
    print(f"wrote {outfile}")