# glacierReader

**Greenland ice-velocity time series from NSIDC, as callable functions.**

Pull velocity for a region of interest straight from the NSIDC/GrIMP catalog, save it as
NetCDF, read it back at a point, and plot it. If you do not know where the ROI is, drag it on a
map of Greenland and the glacier gets named for you.

Everything is a plain function with arguments — a run is reproducible from the ROI centre, the
distance to add in x and y, a date range and a list of product codes. `glacier_reader.py` is
the module; `matlab_glacier_reader.ipynb` is a thin driver over it.

```python
import glacier_reader as gr

display(gr.login())                       # NASA Earthdata, once per session

out = gr.glacier_velocity_workflow(
    center=(-183000.0, -2274200.0),       # EPSG:3413 metres
    dx_m=10000.0,                         # -> a 20 x 20 km ROI
    first_date='2018-01-01', last_date='2019-12-31',
    products=['NSIDC-0725', 'NSIDC-0731'],
    prefix='Jakobshavn',
)

ts, ax = gr.plot_glacier_velocity(out['files'], -183000.0, -2274200.0, method='nearest')
```

## Install

Needs [pixi](https://pixi.sh). The environment pins `gdal >=3.12,<3.13` with `rasterio 1.5.0`:
that pairing is verified against NSIDC here, and concurrent `/vsicurl` reads segfaulted on
GDAL 3.13.3. Lift the bound only after re-running a real download.

```bash
pixi install          # conda-forge + grimpfunc/nisardev from git
pixi run lab          # JupyterLab
pixi run sync-env     # refresh environment.yml for conda users after `pixi add`
```

Downloads need [Earthdata](https://urs.earthdata.nasa.gov) credentials. `gr.login()` sets the
GDAL cookie jar and returns the login widget; log in once and the cookie is reused.

## When you don't know the ROI

Leave `center`/`dx_m` out and a Greenland overview map opens: the most recent **annual velocity
mosaic**, read decimated straight from NSIDC (~5 s, then cached), over the standard
**BedMachine coastline**, with the 234 named flux gates of King et al. (2020) marked. Drag a
box — the readout tracks its size, centre, `dx`/`dy` and the glacier it covers.

Pass `prefix=None` and the glacier the ROI covers is offered as the file name:

```
Glacier within ROI is jakobshavn (71% of its gate inside the ROI).
  also nearby: torssukatak, alangordliup
Press enter to continue with this file name [jakobshavn] or type in name to use to override default:
```

Enter accepts it; anything typed replaces it, cleaned into a file-name-safe form
(`Jakobshavn Isbrae` → `Jakobshavn_Isbrae`). Where a prompt cannot be answered — papermill,
cron, piped stdin — the suggestion is taken silently rather than hanging.

```python
%matplotlib widget                        # in a notebook; inline figures cannot be dragged
out = gr.glacier_velocity_workflow(first_date='2018-01-01', prefix=None)

# or step by step
sel = gr.select_roi_interactive()         # drag a box
sel.center, sel.dx_m, sel.dy_m, sel.name
out = sel.download(products=['NSIDC-0725'])
```

The lookup ranks a gate the ROI actually covers above one it merely clips, and falls back to
the nearest gate within 50 km when the box sits just off a terminus, so an ROI placed
up-glacier still gets a sensible name.

**Backends.** The picker works in a script as well as a notebook, but needs an interactive
matplotlib backend either way: `%matplotlib widget` (ipympl) in a notebook; any GUI backend in
a script (`macosx` is already the default here), where the call blocks until you close the
window. Headless (`Agg`, SSH with no display) cannot drag at all — pass coordinates instead.

**Local data it reads.** Two files, both overridable by environment variable, neither fatal if
missing — the map just opens without a coastline, or without naming:

| What | Default | Override |
|---|---|---|
| Glacier names | `$GLACIER_DATA_HOME/King2020_GreenlandGateCoord.mat` | `GLACIER_GATES_FILE` |
| Coastline | `$GLACIER_DATA_HOME/BedMachine_v5.mat` | `BEDMACHINE_FILE` |

`GLACIER_DATA_HOME` defaults to `~/Documents/GIMP_Working/RECALC_discharge`. The decimated map
and outline are cached in `.cache/` (`GLACIER_READER_CACHE`); delete it to force a refresh.

## What you get

One NetCDF per product — and per GrIMP box for the per-glacier products, since each TSX/optical
tile is its own grid:

```
<out_dir>/<prefix>_<product>[_<box>].nc     e.g. jakobshavn_NSIDC-0725.nc
<out_dir>/<prefix>_roi.json                 the exact inputs that produced them
```

Each file holds `velocity(time, band, y, x)` over `{vx, vy, vv, ex, ey, ev, dT}` in m/yr
(`dT` in days), EPSG:3413, with the ROI, date range, product code and box in its global
attributes. Re-running is cheap: existing files are skipped unless `overwrite=True`.

### No-data: NaN, and only NaN

GrIMP GeoTIFFs flag no-data with sentinels — `-2e9` on `vx`, `vy`, `dT`; `-1` on `vv`, `ex`,
`ey`, `ev` — and whether they survive the read depends on whether the COG declares `nodata` in
its header. Speed makes this worse: `nisardev` derives `vv` as `sqrt(vx² + vy²)`, so an
unmasked sentinel pair becomes `+2.83e9` — a large *positive* number that no threshold on speed
can catch, and one that quietly becomes the maximum of your dataset.

Every array passes through `normalize_nodata`, which thresholds each band *and* makes `vv`/`ev`
inherit the gaps of their components, on the way into a file and on the way back out.
`nodata_report` audits a file raw, without cleaning: `n_sentinel_left` must be 0 everywhere.
`glacier_velocity_workflow` runs that check for you (`check_nodata=True`).

## Products

| Code | Product | Grid |
|---|---|---|
| `NSIDC-0725` | Sentinel-1 annual mosaic | ice-sheet wide, clipped to the ROI |
| `NSIDC-0727` | Sentinel-1 quarterly mosaic | ice-sheet wide |
| `NSIDC-0731` | Sentinel-1 monthly mosaic | ice-sheet wide |
| `NSIDC-0766` | Sentinel-1 6-day mosaic | ice-sheet wide |
| `NSIDC-0481` | TSX/TDX individual glacier | ~50 km boxes, one file per box |
| `NSIDC-0646` | Optical (Landsat/Sentinel-2) | ~50 km boxes, one file per box |

The product version is bumped up to four times if a search comes back empty, so this keeps
working when NSIDC releases a new version.

## The functions

| Stage | Functions |
|---|---|
| Setup | `login`, `configure_dask` |
| ROI | `roi_bbox`, `bbox_lonlat`, `intersect_bbox`, `describe_bbox` |
| Pick an ROI | `select_roi_interactive`, `GreenlandRoiSelector`, `plot_greenland_overview`, `greenland_mosaic_overview`, `greenland_outline` |
| Name an ROI | `suggest_glacier_name`, `prompt_glacier_name`, `glaciers_in_roi`, `load_glacier_gates`, `clean_glacier_name` |
| Search | `cmr_granule_urls`, `search_velocity_cogs`, `summarize_search`, `product_info`, `product_bounds` |
| Load | `load_velocity_series`, `normalize_nodata` |
| Save | `download_velocity_netcdfs`, `save_velocity_netcdf`, `velocity_netcdf_path` |
| Read back | `read_velocity_netcdf`, `resolve_netcdf_paths`, `velocity_time_series`, `sample_velocity_at_point` |
| Plot | `plot_glacier_velocity`, `plot_velocity_time_series` |
| Audit | `nodata_report` |
| MATLAB gates | `matlab_glacier_center`, `matlab_glacier_centers` |
| End to end | `glacier_velocity_workflow` |

All have numpydoc docstrings; `help(gr.<name>)` is the reference.

## Using the output with ModelObsCompare

`moc_py.obs_nsidc` (in the [ModelObsCompare](../../Documents/GIMP_Working/RECALC_discharge/ModelObsCompare)
toolkit) adapts these files into the Dataset its model-vs-observation tools expect. The band
axis here is `band` in nisardev order `(vx, vy, vv, ex, ey, ev, dT)`; moc calls it `component`
and orders it differently, so go through the adapter rather than `moc_py.load_obs_netcdf`:

```python
import moc_py as mp
from moc_py.obs_nsidc import load_nsidc_netcdf, bbox_to_roi

O   = load_nsidc_netcdf(out['files'][0], bands=['vv'])   # -> moc obs Dataset
roi = bbox_to_roi(out['bbox'], name=out['prefix'])       # -> moc Roi (+ ISSM .exp)
M   = mp.load_model('nw_fric1.nc')
mp.plot_diff(mp.spatial_diff(M, O, roi, time=2019.6), roi=roi)
```

`mp.fetch_nsidc(...)` drives this module from the moc side, reusing any file already in
`CONFIG.obs_netcdf_dir` and only downloading what is missing.

## Tests

```bash
pixi run python test_roi_naming.py
```

19 checks on the ROI picker and glacier naming — synthetic gates, no network, no Earthdata
credentials, no BedMachine needed.

## Notes

- **CMR pagination.** `grimpfunc.get_urls` pages with `page_num` and `scroll=false`, which CMR
  has retired; it now returns HTTP 400 and fails with `KeyError: 'feed'` for *every* search.
  `cmr_granule_urls` here uses the supported `CMR-Search-After` header instead, and grimpfunc
  is kept only as a fallback.
- **Connection limits.** NSIDC caps simultaneous connections, so a long download can die with
  an HTTP 404/503 on one tile even when every file is present. `download_velocity_netcdfs`
  retries a product twice; if it still fails, *lower* `num_workers` rather than raising it.
- **Sampling near the terminus.** `method='linear'` returns NaN as soon as one of the four
  surrounding pixels is missing. Use `method='nearest'` at an ice margin, which is exactly
  where a glacier time series is usually wanted.
- `.cache/`, `__pycache__/` and `.DS_Store` are currently tracked in git but are all
  regenerable; a `.gitignore` would be worth adding.

## Layout

```
glacier_reader.py                    the module — everything above
matlab_glacier_reader.ipynb          driver notebook: parameters, run, plots, map view
matlab_glacier_reader.ipynb.prelift.bak   the notebook before the functions moved out
test_roi_naming.py                   ROI picker + naming checks (no network)
pixi.toml / pixi.lock / environment.yml   the environment
.cache/                              decimated overview map + coastline (regenerable)
```

MIT licensed. Velocity products are GrIMP/NSIDC; cite them, and
[grimpfunc/nisardev](https://github.com/fastice), as their own documentation asks.
