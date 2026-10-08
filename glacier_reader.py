"""Glacier velocity reader - GrIMP/NSIDC velocity time series for a region of interest.

Pull velocity time series straight from NSIDC, save them as NetCDF, read them
back at a point, and plot them. Every step is a function with callable
arguments, so a run is reproducible from the ROI centre, the distance to add in
x and y, a date range and a list of NSIDC product codes.

This module is the single source for the code that used to live inline in
``matlab_glacier_reader.ipynb``; the notebook now imports it.

Typical use::

    import glacier_reader as gr

    gr.login()                                  # Earthdata, once per session
    out = gr.glacier_velocity_workflow(
        center=(-183000.0, -2274200.0), dx_m=10000.0,
        first_date='2018-01-01', last_date='2019-12-31',
        products=['NSIDC-0725', 'NSIDC-0731'], prefix='Jakobshavn',
    )
    ts, ax = gr.plot_glacier_velocity(out['files'], -183000.0, -2274200.0,
                                      method='nearest')

If the ROI is not known in advance, omit ``center``/``dx_m`` and pass
``prefix=None``: the annual velocity mosaic is drawn over the standard
Greenland outline, a box is dragged on it, the glacier covered by that box is
looked up in the named flux gates, and the file name is confirmed at a prompt::

    out = gr.glacier_velocity_workflow(first_date='2018-01-01', prefix=None)
    # -> drag a box; "Glacier within ROI is jakobshavn (82% of its gate inside
    #    the ROI)." then Enter to accept, or type a name to override.

The pieces are also usable on their own: :func:`select_roi_interactive`,
:func:`plot_greenland_overview`, :func:`glaciers_in_roi`,
:func:`suggest_glacier_name` and :func:`prompt_glacier_name`.

To feed the result into the ModelObsCompare comparison tools, see
``moc_py.obs_nsidc.load_nsidc_netcdf`` / ``moc_py.obs_nsidc.nsidc_to_obs``,
which relabel the ``band`` dimension to moc's ``component`` convention.

No-data convention
------------------
GrIMP GeoTIFFs use sentinel values (``-2e9`` for ``vx``, ``vy``, ``dT``; ``-1``
for ``vv``, ``ex``, ``ey``, ``ev``), and whether they are masked on read depends
on whether the COG declares ``nodata`` in its header. Speed makes this worse:
``nisardev`` derives ``vv`` as ``sqrt(vx**2 + vy**2)``, so two unmasked
sentinels produce ``2.83e9`` - a large *positive* number that no threshold on
speed can catch. Everything here passes through :func:`normalize_nodata`, so
NaN is the one and only no-data value on the way into a file, on the way back
out, and in every plot. :func:`nodata_report` verifies it.

All coordinates are EPSG:3413 (Polar Stereographic North, metres) unless stated
otherwise.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import dask
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio
import requests
import scipy.io
import xarray as xr
from dask.diagnostics import ProgressBar

import grimpfunc as grimp
import nisardev as nisar


def configure_dask(num_workers: int = 2, progress: bool = True) -> None:
    """Set the dask scheduler used for the concurrent /vsicurl reads.

    Parameters
    ----------
    num_workers : int, optional
        Threads used for remote reads. NSIDC limits simultaneous connections,
        so keep this small. The default is 2.
    progress : bool, optional
        Register a global :class:`dask.diagnostics.ProgressBar`. The default is True.

    Notes
    -----
    The scheduler is named explicitly because setting ``num_workers`` alone no
    longer constrains concurrency in recent dask, and oversubscribed
    ``/vsicurl`` reads make GDAL segfault.
    """
    dask.config.set(scheduler='threads', num_workers=num_workers)
    if progress:
        ProgressBar().register()


def login(cookie_file: str = '~/.grimp_download_cookiejar.txt'):
    """Set up the GDAL cookie jar and return the NASA Earthdata login widget.

    Parameters
    ----------
    cookie_file : str, optional
        Where GDAL stores the Earthdata session cookie. The default is
        ``'~/.grimp_download_cookiejar.txt'``.

    Returns
    -------
    panel widget
        Display it (``display(gr.login())``) and log in once; the cookie is then
        reused. See the GrIMP NSIDCLoginNotebook.

    Notes
    -----
    Downloads need credentials from https://urs.earthdata.nasa.gov.
    """
    path = os.path.expanduser(cookie_file)
    os.environ.update(GDAL_HTTP_COOKIEFILE=path, GDAL_HTTP_COOKIEJAR=path)
    return grimp.NASALogin().view()


NSIDC_VELOCITY_PRODUCTS: dict[str, dict[str, object]] = {
    'NSIDC-0725': {'label': 'Sentinel-1 annual mosaic', 'version': '5',
                   'boxed': False, 'has_dt': True},
    'NSIDC-0727': {'label': 'Sentinel-1 quarterly mosaic', 'version': '5',
                   'boxed': False, 'has_dt': True},
    'NSIDC-0731': {'label': 'Sentinel-1 monthly mosaic', 'version': '5',
                   'boxed': False, 'has_dt': True},
    'NSIDC-0766': {'label': 'Sentinel-1 6-day mosaic', 'version': '2',
                   'boxed': False, 'has_dt': True},
    'NSIDC-0481': {'label': 'TSX/TDX individual glacier', 'version': '3',
                   'boxed': True, 'has_dt': False},
    'NSIDC-0646': {'label': 'Optical (Landsat/Sentinel-2) individual glacier', 'version': '3',
                   'boxed': True, 'has_dt': False},
}

#: Band tokens that can appear in a GrIMP velocity file name.
VELOCITY_BANDS: tuple[str, ...] = ('vx', 'vy', 'vv', 'ex', 'ey', 'dT')

#: GrIMP sentinel no-data values. Anything <= the sentinel is no data.
NO_DATA_SENTINELS: dict[str, float] = {
    'vx': -2.0e9, 'vy': -2.0e9, 'dT': -2.0e9,
    'vv': -1.0, 'ex': -1.0, 'ey': -1.0, 'ev': -1.0,
}

#: Bands nisardev computes as sqrt(x**2 + y**2) instead of downloading, mapped
#: to the components they come from. Their no-data has to be inherited, not
#: thresholded - see normalize_nodata.
DERIVED_BANDS: dict[str, tuple[str, str]] = {'vv': ('vx', 'vy'), 'ev': ('ex', 'ey')}

#: Projected CRS all gridded output is stored in.
GRID_EPSG = 3413


def product_info(product: str) -> dict[str, object]:
    """Look up the registry entry for an NSIDC velocity product code.

    Parameters
    ----------
    product : str
        NSIDC product code, e.g. ``'NSIDC-0725'``.

    Returns
    -------
    dict
        Registry entry with ``label``, ``version``, ``boxed`` and ``has_dt`` keys.

    Raises
    ------
    KeyError
        If the product is not one of the GrIMP velocity products.
    """
    try:
        return NSIDC_VELOCITY_PRODUCTS[product]
    except KeyError:
        valid = ', '.join(NSIDC_VELOCITY_PRODUCTS)
        raise KeyError(f'{product} is not a GrIMP velocity product. Valid codes: {valid}') from None


def roi_bbox(center: tuple[float, float],
             dx_m: float,
             dy_m: float | None = None,
             center_crs: int | str = GRID_EPSG,
             decimals: int = -2) -> dict[str, float]:
    """Build a projected bounding box from an ROI centre and a distance in x and y.

    Parameters
    ----------
    center : tuple of float
        Centre of the ROI as ``(x, y)`` in ``center_crs``. For a geographic CRS
        this is ``(longitude, latitude)``.
    dx_m : float
        Distance in metres to add either side of the centre in x, so the box is
        ``2 * dx_m`` wide.
    dy_m : float, optional
        Same for y. Defaults to ``dx_m`` (a square ROI).
    center_crs : int or str, optional
        CRS of ``center``. The default is EPSG:3413.
    decimals : int, optional
        Passed to :func:`numpy.around` to snap the corners; ``-2`` rounds to
        100 m, ``-3`` to 1 km. The default is ``-2``.

    Returns
    -------
    dict
        ``{'minx': ..., 'miny': ..., 'maxx': ..., 'maxy': ...}`` in EPSG:3413 metres.
    """
    if dy_m is None:
        dy_m = dx_m
    if dx_m <= 0 or dy_m <= 0:
        raise ValueError('dx_m and dy_m must be positive distances in metres')

    x0, y0 = float(center[0]), float(center[1])
    if pyproj.CRS.from_user_input(center_crs).to_epsg() != GRID_EPSG:
        to_grid = pyproj.Transformer.from_crs(center_crs, GRID_EPSG, always_xy=True)
        x0, y0 = to_grid.transform(x0, y0)

    corners = np.around([x0 - dx_m, y0 - dy_m, x0 + dx_m, y0 + dy_m], decimals)
    return dict(zip(['minx', 'miny', 'maxx', 'maxy'], [float(v) for v in corners]))


def bbox_lonlat(bbox: dict[str, float],
                epsg: int = GRID_EPSG,
                n_edge: int = 21) -> tuple[float, float, float, float]:
    """Return the geographic bounding box enclosing a projected bounding box.

    Parameters
    ----------
    bbox : dict
        ``{'minx', 'miny', 'maxx', 'maxy'}`` in ``epsg`` metres.
    epsg : int, optional
        CRS of ``bbox``. The default is EPSG:3413.
    n_edge : int, optional
        Number of points used to densify each edge before transforming. The
        default is 21.

    Returns
    -------
    tuple of float
        ``(lon_min, lat_min, lon_max, lat_max)`` in degrees.

    Notes
    -----
    Densifying matters because straight edges in a polar stereographic
    projection curve in latitude/longitude; using the four corners alone would
    under-cover the ROI. Boxes that span the pole or the antimeridian are not
    handled.
    """
    xs = np.linspace(bbox['minx'], bbox['maxx'], n_edge)
    ys = np.linspace(bbox['miny'], bbox['maxy'], n_edge)
    edge_x = np.concatenate([xs, xs, np.full(n_edge, bbox['minx']), np.full(n_edge, bbox['maxx'])])
    edge_y = np.concatenate([np.full(n_edge, bbox['miny']), np.full(n_edge, bbox['maxy']), ys, ys])

    to_geo = pyproj.Transformer.from_crs(epsg, 4326, always_xy=True)
    lon, lat = to_geo.transform(edge_x, edge_y)
    return float(np.min(lon)), float(np.min(lat)), float(np.max(lon)), float(np.max(lat))


def intersect_bbox(a: dict[str, float], b: dict[str, float]) -> dict[str, float] | None:
    """Intersect two bounding boxes.

    Parameters
    ----------
    a, b : dict
        ``{'minx', 'miny', 'maxx', 'maxy'}`` in the same CRS.

    Returns
    -------
    dict or None
        The overlapping box, or ``None`` if the two do not overlap.
    """
    out = {'minx': max(a['minx'], b['minx']), 'miny': max(a['miny'], b['miny']),
           'maxx': min(a['maxx'], b['maxx']), 'maxy': min(a['maxy'], b['maxy'])}
    if out['minx'] >= out['maxx'] or out['miny'] >= out['maxy']:
        return None
    return out


def describe_bbox(bbox: dict[str, float]) -> str:
    """Return a one-line human-readable summary of a projected bounding box.

    Parameters
    ----------
    bbox : dict
        ``{'minx', 'miny', 'maxx', 'maxy'}`` in EPSG:3413 metres.

    Returns
    -------
    str
        Size in km and the geographic bounds.
    """
    width = (bbox['maxx'] - bbox['minx']) / 1e3
    height = (bbox['maxy'] - bbox['miny']) / 1e3
    lon0, lat0, lon1, lat1 = bbox_lonlat(bbox)
    return (f'{width:.1f} x {height:.1f} km  |  '
            f'lon {lon0:.3f} to {lon1:.3f}, lat {lat0:.3f} to {lat1:.3f}')


#: CMR granule-search endpoint and the provider that serves the GrIMP COGs.
CMR_GRANULE_URL = 'https://cmr.earthdata.nasa.gov/search/granules.json'
CMR_PROVIDER = 'NSIDC_CPRD'


def cmr_granule_urls(short_name: str,
                     version: str,
                     time_start: str,
                     time_end: str,
                     bounding_box: str,
                     filename_filter: str | None = None,
                     page_size: int = 500,
                     max_pages: int = 40,
                     verbose: bool = False) -> list[str]:
    """Query NASA CMR for granule file URLs, paginating with ``search-after``.

    Parameters
    ----------
    short_name : str
        Collection short name, e.g. ``'NSIDC-0725'``.
    version : str
        Collection version, e.g. ``'5'``.
    time_start, time_end : str
        ISO instants, e.g. ``'2018-01-01T00:00:01Z'``.
    bounding_box : str
        ``'lon_min,lat_min,lon_max,lat_max'`` in degrees.
    filename_filter : str, optional
        ``producer_granule_id`` pattern, e.g. ``'*'``. Omitted if None.
    page_size : int, optional
        Granules per request, max 2000. The default is 500.
    max_pages : int, optional
        Safety stop on the pagination loop. The default is 40.
    verbose : bool, optional
        Print each query. The default is False.

    Returns
    -------
    list of str
        File URLs for the matching granules, deduplicated by file name.

    Raises
    ------
    RuntimeError
        If CMR returns an error status.

    Notes
    -----
    This replaces ``grimpfunc.get_urls``, which pages with ``page_num`` and
    ``scroll=false``. CMR retired scrolling, so that call now returns HTTP 400
    and fails with ``KeyError: 'feed'`` for *every* search. The supported
    method is to echo the ``CMR-Search-After`` response header back as a
    request header until no token comes back.
    """
    from grimpfunc.cmr import cmr_filter_urls

    params = {
        'provider': CMR_PROVIDER,
        'short_name': short_name,
        'version': version,
        'page_size': int(page_size),
        'temporal[]': f'{time_start},{time_end}',
        'bounding_box[]': bounding_box,
        'sort_key[]': ['start_date', 'producer_granule_id'],
    }
    if filename_filter:
        params['producer_granule_id[]'] = filename_filter
        params['options[producer_granule_id][pattern]'] = 'true'

    urls: list[str] = []
    headers: dict[str, str] = {}
    for _ in range(max_pages):
        if verbose:
            print(f'CMR: {short_name} v{version} {bounding_box} '
                  f'{"(continued)" if headers else ""}')
        response = requests.get(CMR_GRANULE_URL, params=params,
                                headers=headers, timeout=120)
        if response.status_code != 200:
            raise RuntimeError(
                f'CMR search failed (HTTP {response.status_code}): '
                f'{str(response.text)[:200]}')
        payload = response.json()
        urls += cmr_filter_urls(payload)
        token = response.headers.get('CMR-Search-After')
        if not token or not payload.get('feed', {}).get('entry'):
            break
        headers['CMR-Search-After'] = token

    seen: set[str] = set()
    out = []
    for url in urls:
        name = url.split('/')[-1]
        if name not in seen:
            seen.add(name)
            out.append(str(url))
    return out


def _template_from_url(url: str) -> tuple[str, str] | None:
    """Convert one COG URL into a ``nisardev`` file-name template.

    Parameters
    ----------
    url : str
        Full URL of a single-band GrIMP GeoTIFF.

    Returns
    -------
    tuple of str or None
        ``(box_name, template)`` where ``template`` is the URL with the band
        token replaced by ``*`` and ``.tif`` stripped, and ``box_name`` is the
        GrIMP box (``''`` for ice-sheet-wide mosaics). ``None`` if the URL is
        not a recognised single-band velocity GeoTIFF (e.g. a browse image).

    Notes
    -----
    Only the band token is replaced, unlike ``cmrUrls.getCogs(replace=...)``
    which does a substring replace over the whole URL.
    """
    if not url.endswith('.tif'):
        return None
    head, _, name = url[:-len('.tif')].rpartition('/')
    parts = name.split('_')
    band_index = next((i for i, part in enumerate(parts) if part in VELOCITY_BANDS), None)
    if band_index is None:
        return None
    box = parts[1] if parts[0] in ('TSX', 'OPT', 'CSK') and len(parts) > 1 else ''
    parts[band_index] = '*'
    return box, f'{head}/{"_".join(parts)}'


def search_velocity_cogs(product: str,
                         bbox: dict[str, float],
                         first_date: str,
                         last_date: str,
                         max_version_bump: int = 4,
                         verbose: bool = False) -> dict[str, list[str]]:
    """Search NSIDC for velocity products covering an ROI and date range.

    Parameters
    ----------
    product : str
        NSIDC product code, e.g. ``'NSIDC-0725'``.
    bbox : dict
        ROI as ``{'minx', 'miny', 'maxx', 'maxy'}`` in EPSG:3413 metres.
    first_date, last_date : str
        Inclusive search range as ``'YYYY-MM-DD'``.
    max_version_bump : int, optional
        How many product versions above the registry value to try if a search
        returns nothing. The default is 4.
    verbose : bool, optional
        Print the CMR query URLs. The default is False.

    Returns
    -------
    dict
        Mapping of GrIMP box name (``''`` for mosaics) to a sorted list of COG
        templates. Empty if nothing matched.

    Notes
    -----
    The catalog query goes through :func:`cmr_granule_urls` rather than
    ``grimpfunc.get_urls``, which CMR's retirement of scroll pagination
    broke; grimpfunc is kept as a fallback.
    """
    info = product_info(product)
    for name, value in (('first_date', first_date), ('last_date', last_date)):
        try:
            datetime.strptime(value, '%Y-%m-%d')
        except (TypeError, ValueError):
            raise ValueError(f'{name} must be "YYYY-MM-DD", got {value!r}') from None

    lon0, lat0, lon1, lat1 = bbox_lonlat(bbox)
    bounding_box = f'{lon0:.4f},{lat0:.4f},{lon1:.4f},{lat1:.4f}'
    time_start = f'{first_date}T00:00:01Z'
    time_end = f'{last_date}T23:59:59Z'

    urls: list[str] = []
    for bump in range(max_version_bump + 1):
        version = str(int(info['version']) + bump)
        try:
            urls = cmr_granule_urls(product, version, time_start, time_end,
                                    bounding_box, '*', verbose=verbose)
        except RuntimeError:
            # Fall back to grimpfunc's own search, in case a future grimpfunc
            # fixes its pagination and this module's query drifts instead.
            urls = grimp.get_urls(product, version, time_start, time_end,
                                  bounding_box, None, '*', verbose=verbose)
        if urls:
            break

    groups: dict[str, set[str]] = defaultdict(set)
    for url in urls:
        parsed = _template_from_url(url)
        if parsed is not None:
            box, template = parsed
            groups[box].add(template)
    return {box: sorted(templates) for box, templates in sorted(groups.items())}


def summarize_search(bbox: dict[str, float],
                     first_date: str,
                     last_date: str,
                     products: list[str],
                     verbose: bool = False) -> pd.DataFrame:
    """Count the granules available for each product before downloading anything.

    Parameters
    ----------
    bbox : dict
        ROI as ``{'minx', 'miny', 'maxx', 'maxy'}`` in EPSG:3413 metres.
    first_date, last_date : str
        Inclusive search range as ``'YYYY-MM-DD'``.
    products : list of str
        NSIDC product codes to search.
    verbose : bool, optional
        Print the CMR query URLs. The default is False.

    Returns
    -------
    pandas.DataFrame
        One row per product and box, with the granule count and an example
        file name.
    """
    rows = []
    for product in products:
        groups = search_velocity_cogs(product, bbox, first_date, last_date, verbose=verbose)
        if not groups:
            rows.append({'product': product, 'label': product_info(product)['label'],
                         'box': '', 'n_granules': 0, 'example': ''})
            continue
        for box, templates in groups.items():
            rows.append({'product': product, 'label': product_info(product)['label'],
                         'box': box, 'n_granules': len(templates),
                         'example': templates[0].split('/')[-1]})
    return pd.DataFrame(rows)


def normalize_nodata(da: xr.DataArray) -> xr.DataArray:
    """Replace GrIMP no-data sentinels with NaN so NaN is the only no-data value.

    Parameters
    ----------
    da : xarray.DataArray
        Velocity array with a ``band`` dimension whose labels are GrIMP band
        names (``vx``, ``vy``, ``vv``, ``ex``, ``ey``, ``ev``, ``dT``).

    Returns
    -------
    xarray.DataArray
        ``float32`` copy in which every value at or below the band's sentinel,
        and every non-finite value, is NaN.

    Notes
    -----
    Whether a sentinel survives the read depends on whether the source COG
    declares ``nodata`` in its header, so this cannot be left to rasterio.
    Bands not in :data:`NO_DATA_SENTINELS` are only checked for finiteness.

    Thresholding alone is not enough for the derived speed bands in
    :data:`DERIVED_BANDS`: ``vv`` is computed as ``sqrt(vx**2 + vy**2)``, so a
    pair of unmasked ``-2e9`` sentinels becomes ``+2.83e9``, which passes any
    lower bound on speed. Those bands inherit their components' gaps instead.
    """
    bands = [str(b) for b in da['band'].values]
    thresholds = xr.DataArray(
        [NO_DATA_SENTINELS.get(band, -np.inf) for band in bands],
        coords={'band': da['band']}, dims='band',
    )
    cleaned = da.where(np.isfinite(da) & (da > thresholds))

    # Speed bands are derived as sqrt(x**2 + y**2), so an unmasked sentinel pair
    # yields sqrt(2) * 2e9 -- a large POSITIVE value that no threshold on the
    # speed itself can catch. Inherit the components' gaps instead.
    for derived, (band_x, band_y) in DERIVED_BANDS.items():
        if {derived, band_x, band_y} <= set(bands):
            components_ok = cleaned.sel(band=[band_x, band_y]).notnull().all('band')
            cleaned = cleaned.where((cleaned['band'] != derived) | components_ok)

    cleaned = cleaned.astype('float32')
    stale = [name for name in cleaned.coords if str(name) == '_FillValue']
    if stale:
        cleaned = cleaned.drop_vars(stale)

    cleaned.attrs = {k: v for k, v in da.attrs.items() if k not in ('nodata', '_FillValue')}
    cleaned.attrs['nodata_convention'] = 'NaN'
    return cleaned


def product_bounds(template: str,
                   bands: tuple[str, ...] = ('vx', 'vv', 'vy')) -> dict[str, float]:
    """Read the spatial extent of a product from its COG header.

    Parameters
    ----------
    template : str
        COG template (URL with ``*`` in place of the band token, no ``.tif``).
    bands : tuple of str, optional
        Band tokens to try, in order. The default is ``('vx', 'vv', 'vy')``.

    Returns
    -------
    dict
        ``{'minx', 'miny', 'maxx', 'maxy'}`` in the product's own CRS (EPSG:3413).

    Raises
    ------
    RuntimeError
        If none of ``bands`` could be opened.
    """
    last_error: Exception | None = None
    for band in bands:
        path = f"/vsicurl/?list_dir=no&url={template.replace('*', band)}.tif"
        try:
            with rasterio.open(path) as src:
                bounds = src.bounds
            return {'minx': float(bounds.left), 'miny': float(bounds.bottom),
                    'maxx': float(bounds.right), 'maxy': float(bounds.top)}
        except rasterio.errors.RasterioIOError as err:
            last_error = err
    raise RuntimeError(f'Could not read a header for {template}: {last_error}')


def load_velocity_series(templates: list[str],
                         bbox: dict[str, float] | None = None,
                         use_errors: bool = True,
                         use_dt: bool = False,
                         num_workers: int = 2,
                         verbose: bool = False) -> nisar.nisarVelSeries:
    """Lazily open a list of COG templates as a velocity series.

    Parameters
    ----------
    templates : list of str
        COG templates from :func:`search_velocity_cogs`.
    bbox : dict, optional
        Window to read, ``{'minx', 'miny', 'maxx', 'maxy'}`` in EPSG:3413
        metres. It is clipped to the product extent first, because a window
        reaching outside the raster reads garbage. ``None`` reads the full
        product.
    use_errors : bool, optional
        Include the ``ex``/``ey`` error bands (and the derived ``ev``). The
        default is True.
    use_dt : bool, optional
        Include the ``dT`` band, which only the Sentinel-1 mosaics have. The
        default is False.
    num_workers : int, optional
        Dask workers used for the download. NSIDC throttles simultaneous
        connections, so keep this small. The default is 2.
    verbose : bool, optional
        Passed through to ``nisarVelSeries``. The default is False.

    Returns
    -------
    nisardev.nisarVelSeries
        Series whose ``.subset`` is a lazy ``(time, band, y, x)`` array. No
        pixels are downloaded until ``loadRemote()`` or a write is triggered.
    """
    if not templates:
        raise ValueError('templates is empty - nothing to load')

    read_bbox = bbox
    if bbox is not None:
        read_bbox = intersect_bbox(bbox, product_bounds(templates[0]))
        if read_bbox is None:
            raise ValueError('Requested bbox does not overlap the product extent')

    series = nisar.nisarVelSeries(numWorkers=num_workers, verbose=verbose)
    series.readSeriesFromTiff(templates, bbox=read_bbox, url=True,
                              useVelocity=True, useErrors=use_errors, useDT=use_dt,
                              readSpeed=False, useStack=True)
    return series


def velocity_netcdf_path(out_dir: str | Path, prefix: str, product: str, box: str = '') -> Path:
    """Build the NetCDF path for one product (and box).

    Parameters
    ----------
    out_dir : str or Path
        Directory the file is written to.
    prefix : str
        Leading part of the file name, typically the glacier name.
    product : str
        NSIDC product code.
    box : str, optional
        GrIMP box name for NSIDC-0481/0646. The default is ``''``.

    Returns
    -------
    Path
        ``<out_dir>/<prefix>_<product>[_<box>].nc``
    """
    stem = f'{prefix}_{product}' + (f'_{box}' if box else '')
    return Path(out_dir) / f'{stem}.nc'


def save_velocity_netcdf(series: nisar.nisarVelSeries,
                         path: str | Path,
                         attrs: dict[str, object] | None = None,
                         complevel: int = 4) -> Path:
    """Write every band of a velocity series to a NetCDF file, NaN-masked.

    Parameters
    ----------
    series : nisardev.nisarVelSeries
        Series to write. Its ``.subset`` (the ROI, or the whole product if it
        was never subset) is what gets saved.
    path : str or Path
        Output ``.nc`` path. Parent directories are created; an existing file
        is replaced.
    attrs : dict, optional
        Extra global attributes. Values must be strings or numbers to survive
        the NetCDF round trip.
    complevel : int, optional
        zlib compression level, 0-9. The default is 4.

    Returns
    -------
    Path
        The file that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    velocity = normalize_nodata(series.subset).rename('velocity')
    velocity.attrs.update({
        'long_name': 'ice surface velocity components and errors',
        'units': 'm/yr (vx, vy, vv, ex, ey, ev); days (dT)',
    })
    dataset = velocity.to_dataset()
    dataset.attrs.update({
        'title': 'GrIMP/NSIDC ice velocity subset',
        'source_library': 'grimpfunc + nisardev (github.com/fastice)',
        'nodata_convention': 'NaN (GrIMP sentinels -2e9 / -1 are masked on write and on read)',
        'bands': ', '.join(str(b) for b in velocity['band'].values),
        'epsg': GRID_EPSG,
        'created': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    })
    if attrs:
        dataset.attrs.update(attrs)

    encoding = {'velocity': {'zlib': True, 'complevel': complevel,
                             'dtype': 'float32', '_FillValue': np.float32(np.nan)}}
    if path.exists():
        path.unlink()
    dataset.to_netcdf(path, encoding=encoding)
    return path


def download_velocity_netcdfs(center: tuple[float, float],
                              dx_m: float,
                              dy_m: float | None = None,
                              first_date: str = '2015-01-01',
                              last_date: str | None = None,
                              products: list[str] | tuple[str, ...] = ('NSIDC-0725',),
                              out_dir: str | Path = 'Subsets',
                              prefix: str = 'glacier',
                              center_crs: int | str = GRID_EPSG,
                              use_errors: bool = True,
                              num_workers: int = 2,
                              overwrite: bool = False,
                              load_into_memory: bool = True,
                              retries: int = 2,
                              verbose: bool = True) -> list[Path]:
    """Search, download and save NSIDC velocity data for an ROI, one file per product.

    This is the end-to-end entry point: it takes the ROI centre, the distance to
    add in x and y, the date range and the product codes, and leaves one NetCDF
    per product type (per box for NSIDC-0481/0646) on disk.

    Parameters
    ----------
    center : tuple of float
        Centre of the ROI as ``(x, y)`` in ``center_crs``.
    dx_m : float
        Metres to add either side of the centre in x.
    dy_m : float, optional
        Metres to add either side of the centre in y. Defaults to ``dx_m``.
    first_date, last_date : str
        Inclusive NSIDC search range as ``'YYYY-MM-DD'``. ``last_date`` defaults
        to today.
    products : sequence of str, optional
        NSIDC product codes. The default is ``('NSIDC-0725',)``.
    out_dir : str or Path, optional
        Output directory, created if needed. The default is ``'Subsets'``.
    prefix : str, optional
        Leading part of each file name, typically the glacier name.
    center_crs : int or str, optional
        CRS of ``center``. The default is EPSG:3413.
    use_errors : bool, optional
        Download the ``ex``/``ey`` error bands. The default is True.
    num_workers : int, optional
        Dask workers for the download. The default is 2.
    overwrite : bool, optional
        Re-download files that already exist. The default is False.
    load_into_memory : bool, optional
        Download the whole ROI before writing (shows a progress bar). Set False
        to stream lazily through ``to_netcdf`` for a large ROI. The default is True.
    retries : int, optional
        Extra attempts per product if NSIDC drops a connection mid-download.
        NSIDC limits simultaneous connections, so a long download can fail with
        an HTTP 404/503 on a single tile even though every file is present.
        The default is 2.
    verbose : bool, optional
        Print progress. The default is True.

    Returns
    -------
    list of Path
        The NetCDF files written or already present, in product order.
    """
    if last_date is None:
        last_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    bbox = roi_bbox(center, dx_m, dy_m, center_crs=center_crs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f'ROI: {describe_bbox(bbox)}')

    roi_record = {'prefix': prefix, 'center': list(center), 'center_crs': str(center_crs),
                  'dx_m': dx_m, 'dy_m': dx_m if dy_m is None else dy_m,
                  'first_date': first_date, 'last_date': last_date,
                  'products': list(products), 'bbox_epsg3413': bbox}
    (out_dir / f'{prefix}_roi.json').write_text(json.dumps(roi_record, indent=2))

    written: list[Path] = []
    for product in products:
        info = product_info(product)
        groups = search_velocity_cogs(product, bbox, first_date, last_date)
        if not groups:
            print(f'{product}: no granules for this ROI and date range')
            continue

        for box, templates in groups.items():
            path = velocity_netcdf_path(out_dir, prefix, product, box)
            tag = f'{product}{" " + box if box else ""}'
            if path.exists() and not overwrite:
                print(f'{tag}: {path.name} already exists, skipping (overwrite=True to redo)')
                written.append(path)
                continue

            if verbose:
                print(f'{tag}: {len(templates)} granules -> {path.name}')

            series = None
            for attempt in range(retries + 1):
                try:
                    series = load_velocity_series(templates, bbox=bbox, use_errors=use_errors,
                                                  use_dt=bool(info['has_dt']),
                                                  num_workers=num_workers)
                    if load_into_memory:
                        series.loadRemote()
                    break
                except (ValueError, RuntimeError) as err:
                    print(f'{tag}: skipped ({err})')
                    series = None
                    break
                except rasterio.errors.RasterioIOError as err:
                    series = None
                    if attempt == retries:
                        print(f'{tag}: gave up after {attempt + 1} attempts ({err})')
                        break
                    pause = 10 * (attempt + 1)
                    print(f'{tag}: NSIDC read failed ({err}); retrying in {pause}s')
                    time.sleep(pause)
            if series is None:
                continue

            save_velocity_netcdf(series, path, attrs={
                'nsidc_product': product,
                'product_label': str(info['label']),
                'grimp_box': box,
                'glacier': prefix,
                'n_granules': len(templates),
                'search_first_date': first_date,
                'search_last_date': last_date,
                'roi_center_x': float(center[0]),
                'roi_center_y': float(center[1]),
                'roi_center_crs': str(center_crs),
                'roi_bbox_epsg3413': json.dumps(bbox),
            })
            written.append(path)
            if verbose:
                size_mb = path.stat().st_size / 1e6
                print(f'{tag}: wrote {path} ({size_mb:.1f} MB)')

    return written


def resolve_netcdf_paths(files: str | Path | list[str | Path], pattern: str = '*.nc') -> list[Path]:
    """Normalise a directory, single file, or list of files into a list of paths.

    Parameters
    ----------
    files : str, Path, or list
        A directory to glob, a single ``.nc`` file, or a list of either.
    pattern : str, optional
        Glob applied to directories. The default is ``'*.nc'``.

    Returns
    -------
    list of Path
        Sorted, de-duplicated, existing ``.nc`` files.
    """
    candidates = files if isinstance(files, (list, tuple)) else [files]
    paths: list[Path] = []
    for item in candidates:
        path = Path(item)
        paths.extend(sorted(path.glob(pattern)) if path.is_dir() else [path])

    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f'NetCDF file(s) not found: {", ".join(missing)}')
    return sorted(set(paths))


def read_velocity_netcdf(path: str | Path) -> nisar.nisarVelSeries:
    """Read a saved velocity NetCDF back into a series, with NaN as no-data.

    Parameters
    ----------
    path : str or Path
        A file written by :func:`save_velocity_netcdf` (or by the GrIMP
        subsetter notebook).

    Returns
    -------
    nisardev.nisarVelSeries
        Series whose ``.subset`` has been through :func:`normalize_nodata`.
    """
    series = nisar.nisarVelSeries(verbose=False)
    series.readSeriesFromNetCDF(str(path))
    series.subset = normalize_nodata(series.subset)
    series.xr = series.subset
    series.variables = [str(band) for band in series.subset['band'].values]
    series._mapVariables()
    return series


def sample_velocity_at_point(series: nisar.nisarVelSeries,
                             x: float,
                             y: float,
                             method: str = 'linear') -> pd.DataFrame:
    """Extract every band of a velocity series at one point, for all times.

    Parameters
    ----------
    series : nisardev.nisarVelSeries
        Series to sample.
    x, y : float
        Point coordinates in EPSG:3413 metres.
    method : {'linear', 'nearest'}, optional
        ``'linear'`` bilinearly interpolates and returns NaN if any of the four
        surrounding pixels is no-data; ``'nearest'`` takes the containing pixel,
        which keeps more points near ice margins. The default is ``'linear'``.

    Returns
    -------
    pandas.DataFrame
        One row per time step: ``time``, ``time1``, ``time2`` and one column per
        band. Points outside the grid give all-NaN rows.
    """
    grid = series.subset
    if method == 'nearest':
        point = grid.sel(x=x, y=y, method='nearest')
    elif method == 'linear':
        point = grid.interp(x=x, y=y)
    else:
        raise ValueError(f"method must be 'linear' or 'nearest', got {method!r}")

    frame: dict[str, object] = {'time': pd.to_datetime(point['time'].values)}
    for bound in ('time1', 'time2'):
        if bound in point.coords:
            frame[bound] = pd.to_datetime(point[bound].values)
    for band in [str(b) for b in point['band'].values]:
        frame[band] = np.asarray(point.sel(band=band).values, dtype='float64').ravel()
    return pd.DataFrame(frame)


def velocity_time_series(files: str | Path | list[str | Path],
                         x: float,
                         y: float,
                         coord_crs: int | str = GRID_EPSG,
                         method: str = 'linear',
                         drop_empty: bool = True) -> pd.DataFrame:
    """Build a velocity time series at a coordinate from saved NetCDF files.

    Parameters
    ----------
    files : str, Path, or list
        Directory of ``.nc`` files, a single file, or a list.
    x, y : float
        Point coordinates in ``coord_crs``. For a geographic CRS pass
        ``(longitude, latitude)``.
    coord_crs : int or str, optional
        CRS of ``x``, ``y``. The default is EPSG:3413.
    method : {'linear', 'nearest'}, optional
        Sampling method, see :func:`sample_velocity_at_point`.
    drop_empty : bool, optional
        Drop rows where every band is NaN. The default is True.

    Returns
    -------
    pandas.DataFrame
        Columns ``source``, ``product``, ``box``, ``time``, ``time1``, ``time2``
        and one per band, sorted by source then time. Files whose grid does not
        contain the point are skipped with a warning.
    """
    xp, yp = float(x), float(y)
    if pyproj.CRS.from_user_input(coord_crs).to_epsg() != GRID_EPSG:
        to_grid = pyproj.Transformer.from_crs(coord_crs, GRID_EPSG, always_xy=True)
        xp, yp = to_grid.transform(xp, yp)

    frames = []
    for path in resolve_netcdf_paths(files):
        series = read_velocity_netcdf(path)
        grid = series.subset
        inside = (float(grid.x.min()) <= xp <= float(grid.x.max())
                  and float(grid.y.min()) <= yp <= float(grid.y.max()))
        if not inside:
            warnings.warn(f'{path.name}: ({xp:.0f}, {yp:.0f}) is outside the grid, skipping',
                          stacklevel=2)
            continue

        frame = sample_velocity_at_point(series, xp, yp, method=method)
        with xr.open_dataset(path) as dataset:
            attrs = dict(dataset.attrs)
        frame['source'] = path.stem
        frame['product'] = attrs.get('nsidc_product', 'unknown')
        frame['box'] = attrs.get('grimp_box', '')
        frames.append(frame)

    columns = ['source', 'product', 'box', 'time', 'time1', 'time2', 'vx', 'vy', 'vv']
    if not frames:
        return pd.DataFrame(columns=columns)

    series_df = pd.concat(frames, ignore_index=True)
    if drop_empty:
        bands = [c for c in series_df.columns if c in NO_DATA_SENTINELS]
        series_df = series_df.dropna(subset=bands, how='all')

    lead = [c for c in columns if c in series_df.columns]
    ordered = lead + [c for c in series_df.columns if c not in lead]
    return (series_df[ordered]
            .sort_values(['source', 'time'])
            .reset_index(drop=True))


def _break_on_gaps(times: np.ndarray,
                   values: np.ndarray,
                   max_gap_days: float) -> tuple[np.ndarray, np.ndarray]:
    """Insert a NaN between samples separated by more than ``max_gap_days``.

    Parameters
    ----------
    times : numpy.ndarray
        ``datetime64`` sample times, sorted ascending.
    values : numpy.ndarray
        Values at those times.
    max_gap_days : float
        Gap in days beyond which the line should break.

    Returns
    -------
    tuple of numpy.ndarray
        ``(times, values)`` with NaN values inserted at gap midpoints so
        matplotlib lifts the pen.
    """
    if len(times) < 2:
        return times, values
    out_times, out_values = [times[0]], [values[0]]
    for i in range(1, len(times)):
        gap_days = (times[i] - times[i - 1]) / np.timedelta64(1, 'D')
        if gap_days > max_gap_days:
            out_times.append(times[i - 1] + (times[i] - times[i - 1]) / 2)
            out_values.append(np.nan)
        out_times.append(times[i])
        out_values.append(values[i])
    return np.array(out_times), np.array(out_values)


def plot_velocity_time_series(ts: pd.DataFrame,
                              band: str = 'vv',
                              ax: plt.Axes | None = None,
                              max_gap_days: float = 365.0,
                              show_errors: bool = True,
                              show_span: bool = False,
                              title: str | None = None,
                              ylabel: str | None = None,
                              figsize: tuple[float, float] = (11.0, 5.0)) -> plt.Axes:
    """Plot a velocity time series, one coloured series per source file.

    Parameters
    ----------
    ts : pandas.DataFrame
        Output of :func:`velocity_time_series`.
    band : str, optional
        Band to plot: ``'vv'`` (speed), ``'vx'``, ``'vy'`` or ``'dT'``. The
        default is ``'vv'``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is made if omitted.
    max_gap_days : float, optional
        Break the connecting line across gaps longer than this. The default is 365.
    show_errors : bool, optional
        Draw vertical error bars from the matching error band. The default is True.
    show_span : bool, optional
        Draw horizontal bars spanning each granule's ``time1``-``time2`` window.
        The default is False.
    title, ylabel : str, optional
        Plot labels. Sensible defaults are used if omitted.
    figsize : tuple of float, optional
        Figure size when ``ax`` is None. The default is ``(11, 5)``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    if band not in ts.columns:
        raise ValueError(f'band {band!r} is not in the time series; have {list(ts.columns)}')
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    error_band = {'vv': 'ev', 'vx': 'ex', 'vy': 'ey'}.get(band)
    plotted = 0
    for index, (source, group) in enumerate(ts.groupby('source', sort=True)):
        valid = group.sort_values('time')
        valid = valid[np.isfinite(valid[band])]
        if valid.empty:
            continue

        color = f'C{index % 10}'
        times = valid['time'].to_numpy()
        values = valid[band].to_numpy()

        yerr = None
        if show_errors and error_band in valid.columns:
            yerr = np.where(np.isfinite(valid[error_band].to_numpy()),
                            valid[error_band].to_numpy(), np.nan)
        xerr = None
        if show_span and {'time1', 'time2'} <= set(valid.columns):
            # x is datetime64, so matplotlib needs the spans as timedelta64.
            zero = np.timedelta64(0, 'ns')
            lead = (valid['time'] - valid['time1']).to_numpy()
            trail = (valid['time2'] - valid['time']).to_numpy()
            xerr = np.vstack([np.where(np.isnat(lead), zero, lead),
                              np.where(np.isnat(trail), zero, trail)])

        line_t, line_v = _break_on_gaps(times, values, max_gap_days)
        ax.plot(line_t, line_v, '-', color=color, lw=1.2, alpha=0.85, zorder=1)
        ax.errorbar(times, values, yerr=yerr, xerr=xerr, fmt='o', ms=4,
                    color=color, ecolor=color, elinewidth=0.8, capsize=0,
                    alpha=0.9, zorder=2, label=f'{source} (n={len(valid)})')
        plotted += 1

    if plotted == 0:
        ax.text(0.5, 0.5, f'No finite {band} values at this point',
                transform=ax.transAxes, ha='center', va='center')

    ax.set_xlabel('date')
    ax.set_ylabel(ylabel or ('time separation (days)' if band == 'dT' else f'{band} (m/yr)'))
    ax.set_title(title or f'Ice velocity ({band})')
    ax.grid(alpha=0.3)
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    if plotted:
        ax.legend(fontsize=9, loc='best')
    return ax


def plot_glacier_velocity(files: str | Path | list[str | Path],
                          x: float,
                          y: float,
                          band: str = 'vv',
                          coord_crs: int | str = GRID_EPSG,
                          method: str = 'linear',
                          name: str | None = None,
                          **plot_kwargs) -> tuple[pd.DataFrame, plt.Axes]:
    """Read saved NetCDF files at a coordinate and plot the velocity time series.

    Convenience wrapper around :func:`velocity_time_series` and
    :func:`plot_velocity_time_series`.

    Parameters
    ----------
    files : str, Path, or list
        Directory of ``.nc`` files, a single file, or a list.
    x, y : float
        Point coordinates in ``coord_crs``.
    band : str, optional
        Band to plot. The default is ``'vv'``.
    coord_crs : int or str, optional
        CRS of ``x``, ``y``. The default is EPSG:3413.
    method : {'linear', 'nearest'}, optional
        Sampling method. The default is ``'linear'``.
    name : str, optional
        Glacier name used in the plot title.
    **plot_kwargs
        Passed to :func:`plot_velocity_time_series`.

    Returns
    -------
    tuple
        ``(time_series_dataframe, axes)``.
    """
    ts = velocity_time_series(files, x, y, coord_crs=coord_crs, method=method)
    label = name or 'glacier'
    title = plot_kwargs.pop('title', f'{label} ice velocity ({band}) at ({x:,.0f}, {y:,.0f})')
    ax = plot_velocity_time_series(ts, band=band, title=title, **plot_kwargs)
    return ts, ax


def nodata_report(files: str | Path | list[str | Path]) -> pd.DataFrame:
    """Audit the no-data convention in saved velocity NetCDF files.

    Reads each file raw, without :func:`normalize_nodata`, so any sentinel that
    slipped through shows up.

    Parameters
    ----------
    files : str, Path, or list
        Directory of ``.nc`` files, a single file, or a list.

    Returns
    -------
    pandas.DataFrame
        One row per file and band: ``n_values``, ``nan_fraction``,
        ``n_sentinel_left``, ``min``, ``max``. ``n_sentinel_left`` must be 0.
    """
    rows = []
    for path in resolve_netcdf_paths(files):
        with xr.open_dataset(path) as dataset:
            name = ('velocity' if 'velocity' in dataset.data_vars
                    else next(v for v in dataset.data_vars if v != 'spatial_ref'))
            data = dataset[name]
            for band in [str(b) for b in data['band'].values]:
                values = np.asarray(data.sel(band=band).values, dtype='float64')
                finite = np.isfinite(values)
                sentinel = NO_DATA_SENTINELS.get(band)
                n_sentinel = int(np.sum(finite & (values <= sentinel))) if sentinel is not None else 0
                rows.append({
                    'file': path.name,
                    'band': band,
                    'n_values': int(values.size),
                    'nan_fraction': float(np.mean(~finite)),
                    'n_sentinel_left': n_sentinel,
                    'min': float(np.min(values[finite])) if finite.any() else np.nan,
                    'max': float(np.max(values[finite])) if finite.any() else np.nan,
                })
    return pd.DataFrame(rows)


def matlab_glacier_center(mat_file: str | Path,
                          x_key: str = 'x',
                          y_key: str = 'y',
                          name_key: str = 'glrname') -> tuple[str, tuple[float, float]]:
    """Read a MATLAB flux-gate file and return the glacier name and ROI centre.

    Parameters
    ----------
    mat_file : str or Path
        ``.mat`` file holding gate coordinates in EPSG:3413 metres.
    x_key, y_key : str, optional
        Variable names for the gate coordinates. The defaults are ``'x'``/``'y'``.
    name_key : str, optional
        Variable holding the glacier name. Falls back to the file stem if absent.

    Returns
    -------
    tuple
        ``(name, (x_center, y_center))`` where the centre is the midpoint of the
        gate's bounding box in EPSG:3413 metres.
    """
    mat = scipy.io.loadmat(str(mat_file))
    for key in (x_key, y_key):
        if key not in mat:
            raise KeyError(f'{mat_file} has no variable {key!r}; found {sorted(mat)}')

    x = np.asarray(mat[x_key], dtype='float64').ravel()
    y = np.asarray(mat[y_key], dtype='float64').ravel()

    name = Path(mat_file).stem
    if name_key in mat:
        raw = np.asarray(mat[name_key]).ravel()
        if raw.size:
            name = str(raw[0]).strip().removesuffix('.mat')

    return name, (float((x.min() + x.max()) / 2), float((y.min() + y.max()) / 2))


def matlab_glacier_centers(mat_dir: str | Path, pattern: str = '*.mat') -> pd.DataFrame:
    """Collect ROI centres for every MATLAB flux-gate file in a directory.

    Parameters
    ----------
    mat_dir : str or Path
        Directory of ``.mat`` gate files.
    pattern : str, optional
        Glob pattern. The default is ``'*.mat'``.

    Returns
    -------
    pandas.DataFrame
        Columns ``name``, ``x``, ``y``, ``file``, sorted by name.
    """
    rows = []
    for path in sorted(Path(mat_dir).glob(pattern)):
        try:
            name, (x, y) = matlab_glacier_center(path)
        except (KeyError, ValueError) as err:
            warnings.warn(f'{path.name}: {err}', stacklevel=2)
            continue
        rows.append({'name': name, 'x': x, 'y': y, 'file': str(path)})
    return pd.DataFrame(rows).sort_values('name').reset_index(drop=True)


#: Greenland-wide bounding box in EPSG:3413 metres, used for the overview map
#: and for catalog searches that are not tied to one glacier.
GREENLAND_BBOX: dict[str, float] = {'minx': -680000.0, 'miny': -3380000.0,
                                    'maxx': 880000.0, 'maxy': -620000.0}

#: Root of the local glacier data collection, overridable with
#: ``GLACIER_DATA_HOME``. Only the two files below are read from it.
DATA_HOME = Path(os.environ.get(
    'GLACIER_DATA_HOME', '~/Documents/GIMP_Working/RECALC_discharge')).expanduser()

#: Named flux gates used to name an ROI: King et al. (2020), 234 Greenland
#: glaciers, gate vertices in EPSG:3413. Override with ``GLACIER_GATES_FILE``.
GATES_FILE = Path(os.environ.get(
    'GLACIER_GATES_FILE', DATA_HOME / 'King2020_GreenlandGateCoord.mat')).expanduser()

#: BedMachine v5 (MATLAB v7.3) for the coastline/ice-margin outline. Its
#: ``mask`` is the standard classification 0=ocean, 1=ice-free land,
#: 2=grounded ice, 3=floating ice, 4=non-Greenland land. Override with
#: ``BEDMACHINE_FILE``.
BEDMACHINE_FILE = Path(os.environ.get(
    'BEDMACHINE_FILE', DATA_HOME / 'BedMachine_v5.mat')).expanduser()

#: Where the decimated overview map and outline are cached, so the Greenland
#: picker is slow only the first time.
CACHE_DIR = Path(os.environ.get(
    'GLACIER_READER_CACHE', Path(__file__).resolve().parent / '.cache'))


def greenland_outline(mask_file: str | Path | None = None,
                      stride: int = 20,
                      cache: bool = True) -> list[tuple[np.ndarray, np.ndarray]]:
    """Coastline and ice-margin outline of Greenland, as polylines.

    Contours the BedMachine ``mask`` between ocean and everything else, which
    gives the standard coastline that ice-sheet maps are drawn on. The result
    is cached, so the (large) mask is read once per ``stride``.

    Parameters
    ----------
    mask_file : str or Path, optional
        BedMachine v5 MATLAB v7.3 file. The default is :data:`BEDMACHINE_FILE`.
    stride : int, optional
        Decimation of the 150 m mask before contouring; 20 gives a 3 km
        outline, which is plenty for an overview map. The default is 20.
    cache : bool, optional
        Read/write the cached outline in :data:`CACHE_DIR`. The default is True.

    Returns
    -------
    list of tuple of numpy.ndarray
        One ``(x, y)`` pair per contour segment, in EPSG:3413 metres. Empty if
        the mask file is unavailable - the overview map then just omits the
        outline rather than failing.
    """
    cache_path = CACHE_DIR / f'greenland_outline_s{stride}.npz'
    if cache and cache_path.exists():
        with np.load(cache_path) as z:
            return [(z[f'x{i}'], z[f'y{i}']) for i in range(int(z['n']))]

    path = Path(mask_file) if mask_file else BEDMACHINE_FILE
    if not path.exists():
        warnings.warn(f'No BedMachine mask at {path}; the overview map will have '
                      'no coastline. Set BEDMACHINE_FILE or pass mask_file=.',
                      stacklevel=2)
        return []

    import h5py

    with h5py.File(path, 'r') as h:
        grp = h['greenland']
        mask = np.asarray(grp['mask'][::stride, ::stride], dtype='int8')
        x = np.asarray(grp['x'][:], dtype='float64').ravel()[::stride]
        y = np.asarray(grp['y'][:], dtype='float64').ravel()[::stride]

    # 0 = ocean, everything else is land/ice: the 0.5 contour is the coast.
    fig, ax = plt.subplots()
    try:
        cs = ax.contour(x, y, (mask > 0).astype('float32'), levels=[0.5])
        segments = [(p[:, 0], p[:, 1]) for p in cs.allsegs[0] if len(p) > 3]
    finally:
        plt.close(fig)

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path, n=len(segments),
            **{f'{ax_name}{i}': arr
               for i, (sx, sy) in enumerate(segments)
               for ax_name, arr in (('x', sx), ('y', sy))},
        )
    return segments


def greenland_mosaic_overview(product: str = 'NSIDC-0725',
                              first_date: str = '2021-01-01',
                              last_date: str | None = None,
                              max_px: int = 1200,
                              band: str = 'vv',
                              cache: bool = True,
                              verbose: bool = True) -> dict[str, object] | None:
    """Download a decimated Greenland-wide speed mosaic for the overview map.

    Reads one annual-mosaic COG at a reduced resolution straight from NSIDC -
    only the decimated pixels cross the network, so this is a few seconds, not
    a full mosaic download. The result is cached.

    Parameters
    ----------
    product : str, optional
        Mosaic product to use; must be one of the ice-sheet-wide (non-boxed)
        products. The default is ``'NSIDC-0725'`` (annual).
    first_date, last_date : str
        Search window as ``'YYYY-MM-DD'``; the most recent matching granule is
        used. ``last_date`` defaults to today.
    max_px : int, optional
        Longest side of the decimated array. The default is 1200.
    band : str, optional
        Band to read. The default is ``'vv'`` (speed).
    cache : bool, optional
        Read/write the cached array in :data:`CACHE_DIR`. The default is True.
    verbose : bool, optional
        Print what is being read. The default is True.

    Returns
    -------
    dict or None
        ``{'speed': 2-D array, 'extent': (minx, maxx, miny, maxy), 'source':
        str}``, or ``None`` if nothing could be read (no granule, no
        credentials, no network) - the picker then falls back to the outline
        alone.

    Raises
    ------
    KeyError
        If ``product`` is a boxed (per-glacier) product rather than a mosaic.
    """
    if product_info(product)['boxed']:
        raise KeyError(f'{product} is a per-glacier product; the overview needs a mosaic')

    cache_path = CACHE_DIR / f'greenland_overview_{product}_{band}_{max_px}.npz'
    if cache and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as z:
            return {'speed': z['speed'], 'extent': tuple(z['extent']),
                    'source': str(z['source'])}

    last_date = last_date or datetime.now(timezone.utc).strftime('%Y-%m-%d')
    try:
        groups = search_velocity_cogs(product, GREENLAND_BBOX, first_date, last_date)
    except Exception as err:  # noqa: BLE001 - offline / CMR down: fall back
        warnings.warn(f'Catalog search failed ({err}); overview map will have no '
                      'velocity backdrop.', stacklevel=2)
        return None
    templates = sorted(t for group in groups.values() for t in group)
    if not templates:
        warnings.warn(f'No {product} granule between {first_date} and {last_date}; '
                      'overview map will have no velocity backdrop.', stacklevel=2)
        return None

    template = templates[-1]  # most recent by name (dates sort lexically)
    url = f"/vsicurl/?list_dir=no&url={template.replace('*', band)}.tif"
    if verbose:
        print(f'overview: reading {template.split("/")[-1]} at <= {max_px} px')
    try:
        with rasterio.open(url) as src:
            scale = max(src.width, src.height) / max_px
            shape = (max(1, int(src.height / scale)), max(1, int(src.width / scale)))
            speed = src.read(1, out_shape=shape).astype('float32')
            bounds = src.bounds
    except rasterio.errors.RasterioIOError as err:
        warnings.warn(f'Could not read the mosaic ({err}); is the Earthdata login '
                      'done? Overview map will have no velocity backdrop.', stacklevel=2)
        return None

    # Decimated reads average over sentinels, so threshold generously.
    speed[~np.isfinite(speed)] = np.nan
    speed[speed <= 0] = np.nan
    speed[speed > 2.0e4] = np.nan

    out = {'speed': speed,
           'extent': (float(bounds.left), float(bounds.right),
                      float(bounds.bottom), float(bounds.top)),
           'source': template.split('/')[-1]}
    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, speed=out['speed'],
                            extent=np.asarray(out['extent']), source=out['source'])
    return out


def plot_greenland_overview(ax: plt.Axes | None = None,
                            overview: dict[str, object] | None | str = 'fetch',
                            gates: pd.DataFrame | None = None,
                            show_gates: bool = True,
                            figsize: tuple[float, float] = (7.0, 10.0),
                            vmin: float = 10.0,
                            vmax: float = 3000.0,
                            title: str = 'Greenland ice speed',
                            **overview_kwargs) -> plt.Axes:
    """Draw the Greenland overview: speed mosaic over the standard outline.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is made if omitted.
    overview : dict or None, optional
        Output of :func:`greenland_mosaic_overview`. Fetched when left at the
        default; pass ``None`` explicitly to draw the outline alone.
    gates : pandas.DataFrame, optional
        Named gates from :func:`load_glacier_gates`, drawn as small markers so
        you can see which glacier you are selecting. Loaded if omitted.
    show_gates : bool, optional
        Draw the gate markers. The default is True.
    figsize : tuple of float, optional
        Figure size when ``ax`` is None. The default is ``(7, 10)``.
    vmin, vmax : float, optional
        Speed colour limits in m/yr, log-scaled. The defaults are 10 and 3000.
    title : str, optional
        Axes title. The default is ``'Greenland ice speed'``.
    **overview_kwargs
        Passed to :func:`greenland_mosaic_overview`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on, in EPSG:3413 metres.
    """
    from matplotlib.colors import LogNorm

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if isinstance(overview, str):          # the 'fetch' sentinel
        overview = greenland_mosaic_overview(**overview_kwargs)

    if overview is not None:
        im = ax.imshow(overview['speed'], extent=overview['extent'], origin='upper',
                       cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax),
                       interpolation='nearest')
        cb = ax.figure.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cb.set_label('speed (m/yr)')

    for sx, sy in greenland_outline():
        ax.plot(sx, sy, '-', color='0.35', lw=0.6, zorder=3)

    if show_gates:
        if gates is None:
            gates = load_glacier_gates()
        if not gates.empty:
            ax.plot(gates['x'], gates['y'], '.', color='crimson', ms=2.5,
                    zorder=4, label=f'{len(gates)} named gates')
            ax.legend(loc='upper right', fontsize=8)

    ax.set_xlim(GREENLAND_BBOX['minx'], GREENLAND_BBOX['maxx'])
    ax.set_ylim(GREENLAND_BBOX['miny'], GREENLAND_BBOX['maxy'])
    ax.set_aspect('equal')
    # Data stay in metres (so a selection needs no conversion); only the tick
    # LABELS are in km, because metre ticks collide at this width.
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(mticker.MaxNLocator(5))
        axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v / 1e3:,.0f}'))
    ax.set_xlabel('x (km, EPSG:3413)')
    ax.set_ylabel('y (km, EPSG:3413)')
    ax.set_title(title)
    return ax


class GreenlandRoiSelector:
    """Pick an ROI by dragging a box on the Greenland overview map.

    Use when the ROI is unknown: the annual velocity mosaic over the standard
    Greenland outline shows where the fast ice is, and the named flux gates are
    marked, so a box can be placed without knowing any coordinates in advance.

    Needs an interactive matplotlib backend - ``%matplotlib widget`` in a
    notebook, or any GUI backend in a script.

    Parameters
    ----------
    snap_km : float, optional
        Round the selected box corners to this many km. The default is 0.1
        (100 m, the grid spacing of the mosaics).
    gates : pandas.DataFrame, optional
        Named gates for the suggestion; loaded if omitted.
    **plot_kwargs
        Passed to :func:`plot_greenland_overview`.

    Attributes
    ----------
    bbox : dict or None
        ``{'minx', 'miny', 'maxx', 'maxy'}`` of the current selection.
    center : tuple of float or None
        ``(x, y)`` centre of the current selection, EPSG:3413 metres.
    dx_m, dy_m : float or None
        Half-width and half-height of the selection in metres, ready to pass
        to :func:`download_velocity_netcdfs`.
    name : str
        Suggested glacier name for the current selection (``''`` if none).

    Examples
    --------
    >>> sel = GreenlandRoiSelector()      # drag a box on the figure
    >>> sel.center, sel.dx_m, sel.name
    >>> out = sel.download(prefix=None)   # prompts for the file name
    """

    def __init__(self, snap_km: float = 0.1, gates: pd.DataFrame | None = None,
                 **plot_kwargs):
        from matplotlib.widgets import RectangleSelector

        self.snap_m = float(snap_km) * 1e3
        self.gates = load_glacier_gates() if gates is None else gates
        self.bbox: dict[str, float] | None = None
        self.matches: pd.DataFrame | None = None
        self.name: str = ''
        plot_kwargs.setdefault('title', 'Greenland ice speed - drag a box to select an ROI')
        self.ax = plot_greenland_overview(gates=self.gates, **plot_kwargs)
        self._marker, = self.ax.plot([], [], 'w*', ms=13, mec='k', zorder=6)
        self._readout = self.ax.text(
            0.02, 0.02, 'drag a box to select an ROI', transform=self.ax.transAxes,
            fontsize=9, va='bottom', ha='left',
            bbox=dict(fc='white', ec='0.6', alpha=0.85))
        self._sel = RectangleSelector(self.ax, self._on_select, useblit=True,
                                      interactive=True, button=[1])

    @property
    def center(self) -> tuple[float, float] | None:
        """Centre of the selection as ``(x, y)`` in EPSG:3413 metres."""
        if self.bbox is None:
            return None
        return ((self.bbox['minx'] + self.bbox['maxx']) / 2,
                (self.bbox['miny'] + self.bbox['maxy']) / 2)

    @property
    def dx_m(self) -> float | None:
        """Half-width of the selection in metres (the ``dx_m`` argument)."""
        return None if self.bbox is None else (self.bbox['maxx'] - self.bbox['minx']) / 2

    @property
    def dy_m(self) -> float | None:
        """Half-height of the selection in metres (the ``dy_m`` argument)."""
        return None if self.bbox is None else (self.bbox['maxy'] - self.bbox['miny']) / 2

    def _on_select(self, eclick, erelease) -> None:
        """Store the dragged box, snapped, and update the on-figure readout."""
        x1, x2 = sorted((eclick.xdata, erelease.xdata))
        y1, y2 = sorted((eclick.ydata, erelease.ydata))
        snap = self.snap_m
        self.bbox = {'minx': float(np.floor(x1 / snap) * snap),
                     'miny': float(np.floor(y1 / snap) * snap),
                     'maxx': float(np.ceil(x2 / snap) * snap),
                     'maxy': float(np.ceil(y2 / snap) * snap)}
        self.name, self.matches = suggest_glacier_name(self.bbox, gates=self.gates)
        self._marker.set_data([self.center[0]], [self.center[1]])
        self._readout.set_text(
            f'{describe_bbox(self.bbox)}\n'
            f'centre ({self.center[0]:,.0f}, {self.center[1]:,.0f})  '
            f'dx {self.dx_m / 1e3:.1f} km, dy {self.dy_m / 1e3:.1f} km\n'
            f'glacier: {self.name or "none found"}')
        self.ax.figure.canvas.draw_idle()

    def require_selection(self) -> dict[str, float]:
        """Return the selected bbox, or raise if nothing has been drawn yet.

        Returns
        -------
        dict
            ``{'minx', 'miny', 'maxx', 'maxy'}`` in EPSG:3413 metres.

        Raises
        ------
        RuntimeError
            If no box has been dragged yet.
        """
        if self.bbox is None:
            raise RuntimeError(
                'No ROI selected yet - drag a box on the figure first. If the '
                'figure is not interactive, use %matplotlib widget (notebook) '
                'or a GUI backend (script).')
        return self.bbox

    def prompt_name(self, **kwargs) -> str:
        """Confirm or override the suggested glacier name (see :func:`prompt_glacier_name`)."""
        self.require_selection()
        self.name = prompt_glacier_name(self.bbox, gates=self.gates, **kwargs)
        return self.name

    def download(self, first_date: str = '2015-01-01', last_date: str | None = None,
                 products: list[str] | tuple[str, ...] = ('NSIDC-0725',),
                 prefix: str | None = None, **kwargs) -> dict[str, object]:
        """Run :func:`glacier_velocity_workflow` on the selected ROI.

        Parameters
        ----------
        first_date, last_date : str
            Inclusive NSIDC search range as ``'YYYY-MM-DD'``.
        products : sequence of str, optional
            NSIDC product codes. The default is ``('NSIDC-0725',)``.
        prefix : str, optional
            File-name prefix. ``None`` (default) prompts with the suggested
            glacier name.
        **kwargs
            Passed to :func:`glacier_velocity_workflow`.

        Returns
        -------
        dict
            As :func:`glacier_velocity_workflow`.
        """
        self.require_selection()
        return glacier_velocity_workflow(
            center=self.center, dx_m=self.dx_m, dy_m=self.dy_m,
            first_date=first_date, last_date=last_date, products=products,
            prefix=prefix, gates=self.gates, **kwargs)


def select_roi_interactive(block: bool | None = None,
                           **kwargs) -> GreenlandRoiSelector:
    """Open the Greenland overview map and return the ROI selector.

    Parameters
    ----------
    block : bool, optional
        Hold execution until the figure is closed, which is what a script
        needs. ``None`` (default) blocks only when the backend is not an
        inline/widget notebook backend.
    **kwargs
        Passed to :class:`GreenlandRoiSelector`.

    Returns
    -------
    GreenlandRoiSelector
        Read ``.center``, ``.dx_m``, ``.dy_m``, ``.name`` after dragging a box,
        or call ``.download()``.

    Notes
    -----
    Works in a notebook and in a plain script, but needs an interactive
    matplotlib backend either way:

    * notebook - run ``%matplotlib widget`` first (``ipympl``); the default
      inline backend renders a static image that cannot be dragged.
    * script - any GUI backend (``macosx``, ``tkagg``, ``qtagg``) works, and is
      usually already the default on a desktop. The call then blocks: a window
      opens, you drag a box, and execution resumes when you close it.
    * headless (``Agg``, SSH with no display) - the figure cannot be dragged at
      all, and :meth:`GreenlandRoiSelector.require_selection` raises saying so.
      Pass the ROI coordinates explicitly instead.
    """
    sel = GreenlandRoiSelector(**kwargs)
    backend = plt.get_backend().lower()
    if block is None:
        block = not ('inline' in backend or 'widget' in backend or 'ipympl' in backend)
    if block:
        print('Drag a box on the map, then close the figure window to continue.')
        plt.show(block=True)
    return sel


def load_glacier_gates(gates_file: str | Path | None = None) -> pd.DataFrame:
    """Load the named Greenland flux gates used to name an ROI.

    Parameters
    ----------
    gates_file : str or Path, optional
        MATLAB file with ``glacierName``, ``X`` and ``Y`` cell arrays - King
        et al. (2020), 234 glaciers. The default is :data:`GATES_FILE`.

    Returns
    -------
    pandas.DataFrame
        Columns ``name`` (cleaned), ``raw_name``, ``x``, ``y`` (gate centroid,
        EPSG:3413 m), ``xs``, ``ys`` (full vertex arrays) and ``length_km``,
        sorted by name. Empty with a warning if the file is unavailable.
    """
    path = Path(gates_file) if gates_file else GATES_FILE
    if not path.exists():
        warnings.warn(f'No glacier gate file at {path}; ROI naming is disabled. '
                      'Set GLACIER_GATES_FILE or pass gates_file=.', stacklevel=2)
        return pd.DataFrame(columns=['name', 'raw_name', 'x', 'y', 'xs', 'ys',
                                     'length_km'])

    mat = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    missing = [k for k in ('glacierName', 'X', 'Y') if k not in mat]
    if missing:
        raise KeyError(f'{path} has no {", ".join(missing)}; found {sorted(mat)}')

    rows = []
    for raw, xv, yv in zip(mat['glacierName'], mat['X'], mat['Y']):
        xs = np.asarray(xv, dtype='float64').ravel()
        ys = np.asarray(yv, dtype='float64').ravel()
        if xs.size == 0 or ys.size == 0:
            continue
        span = float(np.hypot(np.diff(xs), np.diff(ys)).sum()) / 1e3 if xs.size > 1 else 0.0
        rows.append({'name': clean_glacier_name(str(raw)), 'raw_name': str(raw).strip(),
                     'x': float(np.nanmean(xs)), 'y': float(np.nanmean(ys)),
                     'xs': xs, 'ys': ys, 'length_km': span})
    return pd.DataFrame(rows).sort_values('name').reset_index(drop=True)


def clean_glacier_name(name: str) -> str:
    """Turn a stored gate name into a file-name-safe glacier name.

    Strips a trailing ``.mat``, collapses whitespace to underscores and drops
    characters that have no business in a file name.

    Parameters
    ----------
    name : str
        Name as stored, e.g. ``'anoritup kangerdlua.mat'``.

    Returns
    -------
    str
        e.g. ``'anoritup_kangerdlua'``.
    """
    stem = str(name).strip()
    if stem.lower().endswith('.mat'):
        stem = stem[:-4]
    stem = '_'.join(stem.split())
    return ''.join(ch for ch in stem if ch.isalnum() or ch in '_-')


def glaciers_in_roi(bbox: dict[str, float],
                    gates: pd.DataFrame | None = None,
                    max_distance_km: float = 50.0) -> pd.DataFrame:
    """Rank named glaciers by how well their flux gate matches an ROI.

    A gate counts as "in" the ROI when any of its vertices falls inside the
    box; those are ranked by the fraction of the gate inside, which prefers a
    glacier the ROI actually covers over one it merely clips. When nothing is
    inside, nearby gates are returned instead, ranked by distance from the ROI
    centre, so an ROI placed just off a terminus still gets a suggestion.

    Parameters
    ----------
    bbox : dict
        ROI as ``{'minx', 'miny', 'maxx', 'maxy'}`` in EPSG:3413 metres.
    gates : pandas.DataFrame, optional
        Output of :func:`load_glacier_gates`; loaded if omitted.
    max_distance_km : float, optional
        How far outside the ROI to look when no gate is inside. The default is 50.

    Returns
    -------
    pandas.DataFrame
        Columns ``name``, ``inside`` (bool), ``frac_inside``, ``n_inside``,
        ``distance_km`` (0 when inside) and ``length_km``, best match first.
        Empty if no gate qualifies.
    """
    gates = load_glacier_gates() if gates is None else gates
    if gates.empty:
        return pd.DataFrame(columns=['name', 'inside', 'frac_inside', 'n_inside',
                                     'distance_km', 'length_km'])

    cx = (bbox['minx'] + bbox['maxx']) / 2
    cy = (bbox['miny'] + bbox['maxy']) / 2
    rows = []
    for row in gates.itertuples(index=False):
        inside = ((row.xs >= bbox['minx']) & (row.xs <= bbox['maxx'])
                  & (row.ys >= bbox['miny']) & (row.ys <= bbox['maxy']))
        n_inside = int(inside.sum())
        # Distance from the ROI to the nearest gate vertex (0 if any is inside).
        dx = np.maximum.reduce([bbox['minx'] - row.xs, np.zeros_like(row.xs),
                                row.xs - bbox['maxx']])
        dy = np.maximum.reduce([bbox['miny'] - row.ys, np.zeros_like(row.ys),
                                row.ys - bbox['maxy']])
        gap_km = float(np.min(np.hypot(dx, dy))) / 1e3
        if n_inside == 0 and gap_km > max_distance_km:
            continue
        rows.append({'name': row.name, 'inside': n_inside > 0,
                     'frac_inside': n_inside / row.xs.size,
                     'n_inside': n_inside, 'distance_km': gap_km,
                     'length_km': row.length_km,
                     'center_distance_km': float(np.hypot(row.x - cx, row.y - cy)) / 1e3})

    if not rows:
        return pd.DataFrame(columns=['name', 'inside', 'frac_inside', 'n_inside',
                                     'distance_km', 'length_km'])
    out = pd.DataFrame(rows)
    return (out.sort_values(['inside', 'frac_inside', 'center_distance_km'],
                            ascending=[False, False, True])
               .reset_index(drop=True))


def suggest_glacier_name(bbox: dict[str, float],
                         gates: pd.DataFrame | None = None,
                         **kwargs) -> tuple[str, pd.DataFrame]:
    """Suggest the glacier name for an ROI.

    Parameters
    ----------
    bbox : dict
        ROI as ``{'minx', 'miny', 'maxx', 'maxy'}`` in EPSG:3413 metres.
    gates : pandas.DataFrame, optional
        Output of :func:`load_glacier_gates`; loaded if omitted.
    **kwargs
        Passed to :func:`glaciers_in_roi`.

    Returns
    -------
    tuple
        ``(name, matches)``: the best-matching glacier name (``''`` if none)
        and the full ranked table from :func:`glaciers_in_roi`.
    """
    matches = glaciers_in_roi(bbox, gates=gates, **kwargs)
    return ('' if matches.empty else str(matches.loc[0, 'name'])), matches


def prompt_glacier_name(bbox: dict[str, float],
                        gates: pd.DataFrame | None = None,
                        default: str | None = None,
                        interactive: bool | None = None,
                        show_alternatives: int = 3) -> str:
    """Confirm the suggested glacier name for an ROI, or type an override.

    Prints the glacier the ROI covers and waits: Enter accepts it as the file
    name, anything typed replaces it.

    Parameters
    ----------
    bbox : dict
        ROI as ``{'minx', 'miny', 'maxx', 'maxy'}`` in EPSG:3413 metres.
    gates : pandas.DataFrame, optional
        Output of :func:`load_glacier_gates`; loaded if omitted.
    default : str, optional
        Name to offer instead of the automatic suggestion.
    interactive : bool, optional
        Ask. ``None`` (default) asks only when stdin is a terminal or a
        notebook kernel, so a papermill or cron run takes the suggestion
        silently instead of hanging.
    show_alternatives : int, optional
        How many runner-up glaciers to list alongside the suggestion. The
        default is 3.

    Returns
    -------
    str
        The accepted name, cleaned for use in a file name. Falls back to
        ``'glacier'`` if there is no suggestion and nothing is typed.
    """
    matches = None
    if default is None:
        suggestion, matches = suggest_glacier_name(bbox, gates=gates)
    else:
        suggestion = clean_glacier_name(default)

    if matches is not None and not matches.empty:
        best = matches.loc[0]
        detail = (f'{best["frac_inside"] * 100:.0f}% of its gate inside the ROI'
                  if best['inside'] else f'{best["distance_km"]:.1f} km outside the ROI')
        print(f'Glacier within ROI is {suggestion} ({detail}).')
        others = matches.loc[1:show_alternatives, 'name'].tolist()
        if others:
            print(f'  also nearby: {", ".join(others)}')
    elif suggestion:
        print(f'Glacier within ROI is {suggestion}.')
    else:
        print('No named glacier found in or near this ROI.')

    if interactive is None:
        interactive = _can_prompt()
    if not interactive:
        chosen = suggestion or 'glacier'
        print(f'  not interactive; using "{chosen}"')
        return clean_glacier_name(chosen)

    shown = suggestion or 'glacier'
    try:
        typed = input(f'Press enter to continue with this file name '
                      f'[{shown}] or type in name to use to override default: ')
    except (EOFError, KeyboardInterrupt):
        typed = ''
    chosen = typed.strip() or shown
    cleaned = clean_glacier_name(chosen)
    if cleaned != chosen:
        print(f'  using "{cleaned}" (cleaned for use in a file name)')
    return cleaned


def _can_prompt() -> bool:
    """True when :func:`input` can actually reach someone (tty or notebook)."""
    try:
        if sys.stdin is not None and sys.stdin.isatty():
            return True
    except (AttributeError, ValueError):  # detached stdin
        pass
    try:  # a Jupyter kernel can prompt even though stdin is not a tty
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == 'ZMQInteractiveShell'
    except ImportError:
        return False


def glacier_velocity_workflow(center: tuple[float, float] | None = None,
                              dx_m: float | None = None,
                              dy_m: float | None = None,
                              first_date: str = '2015-01-01',
                              last_date: str | None = None,
                              products: list[str] | tuple[str, ...] = ('NSIDC-0725',),
                              out_dir: str | Path = 'Subsets',
                              prefix: str | None = 'glacier',
                              center_crs: int | str = GRID_EPSG,
                              num_workers: int = 2,
                              overwrite: bool = False,
                              check_nodata: bool = True,
                              gates: pd.DataFrame | None = None,
                              verbose: bool = True) -> dict[str, object]:
    """Run the whole reader end to end: ROI, search, download, verify.

    This is the notebook's driver as a single call. It builds the ROI, reports
    what NSIDC has, writes one NetCDF per product (per GrIMP box for the boxed
    products) and checks the no-data convention of what it wrote.

    If the ROI is unknown, leave ``center``/``dx_m`` out: the Greenland overview
    map opens so the box can be drawn (see :func:`select_roi_interactive`). If
    the file name is unknown, pass ``prefix=None``: the glacier the ROI covers
    is looked up in the named flux gates and offered for confirmation (see
    :func:`prompt_glacier_name`).

    Parameters
    ----------
    center : tuple of float, optional
        Centre of the ROI as ``(x, y)`` in ``center_crs``. For a geographic CRS
        this is ``(longitude, latitude)``. ``None`` opens the Greenland picker.
    dx_m : float, optional
        Metres to add either side of the centre in x, so the ROI is
        ``2 * dx_m`` wide. ``None`` opens the Greenland picker.
    dy_m : float, optional
        Same for y. Defaults to ``dx_m`` (a square ROI).
    first_date, last_date : str
        Inclusive NSIDC search range as ``'YYYY-MM-DD'``. ``last_date`` defaults
        to today.
    products : sequence of str, optional
        NSIDC product codes, see :data:`NSIDC_VELOCITY_PRODUCTS`. The default is
        ``('NSIDC-0725',)``.
    out_dir : str or Path, optional
        Directory the NetCDF files are written to. The default is ``'Subsets'``.
    prefix : str, optional
        Leading part of each file name, typically the glacier name. ``None``
        suggests the glacier covered by the ROI and asks for confirmation.
    center_crs : int or str, optional
        CRS of ``center``. The default is EPSG:3413.
    num_workers : int, optional
        Dask threads for the download. The default is 2.
    overwrite : bool, optional
        Re-download files that already exist. The default is False.
    check_nodata : bool, optional
        Run :func:`nodata_report` on the result and raise if a sentinel
        survived or a derived speed band is present where its components are
        no-data. The default is True.
    gates : pandas.DataFrame, optional
        Named flux gates for the ROI -> glacier-name lookup; loaded on demand.
    verbose : bool, optional
        Print progress. The default is True.

    Returns
    -------
    dict
        ``{'bbox': ..., 'available': DataFrame, 'files': list of Path,
        'report': DataFrame or None, 'prefix': str, 'center': tuple,
        'dx_m': float, 'dy_m': float}``.

    Raises
    ------
    AssertionError
        If ``check_nodata`` is True and the written files violate the NaN-only
        no-data convention.
    RuntimeError
        If the ROI had to be picked interactively and no box was drawn.
    """
    configure_dask(num_workers=num_workers, progress=verbose)

    if center is None or dx_m is None:
        selector = select_roi_interactive(gates=gates)
        selector.require_selection()
        center, dx_m, dy_m = selector.center, selector.dx_m, selector.dy_m
        center_crs = GRID_EPSG          # the picker works in projected metres
        gates = selector.gates

    bbox = roi_bbox(center, dx_m, dy_m, center_crs=center_crs)

    if prefix is None:
        prefix = prompt_glacier_name(bbox, gates=gates)
    if verbose:
        print(f'{prefix}: {describe_bbox(bbox)}')

    available = summarize_search(bbox, first_date,
                                 last_date or datetime.now(timezone.utc).strftime('%Y-%m-%d'),
                                 list(products))
    if verbose:
        print(available.to_string(index=False))

    files = download_velocity_netcdfs(
        center=center, dx_m=dx_m, dy_m=dy_m, first_date=first_date, last_date=last_date,
        products=products, out_dir=out_dir, prefix=prefix, center_crs=center_crs,
        num_workers=num_workers, overwrite=overwrite, verbose=verbose,
    )

    report = None
    if check_nodata and files:
        report = nodata_report(files)
        assert (report['n_sentinel_left'] == 0).all(), \
            'sentinel no-data values survived the write'
        for name, group in report.groupby('file'):
            gaps = group.set_index('band')['nan_fraction']
            for derived, (band_x, band_y) in DERIVED_BANDS.items():
                if {derived, band_x, band_y} <= set(gaps.index):
                    assert gaps[derived] >= gaps[band_x] - 1e-9, \
                        f'{name}: {derived} is present where {band_x} is no-data'

    return {'bbox': bbox, 'available': available, 'files': files, 'report': report,
            'prefix': prefix, 'center': tuple(center), 'dx_m': dx_m,
            'dy_m': dx_m if dy_m is None else dy_m}
