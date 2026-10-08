"""Self-contained validation of the Greenland ROI picker and naming (no network).

Fabricates a small set of named flux gates - no NSIDC access, no Earthdata
credentials, no BedMachine mask - and asserts that:

  * an ROI sitting on a gate is named after that glacier, and the ranking
    prefers the glacier the ROI actually covers over one it merely clips,
  * an ROI near, but not on, a gate still gets a suggestion, with the distance
    reported, while an ROI far from everything gets none,
  * names are cleaned into file-name-safe form ('.mat' stripped, spaces to
    underscores, punctuation dropped),
  * the prompt accepts the suggestion, accepts a typed override, cleans what
    was typed, and never blocks when it cannot reach a terminal,
  * the selector derives centre / dx_m / dy_m from a dragged box, snaps the
    corners, and refuses to proceed before anything is drawn.

Run:  python test_roi_naming.py
"""

from __future__ import annotations

import sys
import types

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')  # headless: nothing here needs a display

import glacier_reader as gr  # noqa: E402


def _gate(name: str, x0: float, y0: float, length_m: float = 8000.0,
          n: int = 50) -> dict:
    """Build one synthetic across-flow gate centred on ``(x0, y0)``.

    Parameters
    ----------
    name : str
        Stored gate name, as it would appear in the MATLAB file.
    x0, y0 : float
        Gate centre in EPSG:3413 metres.
    length_m : float, optional
        Gate length. The default is 8000.
    n : int, optional
        Vertices. The default is 50.

    Returns
    -------
    dict
        A row for the gates DataFrame.
    """
    xs = np.linspace(x0 - length_m / 2, x0 + length_m / 2, n)
    ys = np.full(n, float(y0))
    return {'name': gr.clean_glacier_name(name), 'raw_name': name,
            'x': float(xs.mean()), 'y': float(ys.mean()), 'xs': xs, 'ys': ys,
            'length_km': length_m / 1e3}


GATES = pd.DataFrame([
    _gate('jakobshavn.mat', -183000.0, -2274200.0),
    _gate('torssukatak.mat', -160000.0, -2250000.0),
    _gate('kangerdlugssuaq.mat', 490000.0, -2290000.0),
    _gate('anoritup kangerdlua.mat', -300000.0, -1500000.0),
])


def _check(label: str, ok: bool) -> bool:
    """Print a pass/fail line and return the result unchanged."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return ok


def run() -> bool:
    """Run every check and return True only if all of them passed."""
    ok = True

    # --- name cleaning -----------------------------------------------------
    ok &= _check("'.mat' stripped and spaces -> underscores",
                 gr.clean_glacier_name('anoritup kangerdlua.mat') == 'anoritup_kangerdlua')
    ok &= _check('punctuation dropped from a typed name',
                 gr.clean_glacier_name("Jakobshavn Isbrae (W)") == 'Jakobshavn_Isbrae_W')

    # --- an ROI on a gate --------------------------------------------------
    on_gate = gr.roi_bbox((-183000.0, -2274200.0), 5000.0)
    name, matches = gr.suggest_glacier_name(on_gate, gates=GATES)
    ok &= _check('ROI on a gate is named after that glacier', name == 'jakobshavn')
    ok &= _check('the covered glacier ranks above the clipped one',
                 bool(matches.loc[0, 'inside']) and matches.loc[0, 'frac_inside'] > 0.5)
    ok &= _check('distance is 0 for a gate inside the ROI',
                 float(matches.loc[0, 'distance_km']) == 0.0)

    # A box covering most of one gate and the tip of another prefers the former.
    both = {'minx': -186000.0, 'miny': -2280000.0, 'maxx': -158000.0, 'maxy': -2248000.0}
    best, ranked = gr.suggest_glacier_name(both, gates=GATES)
    ok &= _check('two gates inside -> the better-covered one wins',
                 best == 'jakobshavn' and len(ranked) >= 2)

    # --- an ROI near, and an ROI far from, any gate ------------------------
    near = gr.roi_bbox((-183000.0, -2262000.0), 4000.0)   # ~8 km up-glacier
    near_name, near_matches = gr.suggest_glacier_name(near, gates=GATES)
    ok &= _check('ROI near a gate still gets a suggestion',
                 near_name == 'jakobshavn' and not bool(near_matches.loc[0, 'inside']))
    ok &= _check('the gap is reported in km',
                 0.0 < float(near_matches.loc[0, 'distance_km']) < 20.0)

    far = gr.roi_bbox((0.0, -1800000.0), 10000.0)         # interior ice sheet
    far_name, far_matches = gr.suggest_glacier_name(far, gates=GATES)
    ok &= _check('ROI far from every gate gets no suggestion',
                 far_name == '' and far_matches.empty)

    # --- the prompt --------------------------------------------------------
    import builtins

    original_input = builtins.input
    try:
        builtins.input = lambda _prompt='': ''
        ok &= _check('empty input accepts the suggestion',
                     gr.prompt_glacier_name(on_gate, gates=GATES, interactive=True)
                     == 'jakobshavn')

        builtins.input = lambda _prompt='': 'Store Glacier'
        ok &= _check('typed input overrides, cleaned for a file name',
                     gr.prompt_glacier_name(on_gate, gates=GATES, interactive=True)
                     == 'Store_Glacier')

        builtins.input = lambda _prompt='': (_ for _ in ()).throw(EOFError)
        ok &= _check('EOF falls back to the suggestion',
                     gr.prompt_glacier_name(on_gate, gates=GATES, interactive=True)
                     == 'jakobshavn')
    finally:
        builtins.input = original_input

    ok &= _check('non-interactive takes the suggestion without asking',
                 gr.prompt_glacier_name(on_gate, gates=GATES, interactive=False)
                 == 'jakobshavn')
    ok &= _check('an ROI with no match falls back to "glacier"',
                 gr.prompt_glacier_name(far, gates=GATES, interactive=False) == 'glacier')
    ok &= _check('default= overrides the lookup',
                 gr.prompt_glacier_name(on_gate, gates=GATES, default='mine.mat',
                                        interactive=False) == 'mine')

    # --- the selector ------------------------------------------------------
    sel = gr.GreenlandRoiSelector(gates=GATES, overview=None, show_gates=True)
    ok &= _check('nothing selected yet -> require_selection raises',
                 _raises(sel.require_selection))

    sel._on_select(types.SimpleNamespace(xdata=-193000.0, ydata=-2284200.0),
                   types.SimpleNamespace(xdata=-173000.0, ydata=-2264200.0))
    ok &= _check('drag -> centre',
                 sel.center == (-183000.0, -2274200.0))
    ok &= _check('drag -> dx_m / dy_m ready for download_velocity_netcdfs',
                 sel.dx_m == 10000.0 and sel.dy_m == 10000.0)
    ok &= _check('drag -> glacier name', sel.name == 'jakobshavn')
    ok &= _check('selection is snapped to the grid',
                 all(abs(v) % 100.0 == 0.0 for v in sel.bbox.values()))
    ok &= _check('the figure readout names the glacier',
                 'jakobshavn' in sel._readout.get_text())
    return ok


def _raises(fn, *args, **kwargs) -> bool:
    """True if calling ``fn`` raises anything."""
    try:
        fn(*args, **kwargs)
    except Exception:  # noqa: BLE001 - any failure is the expected outcome
        return True
    return False


if __name__ == '__main__':
    print('=== ROI picker + glacier naming (synthetic) ===')
    passed = run()
    print('\n' + ('ALL ROI NAMING CHECKS PASSED' if passed else 'SOME CHECKS FAILED'))
    sys.exit(0 if passed else 1)
