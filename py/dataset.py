"""Training-array helpers (numpy only, no browser code).

Per-sample arrays are saved untouched while labeling; the transforms here are applied
when the dataset is built, so that samples of different sizes can be stacked:

* images:  ``interpolate`` (NaN-aware bilinear, block-averaged first when shrinking a lot)
           or ``pad_crop`` (centred, padded with NaN)
* spectra: resampled onto one bias axis; points a file did not measure are NaN
* ``normalize``: ``zscore`` uses each sample's own mean / std (non-NaN values only), ``none`` leaves the values

Arrays keep NaN wherever the data has none (partial scans, bias outside the sweep).
"""
import io
import json
import warnings

import numpy as np


def to_npy(arr):
    """The bytes of a ``.npy`` file holding ``arr``."""
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr), allow_pickle=False)
    return buf.getvalue()


def from_npy(data):
    return np.load(io.BytesIO(_as_bytes(data)), allow_pickle=False)


def _as_bytes(data):
    # a JS Uint8Array arrives from Pyodide as a proxy with to_bytes()
    return data.to_bytes() if hasattr(data, "to_bytes") else bytes(data)


# ----------------------------------------------------------------------------- images

def _pool(a, f):
    """NaN-aware mean over f x f blocks (edges that do not fill a block are dropped)."""
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    blocks = a[:h, :w].reshape((h // f, f, w // f, f) + a.shape[2:])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(blocks, axis=(1, 3))


def resize_interpolate(a, size):
    """Bilinear resize of (H, W) or (H, W, C) to ``size``; NaN pixels are left out of the average."""
    a = np.asarray(a, dtype=np.float64)
    H, W = int(size[0]), int(size[1])
    if a.shape[:2] == (H, W):
        return a.astype(np.float32)
    f = min(a.shape[0] // H, a.shape[1] // W)
    if f >= 2:                       # shrinking a lot: average first, so pixels are not simply skipped
        a = _pool(a, f)
    h, w = a.shape[:2]
    ys = np.clip((np.arange(H) + 0.5) * h / H - 0.5, 0, h - 1)
    xs = np.clip((np.arange(W) + 0.5) * w / W - 0.5, 0, w - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    wy, wx = (ys - y0)[:, None], (xs - x0)[None, :]
    total = np.zeros((H, W) + a.shape[2:])
    weight = np.zeros((H, W) + a.shape[2:])
    for yy, xx, wgt in ((y0, x0, (1 - wy) * (1 - wx)), (y0, x1, (1 - wy) * wx),
                        (y1, x0, wy * (1 - wx)), (y1, x1, wy * wx)):
        v = a[yy][:, xx]
        ok = np.isfinite(v)
        wg = wgt if v.ndim == 2 else wgt[..., None]
        total += np.where(ok, v, 0.0) * wg
        weight += ok * wg
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / weight
    out[weight == 0] = np.nan
    return out.astype(np.float32)


def resize_pad_crop(a, size):
    """Centre crop / pad with NaN to ``size`` (no resampling: the pixel size is unchanged)."""
    a = np.asarray(a, dtype=np.float32)
    H, W = int(size[0]), int(size[1])
    out = np.full((H, W) + a.shape[2:], np.nan, dtype=np.float32)
    h, w = a.shape[:2]
    sy, sx = max((h - H) // 2, 0), max((w - W) // 2, 0)      # crop offset in the source
    dy, dx = max((H - h) // 2, 0), max((W - w) // 2, 0)      # pad offset in the target
    ny, nx = min(h, H), min(w, W)
    out[dy:dy + ny, dx:dx + nx] = a[sy:sy + ny, sx:sx + nx]
    return out


# ----------------------------------------------------------------------------- spectra

def resample_spectrum(a, grid, n_rows):
    """``a``: (rows, N) with row 0 = bias. Returns (n_rows, len(grid)) with row 0 = ``grid``.

    Signals are linearly interpolated onto ``grid``; outside the measured bias range they are NaN.
    Rows the file does not have (e.g. no backward sweep) stay NaN.
    """
    a = np.asarray(a, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    out = np.full((n_rows, len(grid)), np.nan, dtype=np.float32)
    out[0] = grid
    x = a[0]
    for r in range(1, min(a.shape[0], n_rows)):
        ok = np.isfinite(x) & np.isfinite(a[r])
        if ok.sum() < 2:
            continue
        order = np.argsort(x[ok])
        xs, ys = x[ok][order], a[r][ok][order]
        tol = 1e-6 * max(xs[-1] - xs[0], 1e-30)              # the saved bias is float32
        v = np.interp(grid, xs, ys)
        v[(grid < xs[0] - tol) | (grid > xs[-1] + tol)] = np.nan
        out[r] = v
    return out


def pad_rows(a, n_rows):
    a = np.asarray(a, dtype=np.float32)
    out = np.full((n_rows, a.shape[1]), np.nan, dtype=np.float32)
    out[:a.shape[0]] = a[:n_rows]
    return out


# ----------------------------------------------------------------------------- normalize

def sample_stats(arr, kind):
    """(mean, std) of one saved sample, over its non-NaN values only.

    Spectra: the signal rows (forward / backward together); the bias row is not part of it.
    Returns (None, None) when the sample has no finite value.
    """
    a = np.asarray(arr, dtype=np.float64)
    if kind == "spectrum":
        a = a[1:]
    v = a[np.isfinite(a)]
    if not v.size:
        return None, None
    return float(v.mean()), float(v.std())


def zscore(a, mean, std):
    """(a - mean) / std with the sample's own recorded statistics; NaN stays NaN.

    A sample without spread (std 0, e.g. a constant channel) becomes all zeros.
    """
    a = np.asarray(a, dtype=np.float64)
    if mean is None or std is None:
        return a.astype(np.float32)
    out = (a - mean) / std if std > 0 else np.where(np.isfinite(a), 0.0, np.nan)
    return out.astype(np.float32)


def stats(data, kind):
    """Statistics of a saved .npy sample (for arrays saved before they were recorded)."""
    mean, std = sample_stats(from_npy(data), kind)
    return {"sample_mean": mean, "sample_std": std}


# ----------------------------------------------------------------------------- NaN

def nan_handling(out, kind, to_zero, mask):
    """Last step of the build, after the normalization.

    ``to_zero``: values that are not finite become 0 (after a z-score, 0 is the sample's average).
    ``mask``:    a mask is added, 1 where the sample has data and 0 where it had none:

    * image:    (H, W)       -> (2, H, W)           [0] the data, [1] the mask
    * rgb:      (H, W, 3)    -> (H, W, 4)           the mask is the 4th channel (a pixel counts if all 3 are there)
    * spectrum: (rows, P)    -> (rows + rows-1, P)  one mask row per signal row, after the signal rows
                                                    (the bias row has no mask)
    """
    out = np.asarray(out, dtype=np.float32)
    finite = np.isfinite(out)
    data = np.where(finite, out, 0.0).astype(np.float32) if to_zero else out
    if not mask:
        return data
    if kind == "spectrum":
        return np.vstack([data, finite[1:].astype(np.float32)])
    if kind == "rgb":
        return np.concatenate([data, finite.all(axis=-1, keepdims=True).astype(np.float32)], axis=-1)
    return np.stack([data, finite.astype(np.float32)])


# ----------------------------------------------------------------------------- build step

def transform(data, kind, params):
    """One saved sample -> the float32 block that goes into ``<mode>_<kind>_input.npy``.

    ``data``: the bytes of the sample's .npy file. ``kind``: "image" | "rgb" | "spectrum".
    ``params`` (dict or JSON text):
      images:  {"size": [H, W], "method": "interpolate" | "pad_crop"}
      spectra: {"grid": {"min", "max", "points"} | null, "rows": n}
      both:    {"normalize": "zscore" | "none", "mean": ..., "std": ...}   (the sample's own statistics)
               {"nan_to_zero": bool, "nan_mask": bool}                    (see ``nan_handling``)
    Returns {"data": raw little-endian float32 bytes, "shape": [...]}.
    """
    if isinstance(params, str):
        params = json.loads(params)
    a = from_npy(data)
    norm = params.get("normalize", "none") == "zscore"
    mean, std = params.get("mean"), params.get("std")
    if kind == "spectrum":
        rows, grid = int(params["rows"]), params.get("grid")
        if grid:
            out = resample_spectrum(a, np.linspace(grid["min"], grid["max"], int(grid["points"])), rows)
        else:
            out = pad_rows(a, rows)
        if norm:
            out[1:] = zscore(out[1:], mean, std)      # never the bias row
    else:
        size = params["size"]
        out = resize_pad_crop(a, size) if params.get("method") == "pad_crop" else resize_interpolate(a, size)
        if norm:
            out = zscore(out, mean, std)
    out = nan_handling(out, kind, bool(params.get("nan_to_zero")), bool(params.get("nan_mask")))
    out = np.ascontiguousarray(out, dtype="<f4")
    return {"data": out.tobytes(), "shape": list(out.shape)}
