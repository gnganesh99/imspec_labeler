"""Reading and display preparation for the labeler (no browser-specific code).

Nanonis ``.sxm`` / ``.dat`` / ``.3ds`` files are read with the SciFiReaders fork;
plain images with Pillow (tif via SciFiReaders' ImageReader / tifffile). Every
file becomes a dict of *channels*, each holding one entry per scan direction.

``load(path)`` parses a file (cached), ``render(...)`` returns what the UI draws:
images as uint8 colormap indices (255 = no data), spectra as float lists.
"""
import math
import os
import re
import warnings
from collections import OrderedDict

import numpy as np

SPM_TYPES = {".sxm": "sxm", ".dat": "dat", ".3ds": "3ds"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
MAX_DISPLAY_PIXELS = 1024   # longer image side is strided down to at most this
CACHE_SIZE = 8              # parsed files kept in memory (current + prefetched)

_cache = OrderedDict()


def file_type(name):
    """'sxm' | 'dat' | '3ds' | 'img' | None (unsupported)."""
    ext = os.path.splitext(name)[1].lower()
    if ext in SPM_TYPES:
        return SPM_TYPES[ext]
    return "img" if ext in IMAGE_EXTS else None


def list_files(folder):
    """Supported files directly inside ``folder``, naturally sorted."""
    names = [n for n in os.listdir(folder)
             if file_type(n) and os.path.isfile(os.path.join(folder, n))]
    return sorted(names, key=natural_key)


def natural_key(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


# ----------------------------------------------------------------------------- loading

def load(path, key=None):
    """Parse ``path`` (cached under ``key``) and return a summary of its channels."""
    key = key or path
    if key in _cache:
        _cache.move_to_end(key)
    else:
        ftype = file_type(path)
        entries = _read_image(path) if ftype == "img" else _read_spm(path, ftype)
        if not entries:
            raise ValueError("no channels found in " + os.path.basename(path))
        _cache[key] = {"type": ftype, "channels": entries}
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    item = _cache[key]
    return {"type": item["type"],
            "channels": list(item["channels"]),
            "directions": {name: list(dirs) for name, dirs in item["channels"].items()},
            "kinds": {name: next(iter(dirs.values()))["kind"]
                      for name, dirs in item["channels"].items()}}


def is_loaded(key):
    return key in _cache


def drop(key):
    _cache.pop(key, None)


def _read_spm(path, ftype):
    from SciFiReaders import NanonisDatReader, NanonisSXMReader, Nanonis3dsReader
    reader = {"sxm": NanonisSXMReader, "dat": NanonisDatReader, "3ds": Nanonis3dsReader}[ftype]
    datasets = reader(path).read()
    channels = OrderedDict()
    for key, ds in datasets.items():
        base, direction = split_direction(key)
        entry = _entry_from_dataset(ds)
        if entry is not None:
            channels.setdefault(base, OrderedDict())[direction] = entry
    return channels


def split_direction(key):
    """'Z forward' -> ('Z', 'forward'); 'Current [bwd]' -> ('Current', 'backward')."""
    m = re.match(r"^(.*\S)\s+(forward|backward)$", key)
    if m:
        return m.group(1), m.group(2)
    m = re.match(r"^(.*\S)\s*\[bwd\]$", key)
    if m:
        return m.group(1), "backward"
    return key, "forward"


def _axis(ds, i):
    dim = ds._axes[i]
    return {"name": str(dim.name), "units": str(dim.units),
            "values": np.asarray(dim.values, dtype=float),
            "type": str(getattr(dim.dimension_type, "name", dim.dimension_type)).upper()}


def _entry_from_dataset(ds):
    data = np.asarray(ds)
    axes = [_axis(ds, i) for i in range(data.ndim)]
    units, quantity = str(ds.units), str(ds.quantity)
    meta = getattr(ds, "original_metadata", None) or {}
    if data.ndim == 1:
        return {"kind": "spectrum", "y": data.astype(float), "x": axes[0],
                "units": units, "quantity": quantity, "meta": meta}
    if data.ndim == 2:
        return {"kind": "image", "data": data, "y": axes[0], "x": axes[1],
                "units": units, "quantity": quantity, "origin": "lower", "meta": meta}
    if data.ndim == 3:
        spatial = [i for i, a in enumerate(axes) if a["type"] == "SPATIAL"]
        other = [i for i in range(3) if i not in spatial]
        if len(spatial) != 2:
            spatial, other = [0, 1], [2]
        cube = np.moveaxis(data, other[0], -1)
        return {"kind": "cube", "data": cube, "y": axes[spatial[0]], "x": axes[spatial[1]],
                "z": axes[other[0]], "units": units, "quantity": quantity, "origin": "lower",
                "meta": meta}
    return None


def _pixel_axis(n, name):
    return {"name": name, "units": "px", "values": np.arange(n, dtype=float), "type": "SPATIAL"}


def _read_image(path):
    ext = os.path.splitext(path)[1].lower()
    arr = None
    meta = {"File type": ext.lstrip(".").upper(), "File size (bytes)": os.path.getsize(path)}
    if ext in (".tif", ".tiff"):
        # scientific tifs (16/32-bit, stacks) via SciFiReaders/tifffile; Pillow as fallback
        try:
            from SciFiReaders import ImageReader
            ds = ImageReader(path).read()
            arr = np.asarray(ds)
            meta.update(getattr(ds, "metadata", None) or {})
            meta.update(getattr(ds, "original_metadata", None) or {})
        except Exception:
            arr = None
    if arr is None:
        from PIL import Image
        with Image.open(path) as im:
            meta["Mode"] = im.mode
            meta.update({str(k): v for k, v in im.info.items() if isinstance(v, (str, int, float))})
            if im.mode in ("P", "PA", "1", "CMYK", "YCbCr", "LAB", "HSV"):
                im = im.convert("RGBA" if "A" in im.mode or "transparency" in im.info else "RGB")
            elif im.mode == "LA":
                im = im.convert("L")
            arr = np.asarray(im)
    while arr.ndim > 3 or (arr.ndim == 3 and arr.shape[-1] not in (3, 4) and arr.shape[0] == 1):
        arr = arr[0]

    is_rgb = arr.ndim == 3 and arr.shape[-1] in (3, 4)
    h, w = arr.shape[1:] if arr.ndim == 3 and not is_rgb else arr.shape[:2]
    meta["Size (px)"] = "%d x %d" % (w, h)
    common = {"y": _pixel_axis(h, "y"), "x": _pixel_axis(w, "x"),
              "units": "", "quantity": "Intensity", "origin": "upper", "meta": meta}
    channels = OrderedDict()
    if is_rgb:
        rgb = arr[..., :3]
        channels["RGB"] = {"forward": dict(common, kind="rgb", data=rgb)}
        gray = rgb.astype(float) @ np.array([0.299, 0.587, 0.114])
        channels["Gray"] = {"forward": dict(common, kind="image", data=gray)}
        for i, c in enumerate("RGB"):
            channels[c] = {"forward": dict(common, kind="image", data=rgb[..., i])}
    elif arr.ndim == 3:   # stack: frames along axis 0
        z = {"name": "frame", "units": "", "values": np.arange(arr.shape[0], dtype=float),
             "type": "FRAME"}
        channels["Intensity"] = {"forward": dict(common, kind="cube",
                                                 data=np.moveaxis(arr, 0, -1), z=z)}
    else:
        channels["Intensity"] = {"forward": dict(common, kind="image", data=arr)}
    return channels


# ----------------------------------------------------------------------------- rendering

def render(key, channel=None, direction="forward", flatten="none", index=None):
    """Display data for one channel of a loaded file.

    Falls back to the first channel / an available direction when the requested
    one is missing (``fallback`` is True then).
    """
    channels = _cache[key]["channels"]
    fallback = channel is not None and channel not in channels
    if channel not in channels:
        channel = next(iter(channels))
    dirs = channels[channel]
    if direction not in dirs:
        direction = next(iter(dirs))
    entry = dirs[direction]
    out = {"channel": channel, "direction": direction, "fallback": fallback,
           "kind": entry["kind"], "title": entry["quantity"], "origin": entry.get("origin", "upper")}

    if entry["kind"] == "spectrum":
        out.update(_spectrum(dirs))
    elif entry["kind"] == "rgb":
        out.update(_rgb(entry))
    elif entry["kind"] == "image":
        out.update(_image(entry, entry["data"], flatten))
    else:  # cube
        cube = entry["data"]
        n = cube.shape[-1]
        index = n // 2 if index is None else int(min(max(index, 0), n - 1))
        z = entry["z"]
        out.update(_image(entry, cube[..., index], flatten))
        zf, zu = si_scale(np.abs(z["values"]).max(), z["units"])
        out.update(index=index, n=n, slice_label=_label(z["name"], zu),
                   slice_value=float(z["values"][index] * zf))
        if z["type"] == "SPECTRAL":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = np.nanmean(cube.reshape(-1, n), axis=0)
            yf, yu = si_scale(np.nanmax(np.abs(mean)), entry["units"])
            out["spectrum"] = {"x": (z["values"] * zf).tolist(), "x_label": out["slice_label"],
                               "y_label": _label("mean " + entry["quantity"], yu),
                               "series": [{"label": "mean", "y": _clean(mean * yf)}]}
    return out


def _label(name, units):
    return "%s (%s)" % (name, units) if units else name


def _clean(values):
    return [None if not math.isfinite(v) else v for v in np.asarray(values, float).tolist()]


def _spectrum(dirs):
    first = next(iter(dirs.values()))
    x = first["x"]
    xf, xu = si_scale(np.nanmax(np.abs(x["values"])), x["units"])
    ymax = max(float(np.nanmax(np.abs(e["y"]))) if np.isfinite(e["y"]).any() else 0
               for e in dirs.values())
    yf, yu = si_scale(ymax, first["units"])
    return {"x": (x["values"] * xf).tolist(), "x_label": _label(x["name"], xu),
            "y_label": _label(first["quantity"], yu),
            "series": [{"label": d, "y": _clean(e["y"] * yf)} for d, e in dirs.items()]}


def _stride(arr):
    step = math.ceil(max(arr.shape[:2]) / MAX_DISPLAY_PIXELS)
    return arr[::step, ::step] if step > 1 else arr


def _extent(entry):
    def size(ax):
        v = ax["values"]
        return float(abs(v[-1] - v[0]) * len(v) / (len(v) - 1)) if len(v) > 1 else 1.0
    w, h = size(entry["x"]), size(entry["y"])
    units = entry["x"]["units"]
    f, u = si_scale(max(w, h), units)
    return {"extent": [w * f, h * f], "extent_units": u}


def _rgb(entry):
    rgb = _stride(entry["data"])
    if rgb.dtype != np.uint8:   # e.g. 16-bit RGB tif
        rgb = rgb.astype(float)
        rgb = np.clip(rgb / max(np.nanmax(rgb), 1e-12) * 255, 0, 255).astype(np.uint8)
    h, w = rgb.shape[:2]
    rgba = np.empty((h, w, 4), np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = 255
    return dict(_extent(entry), w=w, h=h, data=rgba.tobytes())


def _image(entry, data, flatten):
    """Values in display units as float32 (row 0 = first row of the file; NaN = no data).

    The UI applies transpose / origin, builds the colormap indices and can read exact
    values back for the pixel readout and line profiles.
    """
    img = _stride(np.asarray(data, dtype=float))
    img = apply_flatten(img, flatten)
    finite = np.isfinite(img)
    if finite.any():
        vmin, vmax = np.percentile(img[finite], [1, 99])
        if vmax <= vmin:
            vmin, vmax = float(img[finite].min()), float(img[finite].max())
    else:
        vmin, vmax = 0.0, 1.0
    f, u = si_scale(max(abs(vmin), abs(vmax)), entry["units"])
    h, w = img.shape
    values = np.where(finite, img * f, np.nan).astype(np.float32)
    return dict(_extent(entry), w=w, h=h, data=values.tobytes(),
                vmin=float(vmin * f), vmax=float(vmax * f), units=u)


def apply_flatten(img, mode):
    """'none' | 'offset' (line median) | 'line' (line-wise linear) | 'plane'."""
    if mode in (None, "none"):
        return img
    img = img.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if mode == "offset":
            img -= np.nan_to_num(np.nanmedian(img, axis=1, keepdims=True))
        elif mode == "line":
            x = np.arange(img.shape[1], dtype=float)
            for row in img:
                ok = np.isfinite(row)
                if ok.sum() > 1:
                    row -= np.polyval(np.polyfit(x[ok], row[ok], 1), x)
        elif mode == "plane":
            yy, xx = np.indices(img.shape, dtype=float)
            ok = np.isfinite(img)
            if ok.sum() > 2:
                A = np.c_[xx[ok], yy[ok], np.ones(ok.sum())]
                coef = np.linalg.lstsq(A, img[ok], rcond=None)[0]
                img -= coef[0] * xx + coef[1] * yy + coef[2]
    return img


_PREFIXES = [(1e-15, "f"), (1e-12, "p"), (1e-9, "n"), (1e-6, "µ"), (1e-3, "m"),
             (1, ""), (1e3, "k"), (1e6, "M"), (1e9, "G")]
_SI_UNITS = {"m", "A", "V", "Hz", "N", "s", "F", "W", "S", "T", "K", "Pa"}


def si_scale(max_abs, units):
    """(factor, prefixed units) so that ``max_abs * factor`` lies in [1, 1000)."""
    if units not in _SI_UNITS or not max_abs or not math.isfinite(max_abs):
        return 1.0, units
    for scale, prefix in reversed(_PREFIXES):
        if max_abs >= scale:
            return 1.0 / scale, prefix + units
    return 1.0 / _PREFIXES[0][0], _PREFIXES[0][1] + units


# ----------------------------------------------------------------------------- training arrays

def export(key, channel=None, direction="forward", flatten="none", index=None, with_array=True):
    """The training array for one channel of a loaded file, plus what is needed to interpret it.

    Full resolution, in the units of the file (no display scaling), NaN kept:

    * image / cube: float32 (H, W), flattened as viewed (``flatten``); a cube gives the slice ``index``.
      Row 0 is the first row of the file (see ``origin`` in the returned meta).
    * rgb:          uint8 (H, W, 3)
    * spectrum:     float32 (rows, N): row 0 = the swept axis (bias), then one row per direction

    ``meta`` carries ``sample_mean`` / ``sample_std``: the statistics of the array's non-NaN values
    (spectra: of the signal rows). With ``with_array=False`` only the meta is returned (``npy`` is None).

    Returns {"kind", "npy": bytes of a .npy file, "meta": {...}}. Raises KeyError for an unknown channel.
    """
    import dataset
    channels = _cache[key]["channels"]
    if channel not in channels:
        raise KeyError("channel %r is not in this file" % (channel,))
    dirs = channels[channel]
    if direction not in dirs:
        direction = next(iter(dirs))
    entry = dirs[direction]
    meta = {"channel": channel, "units": entry["units"], "quantity": entry["quantity"]}

    if entry["kind"] == "spectrum":
        x = entry["x"]
        arr = np.asarray([x["values"]] + [e["y"] for e in dirs.values()], dtype=np.float32)
        bias = x["values"][np.isfinite(x["values"])]   # exact values (the saved row is float32)
        meta.update(kind="spectrum",
                    rows=[_label(x["name"], x["units"])] +
                         ["%s %s" % (_label(entry["quantity"], entry["units"]), d) for d in dirs],
                    bias_name=x["name"], bias_units=x["units"], n_points=int(arr.shape[1]),
                    bias_min=float(bias.min()) if bias.size else None,
                    bias_max=float(bias.max()) if bias.size else None)
    else:
        if entry["kind"] == "rgb":
            arr = entry["data"]
            if arr.dtype != np.uint8:   # e.g. 16-bit RGB tif
                arr = arr.astype(float)
                arr = np.clip(arr / max(np.nanmax(arr), 1e-12) * 255, 0, 255).astype(np.uint8)
            meta.update(kind="rgb")
        else:
            data = entry["data"]
            if entry["kind"] == "cube":
                n = data.shape[-1]
                index = n // 2 if index is None else int(min(max(index, 0), n - 1))
                z = entry["z"]
                data = data[..., index]
                meta.update(slice_index=index, slice_value=float(z["values"][index]),
                            slice_name=z["name"], slice_units=z["units"])
            arr = apply_flatten(np.asarray(data, dtype=float), flatten).astype(np.float32)
            meta.update(kind="image", direction=direction, flatten=flatten)

        def size(ax):   # full extent of an axis, in its own units
            v = ax["values"]
            return float(abs(v[-1] - v[0]) * len(v) / (len(v) - 1)) if len(v) > 1 else float(len(v))
        width, height = size(entry["x"]), size(entry["y"])
        meta.update(origin=entry["origin"], extent_units=entry["x"]["units"], width=width, height=height,
                    pixel_x=width / arr.shape[1], pixel_y=height / arr.shape[0])

    mean, std = dataset.sample_stats(arr, meta["kind"])
    meta.update(shape=list(arr.shape), dtype=str(arr.dtype), sample_mean=mean, sample_std=std)
    return {"kind": meta["kind"], "npy": dataset.to_npy(arr) if with_array else None, "meta": meta}


# ----------------------------------------------------------------------------- metadata

def metadata(key, channel=None, direction="forward"):
    """Header of the displayed channel: a few key facts plus every field."""
    item = _cache[key]
    channels = item["channels"]
    if channel not in channels:
        channel = next(iter(channels))
    dirs = channels[channel]
    if direction not in dirs:
        direction = next(iter(dirs))
    raw = dirs[direction].get("meta") or {}
    flat = _flatten_meta(raw)
    return {"facts": _facts(item["type"], flat), "all": [[k, v] for k, v in flat.items()]}


def _fmt(v):
    if isinstance(v, (bytes, bytearray)):
        v = bytes(v).decode("latin-1", "replace")
    if isinstance(v, str):
        out = v.strip()
    elif isinstance(v, (bool, np.bool_)):
        out = str(bool(v))
    elif isinstance(v, (int, np.integer)):
        out = str(int(v))
    elif isinstance(v, (float, np.floating)):
        out = "%.6g" % v
    elif isinstance(v, (list, tuple, np.ndarray)):
        items = np.ravel(v).tolist() if isinstance(v, np.ndarray) else list(v)
        out = "[%d values]" % len(items) if len(items) > 12 else ", ".join(_fmt(x) for x in items)
    else:
        out = str(v)
    return out if len(out) <= 300 else out[:300] + "..."


def _flatten_meta(raw, prefix="", out=None):
    out = {} if out is None else out
    for k, v in raw.items():
        name = prefix + str(k)
        if isinstance(v, dict):
            if not any(str(o).startswith(name + ">") for o in raw):   # sxm repeats these as 'a>b' keys
                _flatten_meta(v, name + ".", out)
        else:
            out[name] = _fmt(v)
    return out


def _facts(ftype, flat):
    low = {k.lower(): v for k, v in flat.items()}

    def get(*names):
        for n in names:
            if low.get(n.lower()):
                return low[n.lower()]
        return ""

    def nums(name):
        try:
            return [float(t) for t in get(name).split(",")]
        except ValueError:
            return []

    def size_nm(name):
        v = nums(name)
        return " x ".join("%.4g" % (x * 1e9) for x in v) + " nm" if v else ""

    def pos_nm(name):
        v = nums(name)
        return ", ".join("%.4g" % (x * 1e9) for x in v) + " nm" if v else ""

    def unit(value, u):
        return "%s %s" % (value, u) if value else ""

    channel = ""
    if get("name"):
        channel = "%s (%s), %s" % (get("name"), get("unit"), get("direction"))
    if ftype == "sxm":
        facts = [("Date", (get("rec_date") + " " + get("rec_time")).strip()),
                 ("Channel", channel),
                 ("Bias", unit(get("bias"), "V")),
                 ("Setpoint", unit(get("z-controller>setpoint"), get("z-controller>setpoint unit"))),
                 ("Scan size", size_nm("scan_range")),
                 ("Pixels", get("scan_pixels").replace(",", " x")),
                 ("Angle", unit(get("scan_angle"), "deg")),
                 ("Offset", pos_nm("scan_offset")),
                 ("Scan direction", get("scan_dir")),
                 ("Acquisition", unit(get("acq_time"), "s")),
                 ("Temperature", unit(get("rec_temp"), "K")),
                 ("Comment", get("comment"))]
    elif ftype == "dat":
        pos = pos_nm("x (m)") and (pos_nm("x (m)") + " ; " + pos_nm("y (m)"))
        facts = [("Experiment", get("experiment")), ("Date", get("date")), ("Channel", channel),
                 ("Position X ; Y", pos), ("Z", pos_nm("z (m)")),
                 ("Settling time", unit(get("settling time (s)"), "s")),
                 ("Integration time", unit(get("integration time (s)"), "s")),
                 ("Comment", get("comment"))]
    elif ftype == "3ds":
        facts = [("Experiment", get("experiment_name")), ("Start", get("start_time")),
                 ("End", get("end_time")), ("Channel", channel),
                 ("Grid size", size_nm("size_xy")), ("Grid pixels", get("dim_px").replace(",", " x")),
                 ("Angle", unit(get("angle"), "deg")),
                 ("Sweep signal", get("sweep_signal")),
                 ("Sweep start", get("bias spectroscopy>sweep start (v)")),
                 ("Sweep end", get("bias spectroscopy>sweep end (v)")),
                 ("Comment", get("comment"))]
    else:
        facts = list(flat.items())[:6]
    return [[k, v] for k, v in facts if v]
