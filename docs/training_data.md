# Training data

ImSpec Labeler can turn your labels into arrays that are ready for training. This page describes what is
saved, how the dataset is built, and every setting that changes the result.

[Back to the README](../README.md)

- [Overview](#overview)
- [1. While labeling: one array per file](#1-while-labeling-one-array-per-file)
- [2. Build dataset: the stacked files](#2-build-dataset-the-stacked-files)
- [3. Build settings](#3-build-settings)
- [4. Normalization](#4-normalization)
- [5. NaN options](#5-nan-options)
- [6. Rebuilding](#6-rebuilding)
- [7. Using the files](#7-using-the-files)
- [8. Running a model on a new file](#8-running-a-model-on-a-new-file)

## Overview

Everything is under **Data → Training dataset** in the app, and everything is written inside
`labeled/training_data/` of your data folder.

```
labeled/
  classification_log.csv          the labels (one row per file and mode)
  dataset.json                    the build settings of this page
  training_data/
    arrays/                       1. one array per labeled file, saved while you label
      scan_001.sxm__Z__forward__offset.npy   (+ .json)
    binary_image_input.npy        2. the stacked files, written by "Build dataset"
    binary_image_output.npy
    binary_image_aux.npy
    binary_image_normalize.json
    binary_image_meta.csv
    binary_image_info.json
```

There are two steps:

1. **While you label**, each labeled file gets its own array, untouched and at full resolution.
2. **Build dataset** stacks those arrays into the files you train on. Resizing, normalization and the NaN
   options happen here, so you can change them and rebuild without labeling again.

## 1. While labeling: one array per file

Switch: *Save a training array for every labeled file*. It is **off by default**; tick it before you label.
The choice is remembered in the browser. Files labeled while it was off are not lost: their arrays are made
from the original files when you build.

Each labeled file gets one `.npy` in `labeled/training_data/arrays/`, and a small `.json` next to it that
describes it (channel, flatten, shape, scan size, bias range, mean, std).

| Data | Saved array | Notes |
| --- | --- | --- |
| Image (`.sxm`, a `.3ds` slice, a grey image) | `float32 (H, W)` | Full resolution, in the units of the file (for example m or A), flattened **as you viewed it**. NaN is kept (partial scans). A `.3ds` saves the slice that was on screen. Row 0 is the first row of the file; `origin` in the `.json` says whether that is the bottom or the top. |
| Spectrum (`.dat`) | `float32 (rows, N)` | Row 0 is the **bias**, row 1 the labeled channel, row 2 its backward sweep if the file has one. |
| Colour photo | `uint8 (H, W, 3)` | |

- **Deferred files** get no array, because they have no output.
- **Undo and overwrite** remove or replace the array together with the log entry.
- **The log** records the array in its `array` column, and the mean and standard deviation of what was labeled
  in `sample_mean` and `sample_std`. Those two are written for every entry, also with saving switched off.
  They use non-NaN values only, in the units of the file, after the flatten. For spectra they describe the
  signal, not the bias.
- **View options are not applied.** Transpose, origin, colormap and contrast only change what you see.

## 2. Build dataset: the stacked files

The **Build dataset** button writes one set of files per **mode** (`binary`, `multiclass`, `score`) and per
**kind** of data (`image`, `spectrum`, `rgb` for colour photos). Images and spectra are never mixed.

| File | Content |
| --- | --- |
| `<mode>_<kind>_input.npy` | `float32 (n, ...)`. One sample per labeled file, files oldest first. |
| `<mode>_<kind>_output.npy` | `(n,)`. The `class_id` for binary and multiclass (for example 1 = Good, 0 = Bad). The score for score mode. |
| `<mode>_<kind>_aux.npy` | `float64 (n, k)`. Extra numbers per sample, taken from the data as measured, before any resizing. See [the aux columns](#aux-columns). |
| `<mode>_<kind>_normalize.json` | How the three arrays were normalized, with the numbers needed to repeat it. |
| `<mode>_<kind>_meta.csv` | One line per sample with the **raw** values: file, channel, flatten, original shape, sizes, mean, std. Same order as the arrays. |
| `<mode>_<kind>_info.json` | Shapes, the layout of a sample, what the class numbers mean, the transform that was used. |

Shapes of `_input.npy`, without the NaN mask:

| Kind | Shape |
| --- | --- |
| `image` | `(n, H, W)` |
| `spectrum` | `(n, rows, P)`: `[:, 0]` is the bias, `[:, 1]` the signal, `[:, 2]` the backward sweep. A file without a backward sweep has NaN there. |
| `rgb` | `(n, H, W, 3)` |

If the class ids of a set are not all numbers (text ids, or older log rows without one), the output holds an
index into the label names instead; `_info.json` lists which number stands for which label.

Deferred files are left out. Labels made before this feature existed, or with saving switched off, are filled in
from the original files when you build.

### Aux columns

| Kind | Column | Meaning |
| --- | --- | --- |
| image, rgb | `width`, `height` | Size of the scan. nm for Nanonis files, pixels for ordinary images. |
| image, rgb | `pixel_x`, `pixel_y` | Size of one pixel. Always 1 for ordinary images. |
| spectrum | `bias_min`, `bias_max` | Bias range of the sweep, in V. |
| spectrum | `n_points` | Number of points of the sweep. |
| all | `sample_mean`, `sample_std` | Mean and standard deviation of the sample (of the signal, for spectra). |

The aux file is written for every kind and every setting. `sample_mean` and `sample_std` are always in it.

## 3. Build settings

The settings are the text box above the button. They are saved to `labeled/dataset.json` when you build, and
loaded when you open the folder again. The box is checked while you type; the button is disabled until it is
valid.

```jsonc
{
  "image":    { "size": null, "method": "interpolate" },
  "spectrum": { "points": null, "bias_range": "union" },
  "normalize": "zscore",
  "nan_to_zero": false,
  "nan_mask": false
}
```

The saved per-file arrays are never changed by any of these.

### Images

| Setting | Values | Effect |
| --- | --- | --- |
| `size` | `null` | All images must already have the same size. If they do not, the build stops and lists the sizes it found. |
| | `[height, width]` | Every image is brought to this size, in pixels. |
| `method` | `"interpolate"` | Resampled to the size. NaN pixels are left out of the average. When shrinking a lot, blocks are averaged first. |
| | `"pad_crop"` | Centred, then cropped or padded with NaN. No resampling, so the pixel size stays the same. |

Resizing loses the physical scale: a 15 nm scan and a 100 nm scan both become the same number of pixels. The
scale is kept in the aux columns `width`, `height`, `pixel_x` and `pixel_y`.

### Spectra

| Setting | Values | Effect |
| --- | --- | --- |
| `bias_range` | `"union"` | One bias axis from the lowest minimum to the highest maximum of all files. A file is NaN where it did not measure. Nothing is cut. |
| | `"overlap"` | Only the range that every file covers. No NaN, but data outside it is dropped. |
| | `[min, max]` | The range you give, in V. NaN where a file did not measure. |
| | `"strict"` | No resampling: each sample keeps its own bias values. All spectra must have the same length. |
| `points` | `null` | As many points as the longest spectrum. |
| | a whole number | That many points on the shared bias axis. |

- The bias axis is always row 0 of every sample. After resampling it is the same for every sample.
- Each file's own range, as measured, is in the aux columns `bias_min`, `bias_max` and `n_points`.
- **Multi-segment sweeps (MLS)** have unevenly spaced bias values. Resampling puts them on one even axis, so a
  finely stepped segment loses resolution unless `points` is large enough. To keep the fine segment, set
  `points` to about `(bias_max - bias_min) / smallest step`. `"strict"` keeps the uneven axis exactly.

## 4. Normalization

`"normalize"` has two values. It is one switch for all three arrays.

| Value | Effect |
| --- | --- |
| `"zscore"` (default) | The arrays are normalized as in the table below. |
| `"none"` | Nothing is normalized. Everything stays in the units of the file. |

All statistics use **non-NaN values only**.

| Array | Scope | Formula |
| --- | --- | --- |
| `_input.npy` | **Per sample.** Each image or spectrum uses its own mean and std. | `(x - sample_mean) / sample_std` |
| `_aux.npy` | **Over the whole dataset.** One mean and one std per column. | `(v - mean) / std`, with a log first for some columns (below) |
| `_output.npy`, score mode | **Over the whole dataset.** One mean and one std for all scores. | `(score - mean) / std` |
| `_output.npy`, binary and multiclass | Never normalized. Class numbers are names of categories, not quantities. | |

Details:

- **Input, per sample.** The mean and std are those of the saved full-resolution array, the same numbers as
  `sample_mean` and `sample_std` in the log. A sample that was resized is therefore close to, but not exactly,
  mean 0 and std 1. There is no second pass over the dataset.
- **Spectra.** Only the signal rows are normalized, forward and backward together. The bias row stays in V.
- **Amplitude is not lost.** Per-sample normalization removes the height range of each sample; that is why
  `sample_mean` and `sample_std` are aux columns.

### Which aux columns get a logarithm first

A log is taken for values that are always positive and span orders of magnitude. Its base does not change a
z-score.

| Kind | Aux column | Log first? |
| --- | --- | --- |
| image, rgb | `width`, `height` | yes |
| image, rgb | `pixel_x`, `pixel_y` | yes |
| image, rgb | `sample_mean` | no (it can be negative; it is near 0 after a flatten) |
| image, rgb | `sample_std` | yes |
| spectrum | `bias_min`, `bias_max` | no (they can be negative; a linear scale) |
| spectrum | `n_points` | no |
| spectrum | `sample_mean` | no |
| spectrum | `sample_std` | yes |

Scores get no log, only the z-score.

### Values without spread

If a column has the same value for every sample, for example every scan is 5 nm wide, its std is 0. The column
then becomes 0 everywhere and is marked `"constant": true` in `normalize.json`. The same holds for scores that
are all equal, and for a sample whose values are all equal (it becomes all zeros).

For a log column, a value of 0 (the std of a constant image) is first raised to the smallest positive value in
that column. That value is recorded as `floor`.

### normalize.json

One file per built set, for example `binary_image_normalize.json`. It records:

- the method and scope for the input;
- for every aux column: whether a log was taken, its `floor`, the `mean`, the `std`, and `constant`;
- for scores: the `mean` and `std`, and how to convert back (`score = z * std + mean`);
- the two NaN options that were used.

With `"normalize": "none"` the file is still written. The arrays then hold the raw values, and the file lists
the mean and std each column would have used.

The raw value of everything, per sample, is always readable in `_meta.csv`.

## 5. NaN options

Two checkboxes next to the button. They are the same settings as `"nan_to_zero"` and `"nan_mask"` in the text
box: ticking a box changes the text, and editing the text changes the box. Both apply to the built files only
(the per-file arrays always keep their NaN), and both are applied last, after the normalization.

| Checkbox | Setting | Effect |
| --- | --- | --- |
| *NaN to zero* | `"nan_to_zero": true` | NaN becomes 0 in `_input.npy` and `_aux.npy`. |
| *Add NaN mask* | `"nan_mask": true` | A mask is added to `_input.npy`: **1 = data, 0 = was NaN**. |

The mask changes the shape of the input:

| Kind | Without mask | With mask |
| --- | --- | --- |
| `image` | `(n, H, W)` | `(n, 2, H, W)`: `[:, 0]` is the data, `[:, 1]` the mask |
| `spectrum`, no file has a backward sweep | `(n, 2, P)` | `(n, 3, P)`: bias, forward, mask of forward |
| `spectrum`, at least one file has a backward sweep | `(n, 3, P)` | `(n, 5, P)`: bias, forward, backward, mask of forward, mask of backward |
| `rgb` | `(n, H, W, 3)` | `(n, H, W, 4)`: the mask is the 4th channel |

- The bias row has no mask, because it is never NaN.
- The number of spectrum rows is set by the whole set: if one file has a backward sweep, every sample gets the
  backward row. For a file without one, that row is NaN and its mask row is 0.
- Tick both to get zero-filled data together with a mask that tells which points were filled.
- After a z-score, 0 is the sample's average, so it is a neutral fill. With `"normalize": "none"`, 0 means zero
  in the units of the file (0 m, 0 A), which is not the average; the mask matters more in that case.
- `_info.json` states the layout that was used.

## 6. Rebuilding

Every build starts from the current `classification_log.csv` and writes all the built files again. The dataset
means and stds are recalculated from what is in the log at that moment.

- **New files** enter the dataset once you have labeled them.
- **Edits to the log by hand** are picked up: changed labels, class numbers and scores, and deleted rows.
  `sample_mean` and `sample_std` are filled back in from the saved array.
- **Per-file arrays are not recalculated.** To change one (another flatten, another channel), label that file
  again in the app.
- **Old sets are not deleted.** If a mode or kind has no samples any more, its files from an earlier build stay
  in `training_data/`. Delete them yourself.
- **Keep `normalize.json` with each trained model.** A rebuild changes the dataset statistics, so a model
  trained earlier needs the file from its own build.

## 7. Using the files

```python
import numpy as np

d = "labeled/training_data/"
X   = np.load(d + "binary_image_input.npy")    # (n, H, W) float32
y   = np.load(d + "binary_image_output.npy")   # (n,)      1 = Good, 0 = Bad
aux = np.load(d + "binary_image_aux.npy")      # (n, 6)    width, height, pixel_x, pixel_y, sample_mean, sample_std

X = np.nan_to_num(X, nan=0.0)                  # not needed if "NaN to zero" was ticked

S = np.load(d + "binary_spectrum_input.npy")   # (n, rows, P)
bias, signal = S[:, 0], S[:, 1]
```

With the NaN mask:

```python
X = np.load(d + "binary_image_input.npy")      # (n, 2, H, W)
data, mask = X[:, 0], X[:, 1]                  # mask: 1 = data, 0 = was NaN
```

Score mode, converting predictions back to scores:

```python
import json
norm = json.load(open(d + "score_image_normalize.json"))["output"]
scores = predictions * norm["std"] + norm["mean"]
```

## 8. Running a model on a new file

Apply the same steps, in the same order, as the build did:

1. Take the same channel and apply the flatten that the log records (`flatten` column).
2. Compute the sample's own mean and std from its non-NaN values (for a spectrum: from the signal).
3. Resize or resample as `_info.json` says (`transform`).
4. Normalize the input with the sample's own mean and std.
5. Build the aux values and normalize each column with the `mean` and `std` from `normalize.json`, after a log
   where `"log": true`. A column marked `"constant": true` is always 0.
6. Apply the NaN options that `normalize.json` records.
