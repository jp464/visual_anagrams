#!/usr/bin/env python3
"""Export edited visual anagrams as a flat Gorilla stimulus set."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REQUIRED_COLUMNS = ("orientation", "concept_1", "concept_2", "seed")
RAW_SOURCE_SIZE = 256
SAFE_FIELD = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
SHINIER_MODES = {
    "none": 0,
    "luminance": 1,
    "histogram": 2,
    "spatial-frequency": 3,
    "histogram-then-spatial-frequency": 5,
    "spatial-frequency-then-histogram": 7,
}
FFT_PADDING_MODES = {
    "none": 0,
    "reflect": 1,
    "symmetric": 2,
    "constant": 3,
}
SHINIER_VERSION_PIN = "0.2.3"


class ExportError(ValueError):
    """Raised when selections or source results cannot be exported safely."""


@dataclass(frozen=True)
class Transform:
    filename_prefix: str
    transformed_view: str
    pil_transpose_name: str


TRANSFORMS = {
    "rotate_180": Transform("r180", "rot180", "ROTATE_180"),
    # Pillow names transpose operations by their counter-clockwise equivalent;
    # ROTATE_270 is therefore a 90-degree clockwise rotation.
    "rotate_cw": Transform("r90cw", "rot90cw", "ROTATE_270"),
}


@dataclass(frozen=True)
class Selection:
    orientation: str
    concept_1: str
    concept_2: str
    seed: str
    row_number: int


@dataclass(frozen=True)
class ExportItem:
    selection: Selection
    class_pair: str
    source: Path
    item_id: str
    identity_filename: str
    transformed_filename: str


@dataclass(frozen=True)
class ShineConfig:
    """Reproducible SHINIER settings for one canonical-image batch."""

    mode: str = "luminance"
    target_mean: float = 0.0
    target_std: float = 0.0
    iterations: int = 5
    seed: int = 0
    legacy_mode: bool = False
    fft_padding: str = "reflect"
    display_background: int = 255

    def __post_init__(self) -> None:
        if self.mode not in SHINIER_MODES:
            raise ExportError(f"Unsupported SHINIER mode: {self.mode!r}.")
        if self.fft_padding not in FFT_PADDING_MODES:
            raise ExportError(
                f"Unsupported SHINIER FFT padding mode: {self.fft_padding!r}."
            )
        if not 0 <= self.target_mean <= 255:
            raise ExportError("SHINIER target mean must be between 0 and 255.")
        if self.target_std < 0:
            raise ExportError(
                "SHINIER target standard deviation must be non-negative."
            )
        if self.iterations <= 0:
            raise ExportError("SHINIER iterations must be a positive integer.")
        if not 0 <= self.display_background <= 255:
            raise ExportError("Display background must be between 0 and 255.")

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    @property
    def effective_iterations(self) -> int:
        return self.iterations if SHINIER_MODES[self.mode] in (5, 7) else 1


def _clean_field(value: str | None, column: str, row_number: int) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        raise ExportError(f"Row {row_number}: {column} must not be empty.")
    return cleaned


def read_selections(path: Path) -> list[Selection]:
    """Read and validate the selection CSV without coercing seeds to numbers."""
    try:
        csv_file = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise ExportError(f"Could not open selection CSV {path}: {exc}") from exc

    with csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ExportError(f"Selection CSV {path} has no header row.")

        fieldnames = [name.strip() for name in reader.fieldnames if name is not None]
        missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise ExportError(
                "Selection CSV is missing required column(s): " + ", ".join(missing)
            )

        selections: list[Selection] = []
        seen: dict[tuple[str, tuple[str, str], str], int] = {}
        for row_number, row in enumerate(reader, start=2):
            if not any((value or "").strip() for value in row.values()):
                continue

            orientation = _clean_field(row.get("orientation"), "orientation", row_number)
            concept_1 = _clean_field(row.get("concept_1"), "concept_1", row_number)
            concept_2 = _clean_field(row.get("concept_2"), "concept_2", row_number)
            seed = _clean_field(row.get("seed"), "seed", row_number)

            if orientation not in TRANSFORMS:
                supported = ", ".join(sorted(TRANSFORMS))
                raise ExportError(
                    f"Row {row_number}: unsupported orientation {orientation!r}; "
                    f"expected one of: {supported}."
                )
            for column, value in (("concept_1", concept_1), ("concept_2", concept_2)):
                if not SAFE_FIELD.fullmatch(value):
                    raise ExportError(
                        f"Row {row_number}: {column}={value!r} must contain only "
                        "letters, numbers, underscores, and hyphens."
                    )
            if not seed.isdigit():
                raise ExportError(f"Row {row_number}: seed={seed!r} must contain only digits.")

            pair_key = tuple(sorted((concept_1, concept_2)))
            key = (orientation, pair_key, seed)
            if key in seen:
                raise ExportError(
                    f"Row {row_number}: duplicate selection (first listed on row {seen[key]})."
                )
            seen[key] = row_number
            selections.append(
                Selection(orientation, concept_1, concept_2, seed, row_number=row_number)
            )

    if not selections:
        raise ExportError(f"Selection CSV {path} contains no selections.")
    return selections


def _result_matches(
    results_dir: Path, selection: Selection
) -> list[tuple[Path, str, bool]]:
    prefix = f"{selection.orientation}.pencil_sketch_of."
    suffixes = [(f".{selection.concept_1}_{selection.concept_2}", False)]
    if selection.concept_1 != selection.concept_2:
        suffixes.append((f".{selection.concept_2}_{selection.concept_1}", True))
    matches: list[tuple[Path, str, bool]] = []

    try:
        candidates = results_dir.iterdir()
    except OSError as exc:
        raise ExportError(f"Could not read results directory {results_dir}: {exc}") from exc

    for candidate in candidates:
        name = candidate.name
        if not candidate.is_dir() or not name.startswith(prefix):
            continue
        for suffix, reversed_pair in suffixes:
            if not name.endswith(suffix):
                continue
            class_pair = name[len(prefix) : len(name) - len(suffix)]
            if class_pair and SAFE_FIELD.fullmatch(class_pair):
                matches.append((candidate, class_pair, reversed_pair))
            break
    return sorted(matches, key=lambda match: match[0].name)


def _find_edited_source(seed_dir: Path, row_number: int) -> Path:
    if not seed_dir.is_dir():
        raise ExportError(f"Row {row_number}: seed directory not found: {seed_dir}")

    try:
        matches = sorted(
            path
            for path in seed_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".png"
            and path.stem.endswith("edited")
        )
    except OSError as exc:
        raise ExportError(
            f"Row {row_number}: could not read seed directory {seed_dir}: {exc}"
        ) from exc

    if not matches:
        raise ExportError(
            f"Row {row_number}: no PNG whose name ends in 'edited' found in {seed_dir}"
        )
    if len(matches) > 1:
        choices = "\n".join(f"  - {path.name}" for path in matches)
        raise ExportError(
            f"Row {row_number}: multiple edited source images found in {seed_dir}:"
            f"\n{choices}"
        )
    return matches[0]


def resolve_items(
    selections: Iterable[Selection],
    results_dir: Path,
    size: int,
    raw_only: bool = False,
) -> list[ExportItem]:
    """Resolve selections to source images and preflight every output name."""
    if size <= 0:
        raise ExportError("Image size must be a positive integer.")
    if not results_dir.is_dir():
        raise ExportError(f"Results directory does not exist: {results_dir}")

    items: list[ExportItem] = []
    output_names: dict[str, int] = {}
    for selection in selections:
        matches = _result_matches(results_dir, selection)
        pair = f"{selection.concept_1}/{selection.concept_2}"
        if not matches:
            raise ExportError(
                f"Row {selection.row_number}: no result directory found for "
                f"{selection.orientation} and pair {pair}."
            )
        if len(matches) > 1:
            choices = "\n".join(f"  - {path.name}" for path, _, _ in matches)
            raise ExportError(
                f"Row {selection.row_number}: multiple result directories match "
                f"{selection.orientation} and pair {pair}:\n{choices}"
            )

        result_dir, class_pair, reversed_pair = matches[0]
        if reversed_pair:
            selection = Selection(
                selection.orientation,
                selection.concept_2,
                selection.concept_1,
                selection.seed,
                selection.row_number,
            )
        seed_dir = result_dir / selection.seed
        source = (
            seed_dir / f"sample_{RAW_SOURCE_SIZE}.png"
            if raw_only
            else _find_edited_source(seed_dir, selection.row_number)
        )
        if not source.is_file():
            raise ExportError(f"Row {selection.row_number}: source image not found: {source}")

        transform = TRANSFORMS[selection.orientation]
        class_slug = class_pair.lower().replace("_", "-")
        item_id = (
            f"{transform.filename_prefix}__{class_slug}__"
            f"{selection.concept_1}--{selection.concept_2}__s{selection.seed}"
        )
        identity_filename = f"{item_id}__identity-{selection.concept_1}.png"
        transformed_filename = (
            f"{item_id}__{transform.transformed_view}-{selection.concept_2}.png"
        )
        for filename in (identity_filename, transformed_filename):
            if filename in output_names:
                raise ExportError(
                    f"Row {selection.row_number}: output filename {filename!r} collides "
                    f"with row {output_names[filename]}."
                )
            output_names[filename] = selection.row_number

        items.append(
            ExportItem(
                selection=selection,
                class_pair=class_pair,
                source=source,
                item_id=item_id,
                identity_filename=identity_filename,
                transformed_filename=transformed_filename,
            )
        )
    return items


def _load_pillow():
    try:
        from PIL import Image
    except ImportError as exc:
        raise ExportError(
            "Pillow is required to validate and rotate images. Install the project's "
            "environment.yml dependencies before running this script."
        ) from exc
    return Image


def validate_sources(items: Iterable[ExportItem], size: int, image_module) -> None:
    for item in items:
        try:
            with image_module.open(item.source) as image:
                image.verify()
            with image_module.open(item.source) as image:
                if image.size != (size, size):
                    raise ExportError(
                        f"Row {item.selection.row_number}: expected a {size}x{size} image, "
                        f"but {item.source} is {image.width}x{image.height}."
                    )
        except ExportError:
            raise
        except (OSError, SyntaxError) as exc:
            raise ExportError(
                f"Row {item.selection.row_number}: invalid source image {item.source}: {exc}"
            ) from exc


def _source_for_manifest(source: Path, results_dir: Path) -> str:
    try:
        return source.relative_to(results_dir.parent).as_posix()
    except ValueError:
        return source.as_posix()


def _manifest_rows(
    item: ExportItem,
    results_dir: Path,
    source_size: int,
    output_size: int,
    shine_config: ShineConfig,
    shine_version: str,
    quality_control: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    selection = item.selection
    common = {
        "item_id": item.item_id,
        "orientation": selection.orientation,
        "class_pair": item.class_pair,
        "concept_1": selection.concept_1,
        "concept_2": selection.concept_2,
        "seed": selection.seed,
        "source_path": _source_for_manifest(item.source, results_dir),
        "source_size": str(source_size),
        "output_size": str(output_size),
        "shine_mode": shine_config.mode,
        "shine_version": shine_version,
        "shine_legacy_mode": str(shine_config.legacy_mode).lower(),
        "shine_seed": str(shine_config.seed),
        "shine_iterations": str(shine_config.effective_iterations),
        "shine_target_mean": (
            "auto" if shine_config.target_mean == 0 else str(shine_config.target_mean)
        ),
        "shine_target_std": (
            "auto" if shine_config.target_std == 0 else str(shine_config.target_std)
        ),
        "shine_fft_padding": shine_config.fft_padding,
        "display_background": str(shine_config.display_background),
    }
    transform = TRANSFORMS[selection.orientation]
    return [
        {
            **common,
            "filename": item.identity_filename,
            "view": "identity",
            "target": selection.concept_1,
            **quality_control[item.identity_filename],
        },
        {
            **common,
            "filename": item.transformed_filename,
            "view": transform.transformed_view,
            "target": selection.concept_2,
            **quality_control[item.transformed_filename],
        },
    ]


def _raw_manifest_row(
    item: ExportItem, results_dir: Path, source_size: int
) -> dict[str, str]:
    selection = item.selection
    return {
        "item_id": item.item_id,
        "filename": f"{item.item_id}__raw.png",
        "orientation": selection.orientation,
        "class_pair": item.class_pair,
        "concept_1": selection.concept_1,
        "concept_2": selection.concept_2,
        "seed": selection.seed,
        "view": "raw",
        "target": selection.concept_1,
        "source_path": _source_for_manifest(item.source, results_dir),
        "source_size": str(source_size),
        "output_size": str(source_size),
    }


def _write_readme(
    path: Path,
    selection_count: int,
    source_size: int,
    output_size: int,
    shine_config: ShineConfig,
    shine_version: str,
) -> None:
    padding = _mask_padding(output_size)
    shine_description = (
        f"SHINIER {shine_version} mode `{shine_config.mode}` was applied to the "
        "complete canonical-image batch within a shared binary circular ROI before "
        "the presentation alpha mask. "
        if shine_config.enabled
        else "SHINIER processing was explicitly disabled. "
    )
    path.write_text(
        "# Gorilla stimuli\n\n"
        f"This export contains {selection_count * 2} PNG files from "
        f"{selection_count} selected visual anagrams at {output_size}x{output_size} "
        f"pixels. Each {source_size}x{source_size} source was converted to grayscale "
        "and centered without resizing on a smooth background fitted from its "
        "border pixels after foreground-like outliers were rejected. The source edge "
        f"was feathered into that background over {_feather_width(source_size)} pixels. "
        + shine_description
        + "Only canonical identity images entered target calculation; transformed "
        "images were created afterward as exact rotations.\n\n"
        "Upload every file in `images/` to the Gorilla Stimuli tab. Gorilla uses a "
        "flat stimulus namespace, so spreadsheet cells should contain the exact value "
        "from the `filename` column in `manifest.csv` (without `images/`).\n\n"
        "Each `item_id` has two rows: an `identity` view and its transformed view. "
        "The transformed file is an exact rotation of the processed identity file. "
        f"Every PNG contains the complete source inside a centered circular mask, a "
        f"{padding}-pixel transparent margin, and transparent pixels outside the "
        "circle. `block_1.csv` contains identity views, `block_2.csv` contains "
        "transformed views, and `block_3.csv` repeats the identity views in a "
        "different shuffled order.\n\n"
        f"Quality-control values in `manifest.csv` were measured in linear-light "
        f"luminance after compositing the PNG in sRGB onto grayscale background "
        f"{shine_config.display_background}. "
        "Histogram RMSE and radial-spectrum NRMSE use the canonical identity set's "
        "mean profile as their reference. Full processing settings and metric "
        "definitions are recorded in `processing.json`.\n",
        encoding="utf-8",
    )


def _write_raw_readme(path: Path, selection_count: int, size: int) -> None:
    path.write_text(
        "# Gorilla stimuli (raw)\n\n"
        f"This export contains {selection_count} PNG files from "
        f"{selection_count} selected visual anagrams at {size}x{size} pixels. "
        "Each source PNG was copied byte-for-byte: it was not decoded, re-encoded, "
        "resized, recolored, masked, padded, or rotated. Only its filename changed "
        "to provide a unique flat Gorilla stimulus name.\n\n"
        "Upload every file in `images/` to the Gorilla Stimuli tab. Gorilla uses a "
        "flat stimulus namespace, so spreadsheet cells should contain the exact value "
        "from the `filename` column in `manifest.csv` (without `images/`). "
        "`block_1.csv` contains the raw views in shuffled order.\n",
        encoding="utf-8",
    )


def _mask_padding(size: int) -> int:
    return max(4, round(size / 64))


def _padded_canvas_size(width: int, height: int) -> int:
    """Return the smallest centered canvas whose circle contains the full image."""
    required_diameter = math.ceil(math.hypot(width, height)) + 2
    canvas_size = required_diameter
    while (
        (canvas_size - width) % 2
        or (canvas_size - height) % 2
        or canvas_size - 2 * _mask_padding(canvas_size) < required_diameter
    ):
        canvas_size += 1
    return canvas_size


def _load_numpy():
    try:
        import numpy
    except ImportError as exc:
        raise ExportError(
            "NumPy is required to create seamless background padding. Install the "
            "project's environment.yml dependencies before running this script."
        ) from exc
    return numpy


def _surface_terms(x, y, numpy_module):
    return numpy_module.column_stack(
        (
            numpy_module.ones(x.size),
            x,
            y,
            x * y,
            x * x,
            y * y,
        )
    )


def _fit_background_surface(image, canvas_size: int, offset_x: int, offset_y: int):
    """Fit and extrapolate a robust low-frequency background from an L image."""
    numpy = _load_numpy()
    pixels = numpy.asarray(image, dtype=numpy.float64)
    height, width = pixels.shape
    x_scale = max((width - 1) / 2, 1)
    y_scale = max((height - 1) / 2, 1)
    source_y, source_x = numpy.mgrid[0:height, 0:width]
    normalized_x = (source_x - (width - 1) / 2) / x_scale
    normalized_y = (source_y - (height - 1) / 2) / y_scale
    band = max(2, round(min(width, height) / 32))
    border = (
        (source_x < band)
        | (source_x >= width - band)
        | (source_y < band)
        | (source_y >= height - band)
    )
    sample_x = normalized_x[border]
    sample_y = normalized_y[border]
    sample_values = pixels[border]

    if sample_values.size < 6:
        background = numpy.full(
            (canvas_size, canvas_size),
            numpy.median(sample_values),
            dtype=numpy.float64,
        )
        return background

    terms = _surface_terms(sample_x, sample_y, numpy)
    inliers = numpy.ones(sample_values.size, dtype=bool)
    coefficients = numpy.linalg.lstsq(terms, sample_values, rcond=None)[0]
    for _ in range(6):
        residuals = sample_values - terms @ coefficients
        residual_center = numpy.median(residuals[inliers])
        deviation = numpy.abs(residuals - residual_center)
        mad = numpy.median(deviation[inliers])
        cutoff = max(4.0, 2.5 * 1.4826 * mad)
        updated = deviation <= cutoff
        if updated.sum() < 6 or numpy.array_equal(updated, inliers):
            break
        inliers = updated
        coefficients = numpy.linalg.lstsq(
            terms[inliers], sample_values[inliers], rcond=None
        )[0]

    canvas_y, canvas_x = numpy.mgrid[0:canvas_size, 0:canvas_size]
    canvas_x = (canvas_x - offset_x - (width - 1) / 2) / x_scale
    canvas_y = (canvas_y - offset_y - (height - 1) / 2) / y_scale
    background = (
        coefficients[0]
        + coefficients[1] * canvas_x
        + coefficients[2] * canvas_y
        + coefficients[3] * canvas_x * canvas_y
        + coefficients[4] * canvas_x * canvas_x
        + coefficients[5] * canvas_y * canvas_y
    )
    low, high = numpy.percentile(sample_values[inliers], (2, 98))
    allowance = max(3.0, (high - low) * 0.15)
    return numpy.clip(background, low - allowance, high + allowance)


def _feather_width(size: int) -> int:
    return max(2, round(size / 16))


def _feather_mask(width: int, height: int, image_module):
    """Create a cosine mask that preserves the interior and softens the edges."""
    numpy = _load_numpy()
    y, x = numpy.mgrid[0:height, 0:width]
    distance = numpy.minimum.reduce((x, width - 1 - x, y, height - 1 - y))
    progress = numpy.clip(
        distance / _feather_width(min(width, height)),
        0,
        1,
    )
    alpha = numpy.rint((0.5 - 0.5 * numpy.cos(numpy.pi * progress)) * 255)
    return image_module.fromarray(alpha.astype(numpy.uint8), mode="L")


def _pad_for_circular_mask(image, image_module):
    """Blend an image into a robustly extrapolated low-frequency background."""
    numpy = _load_numpy()
    canvas_size = _padded_canvas_size(*image.size)
    offset_x = (canvas_size - image.width) // 2
    offset_y = (canvas_size - image.height) // 2
    background = _fit_background_surface(image, canvas_size, offset_x, offset_y)
    padded = image_module.fromarray(
        numpy.clip(numpy.rint(background), 0, 255).astype(numpy.uint8),
        mode="L",
    )
    padded.paste(
        image,
        (offset_x, offset_y),
        _feather_mask(image.width, image.height, image_module),
    )
    return padded


def _circular_alpha_mask(size: tuple[int, int], image_module):
    """Create the antialiased alpha mask used for every final stimulus."""
    from PIL import ImageDraw

    width, height = size
    scale = 4
    mask_size = (width * scale, height * scale)
    padding = _mask_padding(min(width, height)) * scale
    mask = image_module.new("L", mask_size, 0)
    image_draw = ImageDraw.Draw(mask)
    image_draw.ellipse(
        (padding, padding, mask_size[0] - 1 - padding, mask_size[1] - 1 - padding),
        fill=255,
    )
    return mask.resize(size, resample=image_module.Resampling.LANCZOS)


def _apply_circular_mask(image, image_module):
    """Return an RGBA image with a centered circle and a visible clear margin."""
    rgba = image.convert("RGBA")
    rgba.putalpha(_circular_alpha_mask(rgba.size, image_module))
    return rgba


def _load_shinier():
    try:
        import shinier
    except ImportError as exc:
        raise ExportError(
            "SHINIER is required for processed exports. Install the project's "
            f"environment.yml dependencies (shinier=={SHINIER_VERSION_PIN}), or "
            "pass --shine-mode none to explicitly disable matching."
        ) from exc
    return shinier


def _apply_shinier_batch(
    images: Sequence,
    image_module,
    config: ShineConfig,
    work_dir: Path,
    shinier_module=None,
) -> tuple[list, str]:
    """Match one batch of canonical padded images within a shared circular ROI."""
    if not config.enabled:
        return [image.copy() for image in images], "disabled"
    if not images:
        raise ExportError("No padded images were available for SHINIER processing.")

    numpy = _load_numpy()
    shinier_module = shinier_module or _load_shinier()
    version = str(getattr(shinier_module, "__version__", "unknown"))
    mask_dir = work_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    # SHINIER caches this process-level root. The invocation directory remains
    # valid across multiple exports, unlike an individual output staging tree.
    buffer_temp_root = Path.cwd()

    reference_size = images[0].size
    if any(image.size != reference_size for image in images):
        raise ExportError("SHINIER requires every padded image to have the same size.")
    alpha = numpy.asarray(
        _circular_alpha_mask(reference_size, image_module), dtype=numpy.uint8
    )
    # SHINIER ROI masks are binary. Thresholding at half alpha excludes the
    # transparent corners while keeping the soft presentation edge separate.
    roi = numpy.where(alpha >= 128, 255, 0).astype(numpy.uint8)
    mask_path = mask_dir / "circular_roi.png"
    image_module.fromarray(roi, mode="L").save(mask_path, format="PNG")
    arrays = [numpy.asarray(image, dtype=numpy.uint8).copy() for image in images]

    previous_tempdir = tempfile.tempdir
    dataset = None
    try:
        # SHINIER uses tempfile.gettempdir() for disk-backed buffers. Keep those
        # potentially large, process-scoped buffers beside the export destination.
        # The stable parent path also supports multiple exports in one process.
        tempfile.tempdir = str(buffer_temp_root)
        options = shinier_module.Options(
            output_folder=work_dir,
            masks_folder=mask_dir,
            whole_image=3,
            background=300,
            mode=SHINIER_MODES[config.mode],
            seed=config.seed,
            legacy_mode=config.legacy_mode,
            iterations=config.effective_iterations,
            as_gray=True,
            linear_luminance=False,
            gamut_strategy="clip",
            dithering=0,
            conserve_memory=True,
            safe_lum_match=True,
            target_lum=(config.target_mean, config.target_std),
            hist_optim=False,
            hist_specification=4,
            fft_padding_mode=FFT_PADDING_MODES[config.fft_padding],
            fft_padding_value=(
                config.display_background if config.fft_padding == "constant" else 300
            ),
            verbose=-1,
        )
        # SHINIER 0.2.3 expects file provenance when it records mask names.
        dataset = shinier_module.ImageDataset(
            images=arrays, masks=[mask_path], options=options
        )
        processor = shinier_module.ImageProcessor(
            dataset=dataset, options=options, verbose=-1
        )
        results = processor.get_results()
        processed = []
        for index, result in enumerate(results):
            array = numpy.asarray(result)
            if array.ndim == 3 and array.shape[-1] == 1:
                array = array[..., 0]
            elif array.ndim == 3 and array.shape[-1] == 3:
                channel_delta = float(
                    numpy.max(numpy.ptp(array.astype(numpy.float64), axis=2))
                )
                if channel_delta > 1.0:
                    raise ExportError(
                        f"SHINIER returned non-neutral RGB data for grayscale image "
                        f"{index + 1} (maximum channel difference {channel_delta:.3f})."
                    )
                array = array.astype(numpy.float64).mean(axis=2)
            if array.shape != (reference_size[1], reference_size[0]):
                raise ExportError(
                    f"SHINIER returned image {index + 1} with shape {array.shape}; "
                    f"expected {(reference_size[1], reference_size[0])}."
                )
            array = numpy.clip(numpy.rint(array), 0, 255).astype(numpy.uint8)
            processed.append(image_module.fromarray(array, mode="L"))
        if len(processed) != len(images):
            raise ExportError(
                f"SHINIER returned {len(processed)} images for a batch of {len(images)}."
            )
        return processed, version
    except ExportError:
        raise
    except Exception as exc:
        raise ExportError(f"SHINIER processing failed: {exc}") from exc
    finally:
        if dataset is not None:
            try:
                dataset.close()
            except Exception:
                pass
        tempfile.tempdir = previous_tempdir
        shutil.rmtree(work_dir, ignore_errors=True)


def _composited_luminance(image, background: int, numpy_module):
    """Composite in sRGB code space, then return linear-light luminance on 0..255."""
    rgba = numpy_module.asarray(image.convert("RGBA"), dtype=numpy_module.float64)
    alpha = rgba[..., 3] / 255.0
    encoded = (rgba[..., 0] * alpha + background * (1.0 - alpha)) / 255.0
    linear = numpy_module.where(
        encoded <= 0.04045,
        encoded / 12.92,
        ((encoded + 0.055) / 1.055) ** 2.4,
    )
    return linear * 255.0, alpha


def _radial_spectrum_profile(luminance, numpy_module):
    """Return a unit-length radial FFT-magnitude profile with the DC bin removed."""
    centered = luminance - luminance.mean()
    magnitude = numpy_module.abs(
        numpy_module.fft.fftshift(numpy_module.fft.fft2(centered))
    )
    height, width = magnitude.shape
    y, x = numpy_module.ogrid[:height, :width]
    radii = numpy_module.floor(
        numpy_module.hypot(y - height // 2, x - width // 2)
    ).astype(int)
    limit = min(height, width) // 2
    sums = numpy_module.bincount(radii.ravel(), weights=magnitude.ravel())
    counts = numpy_module.bincount(radii.ravel())
    profile = sums[: limit + 1] / numpy_module.maximum(counts[: limit + 1], 1)
    profile[0] = 0.0
    norm = numpy_module.linalg.norm(profile)
    return profile / norm if norm else profile


def _quality_control(
    rendered: dict[str, object],
    identity_filenames: Sequence[str],
    display_background: int,
) -> dict[str, dict[str, str]]:
    """Measure final stimuli after alpha-compositing onto the display background."""
    numpy = _load_numpy()
    measurements = {}
    for filename, image in rendered.items():
        luminance, alpha = _composited_luminance(
            image, display_background, numpy
        )
        roi = alpha >= (128 / 255)
        visible = luminance[roi]
        values = numpy.clip(numpy.rint(visible), 0, 255).astype(numpy.uint8)
        histogram = numpy.bincount(values, minlength=256).astype(numpy.float64)
        histogram /= max(histogram.sum(), 1)
        underlying = numpy.asarray(
            image.convert("RGBA"), dtype=numpy.uint8
        )[..., 0][roi]
        measurements[filename] = {
            "mean": float(visible.mean()),
            "std": float(visible.std()),
            "clipped": float(numpy.mean((underlying == 0) | (underlying == 255))),
            "histogram": histogram,
            "spectrum": _radial_spectrum_profile(luminance, numpy),
        }

    references = [measurements[filename] for filename in identity_filenames]
    target_histogram = numpy.mean(
        [entry["histogram"] for entry in references], axis=0
    )
    target_spectrum = numpy.mean(
        [entry["spectrum"] for entry in references], axis=0
    )
    spectrum_norm = max(float(numpy.linalg.norm(target_spectrum)), 1e-12)
    qc = {}
    for filename, entry in measurements.items():
        qc[filename] = {
            "qc_mean_luminance": f"{entry['mean']:.6f}",
            "qc_rms_contrast": f"{entry['std']:.6f}",
            "qc_clipped_fraction": f"{entry['clipped']:.8f}",
            "qc_histogram_rmse": (
                f"{numpy.sqrt(numpy.mean((entry['histogram'] - target_histogram) ** 2)):.8f}"
            ),
            "qc_radial_spectrum_nrmse": (
                f"{numpy.linalg.norm(entry['spectrum'] - target_spectrum) / spectrum_norm:.8f}"
            ),
        }
    return qc


def _write_processing_metadata(
    path: Path, config: ShineConfig, version: str
) -> None:
    metadata = {
        "shinier": {
            "enabled": config.enabled,
            "version": version,
            "version_pin": SHINIER_VERSION_PIN,
            "mode": config.mode,
            "mode_id": SHINIER_MODES[config.mode],
            "legacy_mode": config.legacy_mode,
            "seed": config.seed,
            "iterations": config.effective_iterations,
            "target_mean": "auto" if config.target_mean == 0 else config.target_mean,
            "target_std": "auto" if config.target_std == 0 else config.target_std,
            "safe_luminance_matching": True,
            "histogram_specification": 1 if config.legacy_mode else 4,
            "dithering": "none",
            "input_transfer": (
                "MATLAB-compatible grayscale code values"
                if config.legacy_mode
                else "gamma-encoded sRGB"
            ),
            "luminance_space": (
                "MATLAB-compatible Rec.601 intensity"
                if config.legacy_mode
                else "CIE xyY Y via Rec.709"
            ),
            "fft_padding": config.fft_padding,
            "target_scope": "all canonical identity images",
            "roi": "binary circular mask derived from final alpha >= 128",
            "fourier_scope": (
                "full padded canvas; SHINIER 0.2.3 does not apply the ROI to FFTs"
            ),
        },
        "quality_control": {
            "display_background_gray": config.display_background,
            "mean_and_rms": "linear-light luminance (0..255) inside alpha >= 128 after sRGB compositing",
            "clipped_fraction": "underlying grayscale values equal to 0 or 255 inside alpha >= 128",
            "histogram_rmse": "RMSE from mean normalized identity linear-luminance histogram inside alpha >= 128",
            "radial_spectrum_nrmse": "relative L2 error from mean unit-length identity radial FFT magnitude in linear luminance after compositing; DC removed",
        },
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _write_block_spreadsheets(
    output_dir: Path, rows: Sequence[dict[str, str]], random_seed: int | None
) -> None:
    """Write transformed and repeated-identity Gorilla blocks."""
    identity = [row["filename"] for row in rows if row["view"] == "identity"]
    transformed = [row["filename"] for row in rows if row["view"] != "identity"]
    if not identity or len(identity) != len(transformed):
        raise ExportError("Cannot create balanced identity and transformed blocks.")

    rng = random.Random(random_seed)
    block_1 = identity.copy()
    block_2 = transformed.copy()
    rng.shuffle(block_1)
    rng.shuffle(block_2)

    block_3 = block_1.copy()
    rng.shuffle(block_3)
    if len(block_3) > 1 and block_3 == block_1:
        # Random shuffling can reproduce its input, especially in small pilots.
        # Rotate by a random non-zero offset so the repeat has a different order.
        offset = rng.randrange(1, len(block_3))
        block_3 = block_3[offset:] + block_3[:offset]

    for block_name, filenames in (
        ("block_1.csv", block_1),
        ("block_2.csv", block_2),
        ("block_3.csv", block_3),
    ):
        with (output_dir / block_name).open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.writer(file)
            writer.writerow(
                ("randomise_trials", "display", "stimulus", "deg", "Fixation")
            )
            writer.writerows(
                (1, "Trial", filename, 10, "fixation_dot.png")
                for filename in filenames
            )


def _write_raw_block_spreadsheet(
    output_dir: Path, rows: Sequence[dict[str, str]], random_seed: int | None
) -> None:
    """Write a single shuffled Gorilla block containing raw images."""
    filenames = [row["filename"] for row in rows if row["view"] == "raw"]
    if not filenames or len(filenames) != len(rows):
        raise ExportError("Cannot create a raw block from non-raw manifest rows.")

    random.Random(random_seed).shuffle(filenames)
    with (output_dir / "block_1.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ("randomise_trials", "display", "stimulus", "deg", "Fixation")
        )
        writer.writerows(
            (1, "Trial", filename, 10, "fixation_dot.png")
            for filename in filenames
        )


def export_items(
    items: Sequence[ExportItem],
    results_dir: Path,
    output_dir: Path,
    source_size: int,
    image_module,
    random_seed: int | None = None,
    raw_only: bool = False,
    shine_config: ShineConfig | None = None,
    shinier_module=None,
) -> None:
    """Write a complete export through a staging directory, then rename it into place."""
    if output_dir.exists():
        raise ExportError(
            f"Output path already exists: {output_dir}. Choose a new --output-dir."
        )
    shine_config = shine_config or ShineConfig()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        images_dir = staging / "images"
        images_dir.mkdir()
        rows: list[dict[str, str]] = []
        output_size: int | None = None

        if raw_only:
            for item in items:
                row = _raw_manifest_row(item, results_dir, source_size)
                shutil.copy2(item.source, images_dir / row["filename"])
                rows.append(row)
            output_size = source_size
            shine_version = "not-run-raw-export"
        else:
            padded_images = []
            for item in items:
                with image_module.open(item.source) as image:
                    processed = image.convert("L")
                    if processed.width != processed.height:
                        raise ExportError(
                            f"Row {item.selection.row_number}: processed image is not "
                            f"square: {processed.size}."
                        )
                    padded = _pad_for_circular_mask(processed, image_module)
                if output_size is None:
                    output_size = padded.width
                elif padded.size != (output_size, output_size):
                    raise ExportError(
                        f"Row {item.selection.row_number}: inconsistent processed size "
                        f"{padded.size}; expected {(output_size, output_size)}."
                    )
                padded_images.append(padded)

            matched_images, shine_version = _apply_shinier_batch(
                padded_images,
                image_module,
                shine_config,
                staging / ".shinier-work",
                shinier_module,
            )
            rendered = {}
            for item, matched in zip(items, matched_images):
                transform = TRANSFORMS[item.selection.orientation]
                transpose = getattr(
                    image_module.Transpose, transform.pil_transpose_name
                )
                identity = _apply_circular_mask(matched, image_module)
                transformed = identity.transpose(transpose)
                identity.save(images_dir / item.identity_filename, format="PNG")
                transformed.save(images_dir / item.transformed_filename, format="PNG")
                rendered[item.identity_filename] = identity
                rendered[item.transformed_filename] = transformed

            quality_control = _quality_control(
                rendered,
                [item.identity_filename for item in items],
                shine_config.display_background,
            )
            for item in items:
                rows.extend(
                    _manifest_rows(
                        item,
                        results_dir,
                        source_size,
                        output_size,
                        shine_config,
                        shine_version,
                        quality_control,
                    )
                )

        if output_size is None:
            raise ExportError("No items were available to export.")
        base_fieldnames = (
            "item_id",
            "filename",
            "orientation",
            "class_pair",
            "concept_1",
            "concept_2",
            "seed",
            "view",
            "target",
            "source_path",
            "source_size",
            "output_size",
        )
        qc_fieldnames = (
            "shine_mode",
            "shine_version",
            "shine_legacy_mode",
            "shine_seed",
            "shine_iterations",
            "shine_target_mean",
            "shine_target_std",
            "shine_fft_padding",
            "display_background",
            "qc_mean_luminance",
            "qc_rms_contrast",
            "qc_clipped_fraction",
            "qc_histogram_rmse",
            "qc_radial_spectrum_nrmse",
        )
        fieldnames = base_fieldnames if raw_only else base_fieldnames + qc_fieldnames
        with (staging / "manifest.csv").open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        if raw_only:
            _write_raw_block_spreadsheet(staging, rows, random_seed)
            _write_raw_readme(staging / "README.md", len(items), source_size)
        else:
            _write_block_spreadsheets(staging, rows, random_seed)
            _write_readme(
                staging / "README.md",
                len(items),
                source_size,
                output_size,
                shine_config,
                shine_version,
            )
            _write_processing_metadata(
                staging / "processing.json", shine_config, shine_version
            )
        staging.rename(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run_export(
    selection_csv: Path,
    results_dir: Path,
    output_dir: Path,
    size: int = 256,
    random_seed: int | None = None,
    raw_only: bool = False,
    shine_config: ShineConfig | None = None,
    shinier_module=None,
) -> tuple[int, int]:
    selections = read_selections(selection_csv)
    source_size = RAW_SOURCE_SIZE if raw_only else size
    items = resolve_items(selections, results_dir, size, raw_only=raw_only)
    image_module = None
    if not raw_only:
        image_module = _load_pillow()
        validate_sources(items, size, image_module)

    export_items(
        items,
        results_dir,
        output_dir,
        source_size,
        image_module,
        random_seed,
        raw_only,
        shine_config,
        shinier_module,
    )
    return len(items), len(items) if raw_only else len(items) * 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export selected visual anagrams into a flat Gorilla stimulus set."
    )
    parser.add_argument(
        "selection_csv",
        type=Path,
        help="CSV with orientation, concept_1, concept_2, and seed columns",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="source results directory (default: results)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gorilla_stimuli"),
        help="new export directory to create (default: gorilla_stimuli)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help=(
            "expected edited-image size in pixels (default: 256; ignored with "
            "--raw-only, which always uses sample_256.png)"
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="seed for reproducible block shuffling (default: random each run)",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="copy one source PNG per selection without any image transformation",
    )
    parser.add_argument(
        "--shine-mode",
        choices=tuple(SHINIER_MODES),
        default="luminance",
        help=(
            "SHINIER batch operation for processed exports (default: luminance; "
            "use none only to explicitly disable matching)"
        ),
    )
    parser.add_argument(
        "--shine-target-mean",
        type=float,
        default=0.0,
        help="target mean luminance for luminance mode; 0 chooses a batch target",
    )
    parser.add_argument(
        "--shine-target-std",
        type=float,
        default=0.0,
        help="target luminance standard deviation; 0 chooses a batch target",
    )
    parser.add_argument(
        "--shine-iterations",
        type=int,
        default=5,
        help="iterations for combined histogram/spatial-frequency modes (default: 5)",
    )
    parser.add_argument(
        "--shine-seed",
        type=int,
        default=0,
        help="SHINIER seed for reproducible matching (default: 0)",
    )
    parser.add_argument(
        "--shine-legacy-mode",
        action="store_true",
        help="use SHINIER's MATLAB-compatible legacy behavior",
    )
    parser.add_argument(
        "--shine-fft-padding",
        choices=tuple(FFT_PADDING_MODES),
        default="reflect",
        help="FFT boundary padding for spatial-frequency modes (default: reflect)",
    )
    parser.add_argument(
        "--display-background",
        type=int,
        default=255,
        help="Gorilla grayscale background used for final QC, 0-255 (default: 255)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        shine_config = ShineConfig(
            mode=args.shine_mode,
            target_mean=args.shine_target_mean,
            target_std=args.shine_target_std,
            iterations=args.shine_iterations,
            seed=args.shine_seed,
            legacy_mode=args.shine_legacy_mode,
            fft_padding=args.shine_fft_padding,
            display_background=args.display_background,
        )
        item_count, image_count = run_export(
            args.selection_csv,
            args.results_dir,
            args.output_dir,
            size=args.size,
            random_seed=args.random_seed,
            raw_only=args.raw_only,
            shine_config=shine_config,
        )
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: could not create export: {exc}", file=sys.stderr)
        return 1

    print(
        f"Exported {image_count} images from {item_count} selections to {args.output_dir}"
    )
    print(f"Manifest: {args.output_dir / 'manifest.csv'}")
    if args.raw_only:
        print(f"Block spreadsheet: {args.output_dir / 'block_1.csv'}")
    else:
        print(
            f"Block spreadsheets: {args.output_dir / 'block_1.csv'}, "
            f"{args.output_dir / 'block_2.csv'}, {args.output_dir / 'block_3.csv'}"
        )
        print(f"Processing metadata: {args.output_dir / 'processing.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
