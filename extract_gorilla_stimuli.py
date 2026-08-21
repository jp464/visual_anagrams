#!/usr/bin/env python3
"""Export selected visual anagrams as a flat Gorilla stimulus set."""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REQUIRED_COLUMNS = ("orientation", "concept_1", "concept_2", "seed")
SAFE_FIELD = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


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
        seen: dict[tuple[str, str, str, str], int] = {}
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

            key = (orientation, concept_1, concept_2, seed)
            if key in seen:
                raise ExportError(
                    f"Row {row_number}: duplicate selection (first listed on row {seen[key]})."
                )
            seen[key] = row_number
            selections.append(Selection(*key, row_number=row_number))

    if not selections:
        raise ExportError(f"Selection CSV {path} contains no selections.")
    return selections


def _result_matches(results_dir: Path, selection: Selection) -> list[tuple[Path, str]]:
    prefix = f"{selection.orientation}.pencil_sketch_of."
    suffix = f".{selection.concept_1}_{selection.concept_2}"
    matches: list[tuple[Path, str]] = []

    try:
        candidates = results_dir.iterdir()
    except OSError as exc:
        raise ExportError(f"Could not read results directory {results_dir}: {exc}") from exc

    for candidate in candidates:
        name = candidate.name
        if not candidate.is_dir() or not name.startswith(prefix) or not name.endswith(suffix):
            continue
        class_pair = name[len(prefix) : len(name) - len(suffix)]
        if class_pair and SAFE_FIELD.fullmatch(class_pair):
            matches.append((candidate, class_pair))
    return sorted(matches, key=lambda match: match[0].name)


def resolve_items(
    selections: Iterable[Selection], results_dir: Path, size: int
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
            choices = "\n".join(f"  - {path.name}" for path, _ in matches)
            raise ExportError(
                f"Row {selection.row_number}: multiple result directories match "
                f"{selection.orientation} and pair {pair}:\n{choices}"
            )

        result_dir, class_pair = matches[0]
        source = result_dir / selection.seed / f"sample_{size}.png"
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


def _manifest_rows(item: ExportItem, results_dir: Path) -> list[dict[str, str]]:
    selection = item.selection
    common = {
        "item_id": item.item_id,
        "orientation": selection.orientation,
        "class_pair": item.class_pair,
        "concept_1": selection.concept_1,
        "concept_2": selection.concept_2,
        "seed": selection.seed,
        "source_path": _source_for_manifest(item.source, results_dir),
    }
    transform = TRANSFORMS[selection.orientation]
    return [
        {
            **common,
            "filename": item.identity_filename,
            "view": "identity",
            "target": selection.concept_1,
        },
        {
            **common,
            "filename": item.transformed_filename,
            "view": transform.transformed_view,
            "target": selection.concept_2,
        },
    ]


def _write_readme(path: Path, selection_count: int, size: int) -> None:
    path.write_text(
        "# Gorilla stimuli\n\n"
        f"This export contains {selection_count * 2} PNG files from "
        f"{selection_count} selected visual anagrams at {size}x{size} pixels.\n\n"
        "Upload every file in `images/` to the Gorilla Stimuli tab. Gorilla uses a "
        "flat stimulus namespace, so spreadsheet cells should contain the exact value "
        "from the `filename` column in `manifest.csv` (without `images/`).\n\n"
        "Each `item_id` has two rows: an `identity` view and its transformed view. "
        "The `target` column records the concept intended for that view. Every PNG "
        "has a centered circular crop, a four-pixel transparent margin, and transparent "
        "pixels outside the circle. `block_1.csv` contains identity views and "
        "`block_2.csv` contains transformed views; both have independently shuffled "
        "stimulus rows.\n",
        encoding="utf-8",
    )


def _apply_circular_mask(image, image_module):
    """Return an RGBA image with a centered circle and a visible clear margin."""
    from PIL import ImageDraw

    rgba = image.convert("RGBA")
    width, height = rgba.size
    scale = 4
    mask_size = (width * scale, height * scale)
    padding = 4 * scale  # Keep the full circumference visibly inside the canvas.
    mask = image_module.new("L", mask_size, 0)
    image_draw = ImageDraw.Draw(mask)
    image_draw.ellipse(
        (padding, padding, mask_size[0] - 1 - padding, mask_size[1] - 1 - padding),
        fill=255,
    )
    mask = mask.resize(rgba.size, resample=image_module.Resampling.LANCZOS)
    rgba.putalpha(mask)
    return rgba


def _write_block_spreadsheets(
    output_dir: Path, rows: Sequence[dict[str, str]], random_seed: int | None
) -> None:
    """Write independently shuffled identity and transformed Gorilla blocks."""
    identity = [row["filename"] for row in rows if row["view"] == "identity"]
    transformed = [row["filename"] for row in rows if row["view"] != "identity"]
    if not identity or len(identity) != len(transformed):
        raise ExportError("Cannot create balanced identity and transformed blocks.")

    rng = random.Random(random_seed)
    for block_name, filenames in (
        ("block_1.csv", identity),
        ("block_2.csv", transformed),
    ):
        rng.shuffle(filenames)
        with (output_dir / block_name).open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.writer(file)
            writer.writerow(("display", "anagram"))
            writer.writerows(("trial", filename) for filename in filenames)


def export_items(
    items: Sequence[ExportItem],
    results_dir: Path,
    output_dir: Path,
    size: int,
    image_module,
    random_seed: int | None = None,
) -> None:
    """Write a complete export through a staging directory, then rename it into place."""
    if output_dir.exists():
        raise ExportError(
            f"Output path already exists: {output_dir}. Choose a new --output-dir."
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        images_dir = staging / "images"
        images_dir.mkdir()
        rows: list[dict[str, str]] = []

        for item in items:
            identity_path = images_dir / item.identity_filename
            transformed_path = images_dir / item.transformed_filename
            transform = TRANSFORMS[item.selection.orientation]
            transpose = getattr(image_module.Transpose, transform.pil_transpose_name)
            with image_module.open(item.source) as image:
                identity = _apply_circular_mask(image, image_module)
                identity.save(identity_path, format="PNG")
                identity.transpose(transpose).save(transformed_path, format="PNG")
            rows.extend(_manifest_rows(item, results_dir))

        fieldnames = (
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
        )
        with (staging / "manifest.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        _write_block_spreadsheets(staging, rows, random_seed)
        _write_readme(staging / "README.md", len(items), size)
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
) -> tuple[int, int]:
    selections = read_selections(selection_csv)
    items = resolve_items(selections, results_dir, size)
    image_module = _load_pillow()
    validate_sources(items, size, image_module)
    export_items(
        items, results_dir, output_dir, size, image_module, random_seed
    )
    return len(items), len(items) * 2


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
        help="source sample size in pixels (default: 256)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="seed for reproducible block shuffling (default: random each run)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        item_count, image_count = run_export(
            args.selection_csv,
            args.results_dir,
            args.output_dir,
            args.size,
            args.random_seed,
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
    print(
        f"Block spreadsheets: {args.output_dir / 'block_1.csv'}, "
        f"{args.output_dir / 'block_2.csv'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
