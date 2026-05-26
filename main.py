from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

Band = list[list[float]]


@dataclass(frozen=True)
class PixelResult:
    row: int
    col: int
    red: float
    nir: float
    ndvi: float | None
    ndvi_class: str


@dataclass(frozen=True)
class AnalysisSummary:
    pixels_total: int
    pixels_valid: int
    average_ndvi: float | None
    minimum_ndvi: float | None
    maximum_ndvi: float | None
    best_pixel: PixelResult | None
    class_distribution: Counter[str]


RED_BAND: Band = [
    [0.11, 0.13, 0.18, 0.25, 0.30],
    [0.09, 0.15, 0.22, 0.28, 0.41],
    [0.07, 0.14, 0.19, 0.25, 0.38],
    [0.16, 0.20, 0.24, 0.32, 0.36],
    [0.09, 0.13, 0.21, 0.34, 0.45],
]

NIR_BAND: Band = [
    [0.55, 0.62, 0.38, 0.35, 0.25],
    [0.58, 0.66, 0.50, 0.36, 0.22],
    [0.61, 0.59, 0.51, 0.40, 0.20],
    [0.54, 0.49, 0.46, 0.33, 0.24],
    [0.63, 0.57, 0.43, 0.29, 0.18],
]

def validate_band(red_band: Band, nir_band: Band) -> None:
    if not red_band or not nir_band:
        raise ValueError("Input bands cannot be empty")
    if len(red_band) != len(nir_band):
        raise ValueError("Red and NIR bands must have the same number of rows")

    expected_width = len(red_band[0])
    if expected_width == 0:
        raise ValueError("Input bands cannot contain empty rows")

    for row_index, (red_row, nir_row) in enumerate(zip(red_band, nir_band)):
        if len(red_row) != expected_width:
            raise ValueError(f"Red band row {row_index} has an inconsistent width")
        if len(nir_row) != expected_width:
            raise ValueError(f"NIR band row {row_index} has an inconsistent width")
        if len(red_row) != len(nir_row):
            raise ValueError(f"Red and NIR rows differ in width at row {row_index}")
        for value in red_row + nir_row:
            if value < 0:
                raise ValueError("Reflectance values cannot be negative")

def calculate_ndvi(red: float, nir: float) -> float | None:
    denominator = nir + red
    if denominator == 0:
        return None
    return (nir - red) / denominator

def classify_ndvi(ndvi: float | None) -> str:
    if ndvi is None:
        return "invalid"
    if ndvi < 0:
        return "water_or_cloud"
    if ndvi < 0.2:
        return "bare_soil"
    if ndvi < 0.5:
        return "temperate_vegetation"
    return "dense_vegetation"

def analyse_bands(red_band: Band, nir_band: Band) -> list[PixelResult]:
    validate_band(red_band, nir_band)
    results: list[PixelResult] = []
    for row_index, (red_row, nir_row) in enumerate(zip(red_band, nir_band)):
        for col_index, (red, nir) in enumerate(zip(red_row, nir_row)):
            ndvi = calculate_ndvi(red=red, nir=nir)
            results.append(
                PixelResult(
                    row=row_index,
                    col=col_index,
                    red=red,
                    nir=nir,
                    ndvi=ndvi,
                    ndvi_class=classify_ndvi(ndvi),
                )
            )
    return results

def build_summary(results: list[PixelResult]) -> AnalysisSummary:
    valid_results = [result for result in results if result.ndvi is not None]
    ndvi_values = [result.ndvi for result in valid_results if result.ndvi is not None]
    best_pixel = (
        max(
            valid_results,
            key=lambda result: result.ndvi if result.ndvi is not None else -2,
        )
        if valid_results
        else None
    )

    return AnalysisSummary(
        pixels_total=len(results),
        pixels_valid=len(valid_results),
        average_ndvi=mean(ndvi_values) if ndvi_values else None,
        minimum_ndvi=min(ndvi_values) if ndvi_values else None,
        maximum_ndvi=max(ndvi_values) if ndvi_values else None,
        best_pixel=best_pixel,
        class_distribution=Counter(result.ndvi_class for result in results),
    )

def format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"

def print_summary(summary: AnalysisSummary) -> None:
    print("NDVI Analysis Summary")
    print("---------------------")
    print(f"Pixels analysed: {summary.pixels_total}")
    print(f"Valid pixels:     {summary.pixels_valid}")
    print(f"Average NDVI:     {format_float(summary.average_ndvi)}")
    print(f"Minimum NDVI:     {format_float(summary.minimum_ndvi)}")
    print(f"Maximum NDVI:     {format_float(summary.maximum_ndvi)}")
    print()
    print("Class distribution:")
    for ndvi_class, count in summary.class_distribution.most_common():
        print(f"- {ndvi_class}: {count} pixels")

    if summary.best_pixel is not None:
        print()
        print("Best vegetation pixel:")
        print(
            f"row={summary.best_pixel.row}, "
            f"col={summary.best_pixel.col}, "
            f"NDVI={format_float(summary.best_pixel.ndvi)}"
        )

def export_to_csv(results: list[PixelResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["row", "col", "red", "nir", "ndvi", "class"],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "row": result.row,
                    "col": result.col,
                    "red": result.red,
                    "nir": result.nir,
                    "ndvi": format_float(result.ndvi),
                    "class": result.ndvi_class,
                }
            )

def print_ascii_map(results: list[PixelResult], width: int) -> None:
    symbols = {
        "dense_vegetation": "D",
        "temperate_vegetation": "T",
        "bare_soil": "B",
        "water_or_cloud": "W",
        "invalid": "?",
    }
    print()
    print("Vegetation map:")
    print("D = dense, T = temperate, B = bare soil, W = water/cloud")

    for index in range(0, len(results), width):
        row = results[index : index + width]
        print(" ".join(symbols[result.ndvi_class] for result in row))

def main() -> None:
    results = analyse_bands(red_band=RED_BAND, nir_band=NIR_BAND)
    summary = build_summary(results)

    print_summary(summary)
    print_ascii_map(results, width=len(RED_BAND[0]))
    export_to_csv(results=results, output_path=Path("results.csv"))
if __name__ == "__main__":
    main()
