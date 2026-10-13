
import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from sklearn.cluster import AgglomerativeClustering


register_heif_opener()

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".heic"}


def calculate_phash(path):
    """Calculate a 64-bit perceptual hash."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return imagehash.phash(img.convert("RGB"), hash_size=8)


def report_progress(callback, percent, message):
    """Send progress to GUI if a callback is provided."""
    if callback is not None:
        callback(min(100, max(0, percent)), message)


def cluster_images(hashes, threshold=8, progress_callback=None):
    """Complete-linkage clustering with precomputed pHash distances."""
    n = len(hashes)

    if n == 0:
        return np.array([], dtype=int)

    if n == 1:
        report_progress(
            progress_callback, 85,
            "Clustering complete (1 image)"
        )
        return np.array([0], dtype=int)

    distances = np.zeros((n, n), dtype=np.float32)

    # Each row represents a known amount of pairwise work.
    total_pairs = n * (n - 1) // 2
    completed_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            distance = hashes[i] - hashes[j]
            distances[i, j] = distance
            distances[j, i] = distance

        completed_pairs += n - i - 1

        percent = 40 + 30 * completed_pairs / total_pairs

        report_progress(
            progress_callback,
            percent,
            f"Calculating distances: "
            f"{completed_pairs:,}/{total_pairs:,} pairs"
        )

    report_progress(
        progress_callback, 70,
        "Running agglomerative clustering..."
    )

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold + 0.5,
        metric="precomputed",
        linkage="complete",
    )

    labels = model.fit_predict(distances)

    report_progress(
        progress_callback, 85,
        "Clustering complete"
    )

    return labels


def process_photos(directory, threshold=8, progress_callback=None):
    """
    Classify images into unique/ and duplicates/.

    progress_callback(percent, message):
        Receives weighted progress in [0, 100].

    Returns a dictionary containing processing statistics.
    """
    directory = Path(directory).expanduser().resolve()

    if not directory.is_dir():
        raise ValueError(f"Directory not found: {directory}")

    unique_dir = directory / "unique"
    duplicates_dir = directory / "duplicates"
    csv_path = directory / "photo_groups.csv"

    for output in (unique_dir, duplicates_dir, csv_path):
        if output.exists():
            raise FileExistsError(
                f"Output already exists: {output}\n"
                "Please move or remove previous results."
            )

    report_progress(
        progress_callback, 0,
        "Scanning image directory..."
    )

    image_paths = sorted(
        (
            p for p in directory.iterdir()
            if p.is_file()
            and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda p: (p.name.lower(), p.name),
    )

    total_images = len(image_paths)

    if total_images == 0:
        raise ValueError("No JPG, JPEG or HEIC images found.")

    valid_paths = []
    hashes = []
    skipped = []

    # ---------------------------------------
    # Stage 1: Calculate pHash (0-40%)
    # ---------------------------------------
    for i, path in enumerate(image_paths, start=1):
        try:
            h = calculate_phash(path)
            valid_paths.append(path)
            hashes.append(h)

        except Exception as e:
            skipped.append((path.name, str(e)))

        percent = 40 * i / total_images

        report_progress(
            progress_callback,
            percent,
            f"Reading images: {i}/{total_images} "
            f"({path.name})"
        )

    if not valid_paths:
        raise ValueError("No readable images found.")

    # ---------------------------------------
    # Stages 2-3: Distances + clustering
    # 40-85%
    # ---------------------------------------
    labels = cluster_images(
        hashes,
        threshold=threshold,
        progress_callback=progress_callback,
    )

    grouped = defaultdict(list)

    for index, label in enumerate(labels):
        grouped[int(label)].append(index)

    groups = [sorted(indices) for indices in grouped.values()]
    groups.sort(key=lambda indices: indices[0])

    unique_count = sum(len(g) for g in groups if len(g) == 1)
    duplicate_count = sum(len(g) for g in groups if len(g) >= 2)
    duplicate_groups = sum(1 for g in groups if len(g) >= 2)

    total_copies = unique_count + duplicate_count

    # ---------------------------------------
    # Stage 4: Copy photos (85-99%)
    # ---------------------------------------
    unique_dir.mkdir()
    duplicates_dir.mkdir()

    csv_rows = []
    copied = 0
    next_number = 1
    group_id = 0

    for indices in groups:
        is_duplicate = len(indices) >= 2

        if is_duplicate:
            group_id += 1

        for index in indices:
            source = valid_paths[index]

            if is_duplicate:
                new_name = f"{next_number}{source.suffix}"
                destination = duplicates_dir / new_name
                next_number += 1
                category = "duplicate"
                current_group_id = group_id
            else:
                new_name = source.name
                destination = unique_dir / new_name
                category = "unique"
                current_group_id = ""

            # Copy original bytes without re-encoding.
            shutil.copy2(source, destination)

            csv_rows.append({
                "group_id": current_group_id,
                "category": category,
                "original_filename": source.name,
                "new_filename": new_name,
                "phash": str(hashes[index]),
            })

            copied += 1
            percent = 85 + 14 * copied / total_copies

            report_progress(
                progress_callback,
                percent,
                f"Copying photos: {copied}/{total_copies} "
                f"({new_name})"
            )

    # ---------------------------------------
    # Stage 5: Export CSV (99-100%)
    # ---------------------------------------
    report_progress(
        progress_callback, 99,
        "Writing photo_groups.csv..."
    )

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group_id",
                "category",
                "original_filename",
                "new_filename",
                "phash",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    stats = {
        "total": total_images,
        "processed": len(valid_paths),
        "unique": unique_count,
        "duplicate": duplicate_count,
        "duplicate_groups": duplicate_groups,
        "skipped": len(skipped),
        "skipped_files": skipped,
        "output_directory": str(directory),
    }

    report_progress(
        progress_callback, 100,
        "Processing completed successfully!"
    )

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Group similar JPG/HEIC photos."
    )

    parser.add_argument("directory")
    parser.add_argument(
        "--threshold",
        type=int,
        default=8,
        choices=range(65),
        metavar="0-64",
    )

    args = parser.parse_args()

    def console_progress(percent, message):
        print(f"\r{percent:6.2f}% | {message}", end="", flush=True)

    result = process_photos(
        args.directory,
        threshold=args.threshold,
        progress_callback=console_progress,
    )

    print("\n\nResults:", result)
