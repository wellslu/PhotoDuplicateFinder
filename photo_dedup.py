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
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.heic'}


def calculate_phash(path):
    """Compute 64-bit perceptual hash, accounting for EXIF orientation."""
    with Image.open(path) as img:
        oriented = ImageOps.exif_transpose(img)
        return imagehash.phash(oriented.convert('RGB'), hash_size=8)


def report_progress(callback, percent, message):
    if callback is not None:
        callback(min(100.0, max(0.0, float(percent))), message)


def cluster_images(hashes, threshold=8, progress_callback=None):
    """Complete-linkage clustering from pairwise Hamming distances."""
    n = len(hashes)
    if n <= 1:
        report_progress(progress_callback, 85, 'Clustering complete')
        return np.zeros(n, dtype=int)

    distances = np.zeros((n, n), dtype=np.float32)
    total_pairs = n * (n - 1) // 2
    completed_pairs = 0
    last_percent = -1

    for i in range(n):
        for j in range(i + 1, n):
            distance = hashes[i] - hashes[j]
            distances[i, j] = distance
            distances[j, i] = distance
        completed_pairs += n - i - 1
        percent = 40 + 30 * completed_pairs / total_pairs
        rounded = int(percent * 10)
        if rounded != last_percent or i == n - 1:
            report_progress(
                progress_callback, percent,
                f'Calculating distances: {completed_pairs:,}/{total_pairs:,} pairs'
            )
            last_percent = rounded

    report_progress(progress_callback, 70, 'Running agglomerative clustering...')
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold + 0.5,
        metric='precomputed',
        linkage='complete',
    )
    labels = model.fit_predict(distances)
    report_progress(progress_callback, 85, 'Clustering complete')
    return labels


def process_photos(directory, threshold=8, progress_callback=None):
    """Move only photos in similar groups into duplicates/; leave unique files intact."""
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f'Directory not found: {directory}')
    if not isinstance(threshold, int) or not 0 <= threshold <= 64:
        raise ValueError('threshold must be an integer between 0 and 64')

    duplicates_dir = directory / 'duplicates'
    csv_path = directory / 'photo_groups.csv'
    for output in (duplicates_dir, csv_path):
        if output.exists():
            raise FileExistsError(
                f'Output already exists: {output}\n'
                'Please move or remove previous results before rerunning.'
            )

    report_progress(progress_callback, 0, 'Scanning image directory...')
    image_paths = sorted(
        (p for p in directory.iterdir()
         if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda p: (p.name.lower(), p.name),
    )
    total_images = len(image_paths)
    if total_images == 0:
        raise ValueError('No JPG, JPEG or HEIC images found.')

    valid_paths = []
    hashes = []
    skipped = []
    for i, path in enumerate(image_paths, start=1):
        try:
            hashes.append(calculate_phash(path))
            valid_paths.append(path)
        except Exception as exc:
            skipped.append((path.name, str(exc)))
        report_progress(
            progress_callback, 40 * i / total_images,
            f'Reading images: {i}/{total_images} ({path.name})'
        )
    if not valid_paths:
        raise ValueError('No readable images found.')

    labels = cluster_images(hashes, threshold, progress_callback)
    grouped = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[int(label)].append(index)

    groups = [sorted(indices) for indices in grouped.values()]
    groups.sort(key=lambda indices: indices[0])
    duplicate_groups = [indices for indices in groups if len(indices) >= 2]
    unique_count = sum(1 for indices in groups if len(indices) == 1)
    duplicate_count = sum(len(indices) for indices in duplicate_groups)

    # Prepare all move destinations and CSV rows before changing source files.
    planned_moves = []
    csv_rows = []
    next_number = 1
    for group_id, indices in enumerate(duplicate_groups, start=1):
        for index in indices:
            source = valid_paths[index]
            new_name = f'{next_number}{source.suffix}'
            planned_moves.append((source, duplicates_dir / new_name, group_id))
            csv_rows.append({
                'group_id': group_id,
                'original_filename': source.name,
                'new_filename': new_name,
                'phash': str(hashes[index]),
            })
            next_number += 1

    # Create only the duplicates folder (even if no duplicate groups exist).
    duplicates_dir.mkdir()
    moved = 0
    for source, destination, group_id in planned_moves:
        if destination.exists():
            raise FileExistsError(f'Refusing to overwrite: {destination}')
        shutil.move(str(source), str(destination))
        moved += 1
        report_progress(
            progress_callback,
            85 + 14 * moved / max(duplicate_count, 1),
            f'Moving duplicate photos: {moved}/{duplicate_count} '
            f'(group {group_id}: {destination.name})'
        )

    report_progress(progress_callback, 99, 'Writing photo_groups.csv...')
    with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(
            f, fieldnames=['group_id', 'original_filename', 'new_filename', 'phash']
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    stats = {
        'total': total_images,
        'processed': len(valid_paths),
        'unique': unique_count,
        'duplicate': moved,
        'duplicate_groups': len(duplicate_groups),
        'skipped': len(skipped),
        'skipped_files': skipped,
        'output_directory': str(directory),
    }
    report_progress(progress_callback, 100, 'Processing completed successfully!')
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Move similar JPG/HEIC photos into duplicates/.'
    )
    parser.add_argument('directory', help='Folder containing the photos')
    parser.add_argument('--threshold', type=int, default=8, choices=range(65), metavar='0-64')
    args = parser.parse_args()

    def console_progress(percent, message):
        print(f'\r{percent:6.2f}% | {message}', end='', flush=True)

    results = process_photos(args.directory, args.threshold, console_progress)
    print('\n\nResults:', results)
