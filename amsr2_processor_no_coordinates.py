#!/usr/bin/env python3
"""
AMSR-2 Processor - Версия без координат
Максимальное сжатие: только температуры + упрощенные метаданные
"""

import pathlib
import datetime as dt
import tqdm
import gportal
import h5py
import numpy as np
import concurrent.futures
import threading
import time
import os
from typing import List, Tuple, Optional, Dict

from config import BASE_DIR, TEMP_DIR, GPORTAL_USERNAME, GPORTAL_PASSWORD

gportal.username = GPORTAL_USERNAME
gportal.password = GPORTAL_PASSWORD

_DS = gportal.datasets()["GCOM-W/AMSR2"]["LEVEL1"]
DS_L1B_TB = _DS["L1B-Brightness temperature（TB）"][0]


class ThreadSafeProgress:
    def __init__(self):
        self.lock = threading.Lock()
        self.processed = 0
        self.total_files = 0

    def set_total(self, total):
        with self.lock:
            self.total_files = total

    def update_processed(self):
        with self.lock:
            self.processed += 1
            return self.processed, self.total_files


def extract_swath_data(h5_path: pathlib.Path) -> Optional[Dict]:
    try:
        with h5py.File(h5_path, "r") as h5:
            var_name = "Brightness Temperature (36.5GHz,H)"
            if var_name not in h5:
                print(f"ERROR: Variable '{var_name}' not found in {h5_path.name}")
                print(f"Available variables: {list(h5.keys())}")
                return None

            # Store temperatures in original format (without float conversion)
            raw_temp = h5[var_name][:]

            # Get scale factor
            scale = 1.0
            if "SCALE FACTOR" in h5[var_name].attrs:
                scale = h5[var_name].attrs["SCALE FACTOR"]
                if isinstance(scale, np.ndarray):
                    scale = scale[0]

            # Check data validity (but don't convert to float)
            valid_mask = raw_temp != 0
            valid_count = np.sum(valid_mask)
            if valid_count == 0:
                print(f"ERROR: No valid temperature data in {h5_path.name}")
                return None

            # Determine orbit type from identifier
            orbit_type = "U"  # Unknown by default
            try:
                identifier = h5_path.stem
                if "_" in identifier:
                    parts = identifier.split("_")
                    if len(parts) >= 3:
                        ad_flag = parts[2][-1]  # Последний символ третьей части
                        if ad_flag in ["A", "D"]:
                            orbit_type = ad_flag
            except:
                # Fallback to old method
                if "A" in h5_path.stem:
                    orbit_type = "A"
                elif "D" in h5_path.stem:
                    orbit_type = "D"

            # Simplified metadata to save space
            metadata = {
                'orbit_type': orbit_type,
                'scale_factor': float(scale),
                'temp_range': (int(np.min(raw_temp[valid_mask])), int(np.max(raw_temp[valid_mask]))),
                'shape': raw_temp.shape
            }

            return {
                'temperature': raw_temp,  # Only temperatures in original format!
                'metadata': metadata
            }

    except Exception as e:
        print(f"Error processing {h5_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def download_process_delete_single(product, temp_dir: pathlib.Path, progress: ThreadSafeProgress) -> Optional[Dict]:
    """Downloads -> Processes -> IMMEDIATELY deletes file"""
    downloaded_file = None
    try:
        # 1. DOWNLOAD
        local_path = gportal.download(product, local_dir=str(temp_dir))
        downloaded_file = pathlib.Path(local_path)

        # 2. PROCESS IMMEDIATELY
        swath_data = extract_swath_data(downloaded_file)

        # 3. DELETE H5 FILE IMMEDIATELY!
        try:
            downloaded_file.unlink()
        except:
            pass

        progress.update_processed()
        return swath_data

    except Exception as e:
        if downloaded_file and downloaded_file.exists():
            try:
                downloaded_file.unlink()
            except:
                pass
        progress.update_processed()
        return None


def process_files_concurrent(products: List, temp_dir: pathlib.Path, max_workers: int = 4) -> List[Dict]:
    """Processes files with immediate deletion"""
    product_list = list(products)
    total_products = len(product_list)

    progress = ThreadSafeProgress()
    progress.set_total(total_products)

    all_swaths = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_product = {
            executor.submit(download_process_delete_single, product, temp_dir, progress): product
            for product in product_list
        }

        with tqdm.tqdm(total=total_products, desc="Processing files") as pbar:
            for future in concurrent.futures.as_completed(future_to_product):
                result = future.result()
                if result is not None:
                    all_swaths.append(result)
                pbar.update(1)

    print(f"Successfully processed: {len(all_swaths)}/{total_products} files")
    return all_swaths


def save_npz(swath_list: List[Dict], base_dir: pathlib.Path, period_name: str, start_datetime: str,
             end_datetime: str) -> pathlib.Path:
    """Save in NPZ format (temperature data only)"""
    print("Saving in NPZ format (Temperature data only)...")

    output_file = base_dir / f"AMSR2_temp_only_{period_name}.npz"

    swath_array = []
    for swath in tqdm.tqdm(swath_list, desc="Preparing swath array"):
        swath_dict = {
            'temperature': swath['temperature'],  # Only temperatures in original format
            'metadata': swath['metadata']         # Simplified metadata
        }
        swath_array.append(swath_dict)

    save_dict = {
        'swath_array': swath_array,
        'period': f"{start_datetime} to {end_datetime}"
    }

    np.savez_compressed(output_file, **save_dict)

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"NPZ file saved: {output_file.name}")
    print(f"NPZ file size: {file_size_mb:.2f} MB")

    return output_file


def load_npz(dataset_path: pathlib.Path) -> List[Dict]:
    """Load NPZ format"""
    with np.load(dataset_path, allow_pickle=True) as data:
        swath_array = data['swath_array']
        return swath_array.tolist()


def fetch_amsr2_data(start_datetime: str, end_datetime: str, max_workers: int = 4) -> pathlib.Path:
    """Main processing function"""

    print(f"Searching for AMSR-2 data: {start_datetime} → {end_datetime}")

    # Create directories
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    search_start = time.time()
    res = gportal.search(
        dataset_ids=[DS_L1B_TB],
        start_time=start_datetime,
        end_time=end_datetime
    )
    search_time = time.time() - search_start

    total_files = res.matched()
    print(f"Found {total_files} files in {search_time:.1f} seconds")

    if total_files == 0:
        print("No data found")
        return None, None

    processing_start = time.time()
    all_products = res.products()
    all_swaths = process_files_concurrent(all_products, TEMP_DIR, max_workers)
    processing_time = time.time() - processing_start

    print(f"Processing completed in {processing_time:.1f} seconds")
    print(f"Extracted {len(all_swaths)} swaths")

    if not all_swaths:
        print("Failed to process data")
        return None

    period_name = f"{start_datetime.replace(':', '').replace('-', '').replace('T', '_')}_to_{end_datetime.replace(':', '').replace('-', '').replace('T', '_')}"

    save_start = time.time()

    # Save in NPZ format
    output_file = save_npz(all_swaths, BASE_DIR, period_name, start_datetime, end_datetime)

    save_time = time.time() - save_start

    try:
        # Clean temporary folder
        for file in TEMP_DIR.glob('*'):
            if file.is_file():
                file.unlink()
        TEMP_DIR.rmdir()
    except:
        pass

    total_time = search_time + processing_time + save_time
    print(f"Total time: {total_time:.1f} seconds ({total_time / 60:.1f} minutes)")

    return output_file


def main():
    """Simple launch"""
    print("=== AMSR-2 Processor - Temperature Only Version ===")

    # Time parameters
    start_input = input("Start datetime (YYYY-MM-DD HH:MM:SS): ").strip()
    end_input = input("End datetime (YYYY-MM-DD HH:MM:SS): ").strip()

    start_datetime = dt.datetime.strptime(start_input, "%Y-%m-%d %H:%M:%S").isoformat()
    end_datetime = dt.datetime.strptime(end_input, "%Y-%m-%d %H:%M:%S").isoformat()

    # Number of threads
    workers_input = input("Number of threads (1-16, default 4): ").strip()
    max_workers = int(workers_input) if workers_input else 4
    max_workers = max(1, min(16, max_workers))

    print(f"\nProcessing: {start_input} → {end_input}")
    print(f"Threads: {max_workers}")
    print("Mode: Temperature data only (no coordinates)")

    output_file = fetch_amsr2_data(start_datetime, end_datetime, max_workers)

    if output_file:
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"\nFile created: {output_file.name}")
        print(f"File size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    main()