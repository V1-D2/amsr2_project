#!/usr/bin/env python3
"""
AMSR-2 Processor - Простая версия
Прямой перенос из Google Colab с выбором формата сохранения
ИСПРАВЛЕННАЯ ВЕРСИЯ на основе рабочего кода
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


def calculate_lat_lon_36ghz(h5):
    lat_89 = None
    lon_89 = None

    for suffix in ["89A", "89B"]:
        lat_key = f"Latitude of Observation Point for {suffix}"
        lon_key = f"Longitude of Observation Point for {suffix}"

        if lat_key in h5 and lon_key in h5:
            lat_89 = h5[lat_key][:]
            lon_89 = h5[lon_key][:]
            break

    if lat_89 is None:
        raise ValueError("89 GHz coordinates not found in file!")

    if lat_89.shape[1] == 486:
        lat_36 = lat_89[:, ::2]
        lon_36 = lon_89[:, ::2]
    else:
        lat_36 = lat_89
        lon_36 = lon_89

    return lat_36, lon_36


def extract_swath_data(h5_path: pathlib.Path) -> Optional[Dict]:
    try:
        with h5py.File(h5_path, "r") as h5:
            var_name = "Brightness Temperature (36.5GHz,H)"
            if var_name not in h5:
                print(f"ERROR: Variable '{var_name}' not found in {h5_path.name}")
                print(f"Available variables: {list(h5.keys())}")
                return None

            raw_temp = h5[var_name][:].astype(np.float64)

            scale = 1.0
            if "SCALE FACTOR" in h5[var_name].attrs:
                scale = h5[var_name].attrs["SCALE FACTOR"]
                if isinstance(scale, np.ndarray):
                    scale = scale[0]

            scaled_temp = np.where(raw_temp == 0, np.nan, raw_temp * scale)
            lat_36, lon_36 = calculate_lat_lon_36ghz(h5)

            if scaled_temp.shape != lat_36.shape or scaled_temp.shape != lon_36.shape:
                print(f"ERROR: Shape mismatch in {h5_path.name} - temp: {scaled_temp.shape}, lat: {lat_36.shape}, lon: {lon_36.shape}")
                return None

            valid_count = np.sum(~np.isnan(scaled_temp))
            if valid_count == 0:
                print(f"ERROR: No valid temperature data in {h5_path.name}")
                return None

            orbit_type = "Unknown"
            if "A" in h5_path.stem:
                orbit_type = "Ascending"
            elif "D" in h5_path.stem:
                orbit_type = "Descending"

            metadata = {
                'filename': h5_path.name,
                'orbit_type': orbit_type,
                'shape': scaled_temp.shape,
                'scale_factor': scale,
                'valid_pixels': valid_count,
                'total_pixels': scaled_temp.size,
                'temp_range': (np.nanmin(scaled_temp), np.nanmax(scaled_temp)),
                'lat_range': (np.nanmin(lat_36), np.nanmax(lat_36)),
                'lon_range': (np.nanmin(lon_36), np.nanmax(lon_36))  # FIXED: was lon_6
            }

            return {
                'temperature': scaled_temp.astype(np.float32),
                'latitude': lat_36.astype(np.float32),
                'longitude': lon_36.astype(np.float32),
                'metadata': metadata
            }

    except Exception as e:
        print(f"Error processing {h5_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def download_process_delete_single(product, temp_dir: pathlib.Path, progress: ThreadSafeProgress) -> Optional[Dict]:
    """Скачивает -> Обрабатывает -> СРАЗУ УДАЛЯЕТ файл"""
    downloaded_file = None
    try:
        # 1. СКАЧИВАЕМ
        local_path = gportal.download(product, local_dir=str(temp_dir))
        downloaded_file = pathlib.Path(local_path)

        # 2. СРАЗУ ОБРАБАТЫВАЕМ
        swath_data = extract_swath_data(downloaded_file)

        # 3. НЕМЕДЛЕННО УДАЛЯЕМ H5 ФАЙЛ!
        try:
            downloaded_file.unlink()
        except:
            pass

        progress.update_processed()
        return swath_data

    except Exception as e:
        print(f"Error in download_process_delete_single: {e}")
        if downloaded_file and downloaded_file.exists():
            try:
                downloaded_file.unlink()
            except:
                pass
        progress.update_processed()
        return None


def process_files_concurrent(products: List, temp_dir: pathlib.Path, max_workers: int = 4) -> List[Dict]:
    """Обрабатывает файлы с немедленным удалением"""
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
                    print(f"✓ Successfully processed: {result['metadata']['filename']}")
                pbar.update(1)

    print(f"Successfully processed: {len(all_swaths)}/{total_products} files")
    return all_swaths


def save_format_1(swath_list: List[Dict], base_dir: pathlib.Path, period_name: str, start_datetime: str,
                  end_datetime: str) -> pathlib.Path:
    """Формат 1: Array of dictionaries (как в первом коде)"""
    print("Saving in Array of dictionaries format...")

    output_file = base_dir / f"AMSR2_swaths_{period_name}.npz"

    swath_array = []
    for swath in tqdm.tqdm(swath_list, desc="Preparing swath array"):
        swath_dict = {
            'temperature': swath['temperature'].astype(np.float32),
            'latitude': swath['latitude'].astype(np.float32),
            'longitude': swath['longitude'].astype(np.float32),
            'metadata': swath['metadata']
        }
        swath_array.append(swath_dict)

    save_dict = {
        'swath_array': swath_array,
        'period': f"{start_datetime} to {end_datetime}",
        'num_swaths': len(swath_list),
        'description': 'AMSR-2 36.5GHz H swath data - Array format'
    }

    np.savez_compressed(output_file, **save_dict)

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"File saved: {output_file.name}")
    print(f"File size: {file_size_mb:.2f} MB")

    return output_file


def save_format_2(swath_list: List[Dict], base_dir: pathlib.Path, period_name: str, start_datetime: str,
                  end_datetime: str) -> pathlib.Path:
    """Формат 2: Separate arrays (как во втором коде)"""
    print("Saving in Separate arrays format...")

    output_file = base_dir / f"AMSR2_swaths_{period_name}.npz"

    arrays_dict = {}
    metadata_list = []

    for i, swath in enumerate(tqdm.tqdm(swath_list, desc="Preparing arrays")):
        arrays_dict[f'temperature_{i}'] = swath['temperature'].astype(np.float32)
        arrays_dict[f'latitude_{i}'] = swath['latitude'].astype(np.float32)
        arrays_dict[f'longitude_{i}'] = swath['longitude'].astype(np.float32)
        metadata_list.append(swath['metadata'])

    arrays_dict['metadata'] = metadata_list
    arrays_dict['period'] = f"{start_datetime} to {end_datetime}"
    arrays_dict['num_swaths'] = len(swath_list)
    arrays_dict['description'] = 'AMSR-2 36.5GHz H swath data - Separate arrays format'

    np.savez_compressed(output_file, **arrays_dict)

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"File saved: {output_file.name}")
    print(f"File size: {file_size_mb:.2f} MB")

    return output_file


def load_format_1(dataset_path: pathlib.Path) -> List[Dict]:
    """Загрузка формата 1"""
    with np.load(dataset_path, allow_pickle=True) as data:
        swath_array = data['swath_array']
        num_swaths = int(data['num_swaths'])
        print(f"Loaded {num_swaths} swaths (Array format)")
        return swath_array.tolist()


def load_format_2(dataset_path: pathlib.Path) -> List[Dict]:
    """Загрузка формата 2"""
    with np.load(dataset_path, allow_pickle=True) as data:
        num_swaths = int(data['num_swaths'])
        swath_list = []

        for i in range(num_swaths):
            swath = {
                'temperature': data[f'temperature_{i}'],
                'latitude': data[f'latitude_{i}'],
                'longitude': data[f'longitude_{i}'],
                'metadata': data['metadata'][i]
            }
            swath_list.append(swath)

        print(f"Loaded {num_swaths} swaths (Separate arrays format)")
        return swath_list


def fetch_amsr2_data(start_datetime: str, end_datetime: str, save_format: int = 1,
                     max_workers: int = 4) -> pathlib.Path:
    """Основная функция обработки"""

    print(f"Searching for AMSR-2 data: {start_datetime} → {end_datetime}")

    # Создаем директории
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
        return None

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
    if save_format == 1:
        output_file = save_format_1(all_swaths, BASE_DIR, period_name, start_datetime, end_datetime)
    else:
        output_file = save_format_2(all_swaths, BASE_DIR, period_name, start_datetime, end_datetime)
    save_time = time.time() - save_start

    try:
        # Очистка временной папки
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
    """Простой запуск"""
    print("=== AMSR-2 Processor ===")

    # Параметры времени
    start_input = input("Start datetime (YYYY-MM-DD HH:MM:SS): ").strip()
    end_input = input("End datetime (YYYY-MM-DD HH:MM:SS): ").strip()

    start_datetime = dt.datetime.strptime(start_input, "%Y-%m-%d %H:%M:%S").isoformat()
    end_datetime = dt.datetime.strptime(end_input, "%Y-%m-%d %H:%M:%S").isoformat()

    # Количество потоков
    workers_input = input("Number of threads (1-16, default 4): ").strip()
    max_workers = int(workers_input) if workers_input else 4
    max_workers = max(1, min(16, max_workers))

    # Формат сохранения
    print("\nSave format:")
    print("1. Array of dictionaries (first code format)")
    print("2. Separate arrays (second code format)")
    format_input = input("Choose format (1 or 2, default 1): ").strip()
    save_format = int(format_input) if format_input in ['1', '2'] else 1

    print(f"\nProcessing: {start_input} → {end_input}")
    print(f"Threads: {max_workers}")
    print(f"Format: {save_format}")

    output_file = fetch_amsr2_data(start_datetime, end_datetime, save_format, max_workers)

    if output_file:
        print(f"\nTesting load...")
        if save_format == 1:
            swaths = load_format_1(output_file)
        else:
            swaths = load_format_2(output_file)

        if len(swaths) > 0:
            example = swaths[0]
            temp = example['temperature']
            print(f"Example swath: {temp.shape}, {np.nanmin(temp):.1f} - {np.nanmax(temp):.1f} K")
        else:
            print("No swaths loaded - check processing")


if __name__ == "__main__":
    main()