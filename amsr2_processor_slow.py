#!/usr/bin/env python3
"""
AMSR-2 Processor - Версия с улучшенной обработкой сетевых ошибок
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
import random
from typing import List, Tuple, Optional, Dict

from config import BASE_DIR, TEMP_DIR, GPORTAL_USERNAME, GPORTAL_PASSWORD

gportal.username = GPORTAL_USERNAME
gportal.password = GPORTAL_PASSWORD

_DS = gportal.datasets()["GCOM-W/AMSR2"]["LEVEL1"]
DS_L1B_TB = _DS["L1B-Brightness temperature（TB）"][0]

# Настройки для работы с нестабильными соединениями
SAVE_INTERVAL = 100
BACKUP_DIR = BASE_DIR / "incremental_backups"
BACKUP_FILENAME = "AMSR2_temp_backup_CURRENT.npz"

# Новые настройки для обработки ошибок
MAX_RETRIES = 3
RETRY_DELAY_BASE = 5  # базовая задержка в секундах
BACKOFF_MULTIPLIER = 2  # множитель для экспоненциального backoff


class ThreadSafeProgress:
    def __init__(self):
        self.lock = threading.Lock()
        self.processed = 0
        self.failed = 0
        self.total_files = 0

    def set_total(self, total):
        with self.lock:
            self.total_files = total

    def update_processed(self, success=True):
        with self.lock:
            if success:
                self.processed += 1
            else:
                self.failed += 1
            return self.processed, self.failed, self.total_files


def extract_swath_data(h5_path: pathlib.Path) -> Optional[Dict]:
    try:
        with h5py.File(h5_path, "r") as h5:
            var_name = "Brightness Temperature (36.5GHz,H)"
            if var_name not in h5:
                print(f"ERROR: Variable '{var_name}' not found in {h5_path.name}")
                return None

            raw_temp = h5[var_name][:]
            scale = 1.0
            if "SCALE FACTOR" in h5[var_name].attrs:
                scale = h5[var_name].attrs["SCALE FACTOR"]
                if isinstance(scale, np.ndarray):
                    scale = scale[0]

            valid_mask = raw_temp != 0
            valid_count = np.sum(valid_mask)
            if valid_count == 0:
                print(f"ERROR: No valid temperature data in {h5_path.name}")
                return None

            orbit_type = "U"
            try:
                identifier = h5_path.stem
                if "_" in identifier:
                    parts = identifier.split("_")
                    if len(parts) >= 3:
                        ad_flag = parts[2][-1]
                        if ad_flag in ["A", "D"]:
                            orbit_type = ad_flag
            except:
                if "A" in h5_path.stem:
                    orbit_type = "A"
                elif "D" in h5_path.stem:
                    orbit_type = "D"

            metadata = {
                'orbit_type': orbit_type,
                'scale_factor': float(scale),
                'temp_range': (int(np.min(raw_temp[valid_mask])), int(np.max(raw_temp[valid_mask]))),
                'shape': raw_temp.shape
            }

            return {
                'temperature': raw_temp,
                'metadata': metadata
            }

    except Exception as e:
        print(f"Error processing {h5_path.name}: {e}")
        return None


def download_with_retry(product, temp_dir: pathlib.Path, max_retries: int = MAX_RETRIES) -> Optional[pathlib.Path]:
    """Скачивание с повторными попытками и экспоненциальным backoff"""
    last_error = None

    for attempt in range(max_retries):
        try:
            # Случайная задержка перед каждой попыткой (кроме первой)
            if attempt > 0:
                delay = RETRY_DELAY_BASE * (BACKOFF_MULTIPLIER ** attempt) + random.uniform(0, 2)
                print(
                    f"Retry {attempt + 1}/{max_retries} for {product.get('title', 'unknown')} after {delay:.1f}s delay")
                time.sleep(delay)

            # Попытка скачивания
            local_path = gportal.download(product, local_dir=str(temp_dir))
            return pathlib.Path(local_path)

        except Exception as e:
            last_error = e
            error_msg = str(e)

            # Проверяем тип ошибки
            if any(err in error_msg.lower() for err in ['ssh', 'connection', 'banner', 'timeout', 'reset']):
                print(f"Network error on attempt {attempt + 1}: {error_msg}")
                if attempt == max_retries - 1:
                    print(f"Failed to download after {max_retries} attempts: {product.get('title', 'unknown')}")
            else:
                # Для не-сетевых ошибок не повторяем
                print(f"Non-network error: {error_msg}")
                break

    return None


def download_process_delete_single_safe(product, temp_dir: pathlib.Path, progress: ThreadSafeProgress) -> Optional[
    Dict]:
    """Безопасная версия с повторными попытками"""
    downloaded_file = None
    try:
        # 1. DOWNLOAD с повторными попытками
        downloaded_file = download_with_retry(product, temp_dir)

        if downloaded_file is None:
            progress.update_processed(success=False)
            return None

        # 2. PROCESS
        swath_data = extract_swath_data(downloaded_file)

        # 3. DELETE
        try:
            downloaded_file.unlink()
        except:
            pass

        success = swath_data is not None
        progress.update_processed(success=success)
        return swath_data

    except Exception as e:
        if downloaded_file and downloaded_file.exists():
            try:
                downloaded_file.unlink()
            except:
                pass
        progress.update_processed(success=False)
        return None


def save_backup(swath_list: List[Dict], backup_dir: pathlib.Path, files_processed: int,
                files_failed: int, start_datetime: str, end_datetime: str) -> None:
    """Сохранить бекап с информацией об ошибках"""
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_file = backup_dir / BACKUP_FILENAME

    swath_array = []
    for swath in swath_list:
        if swath is not None:
            swath_dict = {
                'temperature': swath['temperature'],
                'metadata': swath['metadata']
            }
            swath_array.append(swath_dict)

    save_dict = {
        'swath_array': swath_array,
        'period': f"{start_datetime} to {end_datetime}",
        'backup_info': {
            'files_processed': files_processed,
            'files_failed': files_failed,
            'success_rate': f"{(files_processed - files_failed) / files_processed * 100:.1f}%" if files_processed > 0 else "0%",
            'total_swaths': len(swath_array),
            'timestamp': dt.datetime.now().isoformat()
        }
    }

    np.savez_compressed(backup_file, **save_dict)

    file_size_mb = backup_file.stat().st_size / (1024 * 1024)
    success_rate = (files_processed - files_failed) / files_processed * 100 if files_processed > 0 else 0
    print(f"Backup saved: {files_processed} files processed, {files_failed} failed ({success_rate:.1f}% success rate)")
    print(f"Swaths: {len(swath_array)}, Size: {file_size_mb:.1f} MB")


def process_files_sequential_safe(products: List, temp_dir: pathlib.Path, save_interval: int = SAVE_INTERVAL,
                                  start_datetime: str = "", end_datetime: str = "") -> List[Dict]:
    """Последовательная обработка файлов с улучшенной обработкой ошибок"""
    product_list = list(products)
    total_products = len(product_list)

    progress = ThreadSafeProgress()
    progress.set_total(total_products)

    all_swaths = []
    files_processed = 0
    consecutive_failures = 0
    max_consecutive_failures = 10  # Остановиться после 10 неудач подряд

    with tqdm.tqdm(total=total_products, desc="Processing files") as pbar:
        for i, product in enumerate(product_list):
            result = download_process_delete_single_safe(product, temp_dir, progress)

            if result is not None:
                all_swaths.append(result)
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print(f"\nStopping due to {max_consecutive_failures} consecutive failures")
                    print("This might indicate server issues or network problems")
                    break

            files_processed += 1
            processed, failed, total = progress.processed, progress.failed, progress.total_files

            # Обновляем прогресс-бар с информацией об ошибках
            success_rate = (processed - failed) / processed * 100 if processed > 0 else 0
            pbar.set_postfix({
                'Success': f'{success_rate:.1f}%',
                'Failed': failed,
                'Consecutive fails': consecutive_failures
            })
            pbar.update(1)

            # Сохранить бекап
            if files_processed % save_interval == 0:
                save_backup(all_swaths, BACKUP_DIR, files_processed, failed, start_datetime, end_datetime)

            # Небольшая пауза между файлами для снижения нагрузки на сервер
            time.sleep(0.5)

    # Финальный бекап
    processed, failed, total = progress.processed, progress.failed, progress.total_files
    if files_processed % save_interval != 0:
        save_backup(all_swaths, BACKUP_DIR, files_processed, failed, start_datetime, end_datetime)

    print(f"\nProcessing completed:")
    print(f"  Total files: {total_products}")
    print(f"  Successfully processed: {len(all_swaths)}")
    print(f"  Failed: {failed}")
    print(f"  Success rate: {len(all_swaths) / files_processed * 100:.1f}%")

    return all_swaths


def save_npz(swath_list: List[Dict], base_dir: pathlib.Path, period_name: str, start_datetime: str,
             end_datetime: str) -> pathlib.Path:
    """Save in NPZ format"""
    print("Saving in NPZ format...")

    output_file = base_dir / f"AMSR2_temp_only_{period_name}.npz"

    swath_array = []
    for swath in tqdm.tqdm(swath_list, desc="Preparing swath array"):
        swath_dict = {
            'temperature': swath['temperature'],
            'metadata': swath['metadata']
        }
        swath_array.append(swath_dict)

    save_dict = {
        'swath_array': swath_array,
        'period': f"{start_datetime} to {end_datetime}",
        'processing_info': {
            'total_swaths': len(swath_array),
            'created_at': dt.datetime.now().isoformat()
        }
    }

    np.savez_compressed(output_file, **save_dict)

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"NPZ file saved: {output_file.name}")
    print(f"NPZ file size: {file_size_mb:.2f} MB")

    return output_file


def fetch_amsr2_data_safe(start_datetime: str, end_datetime: str, save_interval: int = SAVE_INTERVAL) -> pathlib.Path:
    """Безопасная версия основной функции"""

    print(f"Searching for AMSR-2 data: {start_datetime} → {end_datetime}")
    print("Using SAFE mode: sequential processing with retry logic")

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
        return None

    processing_start = time.time()
    all_products = res.products()

    # Используем ТОЛЬКО последовательную обработку для стабильности
    all_swaths = process_files_sequential_safe(all_products, TEMP_DIR, save_interval, start_datetime, end_datetime)

    processing_time = time.time() - processing_start

    print(f"Processing completed in {processing_time:.1f} seconds ({processing_time / 60:.1f} minutes)")

    if not all_swaths:
        print("No data was successfully processed")
        return None

    period_name = f"{start_datetime.replace(':', '').replace('-', '').replace('T', '_')}_to_{end_datetime.replace(':', '').replace('-', '').replace('T', '_')}"

    save_start = time.time()
    output_file = save_npz(all_swaths, BASE_DIR, period_name, start_datetime, end_datetime)
    save_time = time.time() - save_start

    # Cleanup
    try:
        backup_file = BACKUP_DIR / BACKUP_FILENAME
        if backup_file.exists():
            backup_file.unlink()
            print("Backup file cleaned up")
    except:
        pass

    try:
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
    """Main function with improved error handling"""
    print("=== AMSR-2 Processor - SAFE Mode (Sequential + Retry Logic) ===")
    print("This version handles network errors and server instability better")

    # Time parameters
    start_input = input("Start datetime (YYYY-MM-DD HH:MM:SS): ").strip()
    end_input = input("End datetime (YYYY-MM-DD HH:MM:SS): ").strip()

    start_datetime = dt.datetime.strptime(start_input, "%Y-%m-%d %H:%M:%S").isoformat()
    end_datetime = dt.datetime.strptime(end_input, "%Y-%m-%d %H:%M:%S").isoformat()

    # Save interval
    interval_input = input(f"Save interval in files (default {SAVE_INTERVAL}): ").strip()
    save_interval = int(interval_input) if interval_input else SAVE_INTERVAL

    print(f"\nProcessing: {start_input} → {end_input}")
    print(f"Mode: Sequential processing with retry logic")
    print(f"Save every: {save_interval} files")
    print(f"Max retries per file: {MAX_RETRIES}")
    print("WARNING: This will be slower but more reliable than parallel processing")

    confirm = input("\nProceed? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Cancelled")
        return

    output_file = fetch_amsr2_data_safe(start_datetime, end_datetime, save_interval)

    if output_file:
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"\nFile created: {output_file.name}")
        print(f"File size: {file_size_mb:.2f} MB")
        print("\nRecommendations:")
        print("1. Wait at least 1 hour before running another large batch")
        print("2. Consider using smaller date ranges (e.g., monthly instead of yearly)")
        print("3. Monitor success rates - if they drop below 80%, take a longer break")


if __name__ == "__main__":
    main()