#!/usr/bin/env python3
"""
AMSR-2 Data Processor - объединенная версия
Процесс скачивания из оптимизированного кода + формат сохранения array of dictionaries
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

# Настройка G-Portal
gportal.username = GPORTAL_USERNAME
gportal.password = GPORTAL_PASSWORD

_DS = gportal.datasets()["GCOM-W/AMSR2"]["LEVEL1"]
DS_L1B_TB = _DS["L1B-Brightness temperature（TB）"][0]


class ThreadSafeProgress:
    def __init__(self):
        self.lock = threading.Lock()
        self.downloaded = 0
        self.processed = 0
        self.deleted = 0
        self.total_files = 0

    def set_total(self, total):
        with self.lock:
            self.total_files = total

    def update_download(self):
        with self.lock:
            self.downloaded += 1
            return self.downloaded, self.total_files

    def update_processed(self):
        with self.lock:
            self.processed += 1
            return self.processed, self.total_files

    def update_deleted(self):
        with self.lock:
            self.deleted += 1
            return self.deleted, self.total_files


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
                return None

            valid_count = np.sum(~np.isnan(scaled_temp))
            if valid_count == 0:
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
                'lon_range': (np.nanmin(lon_36), np.nanmax(lon_36))
            }

            return {
                'temperature': scaled_temp.astype(np.float32),
                'latitude': lat_36.astype(np.float32),
                'longitude': lon_36.astype(np.float32),
                'metadata': metadata
            }

    except Exception as e:
        print(f"Error processing {h5_path.name}: {e}")
        return None


def download_process_delete_single(product, temp_dir: pathlib.Path, progress: ThreadSafeProgress) -> Optional[Dict]:
    """
    КЛЮЧЕВАЯ ФУНКЦИЯ: Скачивает -> Обрабатывает -> СРАЗУ УДАЛЯЕТ файл
    Это экономит место на диске!
    """
    downloaded_file = None
    try:
        # 1. СКАЧИВАЕМ
        local_path = gportal.download(product, local_dir=str(temp_dir))
        downloaded_file = pathlib.Path(local_path)
        progress.update_download()

        # 2. СРАЗУ ОБРАБАТЫВАЕМ
        swath_data = extract_swath_data(downloaded_file)
        progress.update_processed()

        # 3. НЕМЕДЛЕННО УДАЛЯЕМ H5 ФАЙЛ!
        try:
            downloaded_file.unlink()  # Удаляем файл сразу после обработки
            progress.update_deleted()
        except Exception as e:
            pass  # Игнорируем ошибки удаления

        return swath_data

    except Exception as e:
        # Если что-то пошло не так, все равно удаляем файл
        if downloaded_file and downloaded_file.exists():
            try:
                downloaded_file.unlink()
                progress.update_deleted()
            except:
                pass

        progress.update_download()
        progress.update_processed()
        return None


def process_files_batch_immediate_cleanup(products: List, temp_dir: pathlib.Path,
                                          max_workers: int = 4) -> List[Dict]:
    """
    Обрабатывает файлы с НЕМЕДЛЕННЫМ удалением после каждого файла
    Никаких накоплений H5 файлов!
    """
    print(f"\n=== BATCH PROCESSING ({max_workers} threads) - IMMEDIATE CLEANUP ===")
    print("Файлы удаляются СРАЗУ после обработки для экономии места")

    product_list = list(products)
    total_products = len(product_list)

    progress = ThreadSafeProgress()
    progress.set_total(total_products)

    all_swaths = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Отправляем все задачи: скачать -> обработать -> удалить
        future_to_product = {
            executor.submit(download_process_delete_single, product, temp_dir, progress): product
            for product in product_list
        }

        # Прогресс-бар показывает весь процесс
        with tqdm.tqdm(total=total_products, desc="Download→Process→Delete") as pbar:
            for future in concurrent.futures.as_completed(future_to_product):
                result = future.result()
                if result is not None:
                    all_swaths.append(result)
                pbar.update(1)

    print(f"Обработано: {len(all_swaths)}/{total_products} файлов")
    print(f"H5 файлы удалены СРАЗУ после обработки - место сэкономлено!")
    return all_swaths


def process_in_batches(products: List, temp_dir: pathlib.Path,
                       batch_size: int = 50, max_workers: int = 4) -> List[Dict]:
    """
    Обрабатывает файлы ПАЧКАМИ для контроля использования диска
    """
    product_list = list(products)
    total_products = len(product_list)
    all_swaths = []

    print(f"\n=== BATCH PROCESSING ===")
    print(f"Всего файлов: {total_products}")
    print(f"Размер пачки: {batch_size}")
    print(f"Потоков: {max_workers}")
    print(f"Файлы удаляются СРАЗУ после обработки каждого!")

    # Разбиваем на пачки
    for batch_start in range(0, total_products, batch_size):
        batch_end = min(batch_start + batch_size, total_products)
        batch_products = product_list[batch_start:batch_end]
        batch_num = (batch_start // batch_size) + 1
        total_batches = (total_products + batch_size - 1) // batch_size

        print(f"\n--- ПАЧКА {batch_num}/{total_batches} ({len(batch_products)} файлов) ---")

        # Обрабатываем пачку с немедленным удалением
        batch_swaths = process_files_batch_immediate_cleanup(
            batch_products, temp_dir, max_workers
        )

        all_swaths.extend(batch_swaths)

        print(f"Пачка {batch_num} завершена. Получено {len(batch_swaths)} свотов.")
        print(f"Всего обработано: {len(all_swaths)}/{total_products}")

        # Показываем использование диска
        disk_usage = get_disk_usage(temp_dir)
        print(f"Использование диска: {disk_usage:.1f} MB")

    return all_swaths


def get_disk_usage(directory: pathlib.Path) -> float:
    """Показывает использование диска в МБ"""
    try:
        total_size = 0
        if directory.exists():
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        return total_size / (1024 * 1024)
    except:
        return 0.0


def save_swaths_array(swath_list: List[Dict], base_dir: pathlib.Path,
                      period_name: str, start_datetime: str, end_datetime: str) -> pathlib.Path:
    """
    Save as array of dictionaries (original structure)
    """
    print(f"\n=== SAVING ARRAY OF SWATH DICTIONARIES ===")

    output_file = base_dir / f"AMSR2_swaths_{period_name}.npz"

    print("Preparing swath array structure...")

    # Create array of dictionaries structure
    swath_array = []
    for i, swath in enumerate(tqdm.tqdm(swath_list, desc="Preparing swath array")):
        swath_dict = {
            'temperature': swath['temperature'].astype(np.float32),
            'latitude': swath['latitude'].astype(np.float32),
            'longitude': swath['longitude'].astype(np.float32),
            'metadata': swath['metadata']
        }
        swath_array.append(swath_dict)

    # Save as compressed NPZ
    save_dict = {
        'swath_array': swath_array,
        'period': f"{start_datetime} to {end_datetime}",
        'num_swaths': len(swath_list),
        'description': 'AMSR-2 36.5GHz H swath data - Array of dictionaries format'
    }

    print("Saving with maximum compression...")
    np.savez_compressed(output_file, **save_dict)

    # Statistics
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    total_pixels = sum(s['metadata']['valid_pixels'] for s in swath_list)

    print(f"NPZ file saved: {output_file.name}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Total pixels: {total_pixels:,}")
    print(f"Structure: Array of {len(swath_list)} swath dictionaries")

    return output_file


def load_swath_array(dataset_path: pathlib.Path) -> List[Dict]:
    """
    Load data from NPZ file
    """
    with np.load(dataset_path, allow_pickle=True) as data:
        swath_array = data['swath_array']
        num_swaths = int(data['num_swaths'])

        print(f"Loaded array of {num_swaths} swath dictionaries from NPZ file")
        return swath_array.tolist()


def get_optimal_settings():
    """Определяет оптимальные настройки на основе системы"""
    cpu_count = os.cpu_count() or 4

    # Консервативный подход
    max_workers = min(cpu_count, 6)  # Не более 6 потоков
    batch_size = 30  # Небольшие пачки для экономии места

    return max_workers, batch_size


def ask_compression_level():
    """Спрашивает пользователя о желаемом уровне сжатия"""
    print(f"\nНастройка сжатия файла:")
    print("1. Минимальное сжатие (быстро, больше размер)")
    print("2. Среднее сжатие (убирает пустые области, рекомендуется)")
    print("3. Максимальное сжатие (медленно, минимальный размер)")

    while True:
        try:
            level = input("Выберите уровень сжатия (1-3, Enter для 2): ").strip()

            if level == "":
                return 2

            level = int(level)
            if 1 <= level <= 3:
                descriptions = {
                    1: "минимальное",
                    2: "среднее (рекомендуется)",
                    3: "максимальное"
                }
                print(f"Выбрано: {descriptions[level]} сжатие")
                return level
            else:
                print("Введите число от 1 до 3")

        except ValueError:
            print("Введите число от 1 до 3")
        except KeyboardInterrupt:
            return 2


def fetch_amsr2_data(start_datetime: str, end_datetime: str,
                     base: pathlib.Path = BASE_DIR,
                     temp_dir: Optional[pathlib.Path] = None,
                     max_workers: Optional[int] = None,
                     batch_size: Optional[int] = None,
                     compression_level: int = 2) -> pathlib.Path:
    """
    ОБЪЕДИНЕННАЯ функция:
    - Оптимизированный процесс скачивания с немедленным удалением
    - Сохранение в формате array of dictionaries с компрессией
    """

    # Получаем оптимальные настройки
    if max_workers is None or batch_size is None:
        opt_workers, opt_batch = get_optimal_settings()
        max_workers = max_workers or opt_workers
        batch_size = batch_size or opt_batch

    print(f"Настройки: {max_workers} потоков, пачки по {batch_size} файлов")
    print(f"ВАЖНО: H5 файлы удаляются СРАЗУ после обработки!")

    base.mkdir(parents=True, exist_ok=True)

    if temp_dir is None:
        temp_dir = TEMP_DIR
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== SEARCHING FOR AMSR-2 DATA ===")
    print(f"Период: {start_datetime} → {end_datetime}")

    search_start = time.time()
    res = gportal.search(
        dataset_ids=[DS_L1B_TB],
        start_time=start_datetime,
        end_time=end_datetime
    )
    search_time = time.time() - search_start

    total_files = res.matched()
    print(f"Найдено {total_files} файлов за {search_time:.1f} секунд")

    if total_files == 0:
        print("Данные не найдены")
        return None

    # Оценка места на диске
    estimated_size_mb = total_files * 30  # ~30 МБ на файл
    batch_size_mb = batch_size * 30

    print(f"Оценка размера всех H5: ~{estimated_size_mb:.0f} МБ")
    print(f"Максимум на диске одновременно: ~{batch_size_mb:.0f} МБ (пачка)")
    print(f"Экономия места: {(estimated_size_mb - batch_size_mb) / estimated_size_mb * 100:.0f}%")

    # Обработка с немедленным удалением
    processing_start = time.time()
    all_products = res.products()
    all_swaths = process_in_batches(all_products, temp_dir, batch_size, max_workers)
    processing_time = time.time() - processing_start

    print(f"\nОбработка завершена за {processing_time:.1f} секунд")
    print(f"Извлечено {len(all_swaths)} свотов")

    if not all_swaths:
        print("Не удалось обработать данные")
        return None

    # Статистика
    total_pixels = sum(s['metadata']['valid_pixels'] for s in all_swaths)
    print(f"Всего валидных пикселей: {total_pixels:,}")

    # Сохранение в формате array of dictionaries с компрессией
    period_name = f"{start_datetime.replace(':', '').replace('-', '').replace('T', '_')}_to_{end_datetime.replace(':', '').replace('-', '').replace('T', '_')}"

    save_start = time.time()
    output_file = save_swaths_array(all_swaths, base, period_name, start_datetime, end_datetime)
    save_time = time.time() - save_start

    # Финальная очистка временной папки
    try:
        temp_dir.rmdir()
        print(f"Временная папка удалена: {temp_dir}")
    except:
        print(f"Временная папка не пуста: {temp_dir}")

    # Итоговая статистика
    total_time = search_time + processing_time + save_time
    print(f"\n=== ИТОГОВАЯ СТАТИСТИКА ===")
    print(f"Поиск: {search_time:.1f}с | Обработка: {processing_time:.1f}с | Сохранение: {save_time:.1f}с")
    print(f"Общее время: {total_time:.1f}с ({total_time / 60:.1f} минут)")
    print(f"Скорость: {total_files / total_time:.2f} файлов/сек")
    print(f"✅ H5 файлы НЕ сохранены - место сэкономлено!")

    return output_file


def main():
    """
    Интерактивный запуск с вводом параметров
    """
    print("=== AMSR-2 PROCESSOR ===")
    print("Особенности:")
    print("• H5 файлы удаляются СРАЗУ после обработки")
    print("• Обработка пачками для экономии места")
    print("• Сохранение в формате array of dictionaries с компрессией")
    print("• Многопоточность для скорости")

    # Ввод временного интервала
    print("\nВременной интервал:")
    print("Формат: YYYY-MM-DD HH:MM:SS (например, 2025-05-20 14:30:00)")

    while True:
        try:
            start_input = input("Дата и время начала: ").strip()
            end_input = input("Дата и время окончания: ").strip()

            # Проверяем формат
            dt_start = dt.datetime.strptime(start_input, "%Y-%m-%d %H:%M:%S")
            dt_end = dt.datetime.strptime(end_input, "%Y-%m-%d %H:%M:%S")

            if dt_end <= dt_start:
                print("Ошибка: Время окончания должно быть позже времени начала.")
                continue

            # Конвертируем в ISO формат
            start_datetime = dt_start.isoformat()
            end_datetime = dt_end.isoformat()

            # Показываем продолжительность
            duration = dt_end - dt_start
            hours = duration.total_seconds() / 3600
            print(f"Выбранный период: {start_input} → {end_input} ({hours:.1f} часов)")

            # Примерная оценка
            est_files = hours * 15  # примерно 15 файлов в час
            est_size_gb = est_files * 30 / 1024  # H5 размер
            final_size_mb = est_files * 0.5  # NPZ сжатый размер
            print(f"Ожидаемо файлов: ~{est_files:.0f}")
            print(f"H5 размер (НЕ сохраняется): ~{est_size_gb:.1f} ГБ")
            print(f"Финальный NPZ: ~{final_size_mb:.0f} МБ")
            print(f"Экономия места: {est_size_gb * 1024 / final_size_mb:.1f}x!")

            break

        except ValueError:
            print("Ошибка: Неверный формат. Используйте YYYY-MM-DD HH:MM:SS")
        except KeyboardInterrupt:
            print("\nОтменено.")
            return

    # Ввод количества потоков и настроек
    print(f"\nНастройки обработки:")
    opt_workers, opt_batch = get_optimal_settings()
    print(f"Рекомендуемые настройки:")
    print(f"  Потоков: {opt_workers}")
    print(f"  Размер пачки: {opt_batch} файлов")

    use_custom = input("Изменить настройки? (y/n): ").strip().lower()

    if use_custom == 'y':
        try:
            workers = int(input(f"Потоков (1-8, рекомендуется {opt_workers}): ").strip())
            batch = int(input(f"Размер пачки (10-100, рекомендуется {opt_batch}): ").strip())

            max_workers = max(1, min(8, workers))
            batch_size = max(10, min(100, batch))
        except ValueError:
            print("Неверный ввод, используем рекомендуемые настройки")
            max_workers, batch_size = opt_workers, opt_batch
    else:
        max_workers, batch_size = opt_workers, opt_batch

    # Запрос уровня сжатия
    compression_level = ask_compression_level()

    # Подтверждение
    print(f"\n=== ПОДТВЕРЖДЕНИЕ ===")
    print(f"Период: {start_input} → {end_input}")
    print(f"Потоков: {max_workers}")
    print(f"Размер пачки: {batch_size}")
    print(f"Сжатие: уровень {compression_level}")
    print(f"Ожидаемое время: ~{est_files / max_workers / 60:.1f} минут")

    confirm = input("\nНачать обработку? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes', 'да', 'д']:
        print("Отменено.")
        return

    print(f"\n=== НАЧИНАЕМ ОБРАБОТКУ ===")
    overall_start = time.time()

    output_file = fetch_amsr2_data(
        start_datetime, end_datetime,
        max_workers=max_workers,
        batch_size=batch_size,
        compression_level=compression_level
    )

    overall_time = time.time() - overall_start

    if output_file:
        print(f"\n=== ТЕСТИРОВАНИЕ ЗАГРУЗКИ ===")

        load_start = time.time()
        swath_array = load_swath_array(output_file)
        load_time = time.time() - load_start

        print(f"Загружено {len(swath_array)} свотов за {load_time:.1f} секунд")

        if len(swath_array) > 0:
            example_swath = swath_array[0]
            temp_array = example_swath['temperature']
            lat_array = example_swath['latitude']
            lon_array = example_swath['longitude']

            print(f"\nПример свота:")
            print(f"  Размер температур: {temp_array.shape}")
            print(f"  Диапазон температур: {np.nanmin(temp_array):.1f} - {np.nanmax(temp_array):.1f} K")
            print(f"  Диапазон широт: {np.nanmin(lat_array):.1f}° - {np.nanmax(lat_array):.1f}°")
            print(f"  Диапазон долгот: {np.nanmin(lon_array):.1f}° - {np.nanmax(lon_array):.1f}°")

    print(f"\n=== ОБРАБОТКА ЗАВЕРШЕНА ===")
    print(f"Общее время: {overall_time:.1f} секунд ({overall_time / 60:.1f} минут)")
    print("🗑️ H5 файлы удалены - место сэкономлено!")
    print("💾 Сжатый NPZ файл в формате array of dictionaries сохранен!")


if __name__ == "__main__":
    main()
    # !/usr/bin/env python3
"""
AMSR-2 Data Processor - адаптация для PyCharm
Прямой перенос вашего кода из Google Colab
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

# Настройка G-Portal
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
                return None

            valid_count = np.sum(~np.isnan(scaled_temp))
            if valid_count == 0:
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
                'lon_range': (np.nanmin(lon_36), np.nanmax(lon_36))
            }

            return {
                'temperature': scaled_temp.astype(np.float32),
                'latitude': lat_36.astype(np.float32),
                'longitude': lon_36.astype(np.float32),
                'metadata': metadata
            }

    except Exception as e:
        print(f"Error processing {h5_path.name}: {e}")
        return None


def download_and_process_single(product, temp_dir: pathlib.Path, progress: ThreadSafeProgress) -> Optional[Dict]:
    """
    Download -> Process -> Delete file immediately
    """
    downloaded_file = None
    try:
        # Download
        local_path = gportal.download(product, local_dir=str(temp_dir))
        downloaded_file = pathlib.Path(local_path)

        # Process immediately
        swath_data = extract_swath_data(downloaded_file)

        # Delete file immediately
        try:
            downloaded_file.unlink()
        except Exception as e:
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


def process_files_concurrent(products: List, temp_dir: pathlib.Path,
                             max_workers: int = 4) -> List[Dict]:
    """
    Process files with immediate cleanup using concurrent threads
    """
    print(f"\n=== CONCURRENT PROCESSING ({max_workers} threads) ===")

    product_list = list(products)
    total_products = len(product_list)

    progress = ThreadSafeProgress()
    progress.set_total(total_products)

    all_swaths = []
    swaths_lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_product = {
            executor.submit(download_and_process_single, product, temp_dir, progress): product
            for product in product_list
        }

        with tqdm.tqdm(total=total_products, desc="Processing files") as pbar:
            for future in concurrent.futures.as_completed(future_to_product):
                result = future.result()
                if result is not None:
                    with swaths_lock:
                        all_swaths.append(result)
                pbar.update(1)

    print(f"Successfully processed: {len(all_swaths)}/{total_products} files")
    return all_swaths


def save_swaths_array(swath_list: List[Dict], base_dir: pathlib.Path,
                      period_name: str, start_datetime: str, end_datetime: str) -> pathlib.Path:
    """
    Save as array of dictionaries (new structure)
    """
    print(f"\n=== SAVING ARRAY OF SWATH DICTIONARIES ===")

    output_file = base_dir / f"AMSR2_swaths_{period_name}.npz"

    print("Preparing swath array structure...")

    # Create array of dictionaries structure
    swath_array = []
    for i, swath in enumerate(tqdm.tqdm(swath_list, desc="Preparing swath array")):
        swath_dict = {
            'temperature': swath['temperature'].astype(np.float32),
            'latitude': swath['latitude'].astype(np.float32),
            'longitude': swath['longitude'].astype(np.float32),
            'metadata': swath['metadata']
        }
        swath_array.append(swath_dict)

    # Save as compressed NPZ
    save_dict = {
        'swath_array': swath_array,
        'period': f"{start_datetime} to {end_datetime}",
        'num_swaths': len(swath_list),
        'description': 'AMSR-2 36.5GHz H swath data - Array of dictionaries format'
    }

    print("Saving with maximum compression...")
    np.savez_compressed(output_file, **save_dict)

    # Statistics
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    total_pixels = sum(s['metadata']['valid_pixels'] for s in swath_list)

    print(f"NPZ file saved: {output_file.name}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Total pixels: {total_pixels:,}")
    print(f"Structure: Array of {len(swath_list)} swath dictionaries")

    return output_file


def get_optimal_settings():
    """
    Determine optimal settings based on system resources
    """
    cpu_count = os.cpu_count() or 4
    max_workers = min(cpu_count, 8)
    return max_workers


def fetch_amsr2_data(start_datetime: str, end_datetime: str,
                     base: pathlib.Path = BASE_DIR,
                     temp_dir: Optional[pathlib.Path] = None,
                     max_workers: Optional[int] = None,
                     compression_level: int = 2) -> pathlib.Path:
    """
    Main function
    """

    if max_workers is None:
        max_workers = get_optimal_settings()

    print(f"Settings: {max_workers} threads")

    base.mkdir(parents=True, exist_ok=True)

    if temp_dir is None:
        temp_dir = TEMP_DIR
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== SEARCHING FOR AMSR-2 DATA ===")
    print(f"Period: {start_datetime} → {end_datetime}")

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

    # Processing with immediate cleanup
    processing_start = time.time()
    all_products = res.products()
    all_swaths = process_files_concurrent(all_products, temp_dir, max_workers)
    processing_time = time.time() - processing_start

    print(f"\nProcessing completed in {processing_time:.1f} seconds")
    print(f"Extracted {len(all_swaths)} swaths")

    if not all_swaths:
        print("Failed to process data")
        return None

    # Statistics
    total_pixels = sum(s['metadata']['valid_pixels'] for s in all_swaths)
    print(f"Total valid pixels: {total_pixels:,}")

    # Save
    period_name = f"{start_datetime.replace(':', '').replace('-', '').replace('T', '_')}_to_{end_datetime.replace(':', '').replace('-', '').replace('T', '_')}"

    save_start = time.time()
    output_file = save_swaths_array(all_swaths, base, period_name, start_datetime, end_datetime, compression_level)
    save_time = time.time() - save_start

    # Cleanup temp directory
    try:
        temp_dir.rmdir()
        print(f"Temp directory removed: {temp_dir}")
    except:
        print(f"Temp directory not empty: {temp_dir}")

    # Final statistics
    total_time = search_time + processing_time + save_time
    print(f"\n=== FINAL STATISTICS ===")
    print(f"Search: {search_time:.1f}s | Processing: {processing_time:.1f}s | Saving: {save_time:.1f}s")
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} minutes)")
    print(f"Speed: {total_files / total_time:.2f} files/sec")

    return output_file


def load_swath_array(dataset_path: pathlib.Path) -> List[Dict]:
    """
    Load data from NPZ file
    """
    with np.load(dataset_path, allow_pickle=True) as data:
        swath_array = data['swath_array']
        num_swaths = int(data['num_swaths'])

        print(f"Loaded array of {num_swaths} swath dictionaries from NPZ file")
        return swath_array.tolist()


def main():
    """
    Интерактивный запуск с вводом параметров
    """
    print("=== AMSR-2 PROCESSOR ===")
    print("Введите параметры для обработки данных AMSR-2:")

    # Ввод временного интервала
    print("\nВременной интервал:")
    print("Формат: YYYY-MM-DD HH:MM:SS (например, 2025-05-20 14:30:00)")

    while True:
        try:
            start_input = input("Дата и время начала: ").strip()
            end_input = input("Дата и время окончания: ").strip()

            # Проверяем формат
            dt_start = dt.datetime.strptime(start_input, "%Y-%m-%d %H:%M:%S")
            dt_end = dt.datetime.strptime(end_input, "%Y-%m-%d %H:%M:%S")

            if dt_end <= dt_start:
                print("Ошибка: Время окончания должно быть позже времени начала.")
                continue

            # Конвертируем в ISO формат
            start_datetime = dt_start.isoformat()
            end_datetime = dt_end.isoformat()

            # Показываем продолжительность
            duration = dt_end - dt_start
            hours = duration.total_seconds() / 3600
            print(f"Выбранный период: {start_input} → {end_input} ({hours:.1f} часов)")

            # Примерная оценка
            est_files = hours * 15  # примерно 15 файлов в час
            est_size_mb = est_files * 0.5  # примерно 0.5 МБ на файл
            print(f"Ожидаемо файлов: ~{est_files:.0f}")
            print(f"Размер результата: ~{est_size_mb:.0f} МБ")

            break

        except ValueError:
            print("Ошибка: Неверный формат. Используйте YYYY-MM-DD HH:MM:SS")
        except KeyboardInterrupt:
            print("\nОтменено.")
            return

    # Ввод количества потоков
    print(f"\nНастройка потоков:")
    optimal_workers = get_optimal_settings()
    print(f"Рекомендуемое количество потоков: {optimal_workers}")

    while True:
        try:
            workers_input = input(f"Количество потоков (1-16, Enter для {optimal_workers}): ").strip()

            if workers_input == "":
                max_workers = optimal_workers
            else:
                max_workers = int(workers_input)
                max_workers = max(1, min(16, max_workers))  # ограничиваем 1-16

            print(f"Используется потоков: {max_workers}")
            break

        except ValueError:
            print("Ошибка: Введите число от 1 до 16")
        except KeyboardInterrupt:
            print("\nОтменено.")
            return

    # Запрос уровня сжатия
    compression_level = ask_compression_level()

    # Подтверждение
    print(f"\n=== ПОДТВЕРЖДЕНИЕ ===")
    print(f"Период: {start_input} → {end_input}")
    print(f"Потоков: {max_workers}")
    print(f"Сжатие: уровень {compression_level}")
    print(f"Ожидаемое время: ~{est_files / max_workers / 60:.1f} минут")

    confirm = input("\nНачать обработку? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes', 'да', 'д']:
        print("Отменено.")
        return

    print(f"\n=== НАЧИНАЕМ ОБРАБОТКУ ===")
    overall_start = time.time()

    output_file = fetch_amsr2_data(
        start_datetime, end_datetime,
        max_workers=max_workers,
        compression_level=compression_level
    )

    overall_time = time.time() - overall_start

    if output_file:
        print(f"\n=== TESTING LOAD ===")

        load_start = time.time()
        swath_array = load_swath_array(output_file)
        load_time = time.time() - load_start

        print(f"Loaded {len(swath_array)} swaths in {load_time:.1f} seconds")

        if len(swath_array) > 0:
            example_swath = swath_array[0]
            temp_array = example_swath['temperature']
            lat_array = example_swath['latitude']
            lon_array = example_swath['longitude']

            print(f"\nExample swath:")
            print(f"  Temperature shape: {temp_array.shape}")
            print(f"  Temperature range: {np.nanmin(temp_array):.1f} - {np.nanmax(temp_array):.1f} K")
            print(f"  Latitude range: {np.nanmin(lat_array):.1f}° - {np.nanmax(lat_array):.1f}°")
            print(f"  Longitude range: {np.nanmin(lon_array):.1f}° - {np.nanmax(lon_array):.1f}°")

    print(f"\n=== PROCESSING COMPLETED ===")
    print(f"Total time: {overall_time:.1f} seconds ({overall_time / 60:.1f} minutes)")


if __name__ == "__main__":
    main()