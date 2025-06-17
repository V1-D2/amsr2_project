#!/usr/bin/env python3
"""
Backup File Checker - Проверка целостности файла бекапа AMSR-2
Проверяет что файл не поврежден и данные доступны
"""

import numpy as np
import matplotlib.pyplot as plt
import pathlib
from typing import Optional, Dict, List


def check_backup_file(npz_file_path: str, show_images: bool = True, num_images: int = 5) -> bool:
    """
    Проверить файл бекапа на целостность и показать несколько изображений

    Args:
        npz_file_path: путь к .npz файлу
        show_images: показывать ли изображения
        num_images: сколько изображений показать

    Returns:
        True если файл в порядке, False если поврежден
    """

    try:
        print(f"🔍 Checking backup file: {npz_file_path}")
        file_path = pathlib.Path(npz_file_path)

        # Проверить существование файла
        if not file_path.exists():
            print(f"❌ File not found: {npz_file_path}")
            return False

        # Размер файла
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"📦 File size: {file_size_mb:.2f} MB")

        # Загрузить NPZ файл
        print("📂 Loading NPZ file...")
        data = np.load(npz_file_path, allow_pickle=True)

        # Проверить структуру данных
        print("🔎 Checking data structure...")
        print(f"📊 Keys in file: {list(data.keys())}")

        # Загрузить массив swath_array
        if 'swath_array' not in data:
            print("❌ Error: 'swath_array' key not found in backup file")
            return False

        swath_array = data['swath_array']

        # Если это object array, преобразовать в список
        if swath_array.dtype == object:
            swath_list = swath_array.tolist()
        else:
            swath_list = swath_array

        print(f"✅ Found {len(swath_list)} swaths in backup")

        # Проверить первые несколько swaths
        valid_swaths = 0
        damaged_swaths = 0

        for i, swath in enumerate(swath_list[:10]):  # Проверить первые 10
            try:
                if swath is None:
                    continue

                # Проверить структуру swath
                if 'temperature' not in swath or 'metadata' not in swath:
                    print(f"⚠️  Swath {i}: Missing temperature or metadata")
                    damaged_swaths += 1
                    continue

                temp = swath['temperature']
                metadata = swath['metadata']

                # Проверить размеры температурного массива
                if temp is None or len(temp.shape) != 2:
                    print(f"⚠️  Swath {i}: Invalid temperature array")
                    damaged_swaths += 1
                    continue

                valid_swaths += 1

                if i < 3:  # Показать детали для первых 3
                    print(f"✅ Swath {i}: shape={temp.shape}, orbit_type={metadata.get('orbit_type', 'N/A')}")

            except Exception as e:
                print(f"❌ Error checking swath {i}: {e}")
                damaged_swaths += 1

        print(f"📈 Valid swaths: {valid_swaths}, Damaged swaths: {damaged_swaths}")

        # Показать изображения если запрошено
        if show_images and valid_swaths > 0:
            print(f"🖼️  Showing first {min(num_images, len(swath_list))} swath images...")
            visualize_swaths(swath_list, num_images)

        # Проверить дополнительные поля
        if 'period' in data:
            print(f"📅 Period: {data['period']}")

        if 'backup_info' in data:
            backup_info = data['backup_info'].item()
            print(f"🕒 Backup timestamp: {backup_info.get('timestamp', 'N/A')}")
            print(f"📁 Files in backup: {backup_info.get('files_in_backup', 'N/A')}")

        success_rate = valid_swaths / (valid_swaths + damaged_swaths) if (valid_swaths + damaged_swaths) > 0 else 0
        print(f"📊 Success rate: {success_rate:.1%}")

        # Финальная оценка
        if success_rate > 0.95:
            print("✅ Backup file is in GOOD condition")
            return True
        elif success_rate > 0.80:
            print("⚠️  Backup file has some issues but mostly usable")
            return True
        else:
            print("❌ Backup file is DAMAGED")
            return False

    except Exception as e:
        print(f"❌ Error checking backup file: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_swaths(swath_list: List[Dict], num_images: int = 5, rotate_k: int = 1):
    """
    Визуализировать несколько swaths из бекапа

    Args:
        swath_list: список swaths
        num_images: количество изображений для показа
        rotate_k: поворот изображения (1 = 90° CCW)
    """

    valid_swaths = [s for s in swath_list if s is not None and 'temperature' in s]
    num_to_show = min(num_images, len(valid_swaths))

    if num_to_show == 0:
        print("❌ No valid swaths found for visualization")
        return

    for i in range(num_to_show):
        try:
            swath = valid_swaths[i]
            temp = swath['temperature']
            metadata = swath['metadata']

            # Получить информацию о температуре
            valid_mask = temp != 0
            if np.sum(valid_mask) == 0:
                print(f"⚠️  Swath {i}: No valid temperature data")
                continue

            # Применить scale factor если нужно
            scale_factor = metadata.get('scale_factor', 1.0)
            if scale_factor != 1.0:
                temp_scaled = temp.astype(float) * scale_factor
            else:
                temp_scaled = temp.astype(float)

            # Заменить нули на NaN для визуализации
            temp_scaled[~valid_mask] = np.nan

            # Поворот
            temp_rot = np.rot90(temp_scaled, k=rotate_k)

            # Размеры фигуры
            h, w = temp.shape
            ratio = h / w
            height_in = 4
            width_in = ratio * height_in

            # Создать график
            plt.figure(figsize=(width_in, height_in))

            # Использовать только валидные данные для colormap
            valid_data = temp_rot[~np.isnan(temp_rot)]
            if len(valid_data) > 0:
                vmin, vmax = np.percentile(valid_data, [2, 98])  # Убрать выбросы

                im = plt.imshow(temp_rot, cmap="turbo", aspect="auto", vmin=vmin, vmax=vmax)
                plt.colorbar(im, label="Brightness Temperature (K)")

                orbit_type = metadata.get('orbit_type', 'U')
                temp_range = metadata.get('temp_range', (0, 0))

                plt.title(f"Swath {i} - AMSR-2 36.5 GHz H-Pol\n"
                          f"Orbit: {orbit_type}, Range: {temp_range[0]}-{temp_range[1]}K")
                plt.xlabel("Along-track scan # (after rotation)")
                plt.ylabel("Across-track pixel # (after rotation)")
                plt.tight_layout()
                plt.show()

                print(f"✅ Swath {i}: {temp.shape}, valid pixels: {np.sum(valid_mask)}/{temp.size}")
            else:
                print(f"⚠️  Swath {i}: No valid data for visualization")

        except Exception as e:
            print(f"❌ Error visualizing swath {i}: {e}")


def quick_backup_check(backup_file_path: str) -> bool:
    """
    Быстрая проверка бекапа без визуализации
    """
    return check_backup_file(backup_file_path, show_images=False, num_images=0)


def main():
    """
    Основная функция для проверки бекапа
    """
    print("=== AMSR-2 Backup File Checker ===\n")

    # Путь к файлу бекапа
    backup_path = input("Enter path to backup file (or press Enter for default): ").strip()

    if not backup_path:
        # Путь по умолчанию для simple backup версии
        backup_path = "data/TEMP_BACKUP.npz"

    # Опции
    show_images = input("Show images? (y/n, default y): ").strip().lower()
    show_images = show_images != 'n'

    if show_images:
        num_images_str = input("Number of images to show (default 5): ").strip()
        num_images = int(num_images_str) if num_images_str else 5
    else:
        num_images = 0

    # Проверить файл
    print("\n" + "=" * 50)
    is_ok = check_backup_file(backup_path, show_images, num_images)
    print("=" * 50)

    if is_ok:
        print("✅ RESULT: Backup file is OK!")
    else:
        print("❌ RESULT: Backup file has problems!")


if __name__ == "__main__":
    main()