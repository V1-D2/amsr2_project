#!/usr/bin/env python3
"""
AMSR2 Swath Size Checker
Проверяет размеры swaths напрямую из gportal для понимания
почему размеры разные
"""

import pathlib
import datetime as dt
import gportal
import h5py
import numpy as np
from typing import List, Dict, Optional

# Import your config
from config import BASE_DIR, TEMP_DIR, GPORTAL_USERNAME, GPORTAL_PASSWORD

# Setup gportal
gportal.username = GPORTAL_USERNAME
gportal.password = GPORTAL_PASSWORD

_DS = gportal.datasets()["GCOM-W/AMSR2"]["LEVEL1"]
DS_L1B_TB = _DS["L1B-Brightness temperature（TB）"][0]


def check_h5_file_info(h5_path: pathlib.Path) -> Optional[Dict]:
    """
    Проверить информацию о H5 файле включая все переменные и их размеры
    """
    try:
        with h5py.File(h5_path, "r") as h5:
            info = {
                'filename': h5_path.name,
                'variables': {},
                'file_size_mb': h5_path.stat().st_size / (1024 * 1024)
            }

            # Получить информацию о всех переменных
            def collect_info(name, obj):
                if isinstance(obj, h5py.Dataset):
                    info['variables'][name] = {
                        'shape': obj.shape,
                        'dtype': str(obj.dtype),
                        'size_mb': obj.nbytes / (1024 * 1024)
                    }

                    # Для температурных данных получить дополнительную инфо
                    if "Brightness Temperature" in name:
                        # Get scale factor
                        scale = 1.0
                        if "SCALE FACTOR" in obj.attrs:
                            scale = obj.attrs["SCALE FACTOR"]
                            if isinstance(scale, np.ndarray):
                                scale = scale[0]

                        data = obj[:]
                        valid_mask = data != 0
                        valid_count = np.sum(valid_mask)

                        info['variables'][name].update({
                            'scale_factor': float(scale),
                            'valid_pixels': int(valid_count),
                            'total_pixels': int(data.size),
                            'valid_percentage': float(valid_count / data.size * 100),
                            'data_range': (int(np.min(data[valid_mask])) if valid_count > 0 else 0,
                                           int(np.max(data[valid_mask])) if valid_count > 0 else 0)
                        })

            h5.visititems(collect_info)
            return info

    except Exception as e:
        print(f"❌ Error reading {h5_path.name}: {e}")
        return None


def download_and_check_few_files(start_datetime: str, end_datetime: str, max_files: int = 5) -> List[Dict]:
    """
    Скачать несколько файлов и проверить их размеры
    """
    print(f"🔍 Searching for AMSR-2 data: {start_datetime} → {end_datetime}")

    # Create temp directory
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Search for data
    res = gportal.search(
        dataset_ids=[DS_L1B_TB],
        start_time=start_datetime,
        end_time=end_datetime
    )

    total_files = res.matched()
    print(f"📊 Found {total_files} files total")

    if total_files == 0:
        print("❌ No data found")
        return []

    # Get first few products
    all_products = res.products()
    products_to_check = list(all_products)[:max_files]

    print(f"📥 Will download and check first {len(products_to_check)} files")

    results = []

    for i, product in enumerate(products_to_check):
        try:
            print(f"\n📦 Processing file {i + 1}/{len(products_to_check)}")

            # Download file
            print(f"⬇️  Downloading...")
            local_path = gportal.download(product, local_dir=str(TEMP_DIR))
            downloaded_file = pathlib.Path(local_path)

            print(f"📁 Downloaded: {downloaded_file.name}")

            # Check file info
            file_info = check_h5_file_info(downloaded_file)

            if file_info:
                results.append(file_info)

                # Print summary for this file
                print(f"📋 File size: {file_info['file_size_mb']:.2f} MB")
                print(f"📊 Variables found: {len(file_info['variables'])}")

                # Look for temperature variables
                temp_vars = [name for name in file_info['variables'].keys()
                             if "Brightness Temperature" in name]

                if temp_vars:
                    print(f"🌡️  Temperature variables: {len(temp_vars)}")
                    for var_name in temp_vars:
                        var_info = file_info['variables'][var_name]
                        print(f"   • {var_name}: shape={var_info['shape']}, "
                              f"valid={var_info['valid_pixels']}/{var_info['total_pixels']} "
                              f"({var_info['valid_percentage']:.1f}%)")
                else:
                    print("⚠️  No temperature variables found!")

            # Clean up file
            try:
                downloaded_file.unlink()
                print(f"🗑️  Cleaned up: {downloaded_file.name}")
            except:
                pass

        except Exception as e:
            print(f"❌ Error processing file {i + 1}: {e}")
            # Clean up on error
            if 'downloaded_file' in locals() and downloaded_file.exists():
                try:
                    downloaded_file.unlink()
                except:
                    pass

    return results


def analyze_size_variations(results: List[Dict]):
    """
    Анализировать вариации размеров между файлами
    """
    print(f"\n{'=' * 60}")
    print("📊 SIZE VARIATION ANALYSIS")
    print(f"{'=' * 60}")

    if not results:
        print("❌ No results to analyze")
        return

    # Collect all temperature variable shapes
    temp_shapes = {}  # var_name -> [shapes]

    for result in results:
        for var_name, var_info in result['variables'].items():
            if "Brightness Temperature" in var_name:
                if var_name not in temp_shapes:
                    temp_shapes[var_name] = []
                temp_shapes[var_name].append(var_info['shape'])

    # Analyze each temperature variable
    for var_name, shapes in temp_shapes.items():
        print(f"\n🌡️  Variable: {var_name}")
        print(f"📏 Found {len(shapes)} shapes:")

        # Count unique shapes
        unique_shapes = {}
        for shape in shapes:
            shape_str = f"{shape[0]}×{shape[1]}"
            if shape_str not in unique_shapes:
                unique_shapes[shape_str] = 0
            unique_shapes[shape_str] += 1

        # Print shape distribution
        for shape_str, count in sorted(unique_shapes.items()):
            percentage = count / len(shapes) * 100
            print(f"   • {shape_str}: {count} files ({percentage:.1f}%)")

        if len(unique_shapes) > 1:
            print(f"⚠️  Size variation detected! {len(unique_shapes)} different sizes")

            # Analyze along-track dimension (first dimension) variation
            along_track_sizes = [shape[0] for shape in shapes]
            min_size = min(along_track_sizes)
            max_size = max(along_track_sizes)
            avg_size = sum(along_track_sizes) / len(along_track_sizes)

            print(f"📐 Along-track dimension:")
            print(f"   • Min: {min_size}")
            print(f"   • Max: {max_size}")
            print(f"   • Avg: {avg_size:.1f}")
            print(f"   • Variation: {max_size - min_size} pixels")
        else:
            print(f"✅ All files have consistent size: {list(unique_shapes.keys())[0]}")


def main():
    """
    Основная функция для проверки размеров swaths
    """
    print("=== AMSR-2 Swath Size Checker ===\n")

    # Get time parameters
    start_input = input("Start datetime (YYYY-MM-DD HH:MM:SS, default: 2025-01-01 00:00:00): ").strip()
    if not start_input:
        start_input = "2025-01-01 00:00:00"

    end_input = input("End datetime (YYYY-MM-DD HH:MM:SS, default: 2025-01-01 02:00:00): ").strip()
    if not end_input:
        end_input = "2025-01-01 02:00:00"

    start_datetime = dt.datetime.strptime(start_input, "%Y-%m-%d %H:%M:%S").isoformat()
    end_datetime = dt.datetime.strptime(end_input, "%Y-%m-%d %H:%M:%S").isoformat()

    # Number of files to check
    max_files_input = input("Number of files to check (default: 10): ").strip()
    max_files = int(max_files_input) if max_files_input else 10

    print(f"\n🚀 Checking swath sizes for period: {start_input} → {end_input}")
    print(f"📁 Will check first {max_files} files")

    # Download and check files
    results = download_and_check_few_files(start_datetime, end_datetime, max_files)

    # Analyze results
    if results:
        analyze_size_variations(results)

        # Summary
        print(f"\n{'=' * 60}")
        print(f"✅ SUMMARY: Checked {len(results)} files successfully")
        print(f"🔍 This will help determine if size variations are:")
        print(f"   • Original in the satellite data")
        print(f"   • Or introduced during processing")
        print(f"{'=' * 60}")
    else:
        print("\n❌ No files were successfully processed")

    # Cleanup temp directory
    try:
        if TEMP_DIR.exists():
            for file in TEMP_DIR.glob('*'):
                if file.is_file():
                    file.unlink()
            TEMP_DIR.rmdir()
            print(f"🧹 Cleaned up temp directory")
    except:
        pass


if __name__ == "__main__":
    main()