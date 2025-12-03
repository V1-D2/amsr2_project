# AMSR-2 Satellite Data Collection and Processing System

**Automated pipeline for downloading, processing, and archiving AMSR-2 brightness temperature observations from JAXA G-Portal**

---

## Overview

This repository provides a complete system for collecting and processing brightness temperature data from the AMSR-2 (Advanced Microwave Scanning Radiometer 2) instrument aboard the GCOM-W satellite. The system automates the entire workflow from querying JAXA's G-Portal database to extracting temperature observations and storing them in optimized compressed formats.

**Key capabilities:**
- Automated data retrieval from G-Portal API with parallel downloading
- HDF5-to-NPZ conversion with selective variable extraction
- Incremental backup system for long-running collections
- Multiple storage formats optimized for different use cases
- Orbit type detection (ascending/descending passes)
- Comprehensive error handling and retry logic

**Dataset characteristics:**
- Instrument: AMSR-2 aboard GCOM-W satellite
- Variable: Brightness Temperature at 36.5 GHz, Horizontal Polarization
- Temporal coverage: January 2012 - January 2025 (13+ years)
- Swath width: ~1,450 km
- Spatial resolution: ~10 km (nadir) to ~15 km (swath edges)
- Daily observations: ~35 swaths per day

---

## Architecture

### Data Collection Pipeline

The system implements a three-stage pipeline: **Search → Download → Process → Archive**

**1. G-Portal Query**
```python
gportal.search(
    dataset_ids=[DS_L1B_TB],
    start_time=start_datetime,
    end_time=end_datetime
)
```
Queries JAXA's G-Portal API for L1B brightness temperature products matching the specified time range. Returns metadata for available granules.

**2. Parallel Download with Immediate Processing**
```python
download → extract_swath_data → delete_h5_file
```
Downloads HDF5 files to temporary storage, immediately extracts target variables, and deletes the source file to minimize disk usage. Operates in parallel using ThreadPoolExecutor.

**3. Batch Compression and Storage**
```python
swath_array → npz_compressed → disk
```
Accumulates processed swaths in memory and periodically saves compressed NPZ archives with configurable batch sizes.

### Thread-Safe Progress Tracking

```python
class ThreadSafeProgress:
    def __init__(self):
        self.lock = threading.Lock()
        self.processed = 0
        self.total_files = 0
```

Implements mutex-protected counters for tracking processing progress across multiple concurrent download threads.

---

## Data Extraction

### Temperature Variable Extraction

The system targets the 36.5 GHz horizontal polarization channel:

```python
var_name = "Brightness Temperature (36.5GHz,H)"
raw_temp = h5[var_name][:]
```

**Storage optimization:**
- Temperatures stored as **int16** (original HDF5 format)
- Scale factor stored separately in metadata
- Zero values indicate missing/invalid data
- No conversion to float until analysis time

**Physical unit conversion:**
```python
temperature_kelvin = raw_temp.astype(float) * scale_factor
valid_mask = raw_temp != 0
```

### Coordinate Handling

Spatial coordinates are **optionally** included based on processor version:

**Coordinate derivation for 36.5 GHz:**
```python
# 36.5 GHz coordinates derived from 89 GHz observations
lat_89 = h5["Latitude of Observation Point for 89A"][:]
lon_89 = h5["Longitude of Observation Point for 89A"][:]

# Downsample: 89 GHz → 36.5 GHz (every 2nd pixel)
lat_36 = lat_89[:, ::2]
lon_36 = lon_89[:, ::2]
```

The 89 GHz channel has twice the spatial resolution (486 pixels vs. 243 pixels across-track), requiring downsampling to match 36.5 GHz geometry.

### Orbit Type Detection

```python
identifier = h5_path.stem  # Filename without extension
parts = identifier.split("_")
ad_flag = parts[2][-1]  # Last character of 3rd segment
orbit_type = "A" if ad_flag == "A" else "D" if ad_flag == "D" else "U"
```

JAXA filenames encode orbit direction: **A** (ascending, south→north), **D** (descending, north→south), **U** (unknown).

---

## Storage Formats

### Format 1: Temperature-Only (Maximum Compression)

**File:** `amsr2_processor.py`

```python
swath_dict = {
    'temperature': raw_temp,      # int16 array (original format)
    'metadata': {
        'orbit_type': str,         # 'A', 'D', or 'U'
        'scale_factor': float,     # Multiply raw_temp by this
        'temp_range': (int, int),  # Min/max valid values
        'shape': tuple             # Array dimensions
    }
}
```

**Compression ratio:** ~70-80% reduction vs. raw HDF5
**Use case:** Training data for machine learning models (coordinates not needed)

### Format 2: Full Swath Data

**File:** `amsr2_processor_with_changes.py`

```python
swath_dict = {
    'temperature': raw_temp,       # int16 array
    'latitude': lat_36,            # float32 array
    'longitude': lon_36,           # float32 array
    'metadata': {
        'orbit_type': str,
        'scale_factor': float,
        'temp_range': (int, int),
        'lat_range': (float, float),
        'lon_range': (float, float)
    }
}
```

**Compression ratio:** ~60-70% reduction vs. raw HDF5
**Use case:** Geospatial analysis, mapping applications

### NPZ File Structure

```python
with np.load('dataset.npz', allow_pickle=True) as data:
    swath_array = data['swath_array']  # Array of swath dictionaries
    period = data['period']             # ISO time range string
```

**Technical details:**
- Container format: NumPy NPZ (ZIP archive of .npy files)
- Compression: zlib (level 6, automatic)
- Dictionary serialization: Python pickle protocol (requires `allow_pickle=True`)

---

## Incremental Backup System

**File:** `amsr2_processor.py` (primary version)

### Rationale

G-Portal servers occasionally hang or timeout during multi-hour downloads. Without incremental saves, hours of processing can be lost.

### Implementation

```python
SAVE_INTERVAL = 100  # Files per backup
BACKUP_FILENAME = "AMSR2_temp_backup_CURRENT.npz"

if files_processed % SAVE_INTERVAL == 0:
    save_backup(all_swaths, BACKUP_DIR, files_processed, ...)
```

**Backup contents:**
```python
{
    'swath_array': [...],           # All processed swaths
    'backup_info': {
        'files_processed': int,
        'total_swaths': int,
        'timestamp': str
    }
}
```

**Recovery procedure:**
1. User interrupts process (Ctrl+C) or server hangs
2. Backup file contains all data processed up to last interval
3. Rename backup to standard naming convention
4. Resume collection from next time range

---

## Parallel Download Configuration

### Threading Parameters

```python
max_workers: int = 4  # Concurrent download threads
```

**Trade-offs:**

| Threads | Speed | Data Retention | Use Case |
|---------|-------|----------------|----------|
| 1 | 1× (baseline) | 99.97% | High reliability needed |
| 4 | 4× | 99% | Balanced (recommended) |
| 8 | 8× | 95-97% | Speed priority |
| 16 | 12-14× | 83-85% | Maximum speed, acceptable loss |

**Why data loss occurs:**
G-Portal's server capacity limits simultaneous requests. Exceeding this threshold causes:
- Connection timeouts
- Incomplete downloads
- HTTP 503 errors (service unavailable)

### Error Handling and Retry Logic

**File:** `amsr2_processor_slow.py` (network-robust version)

```python
MAX_RETRIES = 3
RETRY_DELAY_BASE = 5  # seconds
BACKOFF_MULTIPLIER = 2

for attempt in range(MAX_RETRIES):
    delay = RETRY_DELAY_BASE * (BACKOFF_MULTIPLIER ** attempt) + random.uniform(0, 2)
    time.sleep(delay)
    # Attempt download...
```

**Exponential backoff sequence:** 5s → 10s → 20s (with jitter)

**Consecutive failure threshold:**
```python
max_consecutive_failures = 10
if consecutive_failures >= max_consecutive_failures:
    # Abort: likely server-side issue
```

---

## File Organization

### Yearly Archives

**Naming convention:**
```
AMSR2_temp_only_YYYYMMDD_HHMMSS_to_YYYYMMDD_HHMMSS.npz
```

**Example:**
```
AMSR2_temp_only_20160101_000000_to_20170101_000000.npz
```

**Statistics per yearly file:**
- Size: ~7 GB
- Swaths: ~12,000
- Coverage: 365 days × 35 swaths/day = ~12,775 expected
- Actual: ~12,000 (due to ~1% download loss)

### Partitioned Archives

**Directory:** `new_data_all/`

**Naming convention:**
```
AMSR2_temp_only_YYYYMMDD_HHMMSS_to_YYYYMMDD_HHMMSS_part_XofY.npz
```

**Partition configuration:**
```python
CHUNK_SIZE = 500  # Swaths per file
```

**Benefits:**
- Reduced memory footprint during loading
- Faster download/transfer times
- Easier distributed processing
- Each file: ~300 MB (vs. ~7 GB for yearly files)

**Partition generation:**
```python
# File: split_amsr_files.py
for i in range(0, len(swath_array), CHUNK_SIZE):
    chunk = swath_array[i:i + CHUNK_SIZE]
    part_number = i // CHUNK_SIZE + 1
    save_partition(chunk, part_number, total_parts)
```

---

## Data Characteristics

### Array Dimensions

**Typical swath shape:**
```
temperature.shape = (n_scans, n_pixels)
                  ≈ (2000-2060, 243)
```

**Dimension semantics:**
- **n_scans (along-track):** Number of scan lines in orbital segment
- **n_pixels (across-track):** Fixed at 243 pixels for 36.5 GHz

**Variation source:** Swaths cover variable orbital segments. Longer segments → more scans.

### Invalid Data Handling

**Zero-value convention:**
```python
valid_mask = temperature != 0
```

**Sources of invalid data:**
- Pixels outside swath coverage
- Land contamination (AMSR-2 optimized for ocean/ice)
- Instrument calibration issues
- Quality control failures

**Processing recommendation:**
```python
temperature[temperature == 0] = np.nan  # For visualization/analysis
```

### Temperature Range

**Typical brightness temperature values:**
- Ocean surface: 80-150 K (cold, low emissivity)
- Sea ice: 200-270 K (intermediate)
- Land/atmosphere: Variable, 200-300 K
- Physical range: 0-350 K (instrument limits)

**Scale factor typical value:** 0.01 (converts int16 → Kelvin)

---

## System Requirements

### Python Dependencies

```bash
pip install numpy>=1.20.0 h5py>=3.0.0 tqdm>=4.60.0 gportal
```

**Library functions:**
- **numpy:** Array operations, NPZ file I/O
- **h5py:** HDF5 file reading
- **tqdm:** Progress bars for long-running operations
- **gportal:** JAXA G-Portal API client (authentication + download)

### Computational Resources

**Disk space:**
- **Temporary:** 100-500 MB per HDF5 file (deleted immediately after processing)
- **Output:** 
  - Yearly files: ~7 GB each
  - Partitioned files: ~300 MB each
  - Total dataset (2012-2025): ~70 GB compressed

**Memory:**
- Minimum: 4 GB RAM
- Recommended: 8+ GB (for higher thread counts)
- Per-swath memory: ~2-5 MB

**Network:**
- Stable connection required (multi-hour downloads)
- Bandwidth: G-Portal typically ~10-50 Mbps

---

## Configuration

**File:** `config.py`

```python
import pathlib

BASE_DIR = pathlib.Path("./data")      # Output directory
TEMP_DIR = pathlib.Path("./temp")      # Temporary HDF5 storage

# G-Portal authentication (free registration at JAXA)
GPORTAL_USERNAME = "your_username"
GPORTAL_PASSWORD = "your_password"
```

**G-Portal registration:** https://gportal.jaxa.jp/gpr/

---

## Usage

### Basic Collection

```bash
python amsr2_processor.py
```

**Interactive prompts:**
```
Start datetime (YYYY-MM-DD HH:MM:SS): 2024-01-01 00:00:00
End datetime (YYYY-MM-DD HH:MM:SS): 2024-02-01 00:00:00
Number of threads (1-16, default 4): 8
Save interval in files (default 100): 500
```

### Long-Running Sessions with Tmux

**Recommended for collections >1 hour:**

```bash
# Create persistent session
tmux new-session -t amsr2_download

# Inside tmux: start processor
python amsr2_processor.py

# Detach: Ctrl+b, then d
# Reattach later: tmux attach -t amsr2_download
```

**Why tmux?** SSH disconnections won't kill the process. Session persists indefinitely on server.

### Network-Robust Collection

**For unstable connections:**

```bash
python amsr2_processor_slow.py
```

**Features:**
- Sequential processing (no parallel threads)
- Automatic retry with exponential backoff
- Consecutive failure detection
- Reduced server load

**Trade-off:** Slower throughput, higher reliability

---

## Data Access

### Loading NPZ Files

```python
import numpy as np

# Load file
with np.load('AMSR2_temp_only_20240101_000000_to_20250101_000000.npz', 
             allow_pickle=True) as data:
    swath_array = data['swath_array']
    period = data['period']

# Access individual swath
swath = swath_array[0]
raw_temp = swath['temperature']
metadata = swath['metadata']

# Convert to physical units
scale_factor = metadata['scale_factor']
temperature_kelvin = raw_temp.astype(float) * scale_factor
temperature_kelvin[raw_temp == 0] = np.nan  # Mask invalid data
```

### Coordinate-Enabled Files

```python
# For files created with amsr2_processor_with_changes.py
swath = swath_array[0]
temperature = swath['temperature']
latitude = swath['latitude']
longitude = swath['longitude']

# Create georeferenced array
valid_mask = temperature != 0
lat_valid = latitude[valid_mask]
lon_valid = longitude[valid_mask]
temp_valid = temperature[valid_mask] * metadata['scale_factor']
```

### Filtering by Orbit Type

```python
# Separate ascending/descending passes
ascending = [s for s in swath_array if s['metadata']['orbit_type'] == 'A']
descending = [s for s in swath_array if s['metadata']['orbit_type'] == 'D']

print(f"Ascending: {len(ascending)}, Descending: {len(descending)}")
```

**Use case:** Diurnal cycle studies (ascending/descending occur at different local times)

---

## Data Verification

**File:** `testOutput.py`

```bash
python testOutput.py
```

**Functionality:**
- Validates NPZ file structure
- Checks swath integrity
- Calculates success rate
- Visualizes sample swaths

**Output:**
```
Found 12000 swaths in backup
Valid swaths: 11950, Damaged swaths: 50
Success rate: 99.6%
Backup file is in GOOD condition
```

---

## Advanced Usage

### Custom Time Ranges

```python
# Collect specific month
start = "2024-03-01 00:00:00"
end = "2024-04-01 00:00:00"
```

### Custom Save Intervals

**For different time ranges:**
- 1 year: `save_interval=2500` (every ~10 days)
- 6 months: `save_interval=1250`
- 1 month: `save_interval=200`

### File Partitioning

**Generate custom partitions:**

```bash
cd split_amsr_files/
python split_amsr_files.py
```

**Configuration variables:**
```python
DATA_DIR = "path/to/yearly/files"
NEW_DATA_DIR = "path/to/output/partitions"
CHUNK_SIZE = 500  # Swaths per partition
```

---

## Error Recovery

### Interrupted Downloads

**Scenario:** Process stops mid-collection

**Example of recovery steps:**
```bash
# 1. Locate backup file
ls /data/incremental_backups/

# 2. Rename with proper timestamp
mv AMSR2_temp_backup_CURRENT.npz \
   AMSR2_temp_only_20240101_000000_to_20240115_120000.npz

# 3. Move to final location
mv AMSR2_temp_only_*.npz /data
```

### Server Timeouts

**Symptoms:**
- Download hangs indefinitely
- No progress updates for >30 minutes

**Action:** Use Ctrl+C to stop, then:
```bash
# Switch to slow/robust processor
python amsr2_processor_slow.py

# Or reduce thread count
# Use 1-2 threads for maximum reliability
```

---

## Known Issues and Limitations

### 1. Coordinate Availability

**Temperature-only format** (`amsr2_processor.py`) excludes geographic coordinates. For geolocation:
- Use full swath format (`amsr2_processor_with_changes.py`), or
- Regenerate coordinates from AMSR-2 orbital parameters (complex)

### 2. Data Gaps

**~1% of observations lost** during collection due to:
- G-Portal server capacity limits
- Network timeouts
- Corrupted downloads (automatically filtered)

**Impact:** Minimal for statistical studies. For time-critical applications, consider sequential downloading.

### 3. Array Dimension Variability

**Along-track dimension varies:** 2000-2060 pixels

**Source:** Variable orbital segment lengths in HDF5 granules

**Handling:**
```python
# Crop to consistent size
target_height = 2000
if temp.shape[0] > target_height:
    temp = temp[:target_height, :]
```

### 4. G-Portal Server Limitations

**Observed behavior:**
- Parallel requests >8 threads → increased failures
- Long sessions (>6 hours) → server instability
- Peak usage times → slower downloads

**Recommendation:** Batch large collections into monthly segments

---

## Repository Structure

```
amsr2_project/
├── amsr2_processor.py                    # Main: temperature-only, incremental backup
├── amsr2_processor_with_changes.py       # Full swath with coordinates
├── amsr2_processor_slow.py               # Network-robust with retry logic
├── amsr2_processor_version_3.py          # Format selection option
├── config.py                             # Configuration (paths, credentials)
├── testOutput.py                         # Data verification utility
├── Checking the difference in size.py    # Diagnostic: analyze HDF5 variations
├── requirements.txt                      # Python dependencies
└── README.md
```

**Recommended processors by use case:**
- **ML training:** `amsr2_processor.py` (smallest files, incremental backup)
- **Geospatial analysis:** `amsr2_processor_with_changes.py` (includes coordinates)
- **Unreliable networks:** `amsr2_processor_slow.py` (retry logic)

---

## Performance Optimization

### Memory Management

```python
# Sequential processing for large datasets
for swath in swath_array:
    process(swath)
    del swath  # Explicit cleanup
    
gc.collect()  # Force garbage collection
```

### Disk I/O

**Write buffering:**
```python
# Accumulate in memory, write periodically
buffer = []
for item in data_stream:
    buffer.append(item)
    if len(buffer) >= BATCH_SIZE:
        write_to_disk(buffer)
        buffer.clear()
```

### Network Optimization

**Recommended thread counts by bandwidth:**
- <10 Mbps: 1-2 threads
- 10-50 Mbps: 4 threads
- 50-100 Mbps: 8 threads
- >100 Mbps: 12-16 threads (monitor loss rate)

---

## Citation

If you use this data collection system in your research, please cite:

```bibtex
@software{amsr2_processor_2025,
  title={AMSR-2 Data Collection and Processing System},
  author={Volodymyr Didur},
  year={2025},
  url={https://github.com/yourusername/amsr2_project}
}
```

**Related datasets:**
- AMSR-2 Level 1B Product: [JAXA G-Portal](https://gportal.jaxa.jp/)
- Dataset DOI: [Contact JAXA for official citation]

---

## License

This project is released under the MIT License.

**Data licensing:** AMSR-2 data is provided by JAXA under their [data policy](https://gportal.jaxa.jp/gpr/information/about). Users must register for G-Portal access and acknowledge JAXA as the data source in publications.

---

## Acknowledgments

- **JAXA** for providing AMSR-2 data through G-Portal
- **GCOM-W/AMSR-2 Project** for instrument operation and data distribution
- **gportal** Python library maintainers for API access tools

---

## Contact

For questions about the data collection system or to report issues, please open an issue on GitHub or contact the repository maintainer.
