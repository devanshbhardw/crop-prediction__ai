"""
Generate synthetic crop dataset for development/testing.
Saves two files:
 - data/sample_crop_data.csv  (small, ~100 rows)
 - data/synthetic_crop_data.csv (larger, default 2000 rows)

Run:
    python scripts/generate_synthetic_data.py --rows 200
"""
import numpy as np
import pandas as pd
import argparse
import os

os.makedirs('data', exist_ok=True)

CROPS = ['maize', 'wheat', 'rice', 'soybean']


def generate_row(rng):
    # environmental and soil variables with plausible ranges
    temperature = rng.normal(loc=25, scale=5)          # °C
    humidity = rng.uniform(30, 95)                     # %
    ph = rng.normal(6.5, 0.6)                          # soil pH
    rainfall = max(0, rng.normal(100, 50))             # mm (seasonal)
    nitrogen = rng.uniform(0.1, 0.6)                   # % (relative nutrient level)
    phosphorus = rng.uniform(0.05, 0.4)
    potassium = rng.uniform(0.1, 0.6)
    crop = rng.choice(CROPS)

    # simple yield function (kg/ha scaled) to simulate relationship
    # different crops have different base yields
    base = {
        'maize': 6000,
        'wheat': 3500,
        'rice': 5000,
        'soybean': 2500
    }[crop]

    # create an interaction-based synthetic yield
    yield_mean = (
        base
        * (1 + 0.02 * (temperature - 25))
        * (1 + 0.01 * (humidity - 60))
        * (1 + 0.1 * (ph - 6.5))
        * (1 + 0.5 * (nitrogen))
        * (1 + 0.4 * (phosphorus))
        * (1 + 0.3 * (potassium))
        * (1 + 0.0005 * rainfall)
    )

    # add noise
    noise = rng.normal(0, yield_mean * 0.12)
    yield_val = max(0, yield_mean + noise)

    return {
        'temperature': round(float(temperature), 2),
        'humidity': round(float(humidity), 2),
        'ph': round(float(ph), 2),
        'rainfall': round(float(rainfall), 2),
        'nitrogen': round(float(nitrogen), 3),
        'phosphorus': round(float(phosphorus), 3),
        'potassium': round(float(potassium), 3),
        'crop': crop,
        'yield': round(float(yield_val), 2)
    }


def generate_dataset(n_rows, seed=42):
    rng = np.random.default_rng(seed)
    rows = [generate_row(rng) for _ in range(n_rows)]
    return pd.DataFrame(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=200, help='Number of rows for the larger synthetic dataset')
    parser.add_argument('--sample-rows', type=int, default=100, help='Number of rows for the sample CSV')
    args = parser.parse_args()

    sample_df = generate_dataset(args.sample_rows, seed=1)
    sample_path = os.path.join('data', 'sample_crop_data.csv')
    sample_df.to_csv(sample_path, index=False)
    print(f'Wrote sample CSV: {sample_path} ({len(sample_df)} rows)')

    big_df = generate_dataset(args.rows, seed=42)
    big_path = os.path.join('data', 'synthetic_crop_data.csv')
    big_df.to_csv(big_path, index=False)
    print(f'Wrote synthetic CSV: {big_path} ({len(big_df)} rows)')