"""
generate_data.py
Creates a simulated residential-property dataset for Greater Noida West (Noida Extension).
Values are modelled on typical market ranges; replace with real listings for real use.
"""
import numpy as np
import pandas as pd

SEED = 42
N = 600
rng = np.random.default_rng(SEED)

# locality -> (base price per sqft in INR, distance to nearest metro in km)
LOCALITIES = {
    "Gaur City 1":  (5600, 4.5),
    "Gaur City 2":  (6100, 5.0),
    "Techzone-4":   (6400, 3.0),
    "Sector 16B":   (6000, 5.5),
    "Sector 10":    (5300, 7.0),
    "Sector 12":    (5000, 7.5),
    "Sector 1":     (5900, 4.0),
    "Sector 4":     (6700, 2.5),
}
FURNISHING_PREMIUM = {"Unfurnished": 0.00, "Semi-Furnished": 0.04, "Furnished": 0.08}
BHK_AREA = {1: (450, 650), 2: (850, 1150), 3: (1250, 1700), 4: (1900, 2500)}

rows = []
loc_names = list(LOCALITIES.keys())
for _ in range(N):
    loc = rng.choice(loc_names)
    base, dist = LOCALITIES[loc]
    bhk = int(rng.choice([1, 2, 3, 4], p=[0.08, 0.42, 0.40, 0.10]))
    lo, hi = BHK_AREA[bhk]
    area = int(rng.integers(lo, hi))
    floor = int(rng.integers(1, 26))
    age = int(rng.integers(0, 12))
    amen = int(rng.integers(3, 11))                       # amenities score 3-10
    furn = rng.choice(list(FURNISHING_PREMIUM), p=[0.45, 0.35, 0.20])
    metro = round(max(0.5, dist + rng.normal(0, 0.6)), 1)

    rate = base
    rate *= 1 + 0.012 * (amen - 6)                         # better amenities -> higher rate
    rate *= 1 - 0.011 * age                                # older -> lower rate
    rate *= 1 + 0.003 * min(floor, 15)                     # higher floor premium (capped)
    rate *= 1 - 0.018 * (metro - 4)                        # closer to metro -> higher
    rate *= 1 + FURNISHING_PREMIUM[furn]
    rate *= rng.normal(1, 0.04)                            # market noise
    price = area * rate
    rows.append([loc, bhk, area, floor, age, amen, furn, metro, round(price / 1e5, 2)])

df = pd.DataFrame(rows, columns=[
    "locality", "bhk", "area_sqft", "floor", "age_years",
    "amenities_score", "furnishing", "metro_km", "price_lakh"])
df.insert(0, "property_id", [f"GNW{1000 + i}" for i in range(N)])

# introduce a few missing values to demonstrate cleaning
for col, n_missing in [("metro_km", 12), ("amenities_score", 8)]:
    idx = rng.choice(df.index, n_missing, replace=False)
    df.loc[idx, col] = np.nan

df.to_csv("../data/gnw_properties.csv", index=False)
print(df.shape)
print(df.head())
