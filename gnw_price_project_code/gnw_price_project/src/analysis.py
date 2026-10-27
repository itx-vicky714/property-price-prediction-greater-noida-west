"""
analysis.py
Cleaning -> EDA -> Model training -> Evaluation -> Charts for the
Greater Noida West Property Price Analysis & Prediction project.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

OUT = "../outputs/charts/"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
NAVY, TEAL, ORANGE, GREY = "#1F3A5F", "#2A9D8F", "#E76F51", "#8D99AE"

# ---------- 1. Load & clean ----------
df = pd.read_csv("../data/gnw_properties.csv")
raw_missing = df.isna().sum().to_dict()
df["metro_km"] = df["metro_km"].fillna(df["metro_km"].median())
df["amenities_score"] = df["amenities_score"].fillna(df["amenities_score"].median())
df["price_per_sqft"] = df["price_lakh"] * 1e5 / df["area_sqft"]
df.to_csv("../data/gnw_properties_clean.csv", index=False)

# ---------- 2. EDA ----------
summary = df[["area_sqft", "price_lakh", "price_per_sqft", "age_years", "metro_km"]].describe().round(2)
loc_stats = df.groupby("locality").agg(
    listings=("property_id", "count"),
    avg_price_lakh=("price_lakh", "mean"),
    avg_rate=("price_per_sqft", "mean")).round(1).sort_values("avg_rate", ascending=False)
bhk_stats = df.groupby("bhk").agg(
    listings=("property_id", "count"),
    avg_area=("area_sqft", "mean"),
    avg_price_lakh=("price_lakh", "mean")).round(1)
furn_stats = df.groupby("furnishing")["price_per_sqft"].mean().round(0)
corr = df[["area_sqft", "bhk", "floor", "age_years", "amenities_score", "metro_km",
           "price_per_sqft", "price_lakh"]].corr().round(2)

# Chart 1: price distribution
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.hist(df["price_lakh"], bins=25, color=NAVY, edgecolor="white")
ax.axvline(df["price_lakh"].mean(), color=ORANGE, ls="--", label=f"Mean = {df['price_lakh'].mean():.1f} L")
ax.axvline(df["price_lakh"].median(), color=TEAL, ls="--", label=f"Median = {df['price_lakh'].median():.1f} L")
ax.set_xlabel("Price (INR Lakh)"); ax.set_ylabel("Number of properties")
ax.set_title("Fig 4.1  Distribution of Property Prices"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT + "c1_price_dist.png", dpi=200); plt.close(fig)

# Chart 2: avg rate per sqft by locality
fig, ax = plt.subplots(figsize=(6.4, 3.6))
s = loc_stats["avg_rate"].sort_values()
ax.barh(s.index, s.values, color=TEAL)
for i, v in enumerate(s.values):
    ax.text(v + 20, i, f"{v:,.0f}", va="center", fontsize=8)
ax.set_xlabel("Average price per sq ft (INR)"); ax.set_xlim(0, s.max() * 1.15)
ax.set_title("Fig 4.2  Average Rate per sq ft by Locality")
fig.tight_layout(); fig.savefig(OUT + "c2_locality_rate.png", dpi=200); plt.close(fig)

# Chart 3: area vs price scatter coloured by BHK
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for b, c in zip([1, 2, 3, 4], [GREY, TEAL, NAVY, ORANGE]):
    d = df[df.bhk == b]
    ax.scatter(d.area_sqft, d.price_lakh, s=14, alpha=0.7, color=c, label=f"{b} BHK")
m, c0 = np.polyfit(df.area_sqft, df.price_lakh, 1)
xs = np.linspace(df.area_sqft.min(), df.area_sqft.max(), 50)
ax.plot(xs, m * xs + c0, color="black", lw=1.2)
ax.set_xlabel("Built-up area (sq ft)"); ax.set_ylabel("Price (INR Lakh)")
ax.set_title("Fig 4.3  Area vs Price (by BHK)"); ax.legend(frameon=False, ncol=4)
fig.tight_layout(); fig.savefig(OUT + "c3_area_price.png", dpi=200); plt.close(fig)

# Chart 4: correlation heatmap
fig, ax = plt.subplots(figsize=(6.4, 4.6))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=40, ha="right", fontsize=8)
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index, fontsize=8)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, fraction=0.04)
ax.set_title("Fig 4.4  Correlation Matrix")
fig.tight_layout(); fig.savefig(OUT + "c4_corr.png", dpi=200); plt.close(fig)

# Chart 5: age and metro effect on rate
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.2))
ag = df.groupby("age_years")["price_per_sqft"].mean()
axs[0].plot(ag.index, ag.values, marker="o", color=ORANGE)
axs[0].set_xlabel("Age of property (years)"); axs[0].set_ylabel("Avg rate / sq ft (INR)")
axs[0].set_title("Age vs Rate", fontsize=10)
df["metro_band"] = pd.cut(df.metro_km, [0, 3, 5, 7, 20], labels=["<3", "3-5", "5-7", ">7"])
mb = df.groupby("metro_band", observed=True)["price_per_sqft"].mean()
axs[1].bar(mb.index.astype(str), mb.values, color=NAVY)
axs[1].set_xlabel("Distance to metro (km)"); axs[1].set_title("Metro Distance vs Rate", fontsize=10)
axs[1].set_ylim(mb.min() * 0.85, mb.max() * 1.05)
fig.suptitle("Fig 4.5  Effect of Age and Metro Distance", fontsize=10)
fig.tight_layout(); fig.savefig(OUT + "c5_age_metro.png", dpi=200); plt.close(fig)

# ---------- 3. Modelling ----------
features = ["locality", "bhk", "area_sqft", "floor", "age_years",
            "amenities_score", "furnishing", "metro_km"]
X = pd.get_dummies(df[features], columns=["locality", "furnishing"], drop_first=True)
y = df["price_lakh"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
}
results, preds = {}, {}
for name, mdl in models.items():
    mdl.fit(X_tr, y_tr)
    p = mdl.predict(X_te)
    preds[name] = p
    results[name] = {
        "R2": round(r2_score(y_te, p), 4),
        "MAE": round(mean_absolute_error(y_te, p), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_te, p))), 2),
        "MAPE": round(float(np.mean(np.abs((y_te - p) / y_te)) * 100), 2),
    }

best = max(results, key=lambda k: results[k]["R2"])

# Chart 6: actual vs predicted for best model
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.scatter(y_te, preds[best], s=16, alpha=0.7, color=TEAL)
lim = [min(y_te.min(), preds[best].min()), max(y_te.max(), preds[best].max())]
ax.plot(lim, lim, color=ORANGE, ls="--", label="Perfect prediction")
ax.set_xlabel("Actual price (Lakh)"); ax.set_ylabel("Predicted price (Lakh)")
ax.set_title(f"Fig 5.1  Actual vs Predicted ({best})", fontsize=10); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT + "c6_actual_pred.png", dpi=200); plt.close(fig)

# Chart 7: feature importance (Random Forest, grouped for readability)
rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns)
grouped = {
    "Area (sq ft)": imp["area_sqft"], "BHK": imp["bhk"],
    "Locality": imp[[c for c in imp.index if c.startswith("locality_")]].sum(),
    "Metro distance": imp["metro_km"], "Property age": imp["age_years"],
    "Amenities": imp["amenities_score"], "Floor": imp["floor"],
    "Furnishing": imp[[c for c in imp.index if c.startswith("furnishing_")]].sum(),
}
gi = pd.Series(grouped).sort_values()
fig, ax = plt.subplots(figsize=(6.0, 3.4))
ax.barh(gi.index, gi.values * 100, color=NAVY)
for i, v in enumerate(gi.values * 100):
    ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
ax.set_xlabel("Importance (%)"); ax.set_xlim(0, gi.max() * 100 * 1.15)
ax.set_title("Fig 5.2  Feature Importance (Random Forest)")
fig.tight_layout(); fig.savefig(OUT + "c7_importance.png", dpi=200); plt.close(fig)

# Chart 8: residual histogram
res = y_te - preds[best]
fig, ax = plt.subplots(figsize=(6.0, 3.0))
ax.hist(res, bins=22, color=GREY, edgecolor="white")
ax.axvline(0, color=ORANGE, ls="--")
ax.set_xlabel("Residual (Actual - Predicted, Lakh)"); ax.set_ylabel("Count")
ax.set_title(f"Fig 5.3  Residual Distribution ({best})")
fig.tight_layout(); fig.savefig(OUT + "c8_residuals.png", dpi=200); plt.close(fig)

# ---------- 4. Sample prediction ----------
def predict(locality, bhk, area, floor, age, amen, furn, metro):
    row = pd.DataFrame([dict(locality=locality, bhk=bhk, area_sqft=area, floor=floor,
                             age_years=age, amenities_score=amen, furnishing=furn, metro_km=metro)])
    row = pd.get_dummies(row, columns=["locality", "furnishing"])
    row = row.reindex(columns=X.columns, fill_value=0)
    return float(models[best].predict(row)[0])

sample_inputs = [
    ("Techzone-4", 3, 1450, 12, 2, 8, "Semi-Furnished", 3.0),
    ("Sector 12", 2, 1000, 8, 5, 6, "Unfurnished", 7.5),
    ("Sector 4", 2, 1100, 15, 1, 9, "Furnished", 2.5),
]
samples = [list(s) + [round(predict(*s), 1)] for s in sample_inputs]

# ---------- 5. Save everything the PDF needs ----------
out = {
    "n_rows": int(len(df)), "raw_missing": raw_missing, "best": best, "results": results,
    "summary": summary.reset_index().values.tolist(), "summary_cols": ["stat"] + list(summary.columns),
    "loc_stats": loc_stats.reset_index().values.tolist(),
    "bhk_stats": bhk_stats.reset_index().values.tolist(),
    "furn_stats": furn_stats.to_dict(),
    "corr_price": corr["price_lakh"].to_dict(),
    "corr_rate": corr["price_per_sqft"].to_dict(),
    "importance": {k: round(v * 100, 1) for k, v in gi.sort_values(ascending=False).items()},
    "age_rate": {int(k): round(v) for k, v in ag.items()},
    "metro_rate": {str(k): round(v) for k, v in mb.items()},
    "samples": samples,
    "mean_price": round(df.price_lakh.mean(), 1), "median_price": round(df.price_lakh.median(), 1),
    "mean_rate": round(df.price_per_sqft.mean()),
}
json.dump(out, open("../outputs/results.json", "w"), indent=2, default=str)
print(json.dumps(out["results"], indent=2))
print("best:", best)
