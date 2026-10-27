# Property Price Analysis and Prediction - Greater Noida West

Assignment 2 | NIET Business School | Roll No. 2527370182 | Section C

A Python / machine-learning project that cleans a residential property dataset for
Greater Noida West (Noida Extension), explores price patterns, and predicts the price
of a property (in INR lakh) from its locality, size, age, floor, amenities, furnishing
and distance to the metro.

> The dataset is simulated (seeded random generation modelled on typical market ranges).
> Replace `data/gnw_properties.csv` with real listings to use it for real decisions.

## Project structure
```
gnw_price_project/
|-- data/            gnw_properties.csv, gnw_properties_clean.csv
|-- src/
|   |-- generate_data.py   builds the dataset
|   |-- analysis.py        cleaning, EDA, models, charts
|   `-- diagrams.py        architecture and mechanism diagrams
|-- outputs/
|   |-- charts/            all figures (PNG)
|   `-- results.json       metrics and summary numbers
|-- requirements.txt
`-- README.md
```

## How to run
```
pip install -r requirements.txt
cd src
python generate_data.py
python analysis.py
python diagrams.py
```

## Results (test set, 20% hold-out)
| Model | R2 | MAE (lakh) | RMSE (lakh) | MAPE |
|---|---|---|---|---|
| Linear Regression | 0.977 | 3.17 | 4.32 | 5.24% |
| Random Forest | 0.963 | 4.11 | 5.53 | 5.88% |
