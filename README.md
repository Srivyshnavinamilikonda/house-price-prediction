# House Price Prediction with Linear Regression

Predicting house sale prices on the Ames, Iowa dataset (Kaggle) using
Linear, Ridge, and Lasso regression.

## Results

| Model | CV RMSE | Test R2 | Test MAE |
|---|---|---|---|
| Linear (baseline) | 0.1302 | 0.8821 | $15,529 |
| Ridge (alpha=10) | 0.1116 | 0.9104 | $14,331 |
| Lasso (alpha=0.0003) | 0.1109 | 0.9177 | $14,057 |

## Approach
1. EDA: price is right-skewed, so I trained on log(SalePrice)
2. Removed 2 outliers (GrLivArea > 4000 with low price)
3. Filled missing values based on data_description.txt
   (NaN often means "feature absent")
4. Preprocessing pipeline: StandardScaler + OneHotEncoder
5. Compared Linear, Ridge, Lasso with 5-fold CV and GridSearchCV
6. Checked residuals to validate model assumptions

## Key findings
- Lasso kept 142 of 312 features and dropped 170, so most one-hot columns
  added no predictive value.
- Above-ground living area (GrLivArea) is the strongest numeric driver:
  +1 standard deviation raises price by about 13%.
- Location matters a lot: StoneBr and Crawford are premium neighborhoods
  (about +11-12%), while Meadow Village is a discount area (about -8%).
- Commercial zoning (MSZoning_C) lowers price by about 21%.
- Newer and higher-quality houses (YearBuilt, OverallQual) sell for more,
  about 6% per standard deviation.
- Caveat: coefficients for rare categories (e.g. Heating_Grav) are based on
  very few houses and are less reliable.

## Tech stack
Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, joblib

## How to run
1. Download train.csv from Kaggle into a `data/` folder
2. `pip install -r requirements.txt`
3. Open `house_price.ipynb` and run all cells

## Limitations and future work
- Linear models can't capture complex interactions
- Next: try tree-based models (Random Forest, XGBoost), and add a Streamlit app