import psycopg2
import pandas as pd
import pickle
import re
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# **Database Connection Parameters**
DB_USER = 'postgres'
DB_PASSWORD = 'D@T@B@SE@13#@!'
DB_HOST = 'localhost'
DB_PORT = '5433'
DB_NAME = 'real_estate_data'

# **Connect to PostgreSQL and Load Data**
try:
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    
    query = "SELECT * FROM public.property_data"
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"✅ Data successfully loaded. Total rows: {df.shape[0]}")

except Exception as e:
    print(f"❌ Error connecting to database: {e}")
    exit()

# **Feature Selection**
all_possible_features = ["bhk", "property_area", "no_of_bathroom", "property_latitude", "property_longitude", 
                         "property_category", "property_type", "currency"]
existing_features = [col for col in all_possible_features if col in df.columns]

if not existing_features:
    raise ValueError("❌ No valid features found in the dataset.")

X = df[existing_features]
print(f"✅ Using features: {existing_features}")

# **Clean `property_price` Column**
def clean_price(price):
    if pd.isna(price):
        return None
    price = str(price).replace("₹", "").replace(",", "").strip()
    match = re.search(r"(\d+(\.\d+)?)", price)
    return float(match.group(1)) if match else None

df["property_price"] = df["property_price"].apply(clean_price)

# **Remove Outliers from `property_price`**
lower_bound, upper_bound = np.percentile(df["property_price"].dropna(), [1, 99])
df = df[(df["property_price"] >= lower_bound) & (df["property_price"] <= upper_bound)]
print("✅ Outliers removed from `property_price`.")

# **Convert `bhk` to Numeric**
X["bhk"] = X["bhk"].astype(str).str.extract(r"(\d+)").astype(float)

# **Convert `property_area` to Numeric**
def clean_area(area):
    if pd.isna(area):
        return None
    area = str(area).lower().replace(",", "").strip()
    match = re.search(r"(\d+)", area)
    if match:
        area_value = float(match.group(1))
        if "sq.m" in area:
            area_value *= 10.764  # Convert sq.m to sq.ft.
        return area_value
    return None

X["property_area"] = X["property_area"].apply(clean_area)

# **Handle Missing Values in Numeric Columns**
for col in ["bhk", "property_area", "no_of_bathroom", "property_latitude", "property_longitude"]:
    X[col] = pd.to_numeric(X[col], errors="coerce")  # Convert all to float
    X[col].fillna(X[col].median(), inplace=True)

# **One-Hot Encode Categorical Features**
categorical_features = ["property_category", "property_type", "currency"]
existing_categoricals = [col for col in categorical_features if col in df.columns]
X = pd.get_dummies(X, columns=existing_categoricals, drop_first=True)

# **Drop NaN Rows in `property_price` to Match `X` and `y`**
df.dropna(subset=["property_price"], inplace=True)
X = X.loc[df.index]  # Ensure `X` and `y` have the same rows

# **Log Transform `property_price` for Better Model Performance**
df["property_price"] = np.log1p(df["property_price"])

# **Apply Standard Scaling**
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print(f"✅ Final Matching Data Size: {X_scaled.shape[0]} rows")

# **Train-Test Split (25% Test Data)**
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df["property_price"], test_size=0.25, random_state=42)

# **Hyperparameter Tuning for XGBoost**
xgb_params = {
    "n_estimators": [300, 500],
    "max_depth": [5, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0]
}

xgb_model = XGBRegressor(random_state=42)
xgb_search = GridSearchCV(xgb_model, xgb_params, cv=5, scoring="r2", n_jobs=-1, verbose=2)
xgb_search.fit(X_train, y_train)

# **Train RandomForest Model**
rf_model = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
rf_model.fit(X_train, y_train)

# **Evaluate Both Models**
y_pred_xgb = np.expm1(xgb_search.best_estimator_.predict(X_test))  # Reverse log transform
y_pred_rf = np.expm1(rf_model.predict(X_test))  # Reverse log transform
y_test = np.expm1(y_test)  # Reverse log transform

def evaluate_model(name, y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"\n🔹 {name} Performance:")
    print(f"   ✅ Accuracy Score (R² Score): {r2:.4f}")
    print(f"   ✅ Prediction Score (Mean Absolute Error - MAE): {mae:.2f}")
    print(f"   ✅ Mean Squared Error (MSE): {mse:.2f}")
    print(f"   ✅ Root Mean Squared Error (RMSE): {rmse:.2f}")

evaluate_model("XGBoost", y_test, y_pred_xgb)
evaluate_model("RandomForest", y_test, y_pred_rf)

# **Choose the Best Model**
best_model = xgb_search.best_estimator_ if r2_score(y_test, y_pred_xgb) > r2_score(y_test, y_pred_rf) else rf_model

# **Save the Best Model**
with open("model.pkl", "wb") as file:
    pickle.dump(best_model, file)

# **Save the Scaler**
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

print("✅ Best Model trained and saved successfully!")
print("🎉 All steps completed successfully!")

