import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED     = 42
DATA_DIR = "/data"
np.random.seed(SEED)

print("1. Loading data...")
sales      = pd.read_csv(f"{DATA_DIR}/sales.csv",            parse_dates=["Date"])
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv", dtype={"Date": str})
promotions = pd.read_csv(f"{DATA_DIR}/promotions.csv",       parse_dates=["start_date","end_date"])
orders_raw = pd.read_csv(f"{DATA_DIR}/orders.csv",           parse_dates=["order_date"])

sales      = sales.sort_values("Date").reset_index(drop=True)
test_dates = pd.to_datetime(sample_sub["Date"])
print(f"   Train: {sales['Date'].min().date()} → {sales['Date'].max().date()}")
print(f"   Test : {test_dates.min().date()} → {test_dates.max().date()} ({len(test_dates)} days)")

print("2. Aggregating signals...")
orders_day = (orders_raw
              .groupby("order_date")
              .agg(n_orders=("order_id","count"))
              .reset_index()
              .rename(columns={"order_date":"Date"}))

print("3. Building promo features...")
all_dates = pd.date_range(sales["Date"].min(), test_dates.max())
promo_df  = pd.DataFrame({"Date": all_dates,
                           "is_promo": 0, "promo_disc": 0.0,
                           "promo_pct": 0, "promo_stackable": 0})
for _, r in promotions.iterrows():
    m = (promo_df["Date"] >= r["start_date"]) & (promo_df["Date"] <= r["end_date"])
    promo_df.loc[m, "is_promo"] = 1
    promo_df.loc[m, "promo_disc"] = np.maximum(promo_df.loc[m, "promo_disc"], r["discount_value"])
    if r["promo_type"] == "percentage": promo_df.loc[m, "promo_pct"] = 1
    if r["stackable_flag"] == 1:        promo_df.loc[m, "promo_stackable"] = 1

print("4. Building master dataframe...")
df = pd.DataFrame({"Date": all_dates})
df = df.merge(sales[["Date","Revenue","COGS"]], on="Date", how="left")
df = df.merge(orders_day,                       on="Date", how="left")
df = df.merge(promo_df,                         on="Date", how="left")
df = df.sort_values("Date").reset_index(drop=True)
df["n_orders"] = df["n_orders"].astype(float)

d = df["Date"]
df["dayofweek"]      = d.dt.dayofweek
df["month"]          = d.dt.month
df["quarter"]        = d.dt.quarter
df["year"]           = d.dt.year
df["day"]            = d.dt.day
df["dayofyear"]      = d.dt.dayofyear
df["is_weekend"]     = d.dt.dayofweek.isin([5,6]).astype(int)
df["is_month_end"]   = d.dt.is_month_end.astype(int)
df["is_month_start"] = d.dt.is_month_start.astype(int)
df["is_qtr_end"]     = d.dt.is_quarter_end.astype(int)
df["trend"]          = (d - d.min()).dt.days

vn_holidays = {(1,1),(4,30),(5,1),(9,2),(9,3),
               (1,27),(1,28),(1,29),(1,30),(1,31),
               (2,1),(2,2),(2,3),(2,4),(2,5)}
df["is_vn_holiday"] = df["Date"].apply(lambda x: 1 if (x.month,x.day) in vn_holidays else 0)

for k in [1,2,3,4]:
    df[f"sw{k}"] = np.sin(2*np.pi*k*d.dt.dayofweek/7)
    df[f"cw{k}"] = np.cos(2*np.pi*k*d.dt.dayofweek/7)
for k in range(1, 9):
    df[f"sy{k}"] = np.sin(2*np.pi*k*d.dt.dayofyear/365.25)
    df[f"cy{k}"] = np.cos(2*np.pi*k*d.dt.dayofyear/365.25)

REV = df["Revenue"].values.copy().astype(float)
COG = df["COGS"].values.copy().astype(float)
ORD = df["n_orders"].values.copy().astype(float)

print("5. Computing lag features...")
for lag in [7, 14, 21, 28]:
    df[f"rev_lag{lag}"] = pd.Series(REV).shift(lag).values
    df[f"ord_lag{lag}"] = pd.Series(ORD).shift(lag).values

for lag in [363, 364, 365, 366, 367, 728, 729, 730, 731]:
    df[f"rev_lag{lag}"] = pd.Series(REV).shift(lag).values
    df[f"ord_lag{lag}"] = pd.Series(ORD).shift(lag).values
    df[f"cog_lag{lag}"] = pd.Series(COG).shift(lag).values

for w in [7, 14, 30, 90, 180, 365]:
    df[f"rev_roll{w}"] = pd.Series(REV).shift(1).rolling(w, min_periods=w//2).mean().values
    df[f"ord_roll{w}"] = pd.Series(ORD).shift(1).rolling(w, min_periods=w//2).mean().values
for w in [30, 90, 365]:
    df[f"rev_rollstd{w}"] = pd.Series(REV).shift(1).rolling(w, min_periods=w//2).std().values

ratio_s = pd.Series(COG) / pd.Series(REV)
df["cogs_ratio_roll90"]  = ratio_s.shift(1).rolling(90,  min_periods=30).mean().values
df["cogs_ratio_roll365"] = ratio_s.shift(1).rolling(365, min_periods=100).mean().values

df["rev_yoy"] = pd.Series(REV) / pd.Series(REV).shift(365)
df["ord_yoy"] = pd.Series(ORD) / pd.Series(ORD).shift(365)

EXCLUDE = {"Date", "Revenue", "COGS"}
FCOLS   = [c for c in df.columns if c not in EXCLUDE]
RCOLS   = [c for c in FCOLS if any(c.startswith(p) for p in
           ["sw","cw","sy","cy","trend","month","quarter","is_","promo"])]

print("6. Training on 2019-2021 (stable period only)...")

train_m = (df["Date"] >= "2019-01-01") & (df["Date"] <= "2021-12-31") & df["Revenue"].notna()
val_m   = (df["Date"] >= "2022-01-01") & (df["Date"] <= "2022-12-31")

print(f"   Train: {train_m.sum()} rows  |  Val: {val_m.sum()} rows")

X_tr = df.loc[train_m, FCOLS]
y_tr = np.log1p(df.loc[train_m, "Revenue"])

HGB_PARAMS = dict(
    max_iter=2000,
    learning_rate=0.01,
    max_depth=6,
    min_samples_leaf=10,
    l2_regularization=0.3,
    max_bins=255,
    random_state=SEED
)

m_rev = HistGradientBoostingRegressor(**HGB_PARAMS)
m_rev.fit(X_tr, y_tr)

sc = StandardScaler()
Xr_tr = sc.fit_transform(X_tr[RCOLS].fillna(0))
ridge = Ridge(alpha=50)
ridge.fit(Xr_tr, y_tr)

def predict_revenue(X, m, r, sc, w=0.92):
    Xr = sc.transform(X[RCOLS].fillna(0))
    return np.maximum(0, w * np.expm1(m.predict(X)) + (1-w) * np.expm1(r.predict(Xr)))

X_va     = df.loc[val_m, FCOLS]
vp_rev   = predict_revenue(X_va, m_rev, ridge, sc)
ratio_va = df.loc[val_m, "cogs_ratio_roll90"].fillna(0.875).values
vp_cog   = vp_rev * ratio_va

y_rev = df.loc[val_m, "Revenue"].values
y_cog = df.loc[val_m, "COGS"].values

print("\n── Validation 2022 ──")
print(f"   [Revenue] MAE={mean_absolute_error(y_rev,vp_rev):>12,.0f}  RMSE={np.sqrt(mean_squared_error(y_rev,vp_rev)):>12,.0f}  R²={r2_score(y_rev,vp_rev):.4f}")
print(f"   [COGS   ] MAE={mean_absolute_error(y_cog,vp_cog):>12,.0f}  RMSE={np.sqrt(mean_squared_error(y_cog,vp_cog)):>12,.0f}  R²={r2_score(y_cog,vp_cog):.4f}")

print("\n7. Retraining on full 2019-2022...")
full_m = (df["Date"] >= "2019-01-01") & df["Revenue"].notna()
X_full = df.loc[full_m, FCOLS]
y_full = np.log1p(df.loc[full_m, "Revenue"])

m_final = HistGradientBoostingRegressor(**HGB_PARAMS)
m_final.fit(X_full, y_full)

sc2 = StandardScaler()
Xr_full = sc2.fit_transform(X_full[RCOLS].fillna(0))
ridge2 = Ridge(alpha=50)
ridge2.fit(Xr_full, y_full)

print(f"   Trained on {full_m.sum()} rows (2019-2022)")

print("\n8. Recursive day-by-day prediction...")
rev_arr = df["Revenue"].values.copy().astype(float)
cog_arr = df["COGS"].values.copy().astype(float)
ord_arr = df["n_orders"].values.copy().astype(float)

test_mask    = df["Date"].isin(test_dates)
test_indices = df.index[test_mask].tolist()

def build_row_features(i, df_row):
    row = df_row.copy()

    for lag in [7, 14, 21, 28]:
        j = i - lag
        row[f"rev_lag{lag}"] = rev_arr[j] if j >= 0 else np.nan
        row[f"ord_lag{lag}"] = ord_arr[j] if j >= 0 else np.nan

    for lag in [363, 364, 365, 366, 367, 728, 729, 730, 731]:
        j = i - lag
        row[f"rev_lag{lag}"] = rev_arr[j] if j >= 0 else np.nan
        row[f"ord_lag{lag}"] = ord_arr[j] if j >= 0 else np.nan
        row[f"cog_lag{lag}"] = cog_arr[j] if j >= 0 else np.nan

    for w in [7, 14, 30, 90, 180, 365]:
        s = max(0, i - w)
        wr = rev_arr[s:i]; wo = ord_arr[s:i]
        row[f"rev_roll{w}"] = np.nanmean(wr) if len(wr) > 0 else np.nan
        row[f"ord_roll{w}"] = np.nanmean(wo) if len(wo) > 0 else np.nan
    for w in [30, 90, 365]:
        s = max(0, i - w)
        wr = rev_arr[s:i]
        row[f"rev_rollstd{w}"] = np.nanstd(wr) if len(wr) > 1 else np.nan

    wc90 = cog_arr[max(0,i-90):i]; wr90 = rev_arr[max(0,i-90):i]
    with np.errstate(invalid='ignore'):
        row["cogs_ratio_roll90"] = np.nanmean(wc90/wr90) if len(wr90) > 0 else 0.875

    wc365 = cog_arr[max(0,i-365):i]; wr365 = rev_arr[max(0,i-365):i]
    with np.errstate(invalid='ignore'):
        row["cogs_ratio_roll365"] = np.nanmean(wc365/wr365) if len(wr365) > 0 else 0.875

    row["rev_yoy"] = (rev_arr[i-365]/rev_arr[i-730]) if (i>=730 and rev_arr[i-730]>0) else 1.0
    row["ord_yoy"] = (ord_arr[i-365]/ord_arr[i-730]) if (i>=730 and ord_arr[i-730]>0) else 1.0

    return row

pred_revs = []; pred_cogs = []; pred_dates = []

for idx, i in enumerate(test_indices):
    row  = build_row_features(i, df.loc[i, FCOLS].copy())
    X_p  = pd.DataFrame([row])[FCOLS]
    Xr_p = sc2.transform(X_p[RCOLS].fillna(0))

    pr = max(0, 0.92 * np.expm1(m_final.predict(X_p)[0]) +
                0.08 * np.expm1(ridge2.predict(Xr_p)[0]))

    ratio = row.get("cogs_ratio_roll90", 0.875)
    if pd.isna(ratio): ratio = 0.875
    pc = max(0, pr * ratio)

    rev_arr[i] = pr
    cog_arr[i] = pc
    j365 = i - 365
    if j365 >= 0 and not np.isnan(ord_arr[j365]) and rev_arr[j365] > 0:
        ord_arr[i] = ord_arr[j365] * (pr / rev_arr[j365])

    pred_revs.append(pr)
    pred_cogs.append(pc)
    pred_dates.append(df.loc[i, "Date"])

    if (idx + 1) % 100 == 0:
        print(f"   {idx+1}/{len(test_indices)} days predicted...", end="\r")

print(f"\n   ✅ Revenue range: {min(pred_revs):,.0f} – {max(pred_revs):,.0f}")
print(f"      COGS    range: {min(pred_cogs):,.0f} – {max(pred_cogs):,.0f}")

print("\n9. Building submission...")
rev_map = dict(zip(pred_dates, pred_revs))
cog_map = dict(zip(pred_dates, pred_cogs))

sample_raw  = pd.read_csv(f"{DATA_DIR}/sample_submission.csv", dtype={"Date": str})
date_parsed = pd.to_datetime(sample_raw["Date"])
sample_raw["Revenue"] = date_parsed.map(rev_map)
sample_raw["COGS"]    = date_parsed.map(cog_map)
submission  = sample_raw[["Date","Revenue","COGS"]]

submission.to_csv("submission.csv", index=False, encoding="utf-8")
print(f"   ✅ Saved {len(submission)} rows → submission.csv")
print(submission.head(5).to_string())