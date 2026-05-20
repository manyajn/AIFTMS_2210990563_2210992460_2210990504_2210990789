import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
 
def get_float(prompt, example):
    while True:
        try:
            return float(input(f"  {prompt} (e.g. {example}): "))
        except ValueError:
            print("  ⚠️  Invalid input. Please enter a number.\n")
 
def get_int(prompt, min_val=1):
    while True:
        try:
            val = int(input(f"  {prompt}: "))
            if val >= min_val:
                return val
            print(f"  ⚠️  Please enter a value >= {min_val}.\n")
        except ValueError:
            print("  ⚠️  Invalid input. Please enter a whole number.\n")
 
# ─────────────────────────────────────────────
#  STEP 1 — Collect TRAINING data from user
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("   LightGBM REVENUE FORECASTING — MANUAL INPUT MODE")
print("=" * 55)
 
print("\n📋 STEP 1: Enter TRAINING data")
print("   (Historical data used to train the model)\n")
 
n = get_int("How many training samples do you want to enter? (min 5)", min_val=5)
 
records = []
for i in range(n):
    print(f"\n  --- Sample {i+1} of {n} ---")
    lag_1w          = get_float("  Last week revenue         (lag_1w)",          "300")
    lag_4w          = get_float("  Revenue 4 weeks ago       (lag_4w)",          "260")
    seasonality_idx = get_float("  Seasonality index         (seasonality_idx)", "1.25")
    pmi_index       = get_float("  PMI index                 (pmi_index)",       "54")
    revenue_actuals = get_float("  Actual revenue (target)   (revenue_actuals)", "320")
    records.append({
        'lag_1w': lag_1w,
        'lag_4w': lag_4w,
        'seasonality_idx': seasonality_idx,
        'pmi_index': pmi_index,
        'revenue_actuals': revenue_actuals
    })
 
data = pd.DataFrame(records)
X = data.drop('revenue_actuals', axis=1)
y = data['revenue_actuals']
 
# ─────────────────────────────────────────────
#  TRAIN MODEL
# ─────────────────────────────────────────────
model = lgb.LGBMRegressor(
    objective='regression',
    learning_rate=0.05,
    num_leaves=31,
    n_estimators=100,
    verbose=-1
)
model.fit(X, y)
print("\n✅ Model trained successfully on your data.\n")
 
# ─────────────────────────────────────────────
#  STEP 2 — Collect TEST input from user
# ─────────────────────────────────────────────
print("=" * 55)
print("📋 STEP 2: Enter values to PREDICT revenue for")
print("=" * 55 + "\n")
 
lag_1w          = get_float("Last week revenue         (lag_1w)",          "350")
lag_4w          = get_float("Revenue 4 weeks ago       (lag_4w)",          "300")
seasonality_idx = get_float("Seasonality index         (seasonality_idx)", "1.40")
pmi_index       = get_float("PMI index                 (pmi_index)",       "58")
 
test = pd.DataFrame({
    'lag_1w':          [lag_1w],
    'lag_4w':          [lag_4w],
    'seasonality_idx': [seasonality_idx],
    'pmi_index':       [pmi_index],
})
 
# ─────────────────────────────────────────────
#  PREDICT
# ─────────────────────────────────────────────
prediction = model.predict(test)
 
print("\n" + "=" * 55)
print(f"  📈 Predicted Next-Week Revenue : {prediction[0]:.2f}")
print("=" * 55)
 
# ─────────────────────────────────────────────
#  SHAP EXPLAINABILITY
# ─────────────────────────────────────────────
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(test)
 
print("\n📊 Feature Contributions (SHAP):")
print("-" * 40)
for feature, value in zip(test.columns, shap_values[0]):
    direction = "▲" if value >= 0 else "▼"
    print(f"  {direction} {feature:<22s}: {value:+.2f}")
 
print(f"\n  Base (average) prediction  : {explainer.expected_value:.2f}")
print(f"  Final prediction           : {prediction[0]:.2f}")
 
# ─────────────────────────────────────────────
#  SHAP FORCE PLOT → PNG
# ─────────────────────────────────────────────
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    test.iloc[0],
    matplotlib=True,
    show=False
)
plt.savefig('shap_explanation.png', bbox_inches='tight', dpi=150)
plt.close()
print("\n✅ SHAP explanation plot saved → shap_explanation.png\n")
 

