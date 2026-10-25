
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load historical sales data
df = pd.read_csv("sales_data.csv")

# 2. Clean and preprocess the data
df = df.dropna()
df = df.drop_duplicates()
df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
df = df.dropna(subset=["Month", "Sales"])

X = df[["Month"]]
y = df["Sales"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 4. Define prediction models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=3, random_state=42
    )
}

results = []
predictions_by_model = {}

# 5. Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    predictions_by_model[name] = predictions

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
        "R2 Score": r2_score(y_test, predictions)
    })

# 6. Display evaluation results
results_df = pd.DataFrame(results)
print("\nMODEL EVALUATION RESULTS")
print(results_df.round(2).to_string(index=False))

best_model_name = results_df.loc[
    results_df["MAE"].idxmin(), "Model"
]
print("\nModel with lower MAE:", best_model_name)

# 7. Predict sales for month 13
best_model = models[best_model_name]
future_month = pd.DataFrame({"Month": [13]})
future_prediction = best_model.predict(future_month)

print(
    "\nPredicted sales for Month 13:",
    round(future_prediction[0], 2)
)

# 8. Visualize model comparison
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["MAE"])
plt.title("Model Comparison - Mean Absolute Error")
plt.ylabel("MAE (lower is better)")
plt.tight_layout()
plt.show()

# 9. Visualize actual vs predicted sales
plt.figure(figsize=(9, 5))
plt.plot(
    X_test["Month"], y_test,
    "o-", label="Actual Sales"
)

for name, predictions in predictions_by_model.items():
    plt.plot(
        X_test["Month"], predictions,
        "x--", label=name
    )

plt.title("Actual vs Predicted Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.show()
