import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
#+++++++++++++++++++++++++++++++++++++++++++++++++


# Step 1: Load the data
df = pd.read_csv('calls2022-2023.csv', sep=';', parse_dates=['date'], dayfirst=True)

# Step 2: Extract weekday, month, and week from the date
df['weekday'] = df['date'].dt.weekday
df['month'] = df['date'].dt.month
df['week'] = df['date'].dt.isocalendar().week

# Introduce lag features
lag_periods = [1, 24]  # Lag for 1 hour and 1 day
for lag in lag_periods:
    df[f'value_lag_{lag}'] = df['value'].shift(lag)

# Step 3: User input for prediction date
input_date = input("Enter a date (dd/mm/yyyy) for prediction: ")
prediction_date = pd.to_datetime(input_date, format='%d/%m/%Y')

# Step 4: Train-test split
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Step 5: Prepare data for XGBoost
#X_train, y_train = train[['hour', 'weekday', 'month', 'week']], train['value']
X_train, y_train = train[['hour', 'weekday', 'month', 'week', 'value_lag_1', 'value_lag_24']], train['value']

# Step 6: Filter historical data for the specified prediction date's weekday, month, and adjacent weeks
cleaned_df_grouped = df.groupby(['hour', 'weekday', 'month', 'week'])
print("Grouped data (before filter):")
print(cleaned_df_grouped['value'].describe())

# Create a mask for the condition
condition_mask = (df['value'] >= 0.3 * df.groupby(['hour'])['value'].transform('mean')) & (df['value'] <= 1.7 * df.groupby(['hour'])['value'].transform('mean'))

# Apply the mask to the DataFrame
cleaned_df = df[condition_mask]
print("Filtered data:")
print(cleaned_df.groupby(['hour', 'weekday', 'month', 'week'])['value'].describe())

# Print the total number of rows before and after filtering
print(f"Total number of rows before filtering: {len(df)}")
print(f"Total number of rows after filtering: {len(cleaned_df)}")

# Calculate historical data based on the specified prediction date's weekday, month, and adjacent weeks
historical_data = cleaned_df[
    ((cleaned_df['weekday'] == prediction_date.weekday()) & (cleaned_df['month'].between(prediction_date.month - 1, prediction_date.month + 1)) & (cleaned_df['week'].between(prediction_date.isocalendar().week - 1, prediction_date.isocalendar().week + 1)))
]

# Use the filtered data to train the model
#X_train_filtered, y_train_filtered = cleaned_df[['hour', 'weekday', 'month', 'week']], cleaned_df['value']
X_train_filtered, y_train_filtered = cleaned_df[['hour', 'weekday', 'month', 'week', 'value_lag_1', 'value_lag_24']], cleaned_df['value']

# Step 7: Hyperparameter Tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
}

grid_search = GridSearchCV(
    xgb.XGBRegressor(objective='reg:squarederror'),
    param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
)

grid_search.fit(X_train_filtered, y_train_filtered)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Use the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Step 8: Make predictions on the test set
#X_test, y_test = test[['hour', 'weekday', 'month', 'week']], test['value']
X_test, y_test = test[['hour', 'weekday', 'month', 'week', 'value_lag_1', 'value_lag_24']], test['value']
y_pred = best_model.predict(X_test)

# Check for NaN values
if any(pd.isna(y_test)) or any(pd.isna(y_pred)):
    print("Error: NaN values found in the test set or predictions.")
else:
    # Step 9: Calculate Mean Squared Error on Test Set
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error on Test Set: {mse}')

    # Step 10: Check Residuals
    residuals = y_test - y_pred

    # Step 11: Plot Historical and Predicted Values
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot historical values for the specified date's weekday, month, and adjacent weeks
    ax.scatter(historical_data['hour'], historical_data['value'], label='Historical Values', color='blue', marker='o', alpha=0.5)

    # Plot predicted values for the user-specified date
    #X_user = pd.DataFrame({'hour': range(9, 25), 'weekday': [prediction_date.weekday()] * 16, 'month': [prediction_date.month] * 16,
    #'week': [prediction_date.isocalendar().week] * 16,
    X_user = pd.DataFrame({'hour': range(9, 25), 'weekday': [prediction_date.weekday()] * 16, 'month': [prediction_date.month] * 16,
    'week': [prediction_date.isocalendar().week] * 16,
    'value_lag_1': [historical_data['value'].iloc[-1]] * 16,  # Use the last known value as the lag feature
    'value_lag_24': [historical_data['value'].iloc[-24]] * 16,  # Use the value from 24 hours ago as the lag feature
    })

    y_user_pred = best_model.predict(X_user)
    label_str = f'Predicted Values ({input_date})'
    ax.scatter(X_user['hour'], y_user_pred, label=label_str, color='red', marker='x')

    ax.set_title('Historical and Predicted Values Plot')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Values')
    ax.set_xticks(range(9, 25))  # Set x-axis ticks for hours from 9 to 24
    ax.set_yticks(range(0, 900, 50))  # Set y-axis ticks

    ax.legend()

    plt.tight_layout()
    plt.show()