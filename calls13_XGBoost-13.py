import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt

# Step 1: Load the data
df = pd.read_csv('calls2022-2023.csv', sep=';', parse_dates=['date'], dayfirst=True)

# Step 2: Extract weekday and month from the date
df['weekday'] = df['date'].dt.weekday
df['month'] = df['date'].dt.month

# Step 3: User input for prediction date
input_date = input("Enter a date (dd/mm/yyyy) for prediction: ")
prediction_date = pd.to_datetime(input_date, format='%d/%m/%Y')

# Step 4: Train-test split
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Step 5: Prepare data for XGBoost
X_train, y_train = train[['hour', 'weekday', 'month']], train['value']

# Step 6: Filter historical data for the specified prediction date's weekday and month, and adjacent months
cleaned_df_grouped = df.groupby(['hour', 'weekday', 'month'])
print("Grouped data (before filter):")
print(cleaned_df_grouped['value'].describe())

# Create a mask for the condition
condition_mask = (df['value'] >= 0.3 * df.groupby(['hour'])['value'].transform('mean')) & (df['value'] <= 1.7 * df.groupby(['hour'])['value'].transform('mean'))

# Apply the mask to the DataFrame
cleaned_df = df[condition_mask]
print("Filtered data:")
print(cleaned_df.groupby(['hour', 'weekday', 'month'])['value'].describe())

# Print the total number of rows before and after filtering
print(f"Total number of rows before filtering: {len(df)}")
print(f"Total number of rows after filtering: {len(cleaned_df)}")



historical_data = cleaned_df[
    ((cleaned_df['weekday'] == prediction_date.weekday()) & (cleaned_df['month'].between(prediction_date.month - 1, prediction_date.month + 1)))
]

# Use the filtered data to train the model
X_train_filtered, y_train_filtered = cleaned_df[['hour', 'weekday', 'month']], cleaned_df['value']

# Step 7: Train the XGBoost model using filtered data
model = xgb.XGBRegressor(objective='reg:squarederror')
model.fit(X_train_filtered, y_train_filtered)

# Step 8: Make predictions on the test set
X_test, y_test = test[['hour', 'weekday', 'month']], test['value']
y_pred = model.predict(X_test)

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

    # Plot historical values for the specified date's weekday and adjacent months
    ax.scatter(historical_data['hour'], historical_data['value'], label='Historical Values', color='blue', marker='o', alpha=0.5)

    # Plot predicted values for the user-specified date
    X_user = pd.DataFrame({'hour': range(9, 25), 'weekday': [prediction_date.weekday()] * 16, 'month': [prediction_date.month] * 16})
    y_user_pred = model.predict(X_user)
    ax.scatter(X_user['hour'], y_user_pred, label='Predicted Values (User Input)', color='red', marker='x')

    ax.set_title('Historical and Predicted Values Plot')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Values')
    ax.set_xticks(range(9, 25))  # Set x-axis ticks for hours from 9 to 24
    ax.set_yticks(range(0, 900,50))  # Set x-axis ticks for hours from 9 to 24
    ax.legend()

    plt.tight_layout()
    plt.show()
