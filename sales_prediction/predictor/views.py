from sklearn.preprocessing import StandardScaler
from django.shortcuts import render
from .forms import SalesDataForm
import pickle
from datetime import date, timedelta
import joblib

import numpy as np
import pandas as pd


def predict_sales(request):
    """Predicts sales based on user input and loaded model.

    Args:
        request: A Django HTTP request object.

    Returns:
        A Django HTTP response object with the rendered template
        and prediction (if successful) or errors (if any).
    """

    model = joblib.load("predictor/regression_pipeline.pkl")
    print(f"Loaded model type: {type(model)}")  # Check the type of the model
    prediction = None

    if request.method == "POST":
        form = SalesDataForm(request.POST)
        if form.is_valid():
            # Extract cleaned data from the form
            cleaned_data = form.cleaned_data

            # Prepare the input data for prediction as a NumPy array
            features = np.array([[
                cleaned_data["Store"],
                cleaned_data["DayOfWeek"],               
                cleaned_data["Customers"],
                cleaned_data["Date"],
                cleaned_data["Open"],
                cleaned_data["Promo"],
                cleaned_data["StateHoliday"],  # Assuming state_holiday is an integer
                cleaned_data["SchoolHoliday"],
            ]])

            # Convert features into a DataFrame (if necessary for your model)
            features_df = pd.DataFrame(features, columns=[
                'Store', 'DayOfWeek', 'Customers', 'Date','Open',
                'Promo', 'StateHoliday', 'SchoolHoliday'
            ])
            features_df['Date'] = pd.to_datetime(features_df['Date'])
            
            # Handle the state holiday logic
            features_df['StateHoliday'] = features_df['StateHoliday'].replace({0: None})
            features_df['StateHoliday'] = features_df['StateHoliday'].fillna('None')
            features_df["HolidayPeriod"] = "Normal"  # Placeholder for holiday period handling

            # Create StateHoliday dummy variables
            # Use boolean indexing to filter for rows where 'StateHoliday' is 'a'
            df_filtered = features_df[features_df['StateHoliday'] == 'a']

            # Assign values based on the filtering result
            features_df.loc[df_filtered.index, 'StateHoliday_a'] = 1
            features_df.loc[features_df['StateHoliday'] == 'b', 'StateHoliday_b'] = 1
            features_df.loc[features_df['StateHoliday'] == 'c', 'StateHoliday_c'] = 1

            # Set remaining values to 0 (assuming other categories are not 'a', 'b', or 'c')
            features_df['StateHoliday_a'] = features_df['StateHoliday_a'].fillna(0)
            features_df['StateHoliday_b'] = features_df['StateHoliday_b'].fillna(0)
            features_df['StateHoliday_c'] = features_df['StateHoliday_c'].fillna(0)

            features_df['HolidayPeriod_Before Holiday'] = 0
            features_df['HolidayPeriod_During Holiday'] = 0
            features_df['HolidayPeriod_Normal'] = 0

            # features_df = pd.get_dummies(features_df, columns=['StateHoliday', 'HolidayPeriod'], drop_first=True)
            # print(features_df.columns)
            # Extract features from the Date column
            features_df['Date'] = pd.to_datetime(features_df['Date'])
            features_df['Weekday'] = features_df['Date'].dt.weekday  # Weekday (0=Monday, 6=Sunday)
            features_df['Is_Weekend'] = features_df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)  # Weekend (1=True, 0=False)
            features_df['Month'] = features_df['Date'].dt.month  # Month
            features_df['Year'] = features_df['Date'].dt.year  # Year
            features_df['Day'] = features_df['Date'].dt.day  # Day
            features_df['Day_of_Year'] = features_df['Date'].dt.dayofyear  # Day of the year

            # Beginning, mid, and end of the month
            features_df['Is_Beginning_Month'] = features_df['Day'].apply(lambda x: 1 if x <= 10 else 0)
            features_df['Is_Mid_Month'] = features_df['Day'].apply(lambda x: 1 if 11 <= x <= 20 else 0)
            features_df['Is_End_Month'] = features_df['Day'].apply(lambda x: 1 if x > 20 else 0)
            
            

            # Define State holidays
            features_df["StateHoliday"] = features_df["StateHoliday"].replace({"0": None})
            holiday_dates = features_df.loc[features_df["StateHoliday"].notnull(), "Date"]

            # Create columns for "Before Holiday", "During Holiday", "After Holiday"
            features_df["HolidayPeriod"] = "Normal"
            for holiday in holiday_dates:
                features_df.loc[features_df["Date"] == holiday, "HolidayPeriod"] = "During Holiday"
                features_df.loc[features_df["Date"] == holiday - pd.Timedelta(days=1), "HolidayPeriod"] = "Before Holiday"
                features_df.loc[features_df["Date"] == holiday + pd.Timedelta(days=1), "HolidayPeriod"] = "After Holiday" 

            
            # Handle holiday period features (ensure the column exists after encoding)
            if 'HolidayPeriod_During Holiday' in features_df.columns:
                features_df['Days_to_Holiday'] = np.where(features_df['HolidayPeriod_During Holiday'] == 1, 0, 7)  # Example: Default 7 days
                features_df['Days_After_Holiday'] = np.where(features_df['HolidayPeriod_During Holiday'] == 1, 0, 7)  # Example: Default 7 days
            else:
                features_df['Days_to_Holiday'] = 7  # Default value if the column does not exist
                features_df['Days_After_Holiday'] = 7  # Default value if the column does not exist

            # features_df['Days_to_Holiday'] = np.where(features_df['HolidayPeriod_During Holiday'] == 1, 0, 7)  # Example: Default 7 days
            # features_df['Days_After_Holiday'] = np.where(features_df['HolidayPeriod_During Holiday'] == 1, 0, 7)  # Example: Default 7 days
            features_df.drop(['Date','StateHoliday', 'HolidayPeriod'], axis=1, inplace=True)

            # Ensure the DataFrame columns match the expected ones
            numeric_columns = ['Customers', 'Day_of_Year', 'Days_to_Holiday', 'Days_After_Holiday']
            scaler = StandardScaler()
            features_df[numeric_columns] = scaler.fit_transform(features_df[numeric_columns])

            # # Check the DataFrame to ensure it's correct
            # print(features_df.columns.to_list())
            # print(model.named_steps['scaler'].feature_names_in_)

            # # Make the prediction using the prepared features
            prediction = model.predict(features_df)[0]  # Assuming the model returns a single value
    else:
        form = SalesDataForm()

    return render(request, "predictor/predict.html", {"form": form, "prediction": prediction})


[]
[
 'HolidayPeriod_Before Holiday' 'HolidayPeriod_During Holiday'
 'HolidayPeriod_Normal']