# Global_Avarage_Temperature_Prediction
Overview
This project uses a Long Short-Term Memory (LSTM) model to predict future global temperatures based on historical temperature data. The data used in this project is sourced from the 'Global Temperatures.csv' dataset, which contains historical records of land average temperatures over time. The model leverages time-series forecasting techniques to make predictions for the next 12 months of temperature data.

Requirements
Python 3.x
Libraries:
pandas
numpy
matplotlib
scikit-learn
tensorflow
You can install the required libraries using pip:

bash
Copy code
pip install pandas numpy matplotlib scikit-learn tensorflow
Data Description
The dataset used in this project is in CSV format and contains the following columns:

dt: Date of the temperature record.
LandAverageTemperature: The average temperature for land in a given year and month.
Data Preprocessing
The following preprocessing steps are performed on the dataset:

Date Conversion: The date column (dt) is converted to a datetime format to facilitate time-based analysis.
Normalization: The temperature values (LandAverageTemperature) are scaled between 0 and 1 using the MinMaxScaler to improve model performance.
Sequence Creation
A function create_sequences is used to create input sequences of length 12 (representing 12 months) to be fed into the LSTM model. This allows the model to learn from previous data and predict future values.

Model Architecture
The model is built using the Keras Sequential API with the following layers:

LSTM Layer 1: A 50-unit LSTM layer with ReLU activation and the ability to return sequences for the next LSTM layer.
LSTM Layer 2: A second 50-unit LSTM layer without returning sequences.
Dense Layer: A fully connected layer with a single output unit to predict the next temperature value.
The model is compiled using the Adam optimizer and mean squared error (MSE) loss function.

Model Training
The model is trained for 50 epochs with a batch size of 32, and the dataset is split into training and testing sets using a 75-25% split. The model performance is evaluated using the validation data.

Future Prediction
After training, the model is used to predict the next 12 months of global temperature data, based on the last 12 months from the dataset. The predictions are transformed back to the original scale of temperatures using the inverse transformation of the MinMaxScaler.

Plotting
The results are visualized using Matplotlib. A plot is generated to display the historical temperature data along with the predicted future temperatures.

Usage
Clone or download the repository.
Replace the dataset path in the script with the location of your own dataset.
Run the Python script to preprocess the data, train the model, and make future predictions.
bash
Copy code
python temperature_prediction.py
Conclusion
The LSTM model can be used for time-series forecasting, and in this case, it is applied to predict future global temperatures based on historical data. The predictions provide insights into the potential trend of global temperatures in the coming months.

Future Improvements
Further tuning of the model hyperparameters to enhance accuracy.
Addition of more features (e.g., location-based data, environmental factors) to improve the model's predictive capabilities.
Exploration of different machine learning models for comparison.
