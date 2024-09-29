House Price Prediction using Linear Regression

Project Overview
This project implements a Linear Regression model to predict house prices using the **Ames Housing dataset**. The features used include living area (`GrLivArea`), number of bedrooms (`BedroomAbvGr`), and number of bathrooms (`FullBath`). The goal is to build a regression model that accurately estimates house prices.

Dataset
The dataset used is `train.csv` from the Ames Housing dataset. Make sure to download and place the CSV file in the project directory.

Features
The following features are used for prediction:
- **GrLivArea**: Above-ground living area (in square feet)
- **BedroomAbvGr**: Number of bedrooms above ground
- **FullBath**: Number of full bathrooms

Installation
Install the required libraries using:

```bash
pip install pandas numpy scikit-learn
```

Usage
Run the script using:

```bash
python house_price_prediction.py
```

Results
The model outputs the following metrics for evaluation:
- **Mean Squared Error (MSE)**
- **R-squared Score (R²)**

Additionally, the project predicts the price of a new house based on user-specified input features.

 Future Improvements
1. Incorporate more features for better accuracy.
2. Experiment with other regression models like Ridge or Lasso.

