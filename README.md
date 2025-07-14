# Vehicle Price Predictor

A machine learning project to predict vehicle prices based on various features using a Random Forest Regressor. The project includes data preprocessing, model training, and saving the trained model and scaler for later use.

## Project Structure

```
vehicle-price-predictor/
├── app.py                # (Optional) Flask or Streamlit app for inference (not included here)
├── train_model.py        # Script to preprocess data and train the model
├── requirements.txt      # Python dependencies
├── data/
│   └── vehicles.csv      # Input dataset
├── models/
│   ├── scaler.pkl        # Saved StandardScaler for mileage and vehicle_age
│   └── vehicle_model.pkl # Trained RandomForestRegressor model
```

## How It Works

1. **Data Loading:** Reads `data/vehicles.csv`.
2. **Preprocessing:**
   - Drops irrelevant columns (`name`, `description`).
   - Converts `year` to `vehicle_age`.
   - Fills missing values.
   - One-hot encodes categorical features.
   - Scales `mileage` and `vehicle_age` using `StandardScaler`.
3. **Model Training:**
   - Splits data into train/test sets.
   - Trains a `RandomForestRegressor` on the data.
4. **Saving Artifacts:**
   - Saves the trained model and scaler to the `models/` directory.
5. **Evaluation:**
   - Prints R2 Score and Mean Squared Error (MSE) on the test set.

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

- Place your dataset as `data/vehicles.csv`.
- Ensure it contains at least the following columns: `price`, `mileage`, `year`, and any other relevant features.

### 3. Train the Model

```bash
python train_model.py
```

- This will preprocess the data, train the model, and save the artifacts in the `models/` directory.

## Requirements

- Python 3.7+
- See `requirements.txt` for package versions.

## Output

- `models/vehicle_model.pkl`: Trained RandomForestRegressor model.
- `models/scaler.pkl`: Fitted StandardScaler for preprocessing.

## Notes

- The script expects a `price` column as the target variable.
- You can use the saved model and scaler in an inference script or web app (e.g., `app.py`).

## License

This project is for educational purposes.
