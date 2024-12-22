# api/routes/data.py
import opendatasets as od
from fastapi import APIRouter, HTTPException
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from typing import List
from typing import Dict, Any

router = APIRouter()


# Step 7 : define the endpoint to load the dataset and return it as a JSON response
@router.get("/load-iris-dataset", response_model=dict)
async def get_iris_data():
    try:
        # Path to the dataset file
        dataset_path = r"TP2 and  3\services\epf-flower-data-science\src\data\Iris\Iris.csv"
        print("Absolute path to dataset:", os.path.abspath(dataset_path))
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(dataset_path)

        # Convert DataFrame to JSON format
        data = df.to_dict(orient="records")  # Each row is converted to a dictionary

        return {"data": data}  # Return the dataset as JSON
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Step 8: Processing the Iris dataset (encoding, scaling, etc.)
@router.get("/process-iris-dataset", response_model=Dict[str, Any])
async def process_iris_data():
    try:
        dataset_path = r"TP2 and  3\services\epf-flower-data-science\src\data\Iris\Iris.csv"
        
        # Load the dataset
        df = pd.read_csv(dataset_path)

        # Drop nan
        df = df.dropna()

        # Drp column Id
        df.drop(columns=["Id"], inplace=True)
        print("Dropped Id column")

        # Label Encoding the 'species' column
        encoder = LabelEncoder()
        df['Species'] = encoder.fit_transform(df['Species'])  # Encoding categorical values to numbers

        # Save the csv
        df.to_csv(r"TP2 and  3\services\epf-flower-data-science\src\data\Iris\Iris_preprocessed.csv", index=False)

        # Debugging: Print the cleaned dataset
        print("Cleaned Data:")
        print(df.head())

        # Return a success message and some data
        return {"message": "Dataset processed successfully", "processed_data": df.to_dict(orient="records")}

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    except Exception as e:
        # Log the error message for debugging purposes
        print(f"Error processing dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


# Step 9: Splitting the Iris dataset into training and testing sets and return as json
@router.get("/split-iris-dataset", response_model=Dict[str, Any])
async def split_iris_data(test_size: float = 0.2):
    try:
        output_dir = r"TP2 and  3\services\epf-flower-data-science\src\data\Iris"
        dataset_path = r"TP2 and  3\services\epf-flower-data-science\src\data\Iris\Iris_preprocessed.csv"

        # Load the preprocessed dataset
        df = pd.read_csv(dataset_path)
        df = df.dropna()

        # Split the dataset into features (X) and labels (y)
        X = df.drop(columns=["Species"])
        y = df["Species"]

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Return the split datasets
        datasets = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }


        for key, value in datasets.items():
            file_path = os.path.join(output_dir, f"{key}.csv")
            if isinstance(value, pd.DataFrame):
                value.to_csv(file_path, index=False)  # Pour X_train et X_test
            else:
                pd.DataFrame(value).to_csv(file_path, index=False, header=[key])  # Pour y_train et y_test

        return {
            "message": "Dataset split successfully",
            "X_train": X_train.to_dict(orient="records"),
            "X_test": X_test.to_dict(orient="records"),
            "y_train": y_train.to_list(),
            "y_test": y_test.to_list(),
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    except Exception as e:
        # Log the error message for debugging purposes
        print(f"Error splitting dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")