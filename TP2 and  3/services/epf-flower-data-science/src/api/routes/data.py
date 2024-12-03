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


# Define the endpoint to load the dataset
@router.get("/load-iris-dataset", response_model=dict)
async def get_iris_data():
    try:
        # Path to the dataset file
        dataset_path = "src/data/iris/Iris.csv"

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
        dataset_path = "src/data/iris/Iris.csv"
        
        # Load the dataset
        df = pd.read_csv(dataset_path)

        # Drop nan
        df = df.dropna()

        # Drp column Id
        df.drop(columns=["Id"])

        # Label Encoding the 'species' column
        encoder = LabelEncoder()
        df['Species'] = encoder.fit_transform(df['Species'])  # Encoding categorical values to numbers

        # Save the csv
        df.to_csv("src/data/iris/Iris_preprocessed.csv", index=False)

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
