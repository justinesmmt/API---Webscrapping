# api/routes/data.py
import opendatasets as od
from fastapi import APIRouter, HTTPException
import os
import pandas as pd

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