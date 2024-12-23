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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import json
import firebase_admin
from firebase_admin import credentials, firestore

router = APIRouter()

# Initialize Firebase app only once
if not firebase_admin._apps:
    cred_path = r"C:\Users\justi\OneDrive - Fondation EPF\Documents\5A semestre 1 EPF\Data sources\repo\api-webscrapping-e7967-firebase-adminsdk-tiigz-02eaac3e18.json"
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

# Step 7 : define the endpoint to load the dataset and return it as a JSON response
@router.get("/load-iris-dataset", response_model=dict)
async def get_iris_data():
    """
  Endpoint to load the Iris dataset and return it as a JSON response.

  This endpoint reads the Iris dataset from a CSV file, converts it
  to JSON format, and returns the data as a dictionary.

  Args:
      None

  Returns:
      dict: A dictionary containing the dataset as a JSON object.

  Raises:
      HTTPException: If the dataset file is not found (404) or any other error occurs (500).
"""
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
    """
  Endpoint to process the Iris dataset.

  This endpoint loads the Iris dataset from a CSV file, preprocesses it by
  dropping missing values and the 'Id' column, label-encoding the 'Species'
  column, and saves the cleaned dataset to a new CSV file.

  Args:
      None

  Returns:
      dict: A dictionary containing a success message and the processed data.

  Raises:
      HTTPException: If the dataset file is not found (404) or if any other error occurs (500).
"""
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
    """
  Endpoint to split the Iris dataset into training and testing sets.

  This endpoint loads the preprocessed Iris dataset, splits it into
  features (`X`) and labels (`y`), then divides the data into training
  and testing sets using a specified test size. The split datasets are
  saved as separate CSV files for further use.

  Args:
      test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.

  Returns:
      dict: A dictionary containing a success message and the training and testing datasets in JSON format.

  Raises:
      HTTPException: If the dataset file is not found (404) or if any other error occurs (500).
"""
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
    
# Step 11: Training the classification model and save the model in src/models
@router.get("/train-iris-model", response_model=Dict[str, Any])
async def train_iris_model():
    """
Endpoint to train the Iris classification model and save it.

This endpoint loads the training data (`X_train` and `y_train`) from CSV files,
retrieves the model parameters from a JSON configuration file, trains a logistic regression
model, and saves the trained model to a specified directory.

Args:   
    None

Returns:
    dict: A dictionary containing a success message and the path to the saved model.

Raises:
    HTTPException: If any file is not found (404) or if any other error occurs (500).
"""
    try:
        output_dir = r"TP2 and  3\services\epf-flower-data-science\src\models"
        X_train_path = r"TP2 and  3\services\epf-flower-data-science\src\data\Iris\X_train.csv"
        y_train_path = r"TP2 and  3\services\epf-flower-data-science\src\data\Iris\y_train.csv"
        model_path = r"TP2 and  3\services\epf-flower-data-science\src\config\model_parameters.json"

        # Load the training data
        X_train = pd.read_csv(X_train_path)
        y_train = pd.read_csv(y_train_path)
        print("Shape of X_train:", X_train.shape)
        print("Shape of y_train:", y_train.shape)
        print("Missing values in X_train:", X_train.isnull().sum().sum())
        print("Missing values in y_train:", y_train.isnull().sum().sum())

        # Get the model parameters in the json file
        with open(model_path, 'r') as file:
            params = json.load(file)

        

        # Initialize the model
        model = LogisticRegression(**params['parameters'])
        print(model.get_params())

        # Train the model
        model.fit(X_train, y_train.values.ravel())

        # Save the model
        model_path = os.path.join(output_dir, "iris_classifier.joblib")
        joblib.dump(model, model_path)

        return {"message": "Model trained and saved successfully", "model_path": model_path}

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    except Exception as e:
        # Log the error message for debugging purposes
        print(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    
# Step 12: Prediction with Trained Model and send back the predictions as json
@router.get("/predict-iris", response_model=Dict[str, Any])
async def predict_iris():
    try:
        model_path = r"TP2 and  3\services\epf-flower-data-science\src\models\iris_classifier.joblib"
        model = joblib.load(model_path)
        X_test_path = r"TP2 and  3\services\epf-flower-data-science\src\data\Iris\X_test.csv"

        # Load the X_test data
        X_test = pd.read_csv(X_test_path)

        # Make predictions
        predictions = model.predict(X_test)

        # Return the predictions
        return {"predictions": predictions.tolist()}

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        # Log the error message for debugging purposes
        print(f"Error predicting with model: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    
# Step 14: Retrieve parameters from Firestore
@router.get("/get-model-parameters", response_model=Dict[str, Any])
async def get_model_parameters():
    """
Endpoint to make predictions with the trained Iris classification model.

This endpoint loads the trained logistic regression model from a `.joblib` file,
loads the test dataset (`X_test`) from a CSV file, makes predictions, and returns
the predictions as a JSON object.

Args:
    None

Returns:
    dict: A dictionary containing the predictions in a list.

Raises:
    HTTPException: If the model or test dataset is not found (404) or if any other error occurs (500).
"""
    try:
        db = firestore.client()

        # Récupérer le document 'parameters' de la collection Firestore
        parameters_ref = db.collection('parameters').document('parameters')
        doc = parameters_ref.get()

        if doc.exists:
            return {"parameters": doc.to_dict()}
        else:
            raise HTTPException(status_code=404, detail="Parameters document not found")

    
    except Exception as e:
        # Log the error message for debugging purposes
        print(f"Error loading model parameters: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    
# Step 15: Update and add Firestore parameters (tested with insomnia)

@router.post("/update-model-parameters", response_model=Dict[str, Any])
async def update_model_parameters(parameters: Dict[str, Any]):
    """
    Endpoint to update model parameters in Firestore.

    This endpoint takes a dictionary of new parameters and updates the 'parameters' document
    in the Firestore database. It then retrieves and returns the updated parameters.

    Args:
        parameters (Dict[str, Any]): A dictionary containing the new model parameters to be updated.

    Returns:
        Dict[str, Any]: A dictionary with a success message and the updated parameters retrieved 
                        from the Firestore database.

    Raises:
        HTTPException: If an error occurs during the update process (500 status code).
    """
    try:
        db = firestore.client()

        # Récupérer le document 'parameters' de la collection Firestore
        parameters_ref = db.collection('parameters').document('parameters')
        parameters_ref.update(parameters)

        return {"message": "Model parameters updated successfully", "parameters": parameters_ref.get().to_dict()}

    except Exception as e:
        # Log the error message for debugging purposes
        print(f"Error updating model parameters: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.post("/add-model-parameters", response_model=Dict[str, Any])
async def add_model_parameters(parameters: Dict[str, Any]):
    """
    Endpoint to add model parameters to Firestore.

    This endpoint takes a dictionary of model parameters and adds them to the 'parameters' document
    in Firestore. If the document already exists, the new parameters will be merged with the existing ones.

    Args:
        parameters (Dict[str, Any]): A dictionary containing the model parameters to be added.

    Returns:
        Dict[str, Any]: A dictionary with a success message and the updated parameters retrieved 
                        from the Firestore database.

    Raises:
        HTTPException: If an error occurs during the adding process (500 status code).
    """
    
    try:
        db = firestore.client()

        # Récupérer le document 'parameters' de la collection Firestore
        parameters_ref = db.collection('parameters').document('parameters')
        parameters_ref.set(parameters, merge=True)

        return {"message": "Model parameters added successfully", "parameters": parameters_ref.get().to_dict()}

    except Exception as e:
        # Log the error message for debugging purposes
        print(f"Error adding model parameters: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")