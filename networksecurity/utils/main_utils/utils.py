import yaml
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os,sys
import numpy as np
import dill
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


def read_yaml_file(file_path:str)->dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    file_path: str
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    

def write_yaml_file(file_path:str, content: object, replace:bool = False):
    """
    Create yaml file 
    file_path: str
    data: dict
    """
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path,"w") as file:
            yaml.dump(content,file)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    

def save_numpy_array_data(file_path: str, array: np.array): 
    """
    Save numpy array data to file 
    file_path : str location of file to save
    array: np.array data to save
    Args:        
        file_path (str): The path where the numpy array will be saved.
        array (np.array): The numpy array to be saved.
    Returns:
        None
    """

    try : 
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj: 
            np.save(file_obj, array)

    except Exception as e: 
        raise NetworkSecurityException(e, sys) from e
    

def save_object(file_path: str, obj: object) -> None: 
    """
    Save a Python object to a file using pickle.
    Args:
        file_path (str): The path where the object will be saved.
        obj (object): The Python object to be saved.
    Returns:
        None
    """
    try: 
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj: 
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e: 
        raise NetworkSecurityException(e, sys) from e
    


## function to load object 
def load_object(file_path: str) -> object:
    """
    Load a Python object from a file using pickle.
    Args:
        file_path (str): The path where the object is saved.
    Returns:
        object: The Python object loaded from the file.
    """
    try: 
        if not os.path.exists(file_path): 
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj: 
            return pickle.load(file_obj)
    except Exception as e : 
        raise NetworkSecurityException(e, sys) 
    

## function to load numpy array 
def load_numpy_array_data(file_path: str) -> np.array: 
    """
    Load numpy array data from file
    Args: 
        file_path: str location of file to load
    Returns: 
        np.array data loaded
    """

    try: 
        with open(file_path, "rb") as file_obj: 
            return np.load(file_obj)
    except Exception as e: 
        raise NetworkSecurityException(e, sys) from e
    


def evaluate_models(X_train, y_train, X_test, y_test, models:dict, param:dict):
    """
    This function evaluates multiple machine learning models using GridSearchCV to find the best hyperparameters for each model. It then fits the models with the best parameters and calculates the R2 score for both training and testing datasets.

    Args:        
        X_train: Training features
        y_train: Training target
        X_test: Testing features
        y_test: Testing target
        models: A dictionary of machine learning models to evaluate
        param: A dictionary of hyperparameters for each model to be used in GridSearchCV.

    Returns:
        A dictionary containing the R2 score for each model on the testing dataset.
    """
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]
            
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        raise NetworkSecurityException(e,sys)