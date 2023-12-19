import numpy as np 
import pandas as pd
import pickle
import json
import os
from flask import request



def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        print("error occured at model loading",e)



def getMileage(jasonpath:str):
    with open(f'constant/{jasonpath}') as json_file:
            data = json.load(json_file)

    return data



