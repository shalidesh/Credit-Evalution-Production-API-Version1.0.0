import sys
import pandas as pd
from utils import load_object
import os
import datetime
import logging


class PredictPipeline:
    def __init__(self,modelpath:str):
        self.model_location=modelpath
        

    def predict(self,features):
        try:
            model_path=os.path.join(self.model_location,"model.pkl")
            preprocessor_path=os.path.join(self.model_location,"preprocessor.pkl")
            print("model before loading")
            model=load_object(file_path=model_path)
            print("model after loading")
            print("preproceesor before loading")
            preprocessor=load_object(file_path=preprocessor_path)
            print("preproceesor after loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            print(preds)
            return preds
        
        except FileNotFoundError:
            logging.error("Model or preprocessor file not found.")
            return None  # Or some default prediction or error response
        except Exception as e:
            logging.error(f"An error occurred during prediction: {str(e)}")
            return None  # Or some default prediction or error response


class CustomDataAutoFinance:
    def __init__(self,
        yom: str,
        model:str,
        # sub_model:str,
        milage: str,
        engine_capacity:str,
        fuel:str,
        transmission: str
        ):

        self.mileage = milage

        self.yom = yom

        self.curr_year=datetime.datetime.now().year

        self.age=self.curr_year - int(self.yom) if int(self.yom) else 0

        self.model = model

        self.engine_capacity = engine_capacity

        self.fuel=fuel

        self.transmission=transmission


    def get_data_as_data_frame(self):
        try:

            custom_data_input_dict = {
                # "Sub_Model":[self.sub_model],
                "Age": [self.age],
                "Mileage": [int(self.mileage)],
                "Engine_Capacity": [self.engine_capacity],
                "Model": [self.model],
                "Fuel_Type": [self.fuel],
                "Transmission": [self.transmission]        
            }

            return pd.DataFrame(custom_data_input_dict)

        
        except Exception as e:
            logging.error(f"An error occurred creating dataframe: {str(e)}")
            return None

