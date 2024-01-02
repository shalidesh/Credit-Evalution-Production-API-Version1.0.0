from flask import Flask,request,render_template,jsonify
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from predict_pipeline import CustomDataAutoFinance,PredictPipeline
from utils import getMileage
from xgboost import XGBRegressor
import os
import json
import datetime
import pandas as pd
from pymongo import MongoClient

from constant.inputsValidation import nissan_list,suzuki_list,toyota_list,Transmission_list,fuel_list


app = Flask(__name__)

suzuki_model_path=os.path.join('models',"suzuki")
toyota_model_path=os.path.join('models',"toyota")
nissan_model_path=os.path.join('models',"nissan")
honda_model_path=os.path.join('models',"honda")

csv_filePath=os.path.join('predctionLogs',"predctions.csv")

suzuki_json='constants_suzuki.json'
toyota_json='constant_toyoto.json'
honda_json='constant_honda.json'
nissan_json='constant_nissan.json'


Fuel_Type=['HYBRID','PETROL'] 


def saveLogs(saveData:dict):
    # Load existing predictions from CSV
    if os.path.exists(csv_filePath):
        df = pd.read_csv(csv_filePath)
    else:
        df = pd.DataFrame(columns=["Timestamps","Manufacture","Submodel","Yom","Engine_capacity","Fuel_Type","Transmission","Predicted_Price"])

    # Append new prediction to dataframe
    new_data = pd.DataFrame([{**saveData}], columns=df.columns)
    df = pd.concat([df, new_data], ignore_index=True)
    
    # Save dataframe to CSV
    df.to_csv(csv_filePath, index=False)


def submodelCheck(manufacture:str,model:str):
    if manufacture=="nissan":
        if model in nissan_list:
            print(f"model :{model}")
            return True
        else:
            return False
        
    if manufacture=="suzuki":
        if model in suzuki_list:
            print(f"model :{model}")
            return True
        else:
            return False
        
    if manufacture=="toyota":
        if model in toyota_list:
            print(f"model :{model}")
            return True
        else:
            return False
        
    else:
        return False


def TransitionCheck(transmission:str):
    if transmission in Transmission_list:
        print(f"transmission :{transmission}")
        return True
    else:
        return False

def fuelTypeCheck(fuel:str):
    if fuel in fuel_list:
        print(f"fuel :{fuel}")
        return True
    else:
        print(f"fuel not :{fuel}")
        return False



def checkInputs(request_data):

    model=request_data.get('model').strip() 
    manufacture = request_data.get('manufacture').strip() 
    fuel=request_data.get('fuel').strip() 
    transmission=request_data.get('transmission').strip() 
    yom=request_data.get('yom').strip() 
    engine_capacity=request_data.get('engine_capacity')

    if yom  and model and manufacture and fuel and transmission and engine_capacity:

        if submodelCheck(manufacture,model) and TransitionCheck(transmission) and fuelTypeCheck(fuel):
            print("Everithing ok")
            return True
        
        else:
            return False
        
    else:
        return False


      
@app.route('/', methods=['GET'])
def index():
    app.logger.info("call the main route")
    response = jsonify({
        "message": "use predict route to get predictions",
    })
    response.status_code = 200
    return response


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():

    ct = datetime.datetime.now()

    request_data = request.get_json(force=True) 

    manufacture = request_data.get('manufacture').strip() 


    if request.method == 'POST':

        if checkInputs(request_data):

            if manufacture=="suzuki":

                yom=request_data.get('yom').strip()

                jasonData=getMileage(suzuki_json)

                milage=jasonData[yom]
                print(milage)
            
                data=CustomDataAutoFinance(
                    yom=request_data.get('yom').strip(),
                    model=request_data.get('model').strip(),
                    milage=milage,
                    engine_capacity=request_data.get('engine_capacity'),
                    fuel=request_data.get('fuel').strip(),
                    transmission=request_data.get('transmission').strip()

                )
            
                pred_df=data.get_data_as_data_frame()
                
                predict_pipeline=PredictPipeline(suzuki_model_path)
                
                results=predict_pipeline.predict(pred_df)
                
                saveData={
                    "Timestamps":ct,
                    "Manufacture": "suzuki",
                    "Submodel":request_data.get('model'),
                    "Yom":request_data.get('yom'),
                    "Engine_capacity":request_data.get('engine_capacity'),
                    "Fuel_Type":request_data.get('fuel'),
                    "Transmission":request_data.get('transmission'),
                    "Predicted_Price":float(results[0])
                }

                saveLogs(saveData)

                selected_data = {key: saveData[key] for key in ('Timestamps', 'Predicted_Price')}

                app.logger.info("finished the prediction process")
                
                response = jsonify(selected_data)
                response.status_code = 200
                return response
                        
            elif manufacture=="toyota":

                yom=request_data.get('yom').strip()

                jasonData=getMileage(toyota_json)

                milage=jasonData[yom]
            
                data=CustomDataAutoFinance(
                    yom=request_data.get('yom').strip(),
                    model=request_data.get('model').strip(),
                    milage=milage,
                    engine_capacity=request_data.get('engine_capacity'),
                    fuel=request_data.get('fuel').strip(),
                    transmission=request_data.get('transmission').strip()
                )
            
                pred_df=data.get_data_as_data_frame()
                
                predict_pipeline=PredictPipeline(toyota_model_path)
                
                results=predict_pipeline.predict(pred_df)
                
                saveData={
                    "Timestamps":ct,
                    "Manufacture": "toyota",
                    "Submodel":request_data.get('model'),
                    "Yom":request_data.get('yom'),
                    "Engine_capacity":request_data.get('engine_capacity'),
                    "Fuel_Type":request_data.get('fuel'),
                    "Transmission":request_data.get('transmission'),
                    "Predicted_Price":float(results[0])
                }

                saveLogs(saveData)

                selected_data = {key: saveData[key] for key in ('Timestamps', 'Predicted_Price')}

                app.logger.info("finished the prediction process")
                
                response = jsonify(selected_data)
                response.status_code = 200
                return response
            
            elif manufacture=="nissan":

                yom=request_data.get('yom').strip()

                jasonData=getMileage(nissan_json)

                milage=jasonData[yom]
                
                data=CustomDataAutoFinance(
                    yom=request_data.get('yom').strip(),
                    model=request_data.get('model').strip(),
                    milage=milage,
                    engine_capacity=request_data.get('engine_capacity'),
                    fuel=request_data.get('fuel').strip(),
                    transmission=request_data.get('transmission').strip()
                )
                pred_df=data.get_data_as_data_frame()
                
                predict_pipeline=PredictPipeline(nissan_model_path)
                
                results=predict_pipeline.predict(pred_df)
                
                saveData={
                    "Timestamps":ct,
                    "Manufacture": "nissan",
                    "Submodel":request_data.get('model'),
                    "Yom":request_data.get('yom'),
                    "Engine_capacity":request_data.get('engine_capacity'),
                    "Fuel_Type":request_data.get('fuel'),
                    "Transmission":request_data.get('transmission'),
                    "Predicted_Price":float(results[0])
                }

                saveLogs(saveData)

                selected_data = {key: saveData[key] for key in ('Timestamps', 'Predicted_Price')}

                app.logger.info("finished the prediction process")
                
                response = jsonify(selected_data)
                response.status_code = 200
                return response
            
            elif manufacture=="honda":

                yom=request_data.get('yom').strip()

                jasonData=getMileage(honda_json)

                milage=jasonData[yom]
                
                data=CustomDataAutoFinance(
                    yom=request_data.get('yom').strip(),
                    model=request_data.get('model').strip(),
                    milage=milage,
                    engine_capacity=request_data.get('engine_capacity'),
                    fuel=request_data.get('fuel').strip(),
                    transmission=request_data.get('transmission').strip()

                )
            
                pred_df=data.get_data_as_data_frame()
                
                predict_pipeline=PredictPipeline(honda_model_path)
                
                results=predict_pipeline.predict(pred_df)
                
                saveData={
                    "Timestamps":ct,
                    "Manufacture": "honda",
                    "Submodel":request_data.get('model'),
                    "Yom":request_data.get('yom'),
                    "Engine_capacity":request_data.get('engine_capacity'),
                    "Fuel_Type":request_data.get('fuel'),
                    "Transmission":request_data.get('transmission'),
                    "Predicted_Price":float(results[0])
                }

                saveLogs(saveData)

                selected_data = {key: saveData[key] for key in ('Timestamps', 'Predicted_Price')}

                app.logger.info("finished the prediction process")
                
                response = jsonify(selected_data)
                response.status_code = 200
                return response
            
            else:
                app.logger.info("allowed only limited makes request")
                response = jsonify({
                    "timestamps":ct,
                    "Predicted_Price":0
                })
                response.status_code = 200
                return response

        else:
            app.logger.info("input validation error")
            response = jsonify({
                        "timestamps":ct,
                        "Predicted_Price":0
                    })
            response.status_code = 200
            return response


    else:
        app.logger.info("allowed only post request")
        response = jsonify({
                        "timestamps":ct,
                        "Predicted_Price":0
                    })
        response.status_code = 500
        return response

    

@app.route('/upload', methods=['GET'])
def upload():
    # Create a MongoDB client
    client = MongoClient('mongodb+srv://shalidesh:shalidesh@vehiclevalution.r1r7ir0.mongodb.net/?retryWrites=true&w=majority')

    # Connect to your database
    db = client['records']

    # Choose your collection
    collection = db['vehicle-prices']

    # Check if the collection has any documents
    if collection.count_documents({}) > 0:
        # If the collection has documents, delete them
        collection.delete_many({})

    # Load your CSV file
    df = pd.read_csv('predctionLogs\predctions.csv')

    # Convert each record to dict and insert into the collection
    collection.insert_many(df.to_dict('records'))

    app.logger.info(" dataset uploaded successfully")
    response = jsonify({
        "message": "dataset uploaded successfully",
    })
    response.status_code = 200
    return response
          

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)    

