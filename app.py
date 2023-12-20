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

    manufacture = request_data.get('manufacture')
    
    if request.method == 'POST':

        if manufacture=="suzuki":

            yom=request_data.get('yom')

            jasonData=getMileage(suzuki_json)

            milage=jasonData[yom]
            print(milage)
           
            data=CustomDataAutoFinance(
                yom=request_data.get('yom'),
                model=request_data.get('model'),
                milage=milage,
                engine_capacity=request_data.get('engine_capacity'),
                fuel=request_data.get('fuel'),
                transmission=request_data.get('transmission')

            )
        
            pred_df=data.get_data_as_data_frame()
            
            predict_pipeline=PredictPipeline(suzuki_model_path)
            
            results=predict_pipeline.predict(pred_df)
            print(results)

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

            app.logger.info("finished the prediction process")
            
            response = jsonify(saveData)
            response.status_code = 200
            return response
                    
        elif manufacture=="toyota":

            yom=request_data.get('yom')

            jasonData=getMileage(toyota_json)

            milage=jasonData[yom]
            
            data=CustomDataAutoFinance(
                yom=request_data.get('yom'),
                model=request_data.get('model'),
                milage=milage,
                engine_capacity=request_data.get('engine_capacity'),
                fuel=request_data.get('fuel'),
                transmission=request_data.get('transmission')

            )
        
            pred_df=data.get_data_as_data_frame()
            
            predict_pipeline=PredictPipeline(toyota_model_path)
            
            results=predict_pipeline.predict(pred_df)
            print(results)

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

            app.logger.info("finished the prediction process")
            
            response = jsonify(saveData)
            response.status_code = 200
            return response
        
        elif manufacture=="nissan":

            yom=request_data.get('yom')

            jasonData=getMileage(nissan_json)

            milage=jasonData[yom]
            print(milage)
        
            data=CustomDataAutoFinance(
                yom=request_data.get('yom'),
                model=request_data.get('model'),
                milage=milage,
                engine_capacity=request_data.get('engine_capacity'),
                fuel=request_data.get('fuel'),
                transmission=request_data.get('transmission')

            )
        
            pred_df=data.get_data_as_data_frame()
            
            predict_pipeline=PredictPipeline(nissan_model_path)
            
            results=predict_pipeline.predict(pred_df)
            print(results)

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

            app.logger.info("finished the prediction process")
            
            response = jsonify(saveData)
            response.status_code = 200
            return response
        
        elif manufacture=="honda":

            yom=request_data.get('yom')

            jasonData=getMileage(honda_json)

            milage=jasonData[yom]
            print(milage)
        
            data=CustomDataAutoFinance(
                yom=request_data.get('yom'),
                model=request_data.get('model'),
                milage=milage,
                engine_capacity=request_data.get('engine_capacity'),
                fuel=request_data.get('fuel'),
                transmission=request_data.get('transmission')

            )
        
            pred_df=data.get_data_as_data_frame()
            print(pred_df)
            
            predict_pipeline=PredictPipeline(honda_model_path)
            
            results=predict_pipeline.predict(pred_df)
            print(results)
            
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

            app.logger.info("finished the prediction process")
            
            response = jsonify(saveData)
            response.status_code = 200
            return response
        
        else:
            response = jsonify({
                "manufacture": request_data.get('manufacture'),
                "submodel":request_data.get('model'),
                "timestamps":ct,
                "results":0
            })
            response.status_code = 200
            return response
                    
            

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port,debug=True)    

