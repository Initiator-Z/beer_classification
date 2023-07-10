from fastapi import FastAPI
from starlette.responses import JSONResponse
from keras.models import load_model
from joblib import load
from src.deploy import Input, Inputs

app = FastAPI()

# load model
model = load_model('../models/model.h5')
# load map for labels
label_map = load('../models/label_map')
# define features
features = Input.model_json_schema()['properties']

@app.get('/')
def read_root():
    endpoints = '''
                "/health/" (GET): Returns status code 200 with a welcome message.
                "/beer/type/" (POST): Makes a prediction for a single input.
                "/beers/type/" (POST): Makes a batch prediction on a provided CSV file of inputs.
                "/model/architecture/" (GET): Lists the neural network model architecture. \n
                '''
    input_parameters =  f'''
                        The following features are required to be provided to the prediction endpoints. 
                        Provide these features as query parameters to "/beer/type" and as columns of the CSV to "/beers/type" with the parameters as headings.
                        {features}
                        '''
    return {'Project objectives': 'This API uses a neural network trained on the BeerAdvocates dataset to predict the type of beer using review rating criterias.\n',
             'List of endpoints': endpoints,
             'Input parameters': input_parameters,
             'Output format': '"/beer/type/" returns a string of the predicted beer type. \n "/beers/type/" returns a CSV file with the predicted beer types included as a new column.\n',
             'Github repo': 'https://github.com/Initiator-Z/beer_classification'
            }

@app.get('/health', status_code=200)
def health_check():
    return 'Welcome! The beer classifier model is ready to predict beers!'

@app.get('/model/architecture')
def model_summary():
    return model.summary()

def predict(feature_array):
    import numpy as np
    # run prediction
    preds = model.predict(feature_array)
    # return predicted type
    index_array = np.argmax(preds)
    pred_type = [label_map[index] for index in index_array]
    return pred_type

@app.post("/beer/type")
def predict_single(input: Input):
    import numpy as np
    # prepare features
    features = input.model_dump()
    feature_array = np.array(list(features.values())).reshape(1, -1)
    # predict
    prediction = predict(feature_array)
    return JSONResponse(prediction[0])

@app.post("/beers/type")
def predict_batch(inputs: Inputs):
    import pandas as pd
    # prepare features
    feature_df = pd.DataFrame(inputs.dict_inputs())
    # predict
    predictions = predict(feature_df) 
    return JSONResponse(predictions)