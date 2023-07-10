from fastapi import FastAPI
from starlette.responses import JSONResponse
from keras.models import load_model
from joblib import load
from src.deploy import Inputs

app = FastAPI()

# load model
model = load_model('../models/model.h5')
# load map for labels
label_map = load('../models/label_map')
# define features


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

def format_features(review_time: int, review_overall: float, review_aroma: float, review_appearance: float, review_palate: float, review_taste: float, beer_abv: float):
    return {'Review time': [review_time],
            'Review overall': [review_overall],
            'Review aroma': [review_aroma],
            'Review appearance': [review_appearance],
            'Review palate': [review_palate],
            'Review taste': [review_taste],
            'Alcohol by volume': [beer_abv]
            }

@app.post("/beer/type")
def predict(review_time: int, review_overall: float, review_aroma: float, review_appearance: float, review_palate: float, review_taste: float, beer_abv: float):
    import numpy as np
    
    # prepare features
    features = format_features(review_time, review_overall, review_aroma, review_appearance, review_palate, review_taste, beer_abv)
    input = np.array(list(features.values())).reshape(1, -1)

    # run prediction
    preds = model.predict(input)

    # return predicted type
    pred_index = np.argmax(preds[0])
    pred_type = label_map[pred_index]

    return JSONResponse(pred_type)

@app.post("/beers/type")
async def predict_batch(inputs: Inputs):
    import pandas as pd
    # Create a dataframe from inputs
    data = pd.DataFrame(inputs.return_dict_inputs())
    data_copy = data.copy() # Create a copy of the data
    labels, probs = make_prediction(data, transformer, model) # Get the labels
    response = output_batch(data, labels) # output results
    return response