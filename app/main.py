from fastapi import FastAPI
from starlette.responses import JSONResponse
from keras.models import load_model
from joblib import load

app = FastAPI()

# load model
model = load_model('../models/model.h5')
# load map for labels
label_map = load('../models/label_map')

@app.get('/')
def read_root():
    return {'project objectives': 'TBC',
             'list of endpoints': 'TBC',
             'expected input parameters': 'TBC',
             'output format of the model': 'TBC',
             'link to the Github repo': 'TBC'
            }

@app.get('/health', status_code=200)
def healthcheck():
    return 'Welcome!'

def format_features(review_time: int, review_overall: float, review_aroma: float, review_appearance: float, review_palate: float, review_taste: float, beer_abv: float):
    return {'Review time': [review_time],
            'Review overall': [review_overall],
            'Review aroma': [review_aroma],
            'Review appearance': [review_appearance],
            'Review palate': [review_palate],
            'Review taste': [review_taste],
            'Alcohol by volume': [beer_abv]
            }

@app.get("/beer/type")
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