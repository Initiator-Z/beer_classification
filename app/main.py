from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
from input_features import Input, Inputs
from pytorch_model import get_device, load_pytorch_model, MyDataset

app = FastAPI()

# get device
device = get_device()
# load model
model = load_pytorch_model('../models/pytorch_model.pth', device)
# load map for labels
label_map = load('../models/label_map')
# load preprocesser
preprocessor = load('../models/preprocesser')
# define features
features = Input.schema()['properties']

def prepare_features(df):
    from torch.utils.data import DataLoader
    # preprocess data
    preprocessed = preprocessor.transform(df)
    # create dataset
    dataset = MyDataset(preprocessed)
    # create data loader
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    return data_loader

def predict(data_loader):
    import torch
    import numpy as np
    with torch.no_grad():
        for features in data_loader:
            # input features
            features = features.float()
            features = features.to(device)
        # predict
        outputs = model(features)
        # number of dims, to account for single sample
        dims = len(outputs.shape)
        # get predicted classes
        _, predicted = torch.max(outputs.data, dims-1)
        index_array = predicted.numpy()
        prediction = np.vectorize(label_map.get)(index_array).tolist()
    return prediction

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
    return JSONResponse({'Project objectives': 'This API uses a neural network trained on the BeerAdvocates dataset to predict the type of beer using review rating criterias.\n',
             'List of endpoints': endpoints,
             'Input parameters': input_parameters,
             'Output format': '"/beer/type/" returns a string of the predicted beer type. \n "/beers/type/" returns a CSV file with the predicted beer types included as a new column.\n',
             'Github repo': 'https://github.com/Initiator-Z/beer_classification'
            })

@app.get('/health', status_code=200)
def health_check():
    return 'Welcome! The beer classifier model is ready to predict beers!'

@app.get('/model/architecture')
def model_summary():
    return JSONResponse(model.summary())

@app.post("/beer/type")
def predict_single(input: Input):
    import numpy as np
    # input to dataframe
    df = pd.DataFrame([input.dict()])
    # prepare features
    data_loader = prepare_features(df)
    # predict
    prediction = predict(data_loader)
    return JSONResponse(prediction)

@app.post("/beers/type")
def predict_batch(inputs: Inputs):
    # inputs to dataframe
    df = pd.DataFrame(inputs.dict_inputs())
    # prepare features
    data_loader = prepare_features(df)
    # predict
    predictions = predict(data_loader)
    return JSONResponse(predictions)