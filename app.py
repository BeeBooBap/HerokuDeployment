from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
from CasePredictor import casePredictor
import uvicorn
from fastapi.responses import JSONResponse
import numpy 
import pandas
from sklearn.preprocessing import LabelEncoder
import pickle

app = FastAPI()

origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def index():
    return {'message': 'hello world'}

def prepare_input(input):
    data = input.dict()

    df = pandas.DataFrame([data.values()], columns=['dateDecision', 'term', 'respondent', 'caseOrigin', 'issue'])

    #file1 = open(r"C:\Users\Ayesha\Desktop\MSc Project\Deployment\HerokuDeployment\model_development\le.obj", "rb")
    #le_loaded = pickle.load(file1)
    #file1.close()
    #df['dateDecision'] = le_loaded.transform(df['dateDecision'])

    file2 = open(r"C:\Users\Ayesha\Desktop\MSc Project\Deployment\HerokuDeployment\model_development\scaler.obj", "rb")
    scaler_loaded = pickle.load(file2)
    file2.close()

    arr = df.values
    new_input = scaler_loaded.transform(arr)

    final_input = numpy.reshape(new_input, (new_input.shape[0], new_input.shape[1] , 1))

    return final_input

@app.post('/predict')
def predict_case(data:casePredictor):
    model = load_model('model_LSTM_CNN.h5')

    data = prepare_input(data)

    prediction = model.predict(data)

    results = {}

    results['Unfavourable'] = "{:.2%}".format(prediction[0][0])
    results['Favourable'] = "{:.2%}".format(prediction[0][1])
    results['Unclear'] = "{:.2%}".format(prediction[0][2])

    return JSONResponse(content=results)
    
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)