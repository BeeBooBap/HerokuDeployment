from fastapi import FastAPI, Form
from starlette.responses import HTMLResponse
from keras.models import load_model
from CasePredictor import casePredictor
import uvicorn
from fastapi.responses import JSONResponse
import numpy 

app = FastAPI()

model = load_model('model_LSTM_CNN.h5')

@app.get('/')
def index():
    return {'message': 'hello world'}

@app.post('/predict')
def predict_case(data:casePredictor):
    data = data.dict()
    dateDecision = data['dateDecision']
    term = data['term']
    respondent = data['respondent']
    caseOrigin = data['caseOrigin']
    issue = data['issue']

    arr = [dateDecision, term, respondent, caseOrigin, issue]
    arr = numpy.array(arr)
    arr = arr.reshape(1, -1)
    prediction = model.predict(arr)

    results = {}

    results['Unfavourable'] = "{:.2%}".format(prediction[0][0])
    results['Favourable'] = "{:.2%}".format(prediction[0][1])
    results['Unclear'] = "{:.2%}".format(prediction[0][2])

    return JSONResponse(content=results)
    
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)