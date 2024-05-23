from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from american import predict


app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
def read_root():
    return "Welcome to my predictor"

@app.post('/api/predict_american_soccer')
async def predict_hockey(data: dict):
    num_params = len(data)

    # Check if the number of parameters is not 2 or 3
    if num_params != 2:
        raise HTTPException(status_code=400, detail='Invalid number of parameters.')

    # Extract the parameters
    if num_params == 2:
        
        team = data.get('team')
        opp_team = data.get('opp_team')

        results = predict(team, opp_team)

        return results

