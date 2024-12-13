# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model, BaseModel
from typing import ClassVar

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("optionclass_api")
print(model.feature_names_in_)

# Define the input model with type annotations
class OptionClassApiInput(BaseModel):
    sexe: float = 1.0
    nb_frère: float = 0.0
    nb_sœur: float = 3.0
    commune_d_origine: int = 44
    habite_avec_les_parents: int = 1
    electricite: int = 0
    conn_sur_les_options: int = 1
    MLG: float = 19.0
    FRS: float = 24.5
    ANG: float = 36.0
    HG: float = 28.5
    SES: float = 23.0
    MATHS: float = 18.0
    PC: float = 20.0
    SVT: float = 33.0
    EPS: float = 28.0
    première_S: float = 12.409090995788574
    deuxième_S: float = 13.313636779785156
    MOY_AN: float = 12.71060562133789


# Define the output model with type annotations
class OptionClassApiOutput(BaseModel):
    prediction: int


# Create input/output pydantic models
input_model = OptionClassApiInput
output_model = OptionClassApiOutput


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    input_df = pd.DataFrame([data.model_dump()])

    # Rename columns to match the expected feature names
    input_df = input_df.rename(columns={
        'nb_frère': 'nb frère',
        'nb_sœur': 'nb sœur',
        'commune_d_origine': "commune d'origine",
        'habite_avec_les_parents': 'Habite avec les parents',
        'electricite': 'electricité',
        'conn_sur_les_options': 'conn sur les options',
        'première_S': '1°S',
        'deuxième_S': '2°S',
        'MOY_AN': 'MOY AN'
    })

    predictions = predict_model(model, data=input_df)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
