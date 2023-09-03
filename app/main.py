from typing import Annotated

import uvicorn
from fastapi import FastAPI, Query
import torch

from src.models.train_model import model
from src.models.predict_model import text_preprocessing
from src.models.predict_model import LABEL_TO_LANGUAGE


app = FastAPI()
# Model initialization
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path  ='classification_model2.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


@app.post("/language-identification/")
async def language_identification(
        texts: Annotated[list[str],
            Query(description="Strings of characters to be fed to the model")]):
    """Language Identification.
    
    Language Identification inference process. The model has been trained on
    a dataset to identify a text between Danish, Swedish, Norwegian or Other.

    """
    texts_tensor = [] 
    for text in texts:
        texts_tensor.append(text_preprocessing(text))
    texts_tensor = torch.stack(texts_tensor)
    labels = model(texts_tensor).argmax(1).tolist()
    labels = [LABEL_TO_LANGUAGE.get(label) for label in labels]
    output = [{text: label} for text, label in zip(texts, labels)]

    return output


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=5000, reload=True)
