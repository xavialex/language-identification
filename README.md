# Language Identification

This project aims to create a Language Identification system for the following languages: *Danish*, *Swedish*, *Norwegian* or *Other*. The project structure is based on [Cookie Cutter for Data Science](https://drivendata.github.io/cookiecutter-data-science/). The Data has been collected for the HuggingFace (HF) [Dataset Hub](https://huggingface.co/datasets/strombergnlp/nordic_langid). The model trained is a basic Transformer Encoder implemented in PyTorch. A Rest API is created with FastAPI, which launches one endpoint for model inference.

## Running the containerized solution

The service is offered in a docker image that can be built with:

`$ docker build -t yolos_object_detection .`

Once the images are built, run the solution with:

`$ docker compose up`

**Note:** This docker image assumes deployment in servers with NVIDIA hardware available.

### FastAPI service

Available through *localhost:8000*. The service can be tried through the documentation in *localhost:8000/docs*. There are two endpoints:  
* **train-speech-emotion-recognition-model/:** Trains a model and saves it to a new location (*models/ser_model* by default). If altered, change the model loading in *app/main.py*.  
* **speech-emotion-recognition:** Inference process for the model located in *models/ser_model*. More information in the More information in the [inference section](## Inference process).

**Important note:** To make the *.zip* deliverable manageable, the necessary model needed for the FastAPI service that should be located in */models/ser_model/* is not included. For the service to work, download it from the link provided or use the Jupyter service or the local code to recreate it, since the */models/* volume is sharable across devices.

### Jupyter service

Available through *localhost:8888*. The notebook of interest is in the *notebooks/* folder. It contains the same logic as the rest of the project with explanations and simplifications to showcase certain aspects of the development of the project.

## Running locally

To run or develop the application locally, it's recommended to install the dependencies in a new virtual environment:

1. To create a Python Virtual Environment (venv) to run the code, type:

    ```python3.11 -m venv my-env```

2. Activate the new environment:
    * Windows: ```my-env\Scripts\activate```
    * macOS and Linux: ```source my-env/bin/activate``` 

3. Install all the dependencies from *requirements.txt*:

    ```pip install -r requirements.txt```

After that, run the service with:

`$ uvicorn app.main:app --reload`

## Inference process

With the service running, it can be tested within the automatically generated documentation in *http://localhost:8000/docs*. The request requires one or several WAV audio files. The model will output the detected labels alongside their corresponding confidences. A sample CURL request is shown below:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/speech-emotion-recognition/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'files=@03a01Fa.wav;type=audio/wav' \
  -F 'files=@03a01Nc.wav;type=audio/wav'
```


