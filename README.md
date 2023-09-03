# Language Identification

This project aims to create a Language Identification system for the following languages: *Danish*, *Swedish*, *Norwegian* or *Other*. The project structure is based on [Cookie Cutter for Data Science](https://drivendata.github.io/cookiecutter-data-science/). The Data has been collected for the HuggingFace (HF) [Dataset Hub](https://huggingface.co/datasets/strombergnlp/nordic_langid). The model trained is a basic Transformer Encoder implemented in PyTorch. A Rest API is created with FastAPI, which launches one endpoint for model inference.

## Running the containerized solution

The service is offered in a docker image that can be built with:

`$ docker build -t language-identification .`

or directly pulled from [Docker Hub]:

`$ docker pull xavialex/language-identification`

Once the image is built, run a container with:

`$ docker run -p 80:80 --rm --gpus all --name lang_id language-identification`

**Note:** This docker image assumes deployment in servers with NVIDIA hardware available.

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

With the service running, it can be tested within the automatically generated documentation in *http://localhost:8000/docs*. The request requires one or several text strings: Once processed, it'll return a JSON response with the structure:

```json
[
    {
        "my first string": "Identified Language",
        "my second string": "Identified Language"
        . . . 
    }
]
```

Requests may be generated with curl:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/language-identification/?texts=mytext1&texts=mytext2' \
  -H 'accept: application/json' \
  -d ''
```


