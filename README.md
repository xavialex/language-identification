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

## Data

The dataset generation logic is in *src/data*. The script *make_dataset.py* takes a dataset from the [HF Hub](https://huggingface.co/datasets/strombergnlp/nordic_langid) with samples in the languages of interest and saves it (or not) for future use. It can be run as a Python module with:

`python -m src.data.make_dataset`

## Training

The model training relies on a simple Transformer Encoder architecture implemented in PyTorch. It'll handle the dataset creation on its own regardless its presence locally. It can be run as a Python module with:

`python -m src.models.train_model`

It'll generate a model in a given location. All the parameters are configured within the *src/models/train_model.py* script.

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

## Conclusions and Next Steps

This is a sample project to showcase a PyTorch implementation from zero in the shortest amount of time. Improvement of the projects results can be tackled targeting every area of the ML lifecycle:
* **Data:** The dataset is small (10k variant instead of 50k) to avoid longer training. Furthermore, additional languages can be added to make the model more broad.
* **Modeling:** The model is created from zero. For text classification tasks, there's a wide variety af pretrained models with great results available to finetune, which would improve model's performance and speed training time (depending on the model of choice). Also, advanced frameworks allow to keep track of experiments, data and model versioning, etc.
* **Inference:** The trained model is used in FastAPI for inference on a GPU (if available). To speed it up, specific Deep Learning (DL) inference engines like Triton may be more suitable. There's also the possibility to perform model quantization, distillation, conversion to other formats (ONNX -> TensorRT for NVIDIA) to reduce even more the inference time.

## References

* [LANGUAGE MODELING WITH NN.TRANSFORMER AND TORCHTEXT](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
* [TEXT CLASSIFICATION WITH THE TORCHTEXT LIBRARY](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)
* [Writing a Transformer Classifier in PyTorch](https://n8henrie.com/2021/08/writing-a-transformer-classifier-in-pytorch/)
