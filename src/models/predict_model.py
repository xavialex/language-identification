import torch
from torch import Tensor

from src.models.train_model import vocab, tokenizer, model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LABEL_TO_LANGUAGE = {0: 'Other', 
                     1: 'Norwegian',
                     2: 'Danish',
                     3: 'Swedish'}


def text_preprocessing(text: str) -> Tensor:
    encoding = vocab(tokenizer(text))
    max_len = 128

    if len(encoding) < max_len:
        encoding = encoding + [0] * (max_len - len(encoding))
    else:
        encoding = encoding[:max_len]

    return torch.tensor(encoding).to(device)


def main():
    model_path  ='classification_model2.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    text = "This is a sample test"
    encoding = text_preprocessing(text)
    label = model(encoding).argmax(1)
    print(f"Identified language for '{text}': {label}")


if __name__ == '__main__':
    main()
    