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
    model_path = 'models/language_identification_classifier.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    input = []
    samples = [
        "jackson er anerkjent som den mest suksessrike artisten gjennom tidene av guinness world records og er blitt betegnet som kongen av pop", 
        "gróa var den norrøne gro den gammaldanske og groa den gammalsvenske forma av namnet", 
        "efter et par år at have været underofficer blev han fænrik ved"	
        ]
    for text in samples:
        input.append(text_preprocessing(text))
    input = torch.stack(input)
    labels = model(input).argmax(1).tolist()
    labels = [LABEL_TO_LANGUAGE.get(label) for label in labels]
    output = {sample: label for sample, label in zip(samples, labels)}
    for sample, label in output.items():
        print(f"Identified language for '{sample}': {label}")


if __name__ == '__main__':
    main()
    