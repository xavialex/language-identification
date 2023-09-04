import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class CustomDataset(Dataset):
    
    def __init__(self, data, tokenizer, vocab, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['sentence']
        label = self.data[idx]['language']
        encoding = self.vocab(self.tokenizer(text))
        if len(encoding) < self.max_len:
            encoding = encoding + [0] * (self.max_len - len(encoding))
        else:
            encoding = encoding[:self.max_len]
        input_ids = torch.tensor(encoding)
        return {'input_ids': input_ids, 'label': label}
    

def yield_tokens(data_iter, tokenizer):
    for text in data_iter:
        yield tokenizer(text['sentence'])
    

def get_tokenizer_vocab(dataset):
    tokenizer = get_tokenizer(tokenizer=None)
    vocab = build_vocab_from_iterator(yield_tokens(dataset, tokenizer), specials=['<unk>'])
    vocab.set_default_index(vocab["<unk>"])
    return tokenizer, vocab



def make_train_val_split_dataloaders(dataset, tokenizer, vocab, max_len, batch_size, split_ratio: float=0.95):
    train_dataset = CustomDataset(dataset, tokenizer, vocab, max_len)
    num_train = int(len(train_dataset) * 0.95)
    split_train_, split_valid_ = random_split(
        train_dataset, [num_train, len(train_dataset) - num_train])
    train_dataloader = DataLoader(
        split_train_, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(
        split_valid_, batch_size=batch_size, shuffle=True)
    
    return train_dataloader, valid_dataloader


def make_test_dataloader(dataset, tokenizer, vocab, max_len, batch_size):
    test_dataset = CustomDataset(dataset, tokenizer, vocab, max_len)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)
    
    return test_dataloader


if __name__ == '__main__':
    pass
