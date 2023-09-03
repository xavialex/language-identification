import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from datasets import load_from_disk
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset


# Define the Transformer model for classification
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.embedding(src)
        embedded = self.positional_encoding(embedded)
        encoded = self.transformer_encoder(embedded)
        pooled = encoded.mean(dim=1)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits

# Define PositionalEncoding module
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(1000, d_model)
        position = torch.arange(0, 1000, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# Define custom dataset
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['sentence']
        label = self.data[idx]['language']
        #"""
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        #"""
        #encoding = torch.tensor(vocab(tokenizer(text)), dtype=torch.long)
        input_ids = encoding['input_ids'].squeeze()
        #input_ids = encoding
        return {'input_ids': input_ids, 'label': label}

# Set random seed
torch.manual_seed(42)

# Define hyperparameters
batch_size = 16
max_length = 128
input_dim = 30522  # Size of BERT vocabulary
hidden_dim = 128
output_dim = 4  # Number of classes
num_heads = 4
num_layers = 2
dropout = 0.1
num_epochs = 5
learning_rate = 2e-5

# Load the dataset
data = load_from_disk('hf_dataset')['train']  # Your HF dataset with 'text' and 'label' columns

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text['sentence'])

#tokenizer = get_tokenizer(tokenizer=None)
#vocab = build_vocab_from_iterator(yield_tokens(data), specials=['<unk>'])

model = TransformerClassifier(input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout)


def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

# Prepare the data
dataset = CustomDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'classification_model.pth')