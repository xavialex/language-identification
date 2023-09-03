import time
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from datasets import load_from_disk
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data.dataset import random_split
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

max_len = 512

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
        """
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        """
        encoding = vocab(tokenizer(text))
        if len(encoding) < max_len:
            encoding = encoding + [0] * (max_len - len(encoding))
        else:
            encoding = encoding[:max_len]
        #input_ids = encoding['input_ids'].squeeze()
        input_ids = torch.tensor(encoding)
        return {'input_ids': input_ids, 'label': label}

# Set random seed
torch.manual_seed(42)

# Define hyperparameters
batch_size = 16
max_length = 128
#input_dim = 30522  # Size of BERT vocabulary

hidden_dim = 128
output_dim = 4  # Number of classes
num_heads = 4
num_layers = 2
dropout = 0.1
num_epochs = 1
learning_rate = 2e-5
model_save_name = 'classification_model2.pth'

# Load the dataset
data = load_from_disk('hf_dataset')['train']  # Your HF dataset with 'text' and 'label' columns

# Initialize tokenizer and model
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text['sentence'])

tokenizer = get_tokenizer(tokenizer=None)
vocab = build_vocab_from_iterator(yield_tokens(data), specials=['<unk>'])

input_dim = len(vocab)
model = TransformerClassifier(input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout)

# Prepare the data
dataset = CustomDataset(data, tokenizer)
#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#TODO
num_train = int(len(dataset) * 0.95)
split_train_, split_valid_ = random_split(
    dataset, [num_train, len(dataset) - num_train]
)
train_dataloader = DataLoader(
    split_train_, batch_size=batch_size, shuffle=True
)
valid_dataloader = DataLoader(
    split_valid_, batch_size=batch_size, shuffle=True
)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)






def train(dataloader):
    model.train()
    total_loss, total_acc, total_count = 0., 0., 0.
    log_interval = 500
    start_time = time.time()

    for idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        predicted_label = model(input_ids)
        loss = criterion(predicted_label, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        total_loss += loss.item()
        total_acc += (predicted_label.argmax(1) == labels).sum().item()
        total_count += labels.size(0)
        if idx % log_interval == 0 and idx > 0:
            lr = scheduler.get_last_lr()[0]
            cur_loss = total_loss / log_interval
            
            acc = total_acc / total_count
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            print(f"| epoch {epoch:3d} | {idx:5d}/{len(dataloader):5d} batches "
                  f"| loss {cur_loss:5.2f} | accuracy {acc:8.3f} " 
                  f"| lr {lr:02.5f} | ms/batch {ms_per_batch:5.2f}")
            total_loss, total_acc, total_count = 0., 0., 0.
            start_time = time.time()


def evaluate(dataloader):
    model.eval()
    total_loss, total_acc, total_count = 0., 0., 0.

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            predicted_label = model(input_ids)
            loss = criterion(predicted_label, labels)
            total_acc += (predicted_label.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
    return total_acc / total_count


for epoch in range(1, num_epochs + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val
    elapsed_time = time.time() - epoch_start_time
    print("-" * 59)
    print(f"| end of epoch {epoch:3d} | time: {elapsed_time:5.2f}s "
          f"| valid accuracy {accu_val:8.3f}")
    print("-" * 59)



"""
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
"""
# Save the trained model
Path('models/').mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), model_save_name)
print(f"Model saved in {model_save_name}")