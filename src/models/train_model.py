import time
import math
from pathlib import Path

from datasets import load_from_disk
import torch
from torch import nn, Tensor
import torch.optim as optim


from src.data.make_dataset import make_dataset
from src.data.dataloader_utils import make_train_val_split_dataloaders
from src.data.dataloader_utils import make_test_dataloader
from src.data.dataloader_utils import get_tokenizer_vocab


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

        
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float=0.1, max_len: int=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) 
            * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# Set random seed
torch.manual_seed(42)

# Load the dataset
data_dir = "data/hf_dataset"
if Path(data_dir).is_dir():
    hf_dataset = load_from_disk(data_dir)
else:
    hf_dataset = make_dataset(save_path=data_dir)
train_data = hf_dataset['train']
test_data = hf_dataset['test']

# Initialize tokenizer and vocab
tokenizer, vocab = get_tokenizer_vocab(train_data)

# Define hyperparameters
max_len = 512
batch_size = 16
input_dim = len(vocab) 
hidden_dim = 128
output_dim = len(train_data.features['language'].names)  # Number of classes
num_heads = 4
num_layers = 2
dropout = 0.1
num_epochs = 100
learning_rate = 2e-5
model_save_name = 'language_identification_classifier.pth'

model = TransformerClassifier(input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout)

# Prepare the data
train_dataloader, valid_dataloader = make_train_val_split_dataloaders(
    train_data, tokenizer, vocab, max_len, batch_size, split_ratio=0.95)
test_dataloader = make_test_dataloader(test_data, tokenizer, vocab, max_len, batch_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def train(dataloader, epoch):
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
                  f"| lr {lr:02.6f} | ms/batch {ms_per_batch:5.2f}")
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


def main():
    total_accu = None
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        train(train_dataloader, epoch)
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

    print("Checking the results of test dataset: ")
    accu_test = evaluate(test_dataloader)
    print(f"Test accuracy {accu_test:8.3f}")

    if model_save_name is not None:
        Path('models/').mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), f"models/{model_save_name}")
        print(f"Model saved in {model_save_name}")


if __name__ == '__main__':
    main()