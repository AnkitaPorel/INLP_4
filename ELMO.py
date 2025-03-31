import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nltk.corpus import brown
import nltk

nltk.download('brown')

class ELMo(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2):
        super(ELMo, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm_layers = nn.ModuleList([
            nn.LSTM(
                embedding_dim if i == 0 else hidden_dim * 2,
                hidden_dim,
                num_layers=1,
                bidirectional=True,
                batch_first=True
            )
            for i in range(num_layers)
        ])
        self.linear = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        layer_outputs = [embeds]
        for i, lstm in enumerate(self.bilstm_layers):
            out, _ = lstm(layer_outputs[-1])
            layer_outputs.append(out)
        final_output = self.linear(out)
        return final_output, layer_outputs[1:]

class BrownDataset(Dataset):
    def __init__(self):
        self.sentences = brown.sents()
        self.vocab = set(brown.words())
        self.word2idx = {word: idx + 2 for idx, word in enumerate(self.vocab)}
        self.word2idx['<UNK>'] = 0
        self.word2idx['<PAD>'] = 1
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        input_ids = [self.word2idx.get(word, 0) for word in sentence]
        return torch.tensor(input_ids)

def collate_fn(batch):
    max_len = max(len(seq) for seq in batch)
    padded_batch = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded_batch[i, :len(seq)] = seq
        padded_batch[i, len(seq):] = 1
    return padded_batch

def train_elmo(model, dataloader, epochs=10, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=1)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            if batch.size(1) < 2:
                continue
            input_seq = batch[:, :-1]
            target_seq = batch[:, 1:]
            output, _ = model(input_seq)
            loss = criterion(output.view(-1, len(dataset.vocab) + 2), target_seq.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")
    torch.save(model.state_dict(), "bilstm.pt")
    print("Model saved as bilstm.pt")

if __name__ == "__main__":
    dataset = BrownDataset()
    with open("vocab.txt", "w") as f:
        f.write("\n".join(dataset.word2idx.keys()))
    print("Vocabulary saved as vocab.txt")
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, collate_fn=collate_fn)
    vocab_size = len(dataset.vocab) + 2
    embedding_dim = 300
    hidden_dim = 256
    model = ELMo(vocab_size, embedding_dim, hidden_dim)
    train_elmo(model, dataloader, epochs=10)