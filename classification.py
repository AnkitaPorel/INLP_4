import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

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

class NewsClassifier(nn.Module):
    def __init__(self, elmo_model, hidden_dim, num_classes=4, mode="trainable"):
        super(NewsClassifier, self).__init__()
        self.elmo = elmo_model
        for param in self.elmo.parameters():
            param.requires_grad = False
        self.gru = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.mode = mode

        if mode == "trainable":
            self.lambda_weights = nn.Parameter(torch.ones(2) / 2)
        elif mode == "frozen":
            self.lambda_weights = torch.ones(2) / 2
        elif mode == "learnable":
            self.combiner = nn.Sequential(
                nn.Linear(hidden_dim * 2 * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * 2)
            )

    def forward(self, x):
        _, elmo_embeds = self.elmo(x)
        if self.mode in ["trainable", "frozen"]:
            weighted_embeds = sum(w * e for w, e in zip(self.lambda_weights, elmo_embeds))
            out, _ = self.gru(weighted_embeds)
        elif self.mode == "learnable":
            combined = torch.cat(elmo_embeds, dim=-1)
            combined = self.combiner(combined)
            out, _ = self.gru(combined)
        out = self.fc(out[:, -1, :])
        return out

class AGNewsDataset(Dataset):
    def __init__(self, csv_file, word2idx):
        self.df = pd.read_csv(csv_file)
        self.texts = self.df["Description"].tolist()
        self.labels = self.df["Class Index"].tolist()
        self.word2idx = word2idx

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        words = self.texts[idx].split()
        input_ids = [self.word2idx.get(word.lower(), 0) for word in words]
        return torch.tensor(input_ids), self.labels[idx] - 1

def collate_fn(batch):
    texts, labels = zip(*batch)
    max_len = max(len(seq) for seq in texts)
    padded_texts = torch.zeros(len(texts), max_len, dtype=torch.long)
    for i, seq in enumerate(texts):
        padded_texts[i, :len(seq)] = seq
        padded_texts[i, len(seq):] = 1
    return padded_texts, torch.tensor(labels)

def train_classifier(classifier, dataloader, epochs=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
    classifier = classifier.to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    classifier.train()
    for epoch in range(epochs):
        total_loss = 0
        for texts, labels in dataloader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")
    torch.save(classifier.state_dict(), f"classifier_{classifier.mode}.pt")

def evaluate_classifier(classifier, dataloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    classifier = classifier.to(device)
    classifier.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for texts, labels in dataloader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = classifier(texts)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Mode: {classifier.mode}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Confusion Matrix:\n{cm}\n")
    return accuracy, f1, precision, recall, cm

if __name__ == "__main__":
    word2idx = {'<UNK>': 0, '<PAD>': 1}
    with open("/kaggle/input/my-dataset/vocab.txt", "r") as f:
        for idx, word in enumerate(f.read().splitlines()):
            word2idx[word] = idx
    vocab_size = len(word2idx)

    train_csv = "/kaggle/input/my-dataset/train.csv"
    test_csv = "/kaggle/input/my-dataset/test.csv"
    train_dataset = AGNewsDataset(train_csv, word2idx)
    test_dataset = AGNewsDataset(test_csv, word2idx)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, collate_fn=collate_fn)

    embedding_dim = 300
    hidden_dim = 256
    elmo = ELMo(vocab_size, embedding_dim, hidden_dim)
    elmo.load_state_dict(torch.load("/kaggle/input/my-dataset/bilstm.pt"))

    for mode in ["trainable", "frozen", "learnable"]:
        print(f"\nTraining with {mode} mode")
        classifier = NewsClassifier(elmo, hidden_dim, mode=mode)
        train_classifier(classifier, train_dataloader, epochs=5)
        evaluate_classifier(classifier, test_dataloader)
