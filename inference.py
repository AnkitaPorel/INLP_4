import torch
import torch.nn as nn
import sys

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
    def __init__(self, elmo_model, hidden_dim, num_classes=4):
        super(NewsClassifier, self).__init__()
        self.elmo = elmo_model
        for param in self.elmo.parameters():
            param.requires_grad = False
        self.gru = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.lambda_weights = nn.Parameter(torch.ones(2) / 2)

    def forward(self, x):
        _, elmo_embeds = self.elmo(x)
        weighted_embeds = sum(w * e for w, e in zip(self.lambda_weights, elmo_embeds))
        out, _ = self.gru(weighted_embeds)
        out = self.fc(out[:, -1, :])
        return out

if __name__ == "__main__":
    word2idx = {'<UNK>': 0, '<PAD>': 1}
    with open("vocab.txt", "r") as f:
        for idx, word in enumerate(f.read().splitlines()):
            word2idx[word] = idx
    vocab_size = len(word2idx)

    embedding_dim = 300
    hidden_dim = 256
    elmo = ELMo(vocab_size, embedding_dim, hidden_dim)
    elmo.load_state_dict(torch.load("bilstm.pt"))

    classifier = NewsClassifier(elmo, hidden_dim)
    classifier.load_state_dict(torch.load(sys.argv[1]))
    classifier.eval()

    description = sys.argv[2]
    words = description.split()
    input_ids = [word2idx.get(word.lower(), 0) for word in words]
    input_tensor = torch.tensor([input_ids])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier = classifier.to(device)
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = classifier(input_tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()

    for i, prob in enumerate(probs):
        print(f"class-{i+1} {prob:.1f}")