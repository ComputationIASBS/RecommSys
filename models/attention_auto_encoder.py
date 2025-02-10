import torch
import torch.nn as nn

class AttentionAutoEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=64, dropout=0.5, users_features=None):
        super(AttentionAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.dropout = dropout
        self.users_features = users_features

        self.fc1_attention = nn.Linear(users_features.size(1), embedding_dim * 2, bias=True)
        self.fc2_attention = nn.Linear(users_features.size(1), embedding_dim * 2, bias=True)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.alpha = nn.Parameter(torch.Tensor([0.5]))
        self.encoder = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_dim, embedding_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LeakyReLU()
            )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim * 2, input_dim),
            nn.LeakyReLU()
            )

    def forward(self, x):
        encoded = self.encoder(x)
        users_features_1 = self.fc1_attention(self.users_features)
        users_features_2 = self.fc2_attention(self.users_features)
        users_features_1 = users_features_1 / torch.norm(users_features_1,dim=1).reshape(-1,1)
        users_features_2 = users_features_2 / torch.norm(users_features_2,dim=1).reshape(-1,1)
        sim_matrix = users_features_1 @ users_features_2.T
        sim_matrix = torch.softmax(sim_matrix, dim=1)

        encoded_hat = sim_matrix @ encoded


        encoded = self.alpha * encoded_hat + (1-self.alpha) * encoded
        encoded = self.layer_norm(encoded)
        decoded = self.decoder(encoded)

        return decoded