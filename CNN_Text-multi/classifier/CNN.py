import torch
import torch.nn as nn
import torch.nn.functional as F 

class CNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):

        super().__init__()

        self.embeds = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels= 1,
                                              out_channels= n_filters,
                                              kernel_size = (fs, embedding_dim))
                                              for fs in filter_sizes
                                    ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)



    def forward(self, text):

        text = text.permute(1, 0)
        embeds = self.embeds(text)
        embeds = embeds.unsqueeze(1)
        conved = [F.relu(conv(embeds)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim = 1))

        return self.fc(cat)

    