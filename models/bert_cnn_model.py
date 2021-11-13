import torch
import torch.nn as nn
import torch.nn.functional as F

class BERTCNNSentiment(nn.Module):
    def __init__(self,
                 bert,
                 output_dim,
                 dropout,
                 n_filters,
                 filter_sizes):
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.conv_0 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[0], embedding_dim))
        
        self.conv_1 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[1], embedding_dim))
        
        self.conv_2 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[2], embedding_dim))
        self.conv_3 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[3], embedding_dim))
        self.conv_4 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[4], embedding_dim))
        

        self.fc = nn.Linear(len(filter_sizes) * n_filters, 64)
        self.out = nn.Linear(64, output_dim)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):
        with torch.no_grad():
            embedded = self.bert(text)[0]
        embedded = embedded.unsqueeze(1)
        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))
        conved_3 = F.relu(self.conv_3(embedded).squeeze(3))
        conved_4 = F.relu(self.conv_4(embedded).squeeze(3))

        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        pooled_3 = F.max_pool1d(conved_3, conved_3.shape[2]).squeeze(2)
        pooled_4 = F.max_pool1d(conved_4, conved_4.shape[2]).squeeze(2)

        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2, pooled_3, pooled_4), dim = 1))
        out = self.fc(cat)
        return self.out(out)