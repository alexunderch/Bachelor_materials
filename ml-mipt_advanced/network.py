import torch
from torch import nn
import torch.nn.functional as F

class ThreeInputsNet(nn.Module):
    def __init__(self, n_tokens, n_cat_features, concat_number_of_features, hid_size = 64):
        super(ThreeInputsNet, self).__init__()
        self.title_emb = nn.Embedding(n_tokens, embedding_dim = hid_size)
        self.title_conv1 = nn.Conv1d(in_channels = hid_size, out_channels = 2 * hid_size, kernel_size = 2, padding = 1)
        self.title_relu1 = nn.ReLU()
        self.title_bn1 = nn.BatchNorm1d(2 * hid_size)
        self.title_amp = nn.AdaptiveMaxPool1d(2)
        # <YOUR CODE HERE>        
        
        self.full_emb = nn.Embedding(num_embeddings = n_tokens, embedding_dim = hid_size)
        self.full_conv1 = nn.Conv1d(in_channels = hid_size, out_channels = 2 * hid_size, kernel_size = 3)
        self.full_relu1 = nn.ReLU()
        self.full_bn1 = nn.BatchNorm1d(2 * hid_size)
        self.full_conv2 = nn.Conv1d(in_channels = 2 * hid_size, out_channels = 2 * hid_size, kernel_size = 3)
        self.full_relu2 = nn.ReLU()
        self.full_bn2 = nn.BatchNorm1d(2 * hid_size)
        self.full_amp = nn.AdaptiveMaxPool1d(2)
        
        
        # <YOUR CODE HERE>
        resid_features = concat_number_of_features - 8 * hid_size
        self.category_in = nn.Linear(n_cat_features, resid_features)
        self.category_bn1 = nn.BatchNorm1d(resid_features)
        self.category_dropout = nn.Dropout(.33)
        self.category_out = nn.Linear(resid_features, resid_features) # <YOUR CODE HERE>


        # Example for the final layers (after the concatenation)
        self.inter_dense = nn.Linear(in_features = concat_number_of_features, out_features = hid_size * 3)
        self.final_dense = nn.Linear(in_features = hid_size * 3, out_features = 1)

    def forward(self, whole_input):
        input1, input2, input3 = whole_input
        title_beg = self.title_emb(input1).permute((0, 2, 1))
        title = self.title_conv1(title_beg)# <YOUR CODE HERE>
        title = self.title_relu1(title)
        title = self.title_bn1(title)
        title = self.title_amp(title)

        full_beg = self.full_emb(input2).permute((0, 2, 1))
        full = self.full_conv1(full_beg)# <YOUR CODE HERE>
        full = self.full_relu1(full)
        full = self.full_bn1(full)
        full = self.full_conv2(full)# <YOUR CODE HERE>
        full = self.full_relu2(full)
        full = self.full_bn2(full)
        full = self.full_amp(full)      
        
        category = self.category_in(input3) # <YOUR CODE HERE>
        category = self.category_bn1(category)
        category = self.category_dropout(category)
        category = self.category_out(category)        
        
        concatenated = torch.cat(
            [title.view(title.size(0), -1),
            full.view(full.size(0), -1),
            category.view(category.size(0), -1)],
            dim = 1)
        
        out = self.inter_dense(concatenated)# <YOUR CODE HERE>
        out = self.final_dense(out)
        
        return out
