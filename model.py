import torch
import torch.nn as nn
import torch.nn.functional as F
'''
    The activation, dropout, etc. Modules in torch.nn are provided primarily to make it easy to use those operations 
    in an nn.Sequential container. Otherwise it’s simplest to use the functional form for any operations that don’t 
    have trainable or configurable parameters.
'''
from torch.autograd import Variable

class CNN_Text(nn.Module):
    
    def __init__(self, args, vocab):
        super(CNN_Text, self).__init__()
        '''
            args: command-line arguments with model hyper-parameters            
        '''
        self.args = args

        vocab_size = vocab.vocab_size_pretrain
        emb_dim = vocab.emb_dim
        class_num = args.class_num
        input_channels = 1
        output_channels = args.kernel_num
        kernel_sizes = args.kernel_sizes

        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        # initialize with pre-trained embeddings
        if vocab.pretrain_embeddings is not None:
            self.embed.weight = nn.Parameter(torch.from_numpy(vocab.pretrain_embeddings))
            if self.args.static: self.embed.requires_grad = False

        self.convs1 = nn.ModuleList([nn.Conv2d(input_channels, output_channels, (k, emb_dim)) for k in kernel_sizes])

        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(kernel_sizes)*output_channels, class_num) # default: bias=True

        # kaiming initialization
        for conv in self.convs1: nn.init.kaiming_uniform(conv.weight)
        nn.init.kaiming_uniform(self.fc1.weight)

    def forward(self, x):

        x = self.embed(x)  # (batch_size, sequence_length, emb_dim)
        '''
            A PyTorch Variable is a wrapper around a PyTorch Tensor, and represents a node in a computational graph. 
            If x is a Variable then x.data is a Tensor giving its value, 
            and x.grad is another Variable holding the gradient of x with respect to some scalar value.
        '''
        # if self.args.static:
        #     x = Variable(x)

        # Input (N,Cin,Hin,Win) : (batch_size, input_channels=1, sequence_length, emb_dim)
        x = x.unsqueeze(1)

        # Output (N,Cout,Hout,Wout) : (batch_size, output_channels, sequence_length, 1)
        conv_out = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]

        # kernel_size : the size of the window to take a max over
        pool_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conv_out] # (batch_size, output_channels)

        cat_out = torch.cat(pool_out, 1)

        cat_out_dropout = self.dropout(cat_out)  # (batch_size, types_of_kernels*output_channels)

        logit = self.fc1(cat_out_dropout)  # (batch_size, class_num)

        return logit
