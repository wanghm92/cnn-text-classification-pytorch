import torch, sys
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

        self.convs1 = nn.ModuleList([nn.Conv2d(input_channels, output_channels, (k, emb_dim), padding=((k-1)//2, 0))
                                     for k in kernel_sizes])
        self.convs2 = nn.ModuleList([nn.Conv2d(input_channels, output_channels, (k, emb_dim), padding=((k-1)//2, 0))
                                     for k in kernel_sizes])

        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(kernel_sizes)*output_channels*2, class_num) # default: bias=True

        # kaiming initialization
        for conv in self.convs1: nn.init.kaiming_uniform(conv.weight)
        for conv in self.convs2: nn.init.kaiming_uniform(conv.weight)
        nn.init.kaiming_uniform(self.fc1.weight)

    def forward(self, query, doc):

        query = self.embed(query)  # (batch_size, sequence_length, emb_dim)
        doc = self.embed(doc)  # (batch_size, sequence_length, emb_dim)
        '''
            A PyTorch Variable is a wrapper around a PyTorch Tensor, and represents a node in a computational graph. 
            If x is a Variable then x.data is a Tensor giving its value, 
            and x.grad is another Variable holding the gradient of x with respect to some scalar value.
        '''
        # if self.args.static:
        #     x = Variable(x)

        # Input (N,Cin,Hin,Win) : (batch_size, input_channels=1, sequence_length, emb_dim)
        query = query.unsqueeze(1)
        doc = doc.unsqueeze(1)

        # Output (N,Cout,Hout,Wout) : (batch_size, output_channels, sequence_length, 1)
        query_conv_out = [F.relu(conv(query)).squeeze(3) for conv in self.convs1]
        doc_conv_out = [F.relu(conv(doc)).squeeze(3) for conv in self.convs2]

        # temp = query_conv_out[0].permute(0,2,1) * doc_conv_out[0].permute(0,2,1)
        # print(temp)
        # sys.exit(0)

        # kernel_size : the size of the window to take a max over
        query_pool_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in query_conv_out] # (batch_size, output_channels)
        doc_pool_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in doc_conv_out] # (batch_size, output_channels)

        query_cat_out = torch.cat(query_pool_out, 1)
        doc_cat_out = torch.cat(doc_pool_out, 1)

        final_cat_out = torch.cat([query_cat_out, doc_cat_out], 1)

        cat_out_dropout = self.dropout(final_cat_out)  # (batch_size, types_of_kernels*output_channels)

        logit = self.fc1(cat_out_dropout)  # (batch_size, class_num)

        # for i in [query_conv_out, doc_conv_out, query_pool_out, doc_pool_out]:
        #     for x in i:
        #         print(x.size())
        # for i in [query_cat_out, doc_cat_out, final_cat_out, cat_out_dropout, logit]:
        #     print(i.size())

        # sys.exit(0)

        return logit
