import os, argparse, datetime, logging, codecs, sys
import model, train
import torch
from qangaroo_data_reader import QAngarooDataReader
from vocab import Vocab

if __name__ == "__main__":

    #-------------- Logging ----------------#
    program = os.path.basename(sys.argv[0])
    L = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    L.info("Running %s" % ' '.join(sys.argv))
    #-------------- End Logging ------------#

    #-------------- Argparse ----------------#
    parser = argparse.ArgumentParser(description='CNN text classificer')

    # files
    parser.add_argument('-train', dest='train_file', required=True)
    parser.add_argument('-valid', dest='valid_file', default=os.path.abspath('../output/base_cnn/temp.val'))
    parser.add_argument('-test', dest='test_file', default=os.path.abspath('../output/base_cnn/temp.test'))
    parser.add_argument('-output', dest='output_file', default=os.path.abspath('../output/base_cnn/temp.out'))
    parser.add_argument('-pre_emb', dest='pre_emb')
    parser.add_argument('-concat', action='store_true', default=False, help='whether to concat query and document')

    # learning
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size for training [default: 64]')
    parser.add_argument('-log_interval',  type=int, default=10,   help='steps to log training status [default: 1]')
    parser.add_argument('-test_interval', type=int, default=1000, help='steps to wait before testing [default: 100]')
    parser.add_argument('-save_interval', type=int, default=500, help='steps to wait before saving [default:500]')
    parser.add_argument('-save_dir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('-early_stop', type=int, default=1000, help='iterations to stop w/o performance increasing')
    parser.add_argument('-save_best', type=bool, default=True, help='whether to save at best performance')
    parser.add_argument('-is_shuffle', action='store_true', default=True, help='whether to shuffle mini-batches')
    # data
    parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
    # model
    parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    parser.add_argument('-max_norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
    parser.add_argument('-embed-dim', type=int, default=100, help='number of embedding dimension [default: 100]')
    parser.add_argument('-class_num', type=int, default=2, help='number of output classes [default: 100]')
    parser.add_argument('-kernel_num', type=int, default=300, help='number of each kind of kernel')
    parser.add_argument('-kernel_sizes', type=str, default='3,4,5', help='kernel sizes for convolution')
    parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
    # device
    parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1=cpu [default: -1]')
    parser.add_argument('-no_cuda', action='store_true', default=False, help='disable the gpu')
    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
    parser.add_argument('-mode', type=str, default='train', help='train or test')
    args = parser.parse_args()
    # TODO: load and dump args with JSON files

    #-------------- End Argparse ------------#
    # update args
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()
    del args.no_cuda
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # print args
    L.info("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))
    sys.stdout.flush()
    # process file paths
    train_file = os.path.abspath(args.train_file)
    valid_file = os.path.abspath(args.valid_file)
    test_file = os.path.abspath(args.test_file)
    output_file = os.path.abspath(args.output_file)
    # pre_emb_file = os.path.abspath(args.pre_emb)
    pre_emb_file = os.path.abspath(args.pre_emb)
    output_path = os.path.dirname(output_file)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # load data
    L.info("Loading training data ...")
    train_set = QAngarooDataReader(train_file, args)
    L.info("Building vocab ...")
    vocab = Vocab(train_set, emb_path=pre_emb_file)
    L.info("Indexing training samples ...")
    train_set.index_samples(vocab)
    L.info("Making training batches")
    train_set.split_batches()
    print("Total number of training samples = {}".format(train_set.get_number_of_samples()))
    sys.stdout.flush()
    L.info("Loading validation data ...")
    valid_set = QAngarooDataReader(valid_file, args, is_test=True)
    L.info("Indexing validation samples ...")
    valid_set.index_samples(vocab)
    L.info("Making validation batches")
    valid_set.split_batches()
    print("Total number of valid samples = {}".format(valid_set.get_number_of_samples()))

    # create model
    cnn = model.CNN_Text(args, vocab) # TODO: READ

    # load pre-trained model parameters
    if args.snapshot is not None:
        L.info('Loading model from {}...'.format(args.snapshot))
        cnn.load_state_dict(torch.load(args.snapshot)) # TODO: READ

    # assign model to GPU
    if args.cuda:
        # torch.cuda.device(args.device)
        cnn = cnn.cuda()

    # train or predict
    text_field = None
    label_field = None
    test_iter = None
    train_iter = None
    dev_iter = None

    if args.predict is not None:
        label = train.predict(args.predict, cnn, text_field, label_field, args.cuda) # TODO: READ
        print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
    elif args.mode == 'test':
        try:
            train.eval(test_iter, cnn, args) # legacy problem
        except Exception as e:
            print("\nSorry. The test dataset doesn't exist.\n")
    else:
        L.info('Start Training ...')
        try:
            train.train(train_set, valid_set, cnn, args)
        except KeyboardInterrupt:
            print('\n' + '-' * 89)
            L.info('Exiting from training early')


