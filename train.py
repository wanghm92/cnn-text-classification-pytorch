import os
import sys
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import logging, sys
# -------------- Logging ----------------#
program = os.path.basename(sys.argv[0])
L = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

# -------------- End Logging ------------#

def train(train_set, data_set, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps, last_step = 0, 0
    best_acc, best_f1, best_recall = 0.0, 0.0, 0.0
    epoch_acc, epoch_f1, epoch_recall = 0.0, 0.0, 0.0

    # Sets the module in training mode.
    model.train()
    for epoch in range(1, args.epochs+1):
        for i in range(train_set.get_num_batch()):
            _, feature, target = train_set.next_batch() # shuffled for training
            feature = Variable(torch.from_numpy(feature))
            target = Variable(torch.from_numpy(target))

            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)

            '''
                Calling .backward() mutiple times accumulates the gradient (by addition) for each parameter. 
                This is why you should call optimizer.zero_grad() after each .step() call.
                Second call is only possible after you have performed another forward pass.
            '''

            loss = F.cross_entropy(logit, target)
            loss.backward()

            '''
                performs a parameter update based on the current gradient 
                (stored in .grad attribute of a parameter) and the update rule
            '''
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/train_set.batch_size
                sys.stdout.write('\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(
                    steps, loss.data[0], accuracy, corrects,train_set.batch_size))
                sys.stdout.flush()
            if steps % args.test_interval == 0:
                dev_f1, dev_recall, dev_prec, dev_acc = eval(data_set, model, args)
                if dev_recall > best_recall:
                    best_recall = dev_recall
                    print('*** BEST RECALL = {:.3f}% *** Precision: {:.3f}% Acc: {:.3f}%'.
                          format(best_recall, dev_prec, dev_acc))
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best_recall', steps)
                if dev_f1 > best_f1:
                    best_f1 = dev_f1
                    print('*** BEST F1 Score = {:.3f} *** Precision: {:.3f}% Recall: {:.3f}% Acc: {:.3f}%'
                          .format(best_f1, dev_prec, dev_recall, dev_acc))
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best_f1', steps)
                if steps - last_step >= args.early_stop:
                    print('early stop by {} steps.'.format(args.early_stop))
                print('\n')
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)

        L.info('Finished Training epoch ${}'.format(epoch))
        L.info('Evaluating on validation set ...')
        f1, recall, prec, acc = eval(data_set, model, args)
        print('F1: {:.3f}% Recall: {:.3f}% Precision: {:.3f}% Acc: {:.3f}%'.format(f1, recall, prec, acc))
        if f1 > epoch_f1:
            epoch_f1 = f1
            print('+++ BEST EPOCH F1 +++')
            save(model, args.save_dir, 'best_epoch_f1', epoch, middle_str='epoch')
        if recall > epoch_recall:
            epoch_recall = recall
            print('+++ BEST EPOCH Recall +++')
            save(model, args.save_dir, 'best_epoch_recall', epoch, middle_str='epoch')

        print('********** Best Recall during this epoch = {:.3f}% **********'.format(best_recall))
        print('********** Best F1 Score during this epoch = {:.3f}% **********'.format(best_f1))

def eval(data_set, model, args):
    model.eval()
    corrects, avg_loss, positives, true_pos, relevant = 0, 0, 0, 0, 0
    # for batch in data_iter:
    for i in range(data_set.get_num_batch()):
        _, feature, target = data_set.get_batch(i)
        feature = Variable(torch.from_numpy(feature))
        target = Variable(torch.from_numpy(target))

        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)

        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]

        '''
            max: return a tuple(data, indices)
            view: Returns a new tensor with the same data as the self tensor but of a different size.
            Variable: A PyTorch Variable is a wrapper around a PyTorch Tensor, and represents a node in a computational graph.
                      If x is a Variable then x.data is a Tensor giving its value,
                      x.grad is another Variable holding the gradient of x with respect to some scalar value
        '''
        selected = torch.max(logit, 1)[1].view(target.size()).data
        corrects += torch.eq(selected, target.data).sum()
        positives += selected.sum()
        true_pos += (selected * target.data).sum()
        relevant += target.data.sum()

    precision = 100.0*true_pos/positives
    recall = 100.0*true_pos/relevant
    f1_score = 0.02*precision*recall/(precision+recall)
    size = data_set.get_number_of_samples()
    avg_loss /= size
    acc = 100.0 * corrects/size
    # print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, accuracy, corrects, size))
    print('\nEval--loss: {:.6f} F1: {:.3f} Precision: {:.3f}%({}/{}) Recall: {:.3f}%({}/{}) Acc: {:.3f}%({}/{}))'.
          format(avg_loss, f1_score, precision, true_pos, positives, recall, true_pos, relevant, acc, corrects, size))
    sys.stdout.flush()
    return f1_score, recall, precision, acc

def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = text_field.tensor_type(text)
    x = autograd.Variable(x, volatile=True)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    #return label_feild.vocab.itos[predicted.data[0][0]+1]
    return label_feild.vocab.itos[predicted.data[0]+1]


def save(model, save_dir, save_prefix, steps, middle_str='steps'):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_{}_{}.pt'.format(save_prefix, middle_str, steps)
    torch.save(model.state_dict(), save_path)