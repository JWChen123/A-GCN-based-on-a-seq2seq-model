import torch
from model import DecoderModel1, newModel
from train import generate_input_long_history, generate_queue
from train import RnnParameterData
import torch.nn as nn
import torch.optim as optim
import math
import json
import time
import numpy as np
import argparse
import os


def run(args):
    parameters = RnnParameterData(
        loc_emb_size=args.loc_emb_size,
        hidden_size=args.hidden_size,
        dropout_p=args.dropout_p,
        data_name=args.data_name,
        lr=args.learning_rate,
        lr_step=args.lr_step,
        lr_decay=args.lr_decay,
        L2=args.L2,
        optim=args.optim,
        clip=args.clip,
        epoch_max=args.epoch_max,
        data_path=args.data_path,
        save_path=args.save_path)
    candidate = parameters.data_neural.keys()
    print('prepare the data')
    if args.data_name == 'foursquare_2012':
        data_train, train_idx, adj_train, loc_train = generate_input_long_history(
            parameters.data_neural, 'train', candidate)
        data_test, test_idx, adj_test, loc_test = generate_input_long_history(
            parameters.data_neural, 'test', candidate)
    elif args.data_name == 'datagowalla':
        data_train, train_idx, adj_train, loc_train = generate_input_long_history(
            parameters.data_neural, 'train', candidate)
        data_test, test_idx, adj_test, loc_test = generate_input_long_history(
            parameters.data_neural, 'test', candidate)

    print('set the parameters')
    # initial model

    # Encoder
    encoder = newModel(parameters)
    # Decoder
    if args.attn_state is True:
        decoder = DecoderModel(parameters)
    else:
        decoder = DecoderModel1(parameters)

    # Move models to GPU
    if args.USE_CUDA:
        encoder.cuda()
        decoder.cuda()
    SAVE_PATH = args.save_path
    try:
        os.mkdir(SAVE_PATH)
    except FileExistsError:
        pass
    # 度量标准
    metrics = {
        'train_loss': [],
        'valid_loss': [],
        'ppl': [],
        'accuracy': [],
        'accuracy_top5': []
    }
    lr = parameters.lr  # 学习速率

    # Initialize optimizers and criterion

    encoder_optimizer = optim.Adam(
        encoder.parameters(), lr=parameters.lr, weight_decay=parameters.L2)
    decoder_optimizer = optim.Adam(
        decoder.parameters(), lr=lr, weight_decay=parameters.L2)
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(
        encoder_optimizer,
        'max',
        patience=parameters.lr_step,
        factor=parameters.lr_decay,
        threshold=1e-3)  # 动态学习率
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(
        decoder_optimizer,
        'max',
        patience=parameters.lr_step,
        factor=parameters.lr_decay,
        threshold=1e-3)  # 动态学习率

    criterion = nn.NLLLoss().cuda()

    print('begin the train')
    if args.pretrain == 0:
        # Keep track of time elapsed and running averages
        start = time.time()
        for epoch in range(1, args.epoch_max + 1):
            # Run the train function
            loss, encoder, decoder = run_new(
                args, data_train, adj_train, loc_train, train_idx, 'train', lr,
                parameters.clip, encoder, decoder, encoder_optimizer,
                decoder_optimizer, criterion)

            print_summary = '%s (%d %d%%) %.4f %g' % (
                time_since(start, epoch / args.epoch_max), epoch,
                epoch / args.epoch_max * 100, loss, lr)
            print(print_summary)
            metrics['train_loss'].append(loss)

            valid_loss, avg_acc, avg_acc_top5, avg_ppl = run_new(
                args, data_test, adj_test, loc_test, test_idx, 'test', lr,
                parameters.clip, encoder, decoder, encoder_optimizer,
                decoder_optimizer, criterion)
            print(valid_loss, avg_acc, avg_acc_top5, avg_ppl)
            metrics['valid_loss'].append(valid_loss)
            metrics['ppl'].append(avg_ppl)
            metrics['accuracy'].append(avg_acc)
            metrics['accuracy_top5'].append(avg_acc_top5)
            # metrics['accuracy_top10'].append(avg_acc_top10)
            save_name_tmp1 = 'ep_' + str(epoch) + 'encoder.m'
            save_name_tmp2 = 'ep_' + str(epoch) + 'decoder.m'
            torch.save(encoder.state_dict(), args.save_path + save_name_tmp1)
            torch.save(decoder.state_dict(), args.save_path + save_name_tmp2)
            scheduler1.step(avg_acc)
            scheduler2.step(avg_acc)
            lr_last = lr
            lr = (encoder_optimizer.param_groups[0]['lr'] +
                  decoder_optimizer.param_groups[0]['lr']) / 2
            if lr_last > lr:
                load_epoch = np.argmax(metrics['accuracy'])
                load_name_tmp1 = 'ep_' + str(load_epoch + 1) + 'encoder.m'
                load_name_tmp2 = 'ep_' + str(load_epoch + 1) + 'decoder.m'
                encoder.load_state_dict(
                    torch.load(args.save_path + load_name_tmp1))
                decoder.load_state_dict(
                    torch.load(args.save_path + load_name_tmp2))
                print('load epoch={} model state'.format(load_epoch + 1))

        metrics_view = {
            'train_loss': [],
            'valid_loss': [],
            'accuracy': [],
            'accuracy_top5': [],
            'ppl': []
        }
        for key in metrics_view:
            metrics_view[key] = metrics[key]
        json.dump(
            {
                'metrics': metrics_view,
                'param': {
                    'hidden_size': parameters.hidden_size,
                    'L2': parameters.L2,
                    'lr': parameters.lr,
                    'loc_emb': parameters.loc_emb_size,
                    'loc_graph_emb': parameters.loc_graph_emb_size,
                    'dropout': parameters.dropout_p,
                    'clip': parameters.clip,
                    'lr_step': parameters.lr_step,
                    'lr_decay': parameters.lr_decay
                }
            },
            fp=open('./results/' + 'tmp_res' + '.txt', 'w'),
            indent=4)


def getacc(decoder_output, target):
    _, topi = decoder_output.data.topk(10)
    acc = np.zeros(3)
    index = topi.view(-1).cpu().numpy()

    if target == index[0] and target > 0:
        acc[0] += 1
    if target in index[:5] and target > 0:
        acc[1] += 1
    if target in index[:10] and target > 0:
        acc[2] += 1
    return acc


def run_new(args,
            data,
            adj_zz,
            loc_tt,
            run_idx,
            mode,
            lr,
            clip,
            model1,
            model2,
            model1_optimizer,
            model2_optimizer,
            criterion,
            mode2=None):
    run_queue = None
    if mode == 'train':
        model1.train(True)
        model2.train(True)
        run_queue = generate_queue(run_idx, 'random', 'train')
    elif mode == 'test':
        model1.train(False)
        model2.train(False)
        run_queue = generate_queue(run_idx, 'normal', 'test')
    total_loss = []
    queue_len = len(run_queue)

    users_acc = {}
    users_ppl = {}
    for _ in range(queue_len):
        acc = np.zeros(3)
        model1_optimizer.zero_grad()
        model2_optimizer.zero_grad()
        loss = 0
        ppl_list = []
        u, i = run_queue.popleft()
        if u not in users_acc:
            users_acc[u] = [0, 0, 0]
            users_ppl[u] = ppl_list
        loc = data[u][i]['loc'].cuda()
        adj = adj_zz[u][i].cuda()
        loc_map = loc_tt[u][i].cuda()
        target = data[u][i]['target'].cuda()
        target_len = target.data.size()[0]
        target = target.reshape(target_len, 1)
        # encoder_outputs,hidden_state
        history, c1_gcn, (h1, c1) = model1(loc, adj, loc_map, target_len)

        # decoder_input,context
        decoder_input = torch.LongTensor([[loc[-target_len]]])
        if args.attn_state is True:
            decoder_context = torch.zeros(1, model2.hidden_size)
            if args.USE_CUDA:
                decoder_input = decoder_input.cuda()
                decoder_context = decoder_context.cuda()
            h2 = h1
            c2 = c1
            for di in range(target_len):
                decoder_output, decoder_context, (h2, c2) = model2(
                    decoder_input, decoder_context, h2, c2, history)
                loss += criterion(decoder_output, target[di])
                if mode == 'test':
                    acc = np.add(acc, getacc(decoder_output,
                                             target[di].item()))
                decoder_input = target[di]
        else:
            decoder_input = decoder_input.cuda()
            # use BLSTM as encoder
            h2 = h1
            c1_tmp1 = torch.add(c1[0].unsqueeze(0), c1_gcn)
            c1_tmp2 = torch.add(c1[1].unsqueeze(0), c1_gcn)
            c1 = torch.cat([c1_tmp1, c1_tmp2], 0)
            for di in range(target_len):
                decoder_output, (h2, c2) = model2(decoder_input, h2, c1)
                loss += criterion(decoder_output, target[di])
                if mode == 'test':
                    acc = np.add(acc, getacc(decoder_output,
                                             target[di].item()))
                    users_ppl[u].append(
                        math.log10(
                            np.exp(
                                criterion(decoder_output, target[di]).item())))
                decoder_input = target[di]

        if mode == 'train':
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model1.parameters(), clip)
            torch.nn.utils.clip_grad_norm_(model2.parameters(), clip)
            model1_optimizer.step()
            model2_optimizer.step()

        if mode == 'test':
            users_acc[u][0] += target_len
            users_acc[u][1] += acc[0]
            users_acc[u][2] += acc[1]
            # users_acc[u][3] += acc[2]
        avgloss = loss.data.item() / target_len
        total_loss.append(avgloss)
    epoch_loss = np.mean(total_loss)
    if mode == 'train':
        return epoch_loss, model1, model2
    elif mode == 'test':
        users_rnn_acc = {}
        users_rnn_acc_top5 = {}
        users_rnn_ppl = {}
        # users_rnn_acc_top10 = {}
        for u in users_acc:
            tmp_acc = users_acc[u][1] / users_acc[u][0]
            top5_acc = users_acc[u][2] / users_acc[u][0]
            # top10_acc = users_acc[u][3] / users_acc[u][0]
            tmp_ppl = np.mean(users_ppl[u])
            users_rnn_acc[u] = tmp_acc
            users_rnn_acc_top5[u] = top5_acc
            users_rnn_ppl[u] = tmp_ppl
            # users_rnn_acc_top10[u] = top10_acc
        avg_acc = np.mean([users_rnn_acc[x] for x in users_rnn_acc])
        avg_ppl = np.mean([users_rnn_ppl[x] for x in users_ppl])

        avg_acc_top5 = np.mean(
            [users_rnn_acc_top5[x] for x in users_rnn_acc_top5])
        # avg_acc_top10 = np.mean(
        #     [users_rnn_acc_top10[x] for x in users_rnn_acc_top10])
        return epoch_loss, avg_acc, avg_acc_top5, avg_ppl


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    np.random.seed(1)
    torch.manual_seed(1)
    parser.add_argument(
        '--loc_emb_size',
        type=int,
        default=300,
        help="location embeddings size")
    parser.add_argument(
        '--loc_graph_emb_size',
        type=int,
        default=100,
        help="location embeddings size")
    parser.add_argument('--USE_CUDA', type=int, default=1)
    parser.add_argument('--pretrain', type=int, default=0)
    parser.add_argument('--bidirectional', type=bool, default=False)
    parser.add_argument('--attn_state', type=bool, default=False)
    parser.add_argument('--hidden_size', type=int, default=300)
    parser.add_argument('--dropout_p', type=float, default=0.6)
    parser.add_argument(
        '--data_name',
        type=str,
        default='foursquare_2012',
        choices=['foursquare', 'datagowalla', 'foursquare_2012'])
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lr_step', type=int, default=2)
    parser.add_argument('--lr_decay', type=float, default=0.2)
    parser.add_argument(
        '--optim', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument(
        '--L2',
        type=float,
        default=1 * 1e-6,
        help=" weight decay (L2 penalty)")
    parser.add_argument('--clip', type=float, default=2.0)
    parser.add_argument('--epoch_max', type=int, default=25)
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument(
        '--save_path', type=str, default='./results/checkpoint/')
    args = parser.parse_args()
    run(args)