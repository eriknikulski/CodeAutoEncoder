from __future__ import unicode_literals, print_function, division
import argparse
import time
import math

from comet_ml import Experiment
import torch
import torch.nn as nn
from torch import optim

import const
import keys
import loader
import model
import pad_collate


parser = argparse.ArgumentParser(description='ML model for sequence to sequence translation')
parser.add_argument('-d', '--data', choices=['java', 'synth'], help='The data to be used.')


def print_time(prefix=''):
    def wrapper(func):
        def inner(*args, **kwargs):
            begin = time.time()
            res = func(*args, **kwargs)
            end = time.time()
            print(prefix + 'time taken: ', int(end - begin), 's')
            return res
        return inner
    return wrapper


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


@print_time()
def train_loop(encoder, decoder, dataloader, loss_fn, encoder_optimizer, decoder_optimizer, experiment, max_length=const.MAX_LENGTH):
    losses = []
    size = len(dataloader.dataset)
    current_batch_size = const.BATCH_SIZE
    encoder.setBatchSize(current_batch_size)
    decoder.setBatchSize(current_batch_size)

    for batch, (inputs, targets, urls) in enumerate(dataloader):
        if len(inputs) < const.BATCH_SIZE:
            current_batch_size = len(inputs)
            encoder.setBatchSize(current_batch_size)
            decoder.setBatchSize(current_batch_size)

        loss = 0

        input_length = inputs[0].size(0)
        target_length = targets[0].size(0)

        encoder_hidden = encoder.initHidden()
        encoder_output, encoder_hidden = encoder(inputs, encoder_hidden)

        decoder_input = torch.tensor([[const.SOS_token] * current_batch_size], device=const.DEVICE)
        decoder_hidden = torch.cat(tuple(el for el in encoder_hidden[0]), dim=1).view(1, current_batch_size, -1)

        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += loss_fn(decoder_output, targets[:,di].flatten())
        loss = loss / target_length

        # Backpropagation
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        experiment.log_metric('batch_loss', loss.item())
        losses.append(loss.item())

        if batch % const.TRAINING_PER_BATCH_PRINT == 0:
            loss, current = loss.item(), batch * const.BATCH_SIZE
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return losses


def test_loop(encoder, decoder, dataloader, loss_fn, experiment, epoch_num):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    current_batch_size = const.BATCH_SIZE_TEST
    encoder.setBatchSize(current_batch_size)
    decoder.setBatchSize(current_batch_size)

    inputs = None

    with torch.no_grad():
        for inputs, targets, urls in dataloader:
            if len(inputs) < const.BATCH_SIZE_TEST:
                current_batch_size = len(inputs)
                encoder.setBatchSize(current_batch_size)
                decoder.setBatchSize(current_batch_size)
            encoder_hidden = encoder.initHidden()

            input_length = inputs[0].size(0)
            target_length = targets[0].size(0)

            loss = 0
            output = []

            _, encoder_hidden = encoder(inputs, encoder_hidden)

            decoder_input = torch.tensor([[const.SOS_token] * current_batch_size], device=const.DEVICE)
            decoder_hidden = torch.cat(tuple(el for el in encoder_hidden[0]), dim=1).view(1, current_batch_size, -1)

            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                output.append(topi)
                loss += loss_fn(decoder_output, targets[:, di].flatten())

            test_loss += loss.item() / target_length
            result = torch.cat(output).view(1, -1, current_batch_size).T
            correct += (result == targets).all(axis=1).sum().item()

    inputs = [' '.join(decoder.lang.seqFromTensor(el.flatten())) for el in inputs[:5]]
    result = [' '.join(decoder.lang.seqFromTensor(el.flatten())) for el in result[:5]]
    experiment.log_text(epoch_num + str(inputs) + ' ===> ' + str(result))

    test_loss /= num_batches
    experiment.log_metric('test_batch_loss', test_loss)
    experiment.log_metric('accuracy', 100*correct)
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss


@print_time('\nTotal ')
def go_train(encoder, decoder, dataloader, test_dataloader, epochs=const.EPOCHS):
    losses_train = []
    losses_test = []
    loss_fn = nn.NLLLoss()
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=const.LEARNING_RATE, momentum=const.MOMENTUM)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=const.LEARNING_RATE, momentum=const.MOMENTUM)

    encoder_scheduler = optim.lr_scheduler.StepLR(encoder_optimizer, step_size=10, gamma=0.1)
    decoder_scheduler = optim.lr_scheduler.StepLR(decoder_optimizer, step_size=10, gamma=0.1)

    experiment = Experiment(
        api_key=keys.COMET_API_KEY,
        project_name="seq2seqtranslation",
        workspace="eriknikulski",
    )

    experiment.log_parameters(const.HYPER_PARAMS)

    with experiment.train():
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            losses_train.extend(train_loop(encoder, decoder, dataloader, loss_fn, encoder_optimizer, decoder_optimizer, experiment))
            with experiment.test():
                losses_test.append(test_loop(encoder, decoder, test_dataloader, loss_fn, experiment, epoch + 1))
            encoder_scheduler.step(losses_test[-1])
            decoder_scheduler.step(losses_test[-1])

            experiment.log_metric('learning_rate_encoder', encoder_optimizer.param_groups[0]['lr'], step=epoch)
            experiment.log_metric('learning_rate_decoder', decoder_optimizer.param_groups[0]['lr'], step=epoch)

    print("Done!")
    print(f'LR: {const.LEARNING_RATE}')


def run(data):
    if data == 'java':
        data_path = const.JAVA_PATH
    else:
        data_path = const.SYNTH_PATH

    train_data = loader.CodeDataset(const.PROJECT_PATH + data_path + 'train/', only_labels=True)
    train_dataloader = loader.DataLoader(train_data, batch_size=const.BATCH_SIZE, shuffle=True,
                                         collate_fn=pad_collate.PadCollate())

    test_data = loader.CodeDataset(const.PROJECT_PATH + data_path + 'test/', only_labels=True)
    test_dataloader = loader.DataLoader(test_data, batch_size=const.BATCH_SIZE, shuffle=True,
                                        collate_fn=pad_collate.PadCollate())

    valid_data = loader.CodeDataset(const.PROJECT_PATH + data_path + 'valid/', only_labels=True)
    valid_dataloader = loader.DataLoader(valid_data, batch_size=const.BATCH_SIZE_TEST, shuffle=True,
                                         collate_fn=pad_collate.PadCollate())

    input_lang, output_lang = train_data.get_langs()

    encoder = model.EncoderRNN(input_lang.n_words, const.HIDDEN_SIZE, const.BATCH_SIZE, input_lang).to(const.DEVICE)
    decoder = model.DecoderRNN(const.BIDIRECTIONAL * const.ENCODER_LAYERS * const.HIDDEN_SIZE,
                               output_lang.n_words, const.BATCH_SIZE, output_lang).to(const.DEVICE)

    go_train(encoder, decoder, train_dataloader, test_dataloader)

    torch.save(encoder.state_dict(), const.ENCODER_PATH)
    torch.save(decoder.state_dict(), const.DECODER_PATH)

    print('saved models')


if __name__ == '__main__':
    args = parser.parse_args()
    run(args.data)
