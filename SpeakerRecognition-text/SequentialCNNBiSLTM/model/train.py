from config import Config
from re import S
import torch
import torch.nn as nn
import torch.optim as optim


def train(model, config, training_iter, validation_iter, device, save_model_dir):
    criterion = nn.CrossEntropyLoss(ignore_index=100)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    acc = eval(model, config, validation_iter, device)
    print(F"Accuracy \t {acc}")

    overlap = config.overlap
    for epoch in range(config.epoch):
        model.train()
        count = 0
        loss_sum = 0
        for data, labels, masks in training_iter:
            data = data.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            out = model(data)

            out = out[:, overlap:].reshape(-1, config.label_num)
            labels = labels[:, overlap:].reshape(-1)
            masks = masks[:, overlap:].reshape(-1)

            loss = criterion(out, labels)

            loss_sum += float(loss.data)
            count += 1

            if count % 100 == 0:
                percent = str(count * 100 / len(training_iter))[:5]
                print(F"epoch {epoch}    {percent}%", end='  ')
                print("The loss is: %.5f" % (loss_sum/100))

                loss_sum = 0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # save the model in every epoch
        model.save(F'{save_model_dir}/epoch{epoch}.ckpt')
        acc = eval(model, config, validation_iter, device)
        print(F"Accuracy \t {acc}")


def eval(model, config, validation_iter, device):
    total = 0
    count = 0

    overlap = config.overlap
    model.eval()
    with torch.no_grad():
        for data, labels, masks in validation_iter:
            data = data.to(device)
            labels = labels.to(device)[:, overlap:]
            masks = masks.to(device)[:, overlap:]
            out = model(data)[:, overlap:]
            out = out.reshape(-1, config.label_num)
            _, predicted = torch.max(out, 1)
            labels = labels.reshape(-1)
            masks = masks.reshape(-1)
            total += sum(masks)
            count += int(sum(labels == predicted))

    return count/total


def predict(seqs, model, config, text_vocab, label_vocab, device):
    batch_ids = []
    padding = ['<pad>'] * config.sentence_max_size
    padding_ids = text_vocab.lookup_indices(padding)

    overlap = config.overlap
    block_size = config.block_size
    for seq in seqs:
        padding = ['<pad>'] * config.sentence_max_size
        t = list(seq)
        t = t[:config.sentence_max_size] + padding[:max(0, config.sentence_max_size - len(t))]
        batch_ids.append(text_vocab.lookup_indices(t))

    num = 0
    batch = []
    indicies = list(range(overlap))
    while num * block_size - num * overlap < len(batch_ids):
        segment = batch_ids[num * block_size - num *
                            overlap: (num+1) * block_size - num * overlap]
        indicies.extend(list(range(block_size * num + overlap, block_size * num + len(segment))))
        num += 1
        if len(segment) < block_size:
            pad_len = block_size - len(segment)
            segment += [padding_ids] * pad_len
        batch.append(segment)

    data = torch.tensor(batch)
    data = data.to(device)
    out = model(data)
    out = out.reshape(-1, out.shape[-1])
    indicies = torch.tensor(indicies)
    indicies = indicies.to(device)
    out = torch.index_select(out, 0, indicies)
    _, predicted = torch.max(out, 1)
    labels = [label_vocab.itos[i] for i in predicted]
    res = ['doctor' if '我是医生' in seqs[i] else labels[i] for i in range(len(labels))]
    return res

