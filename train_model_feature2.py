# import dataset_new as dataset
import dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torchvision
import os
import math
import pickle
import time
import matplotlib.pyplot as plt
import platform


class q_LSTM(nn.Module):
    def __init__(self, dictionary, hidden_size, word_dim=150, num_layers=2):
        super(q_LSTM, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dict = dictionary
        self.lstm = nn.LSTM(input_size=word_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.word_vectors = nn.Embedding(len(self.dict.tokens), word_dim).weight.data

    def forward(self, question):
        question_embeds = self.word_vectors[question].to(self.device)  # [batch_size, seq_length, emb_dim]
        lstm_out, _ = self.lstm(question_embeds)  # [batch_size, seq_length, hidden_dim]
        out_tensor = lstm_out.permute(1, 0, 2)[-1]  # [batch_size, hidden_dim]
        return lstm_out.to(self.device), out_tensor.to(self.device)


class CNN(nn.Module):
    def __init__(self, img_dim):
        super(CNN, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.img_dim = math.floor(7) ** 2

    def forward(self, x):
        with torch.cuda.amp.autocast():
            out = self.layer1(x)  # [batch_size, out_chanel, image_size/max_pool,  image_size/max_pool ]
            out = self.layer2(out)  # [batch_size, out_chanel, image_size/max_pool**2,  image_size/max_pool**2]
            out = self.layer3(out)  # [batch_size, out_chanel, image_size/max_pool**3,  image_size/max_pool**3]
            out = self.layer4(out)
            out = self.layer5(out)
            out = out.reshape([out.size(0), out.size(2) * out.size(3), out.size(1)])  # for att 2
            return out.to(self.device)


class fully(nn.Module):
    def __init__(self, dictionary, hidden_size, CNN, class_num, word_dim=150, num_layers=1, img_size=224, drop=0.05):
        super(fully, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lstm = q_LSTM(dictionary, hidden_size, word_dim=word_dim, num_layers=num_layers).to(self.device)
        self.cnn = CNN(img_size).to(self.device)
        self.relu = nn.ReLU()
        img_dim_cnn = self.cnn.img_dim
        self.i_weights = nn.Sequential(nn.Linear(hidden_size + img_dim_cnn, 512),
                                       nn.ReLU(),
                                       nn.Linear(512, 1),
                                       nn.Softmax(dim=1))
        self.i_weights = nn.Sequential(nn.Linear(hidden_size + 256, 512),
                                       nn.ReLU(),
                                       nn.Linear(512, 1),
                                       nn.Softmax(dim=1))
        self.q_weights = nn.Sequential(nn.Linear(hidden_size + 256 * img_dim_cnn, 512),
                                       nn.ReLU(),
                                       nn.Linear(512, 1),
                                       nn.Softmax(dim=1))
        self.fc_img = nn.Sequential(nn.Linear(256, hidden_size),
                                    nn.ReLU())
        self.fc_q = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                  nn.ReLU())
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=drop)
        self.fc = nn.Linear(hidden_size, class_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, i):
        q_out_all, q_out = self.lstm(q)
        i_out = self.cnn(i)  # [batch,img**2, 256(out_chanel)]
        image = i_out.view(i_out.size(0), -1)

        # image attention
        i_expand_dim = [i_out.shape[1],
                        q_out.shape[0],
                        q_out.shape[1]]
        i_q_att = torch.cat((i_out, q_out.expand(i_expand_dim).permute(1, 0, 2)), dim=2)
        i_weights = self.i_weights(i_q_att)  # [batch_size , 49,1]
        i_out = torch.mul(i_weights, i_out)
        i_out = torch.sum(i_out, dim=1)
        i_out = self.fc_img(i_out)  # [batch, hidden_size]

        # question att
        q_expand_dim = [q_out_all.shape[1],
                        image.shape[0],
                        image.shape[1]]
        q_i_att = torch.cat((q_out_all, image.expand(q_expand_dim).permute(1, 0, 2)), dim=2)
        q_weights = self.q_weights(q_i_att)  # [batch_size , filter,1]
        q_out_all = torch.mul(q_weights, q_out_all)
        q_out_all = torch.sum(q_out_all, dim=1)  # [batch, hidden_size]
        q_out_all = self.fc_q(q_out_all)  # [batch_size, hidden_size]

        # fully
        out = torch.mul(q_out_all, i_out).to(self.device)
        out = self.relu(self.linear(self.dropout(out))).to(self.device)
        out = self.fc(self.dropout(out))
        return out.to(self.device)


def accuracy(img_id, question_id, answers_pred, name='train'):
    target = pickle.load(open(f'data/cache/{name}_target_new.pkl', 'rb'))
    acc = 0.
    for i_id, q_id, a_pred in zip(img_id, question_id, answers_pred):
        a_pred = torch.argmax(a_pred).item()
        a_real = target[(q_id.item(), i_id.item())]
        if a_pred in a_real.keys():
            acc += a_real[a_pred]
    return acc / len(img_id)


def train(train_loader, val_loader, dictionary, hidden_size, class_num, epochs, word_dim=150, num_layers=2,
          img_size=224, lr=0.001, drop=0.5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = fully(dictionary, hidden_size, CNN, class_num, word_dim=word_dim, num_layers=num_layers,
                  img_size=img_size, drop=drop).to(device)
    flip_img = torchvision.transforms.RandomHorizontalFlip(p=0.5)

    loss_function = nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
    train_loss = list()
    train_acc = list()
    val_loss = list()
    val_acc = list()
    print('start trainning')
    for epoch in range(epochs):
        print('epoch:', epoch)
        start_time = time.time()
        tmp_train_loss = 0
        tmp_acc = 0
        model.train()
        T = time.time()
        for idx, (images, image_id, questions, question_id, answers) in enumerate(train_loader):
            if images == [0]:
                continue
            images = images.to(device)
            questions = questions.to(device)
            answers = answers.to(device)
            images = flip_img(images)
            answers_pred = model(questions, images).to(device)
            loss = loss_function(answers_pred, answers)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25, norm_type=2)
            optimizer.step()
            tmp_train_loss += loss.item()
            tmp_acc += accuracy(image_id, question_id, answers_pred)

            if idx % 500 == 0:
                print(f'{idx} loss ={tmp_train_loss / (idx + 1)} and acc = {tmp_acc / (idx + 1)}')
                print(f'time for 100 batch:{time.time() - T}')
                T = time.time()
        train_loss.append(tmp_train_loss / len(train_loader))
        train_acc.append(tmp_acc / len(train_loader))

        model.eval()
        with torch.no_grad():
            tmp_test_loss = 0
            tmp_test_acc = 0
            for idx, (images, image_id, questions, question_id, answers) in enumerate(val_loader):
                images = images.to(device)
                questions = questions.to(device)
                answers = answers.to(device)
                answers_pred = model(questions, images).to(device)
                tmp_test_loss += loss_function(answers_pred, answers)
                tmp_test_acc += accuracy(image_id, question_id, answers_pred, 'val')
            val_loss.append(tmp_test_loss / len(val_loader))
            val_acc.append(tmp_test_acc / len(val_loader))
        test_acc = tmp_test_acc / len(val_loader)
        print("Train: Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(epoch + 1, epochs,
                                                                          tmp_train_loss / len(train_loader),
                                                                          tmp_acc / len(train_loader)))
        print("Test: Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(epoch + 1, epochs,
                                                                         tmp_test_loss / len(val_loader),
                                                                         tmp_test_acc / len(val_loader)))
        print(f'total time: {time.time() - start_time}')
        if test_acc >= max(val_acc) and test_acc > 0.47:
            print(
                f"========== Saving epoch {epoch + 1} model with extra attention with flipping extra param with validation "
                f"accuracy = {round(tmp_test_acc / len(val_loader), 5)} ========")
            torch.save(model, os.path.join("weights",
                                           f"vqa_epoch_{epoch + 1}_attention_1280_160_0.002_val_acc={round(test_acc, 5)}.pkl"))
        torch.cuda.empty_cache()
    return train_loss, train_acc, val_loss, val_acc


def plot_graphs(train_accuracy, train_loss, test_accuracy, test_loss):
    x = list(range(1, len(train_accuracy) + 1))
    plt.plot(x, train_accuracy, c="salmon", label="Mini test Accuracy", marker='o')
    plt.plot(x, test_accuracy, c="darkgreen", label="Test Accuracy", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.xticks(x, rotation=45)
    plt.legend()
    plt.title("Test and mini test accuracy along epochs")
    plt.show()

    plt.plot(x, train_loss, c="salmon", label="Mini test Loss", marker='o')
    plt.plot(x, test_loss, c="darkgreen", label="Test Loss", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(x, rotation=45)
    plt.legend()
    plt.title("Test and mini test loss along epochs")
    plt.show()


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    if 'Linux' in platform.platform():
        import resource

        torch.cuda.empty_cache()
        # https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

    batch_size = 160
    hidden_size = 1280
    word_dim = 300
    num_layers = 1
    img_size = 224
    lr = 0.002
    drop = 0.05
    epochs = 25
    num_workers = 12 if 'Linux' in platform.platform() else 0
    class_num = len(pickle.load(open('data/cache/trainval_ans2label.pkl', 'rb')))

    print(
        f' batch size = {batch_size}, hidden size = {hidden_size}, embd dim ={word_dim}, num layers = {num_layers},'
        f' imag size = {img_size}, lr = {lr}, drop = {drop}, mum workers= {num_workers}, epochs = {epochs} ')
    dictionary = dataset.TokenDictionary()
    train_data = dataset.VQAFeatureDataset('train', dictionary=dictionary)
    print('finish prep train')
    val_data = dataset.VQAFeatureDataset('val', dictionary=dictionary)
    print('finish prep val')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    train_loss, train_acc, val_loss, val_acc = train(train_loader, val_loader, dictionary, hidden_size, class_num,
                                                     epochs, word_dim=word_dim, num_layers=num_layers,
                                                     img_size=img_size, lr=lr, drop=drop)
    plot_graphs(train_acc, train_loss, val_acc, val_loss)


if __name__ == '__main__':
    main()
