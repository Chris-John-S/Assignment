import os

import cv2
import torch
import argparse
import warnings
import numpy as np
import torch.nn as nn
from torch import optim
import torchvision.models as models
from torch.utils.data import SubsetRandomSampler

warnings.filterwarnings('ignore')

categories = {
    "cat": 0,
    "dog": 1
}
labels = ["cat", "dog"]


# Classification Net
class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.basenet = models.mobilenet_v3_small(pretrained=False)
        self.basenet.load_state_dict(torch.load('./image_classification/mobilenetv3_small_pretrained.pth'))
        self.fc = nn.Linear(1000, 2)
        self.init_weight()

    def init_weight(self):
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.zero_()

    def forward(self, x):
        with torch.no_grad():
            x = self.basenet(x)
        x = self.fc(x)
        x = torch.nn.functional.leaky_relu(x)
        return x


# Normalize image
def normalize_mean_variance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)
    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img


# Load training data
def load(filename, img_size=256):
    id = categories[os.path.basename(filename)[:3]]

    img = cv2.imread(filename, 1)
    h, w, _ = img.shape
    if h > w:
        w = int(w * img_size / h)
        h = int(img_size)
    elif w > h:
        h = int(h * img_size / w)
        w = int(img_size)
    else:
        h = 256
        w = 256
    img = normalize_mean_variance(np.array(cv2.resize(img, (w, h))))
    img = np.pad(img, ((0, 256 - h), (0, 256 - w), (0, 0)), mode='constant', constant_values=(0, 0))
    return torch.from_numpy(img.transpose([2, 0, 1])), id


# Load all imgs in the path
def get_imgs(path):
    extensions = [
        "jpg", "jpeg", "JPG", "JPEG",
        "png", "PNG", "bmp", "BMP"
    ]
    data_cnt = 0
    files = []
    for dirname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            abspath = os.path.abspath(dirname + "\\" + filename)
            name, _, extension = abspath.rpartition('.')
            if extensions.count(extension) != 0:
                data_cnt += 1
                files.append(abspath)
    return files, data_cnt


# train with 3 images of two cats and a dog
def train(args):
    files, data_cnt = get_imgs(args.train_dir)
    # ----------------------------Constants------------------------------------
    print(data_cnt)
    dataset_len = data_cnt

    validation_split = 0.1
    n_epoch = 1000
    batch_size = args.batch_size
    batch_size_v = args.batch_size
    img_size = 256
    lr = 0.1 ** args.init_lr
    # ----------------------------initializing model, optimizer, loss functions-----------------
    device = ('cuda:' + args.cuda_num) if torch.cuda.is_available() else 'cpu'
    print(torch.cuda.is_available())

    loss_fn = nn.CrossEntropyLoss()
    model = NET().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ---------------------------loading checkpoint---------------------------------

    if args.save != "None":
        checkpoint = torch.load(args.save, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        del checkpoint
        torch.cuda.empty_cache()

    # ----------------------------learning-----------------------------------
    for epoch in range(n_epoch):
        print('Epoch {}/{}'.format(epoch, n_epoch - 1))
        print('-' * 10)
        # -------------------------dataIndex loader---------------------------------
        with torch.no_grad():
            indices = list(range(dataset_len))
            val_len = int(np.floor(validation_split * dataset_len))
            validation_idx = np.random.choice(indices, size=val_len, replace=False)
            train_idx = list(set(indices) - set(validation_idx))

            train_sampler = SubsetRandomSampler(train_idx)
            validation_sampler = SubsetRandomSampler(validation_idx)

            train_loader = torch.utils.data.DataLoader(indices, sampler=train_sampler, batch_size=batch_size)
            validation_loader = torch.utils.data.DataLoader(indices, sampler=validation_sampler,
                                                            batch_size=batch_size_v)
            data_loaders = {"train": train_loader, "val": validation_loader}
            data_lengths = {"train": len(train_idx), "val": val_len}

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
                print('Training...')
            else:
                model.train(False)
                print('Validating...')
            running_loss = 0.0
            running_accuracy = 0.0

            count = 0
            # ------------------------ mini-batch -----------------------------
            for batch_mask in data_loaders[phase]:
                with torch.no_grad():
                    # ----------------- loading data with label ------------------------------
                    x = torch.zeros((len(batch_mask), 3, img_size, img_size))
                    y = torch.ones(len(batch_mask))
                    cnt = 0
                    for i in batch_mask:
                        x[cnt], y[cnt] = load(files[i.data.numpy()], img_size)
                        cnt += 1
                # ----------------- prediction ----------------------
                y_pred = model(x.to(device))
                y = y.to(device).long()
                accuracy = torch.sum(y_pred.argmax(dim=1) == y) / len(batch_mask)

                # ------------------ loss, backpropagation -------------------
                if phase == 'train':
                    loss = loss_fn(y_pred, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        loss = loss_fn(y_pred, y)

                # ------------------------------ show result ---------------------------
                print('Count: {}\taccuracy: {:.4f}\tloss: {:.4f}'.format(count, accuracy.item(), loss.item()))
                count += 1

                running_loss += torch.Tensor.cpu(loss).data.numpy() * len(batch_mask)
                running_accuracy += torch.Tensor.cpu(accuracy).data.numpy() * len(batch_mask)
                del y_pred
                del y
                del x
                torch.cuda.empty_cache()

            if data_lengths[phase] == 0:
                continue

            # ------------------------------ calc total loss/accuracy -------------------------
            epoch_loss = running_loss / data_lengths[phase]
            epoch_accuracy = running_accuracy / data_lengths[phase]
            print('{}\tAccuracy: {:.4f}\tLoss: {:.4f}'.format(phase, epoch_accuracy, epoch_loss))

            # ---------------------  save model ---------------------
            if epoch == 999:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                    # }, "lastState1.pkl")
                }, 'lastState1_{:.4f}_{:.4f}.pkl'.format(epoch_accuracy, epoch_loss))


# Load image for test/evaluation
def load_img_only(filename, img_size=256):
    img = cv2.imread(filename, 1)
    h, w, _ = img.shape
    if h > w:
        w = int(w * img_size / h)
        h = int(img_size)
    elif w > h:
        h = int(h * img_size / w)
        w = int(img_size)
    else:
        h = 256
        w = 256
    img = normalize_mean_variance(np.array(cv2.resize(img, (w, h))))
    img = np.pad(img, ((0, 256 - h), (0, 256 - w), (0, 0)), mode='constant', constant_values=(0, 0))
    return torch.from_numpy(img.transpose([2, 0, 1])).unsqueeze(0)


# Classify images
def classify(img_path, model_path="./image_classification/lastState1_1.0000_0.0006.pkl"):
    extensions = [
        "jpg", "JPG", "png", "PNG",
        "bmp", "BMP", "jpeg", "JPEG"
    ]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model from checkpoint
    model = NET().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if os.path.isfile(img_path):
        name, _, extension = img_path.rpartition('.')
        if extensions.count(extension) != 0:
            data = load_img_only(img_path, 256)
            res = model(data.to(device))
            idx = res.argmax(axis=1).cpu().squeeze().data.numpy()
            print(img_path, "\t", labels[idx])

    elif os.path.isdir(img_path):
        for dirname, dirnames, filenames in os.walk(img_path):
            for filename in filenames:
                abspath = os.path.abspath(dirname + "\\" + filename)
                name, _, extension = abspath.rpartition('.')
                if extensions.count(extension) != 0:
                    data = load_img_only(abspath, 256)
                    res = model(data.to(device))
                    idx = res.argmax(axis=1).cpu().squeeze().data.numpy()
                    print(abspath, "\t", labels[idx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", "-td", type=str, default="train_data")
    parser.add_argument("--save", "-s", type=str, default="None")
    parser.add_argument("--batch-size", "-bs", type=int, default=3)
    parser.add_argument("--cuda-num", "-cn", type=str, default="0")
    parser.add_argument("--init-lr", "-lr", type=int, default=4)
    parser.add_argument("--multi-step", "-ms", type=list, default=[5, 10, 100])

    args = parser.parse_args()
    # train
    # train(args)

    # Test trained model
    classify("./test")
