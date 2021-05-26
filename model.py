import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np

image_w = 12
image_size = image_w * image_w


class Model:
    def __init__(self, data_path, normalize=True, epochs=100):
        self.optimizer = None
        self.data_path = data_path
        self.normalize = normalize
        self.model = None
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.transform = transforms.Compose([transforms.Resize(image_w),
                                        transforms.CenterCrop(image_w),
                                        transforms.ToTensor()])
        if self.normalize:
            self.transform = transforms.Compose([self.transform, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def create_model(self, num_classes):
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(image_size * 3, 30)
                self.drop1 = nn.Dropout(p=0.2)
                self.fc2 = nn.Linear(30, num_classes)

            def forward(self, x):
                x = x.view(-1, image_size * 3)
                x = torch.sigmoid(self.fc1(x))
                x = x.drop1(x)
                x = F.softmax(self.fc2(x), dim=1)
                return x
        return MyModel()

    def build_and_train(self):
        transform = transforms.Compose([self.transform,
                                        transforms.ColorJitter(brightness=0.1, contrast=(1, 1.2), saturation=0.1, hue=0.05)])

        dataset = datasets.ImageFolder(self.data_path, transform=transform)
        self.labels = list(dataset.class_to_idx.keys())
        train_size = int(len(dataset) * 0.8)
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_ds, batch_size = 32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)

        #  visualization of images
        # images, labels = next(iter(train_loader))
        # imshow(make_grid(images[:32]))
        # plt.show()

        self.model = self.create_model(5)  # create model with 5 classes
        return self.train(train_loader, val_loader)

    def get_accuracy(self, data_loader):
        total = 0
        correct = 0
        self.model.eval()
        for _, data in enumerate(data_loader):
            inputs, labels = data
            with torch.no_grad():
                y_pred = self.model(inputs)
                correct += (torch.argmax(y_pred, dim=1) == labels).type(torch.FloatTensor).sum().item()
                total += labels.size(0)
        return correct / total

    def train(self, train_loader, val_loader):
        train_accuracy = np.zeros((self.epochs,))
        valid_accuracy = np.zeros((self.epochs,))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.002)

        for epoch in range(self.epochs):  # loop over the dataset multiple times
            self.model.train()
            for i, data in enumerate(train_loader):
                # get the inputs
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            valid_accuracy[epoch] = self.get_accuracy(val_loader)
            train_accuracy[epoch] = self.get_accuracy(train_loader)
            print('[%d epoch] train acc: %.3f, validation acc: %.3f' % (
                epoch + 1, train_accuracy[epoch], valid_accuracy[epoch]))
        return train_accuracy, valid_accuracy

    def predict(self, image_path):
        self.model.eval()
        with torch.no_grad():
            pil_img = Image.open(image_path).convert('RGB')
        img = self.transform(pil_img)
        prob = self.model(img)
        prob = prob.detach().numpy()
        return {self.labels[i]: prob[0, i] for i in range(prob.size)}


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(20, 15))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


if __name__ == '__main__':
    model = Model("static/data", epochs=50)
    model.build_and_train()
    print(model.predict('data/crying/img_8.png'))
    print(model.labels)

