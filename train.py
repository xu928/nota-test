import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn as nn
import io


def mnist_train(training_epochs=1, batch_size=32):
    USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
    device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
    print("device:", device)

    # MNIST dataset
    mnist_train = dsets.MNIST(root='MNIST_data/',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)

    # dataset loader
    data_loader = DataLoader(dataset=mnist_train,
                                              batch_size=batch_size, # 배치 크기는 100
                                              shuffle=True,
                                              drop_last=True)

    linear = nn.Linear(784, 10, bias=True).to(device)
    criterion = nn.CrossEntropyLoss().to(device) # 내부적으로 소프트맥스 함수를 포함하고 있음.
    optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)

    for epoch in range(training_epochs): # 앞서 training_epochs의 값은 15로 지정함.
        avg_cost = 0
        total_batch = len(data_loader)

        for X, Y in data_loader:
            # 배치 크기가 100이므로 아래의 연산에서 X는 (100, 784)의 텐서가 된다.
            X = X.view(-1, 28 * 28).to(device)
            # 레이블은 원-핫 인코딩이 된 상태가 아니라 0 ~ 9의 정수.
            Y = Y.to(device)

            optimizer.zero_grad()
            hypothesis = linear(X)
            cost = criterion(hypothesis, Y)
            cost.backward()
            optimizer.step()

            avg_cost += cost / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print('Learning finished')
    return linear

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    import json
    imagenet_class_index = json.load(open('class_index.json'))
    model = models.densenet121(pretrained=True)
    model.eval()
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]