import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import random
from collections import Counter
import torch
import time
from Resnet18 import count_model_parameters
import numpy as np

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        """
        Gaussian Noise 추가
        :param mean: 평균
        :param std: 표준편차
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Tensor에 Gaussian Noise 추가
        :param tensor: 입력 이미지 텐서
        :return: Gaussian Noise가 추가된 텐서
        """
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"



### 로컬에 있는 이미지 파일을 데이터의 형태로 불러오기
class PotatoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: 상위 디렉토리 경로 (예: 'Potato_Leaf_Disease_in_uncontrolled_env/')
        transform: torchvision.transforms 객체
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = [] 
        self.class_to_idx = {} # 딕셔너리로 해당 클래스의 index를 넣어줄 것임.

        # 클래스 디렉토리 가져오기
        classes = sorted(os.listdir(root_dir)) # Potato_Leaf_Disease_Dataset_in_Uncontrolled_Environment 안에 있는 모든 디렉토리 이름을 리스트로 반환함.
        print("Root Directory 확인:", root_dir)
        assert os.path.exists(root_dir), "Root Directory가 존재하지 않습니다!"

        for idx, class_name in enumerate(classes):
            self.class_to_idx[class_name] = idx
            class_dir = os.path.join(root_dir, class_name) # PotatoPlants/Potato__Early_blight와 같이 경로가 합쳐짐.
            print(str(class_dir))
            for fname in os.listdir(class_dir): # 이 class_dir에 있는 모든 파일들의 .png, .jpg, .jpeg로 끝나는 이미지 파일들을 리스트로 변경함.
                if fname.endswith(('.png', '.JPG', '.jpeg', '.jpg')):
                    self.data.append(os.path.join(class_dir, fname)) # 이미지의 전체 경로를 data리스트에 추가
                    self.labels.append(idx) # 이미지의 index를 labels 리스트에 추가.

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')  # RGB로 변환
        image = self.transform(image)
        # 레이블 가져오기
        label = self.labels[idx]
        return image, label
    
    
def augment_class(dataset, target_class=0, target_count=7000, transform=None, output_dir = ''):
    """
    Class 2(건강한 식물) 데이터를 증강하여 target_count(1000개)에 맞춥니다.
    dataset: PotatoDataset 객체
    target_class: 증강할 클래스 (기본값 0)
    target_count: 목표 샘플 개수
    transform: 증강 변환
    """

    # Class 2의 데이터 필터링
    class_indices = [i for i, label in enumerate(dataset.labels) if label == target_class]
    class_images = [dataset.data[i] for i in class_indices]
    
    # 부족한 개수 계산
    required_count = target_count - len(class_images)
    if required_count <= 0:
        print(f"Class {target_class} 이미 {target_count}개 이상입니다. 증강이 필요 없습니다.")
        return dataset

    # 증강된 이미지와 라벨 저장
    augmented_images = []
    augmented_labels = []
    to_pil = ToPILImage()

    for _ in range(required_count):
        original_image_path = random.choice(class_images)  # 기존 이미지 중 하나 선택
        image = Image.open(original_image_path).convert('RGB')  # 이미지 열기
        if transform:
            augmented_image = transform(image)  # 증강 적용
            augmented_image = to_pil(augmented_image)

        save_path = os.path.join(output_dir, f'augmented3_{_}.jpg')
        augmented_image.save(save_path)  # 로컬에 저장

    # 기존 데이터에 증강 데이터 추가
    dataset.data.extend(augmented_images)
    dataset.labels.extend(augmented_labels)

    return dataset
    
    

### Student 모델은 아주 작은 모델
class Light_S_Model(nn.Module):
    def __init__(self):
        super(Light_S_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 15, kernel_size=3, padding='same') # 224
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2) # 112
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(15)

        self.conv2 = nn.Conv2d(15, 32, kernel_size=3, padding='same') # 56
        self.batchnorm2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding='same') # 28
        self.batchnorm3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding='same') # 14
        self.batchnorm4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding='same') # 7
        self.batchnorm5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 100, kernel_size=1, padding='same')
        self.batchnorm6 = nn.BatchNorm2d(100)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(100*7*7, 100)
        self.batchnorm_fc1 = nn.BatchNorm1d(100)

        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(100, 7)
        
    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.batchnorm1(x)
        # print(x.shape)
        x = self.conv2(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.batchnorm2(x)

        x = self.conv3(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.batchnorm3(x)

        x = self.conv4(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.batchnorm4(x)

        x = self.conv5(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.batchnorm5(x)

        x = self.conv6(x)
        x = self.batchnorm6(x)

        x = self.flatten(x)
        # print(x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.batchnorm_fc1(x)

        # print(x.shape)
        x = self.fc2(x)
        # print(x.shape)
        return x

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 사용 가능한 gpu가 있으면 cuda로, 아니면 cpu로.
    
    image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지 크기 조정 (원래 256 x 256)인데 (128 x 128)로 다운샘플링도 할지말지 고민중.
    # 이 밑에는 Data Augmentation 기본 기법들 추가는 나중에 할듯.
    transforms.RandomHorizontalFlip(), # 랜덤 수평으로 뒤집는다.
    transforms.RandomRotation(10), # 랜덤으로 +- 10도를 회전시킴
    # transforms.ColorJitter(brightness = 0.05, contrast=0.05), # 밝기와 대비를 5%내로 랜덤 조정해줌.  
    transforms.ToTensor(),  # 텐서로 변환
    AddGaussianNoise(mean=0.0, std=0.05),  # Gaussian Noise 추가 (std는 조정 가능)
    ])
    
    dataset = PotatoDataset(root_dir = 'Potato_Leaf_Disease_in_uncontrolled_env', transform=image_transform)

    balanced_dataset = augment_class(dataset, target_class=0, target_count=800, transform=image_transform, output_dir='Potato_Leaf_Disease_in_uncontrolled_env/Bacteria')
    balanced_dataset = augment_class(dataset, target_class=1, target_count=800, transform=image_transform, output_dir='Potato_Leaf_Disease_in_uncontrolled_env/Fungi')
    balanced_dataset = augment_class(dataset, target_class=2, target_count=800, transform=image_transform, output_dir='Potato_Leaf_Disease_in_uncontrolled_env/Healthy')
    balanced_dataset = augment_class(dataset, target_class=3, target_count=800, transform=image_transform, output_dir='Potato_Leaf_Disease_in_uncontrolled_env/Nematode')
    balanced_dataset = augment_class(dataset, target_class=4, target_count=800, transform=image_transform, output_dir='Potato_Leaf_Disease_in_uncontrolled_env/Pest')
    balanced_dataset = augment_class(dataset, target_class=5, target_count=800, transform=image_transform, output_dir='Potato_Leaf_Disease_in_uncontrolled_env/Phytopthora') 
    balanced_dataset = augment_class(dataset, target_class=6, target_count=800, transform=image_transform, output_dir='Potato_Leaf_Disease_in_uncontrolled_env/Virus')
    
    # 데이터셋 크기
    dataset_size = len(dataset)
    print(dataset_size)
    train_size = int(0.8 * dataset_size)  # 80%를 train으로 사용.
    val_size = int(0.1 * dataset_size) # 10%를 Validation Set으로 사용.
    test_size = dataset_size - train_size - val_size # 10%를 test set으로 사용.
    remain_size = dataset_size - train_size # random_split을 두 번 사용해서 데이터를 나눌 것임.
    
    # 라벨 분포 계산
    label_counts = Counter(dataset.labels)

    # 출력
    print(f"Label Counts: {dict(label_counts)}")
    
    # Train-Test-Validation Split
    train_dataset, remain_dataset = random_split(dataset, [train_size, remain_size])
    val_dataset, test_dataset = random_split(remain_dataset, [val_size, test_size])
    

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True) # batch단위로 나누어 준다.
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False) # 여러 번 성능을 테스트 할 것이기 때문에 Shuffle=False로 해서 항상 같은 데이터를 유지하게 함.
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False) # 위와 같은 이유로 Shuffle=False임.

    for (images, labels) in train_loader: # Train_loader의 한 batch를 보면, 안에 잘 섞인 채로 들어간 것을 볼 수 있다.
        print(images.shape)
        print(labels)
        break 

    '''
    to_pil = transforms.ToPILImage()
    
    fig, axs = plt.subplots(8, 4)
    axs = axs.flatten()

    for i in range(32):
        image = images[i]
        label = labels[i].item()
        axs[i].imshow(to_pil(image))
        axs[i].set_title(f'Label : {label}')
        axs[i].axis('off')
    plt.show()
    ''' 
    
    print(dataset.class_to_idx) # Early_blight : 0, Late_blight : 1, Potato_healthy : 2 임.
    
    model = Light_S_Model().to(device) # 모델을 GPU로 이동 
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001) # 학습률을 0.001로 설정
    criterion = nn.CrossEntropyLoss() # 손실함수는 CrossEntropy로
    losses = []
    accuracys = []

    count_model_parameters(model)
    
    def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, max_patience):
        patience = 0 # patience를 통해서 모델의 val_loss가 더이상 내려가지 않고, 3번 이상 연속으로 증가하면 학습을 조기종료한다.
        best_model = model.state_dict() # 최적의 상태일 때 parameter를 저장하기 위함.
        best_val_loss = float('inf') # 일단 초기에는 무한대로 설정해놓고, 최적의 loss를 계속 바꿔감.
        for epoch in range(epochs):
            model.train() # 학습 모드로 변경
            train_loss = 0 # loss값
            train_correct = 0 # 맞은 개수
            train_total = 0 # 문제 개수
            # 각 epoch마다 위의 세 개를 초기화함.
            for inputs, targets in train_loader: # 각 배치마다씩으로 parameter를 업데이트함.
                inputs, targets = inputs.to(device), targets.to(device)  # 데이터를 GPU로 이동
                optimizer.zero_grad() # 초기 기울기 초기화
                outputs = model(inputs) # forward 진행
                loss = criterion(outputs, targets) # loss값 계산
                loss.backward() # loss를 토대로 가중치 계산
                optimizer.step() # 가중치로 parameter 업데이트 
                train_loss += loss.item() # epoch의 전체 loss 추가
                
                _, predicted = torch.max(outputs, dim=1) # softmax를 통해 나온 값 중 가장 큰게 예측값일 것임.
                train_correct += (predicted == targets).sum().item() # 맞춘거 더하기
                train_total += targets.size(dim=0) # 전체 문제 개수 더하기
                
            train_accuracy = (train_correct / train_total) # 이번 epoch의 acc
            train_loss /= len(train_loader) # 이번 epoch의 loss(평균)
            
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            # validation 진행
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)  # 데이터를 GPU로 이동
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    # 여기서는 acc와 loss만 계산하고, 가중치 업데이트나 이런건 안함.
                    _, predicted = torch.max(outputs, dim=1)
                    val_correct += (predicted == targets).sum().item()
                    val_total += targets.size(dim=0)
                    
            val_accuracy = (val_correct / val_total)
            val_loss /= len(val_loader)
            # 그래프로 그리기 위해서 저장해놓는다.
            losses.append(val_loss)
            accuracys.append(val_accuracy)
            print(f'{epoch+1}s : train_accuracy - {train_accuracy:.3f} train_loss - {train_loss:.3f}, val_accuracy - {val_accuracy:.3f}, val_loss - {val_loss:.3f}')
            
            # 조기종료를 위해 현재까지 최적의 loss와 이번 epoch의 loss 비교
            if val_loss < best_val_loss: 
                best_val_loss = val_loss # 이번이 더 낮다면 업데이트
                patience = 0 # 그리고 patience도 0으로 초기화
                best_model = model.state_dict() # 지금 가중치 저장.
            else: # loss가 증가했다면 patience를 1 증가시키고 
                patience += 1 
                if patience >= max_patience: # 지정해둔 max_patience와 비교해서 같아지면 학습을 종료해버림.
                    break
        
        torch.save(best_model, 'model/light_Student_bestmodel.pth') # 최적의 결과였던 모델 parameter를 저장함.
        
    train_model(model, train_loader, val_loader, optimizer, criterion, epochs=50, max_patience=5)

    test_loss = 0
    test_correct = 0
    test_total = 0
    model.eval() # 모드를 evaluation 모드로 변환.

    start_time = time.time() # test_data predict에 걸리는 시간 계산
    
    with torch.no_grad(): # test_data에 대해서는 파라미터 가중치를 변경할 필요가 없기 때문.
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # 데이터를 GPU로 이동
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, dim=1)
            test_correct += (predicted == targets).sum().item()
            test_total += targets.size(dim=0)

    end_time = time.time()

    print(f'Test Set의 Accuracy : {(test_correct/test_total):.3f}, Test Set의 loss : {(test_loss / len(test_loader)):.3f}') # 

    fig, ax = plt.subplots(1, 2)
    
    ax[0].plot(range(1, len(losses)+1), losses)
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('Validation Loss')
    ax[0].set_title('light_Student_Validation Loss')
    ax[1].plot(range(1, len(losses)+1), accuracys)
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('Validation Acc')
    ax[1].set_title('light_Student_Validation Accuracy')
    plt.show()

    elapsed_time = end_time - start_time
    print(f"Test time: {elapsed_time:.4f} seconds")