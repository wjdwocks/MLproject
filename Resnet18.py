import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import random
from collections import Counter
import torchvision.models as models
import torch
import time
import cv2

def count_model_parameters(model):
    """
    모델의 파라미터 개수를 계산
    :param model: PyTorch 모델
    :return: 학습 가능한 파라미터 개수와 전체 파라미터 개수
    """
    total_params = sum(p.numel() for p in model.parameters())  # 전체 파라미터 개수
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 학습 가능한 파라미터 개수

    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    return total_params, trainable_params

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
        self.labels = [] # 
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
    
    
def augment_class(dataset, target_class=0, target_count=1000, transform=None):
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

    for _ in range(required_count):
        original_image_path = random.choice(class_images)  # 기존 이미지 중 하나 선택
        image = Image.open(original_image_path).convert('RGB')  # 이미지 열기
        if transform:
            augmented_image = transform(image)  # 증강 적용
        augmented_images.append(original_image_path)  # 경로 저장
        augmented_labels.append(target_class)  # 라벨 추가

    # 기존 데이터에 증강 데이터 추가
    dataset.data.extend(augmented_images)
    dataset.labels.extend(augmented_labels)

    return dataset

### Residual Net-18 모델 정의
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # 1번째 Conv Layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # 여기서 만약 2Stage 이상의 첫 번째 블록이었다면 stride = 2였을 것임.
        # 두 번재 블록이었다면 stride는 기본값인 1이었을 것이고.
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        # 2번째 Conv Layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 입력과 출력 채널이 다를 경우 차원을 조정해줌.
        # stride != 1 의 의미 : Stage 2, 3, 4의 첫 번째 블록에서는 stride=2로 설정되므로, 
        # 첫 번째 Residual Block에서 stride = 2인 경우 다운샘플링이 필요하기에 
        # shortcut 경로에서도 같은 stride를 사용하여 출력과 입력의 해상도를 일치시킴

        # in_channels != out_channels 의 의미
        # 각 Stage의 첫 번째 Residual Block은 이전 Stage와 채널 수가 다름.
        # ex) Stage 1에서 out_channels = 64로 끝나면 Stage 2에서는 in_channels = 128 로 시작해야 함.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels: 
            self.shortcut = nn.Sequential( # 그렇기 때문에 입력 채널 수를 kernel_size = 1로 하여 채널 수만 2배로 늘려준다.
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        # 즉, 각 Stage의 첫 번재 블록에서 stride = 2가 되고, 각 특성맵의 크기는 반으로 줄어드는 대신, 채널의 깊이가 2배가 된다.

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x))) # 각 블록의 첫 번째 Conv층
        out = self.bn2(self.conv2(out)) # 두 번째 Conv층
        out += self.shortcut(x) # 단차 더해줌 여기서 Conv를 겪으며 바뀐 특성맵의 크기(해상도) out과 x가 다르면 안되기 때문에 shortcut(x)로 그 특성맵의 크기를 맞춰준다.
        # 여기서 각 Stage의 첫 번째 블록에서는 특성맵의 크기가 다르기 때문에 out += shortcut(x) 가 되고,
        # 두 번째 블록에서는 out += x가 되는 것이다.
        out = self.relu2(out) # 활성화 함수 적용
        return out # 반환
    


class Resnet18(nn.Module):
    def __init__(self, num_classes = 7):
        super(Resnet18, self).__init__()
        # 초기 Conv 레이어 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(2, 2)
        
        # maxPooling도 해상도가 반으로 줄어드니 사용하지 않음.
        self.stage1 = self._make_stage(64, 64, num_blocks = 2, stride=1) 
        self.stage2 = self._make_stage(64, 128, num_blocks = 2, stride=2)
        self.stage3 = self._make_stage(128, 256, num_blocks = 2, stride=2)
        self.stage4 = self._make_stage(256, 512, num_blocks = 2, stride=2)
        self.relu = nn.ReLU()
        # Adaptive Pooling과 FC Layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 출력되는 특성 맵의 목표 크기를 tuple로 넘겨줌.
        self.fc = nn.Linear(512, num_classes)

    # 각 스테이지를 꾸리는 함수
    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 초기 Conv층, BN, ReLU 지나는데 
        # 256, 256으로 시작
        # print(x.shape)
        out = self.relu(self.bn1(self.conv1(x))) # 128, 128
        # print(out.shape)
        out = self.maxpool(out) # 64, 64
        # print(out.shape)

        # 각 Residual Stage를 건넘
        out = self.stage1(out) # 64, 64 - 첫 번째 Residual Stage에서는 Pooling이 적용 안됨.
        # print(out.shape)
        out = self.stage2(out) # 32, 32
        # print(out.shape)
        out = self.stage3(out) # 16, 16
        # print(out.shape)
        out = self.stage4(out) # 8, 8
        # print(out.shape)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # batch단위로 나누어 준다.
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # 여러 번 성능을 테스트 할 것이기 때문에 Shuffle=False로 해서 항상 같은 데이터를 유지하게 함.
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) # 위와 같은 이유로 Shuffle=False임.

    for (images, labels) in train_loader: # Train_loader의 한 batch를 보면, 안에 잘 섞인 채로 들어간 것을 볼 수 있다.
        print(images.shape)
        print(labels)
        break 
    
    print(dataset.class_to_idx) # Early_blight : 0, Late_blight : 1, Potato_healthy : 2 임.
    
    model = models.resnet18(pretrained=False).to(device) # 모델을 GPU로 이동 
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
        
        torch.save(best_model, 'model/Resnet18_bestmodel.pth') # 최적의 결과였던 모델 parameter를 저장함.
        
    train_model(model, train_loader, val_loader, optimizer, criterion, epochs=50, max_patience=10)

    test_loss = 0
    test_correct = 0
    test_total = 0
    model.eval() # 모드를 evaluation 모드로 변환.
    
    with torch.no_grad(): # test_data에 대해서는 파라미터 가중치를 변경할 필요가 없기 때문.
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # 데이터를 GPU로 이동
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, dim=1)
            test_correct += (predicted == targets).sum().item()
            test_total += targets.size(dim=0)
            
    print(f'Test Set의 Accuracy : {(test_correct/test_total):.3f}, Test Set의 loss : {(test_loss / len(test_loader)):.3f}') # 

    fig, ax = plt.subplots(1, 2)
    
    ax[0].plot(range(1, len(losses)+1), losses)
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('Validation Loss')
    ax[0].set_title('Resnet18_Validation Loss')
    ax[1].plot(range(1, len(losses)+1), accuracys)
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('Validation Acc')
    ax[1].set_title('Resnet18_Validation Accuracy')
    plt.show()