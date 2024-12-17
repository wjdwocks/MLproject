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


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 사용 가능한 gpu가 있으면 cuda로, 아니면 cpu로.
    
    image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지 크기 조정 (원래 256 x 256)인데 (128 x 128)로 다운샘플링도 할지말지 고민중.
    # 이 밑에는 Data Augmentation 기본 기법들 추가는 나중에 할듯.
    transforms.RandomHorizontalFlip(), # 랜덤 수평으로 뒤집는다.  
    transforms.ToTensor(),  # 텐서로 변환
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
    
    model = models.resnet50(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),  # Dropout 추가, 일단은 0.5로 설정되었는데 0.1~0.3정도는 과적합 방지 효과가 줄어드는데 정보손실 적음, 0.5이상은 과적합 방지 효과는 크지만 학습 성능 안 좋아질 수 있다네 한 번 파인 튜닝 해봐야될듯?
        nn.Linear(512, 7)
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=1e-4) # weight decay 추가, 이 값도 보통 0~0.1로 설정한다고함, 근데 1e-4가 일반적이래
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
        
        torch.save(best_model, 'model/new_Resnet50_regularization_bestmodel.pth') # 최적의 결과였던 모델 parameter를 저장함.
        
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
    ax[0].set_title('Resnet50_regularization_Validation Loss')
    ax[1].plot(range(1, len(losses)+1), accuracys)
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('Validation Acc')
    ax[1].set_title('Resnet50_regularization_Validation Accuracy')
    plt.show()
