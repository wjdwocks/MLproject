import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import random
from collections import Counter
import torch
import time
import torchvision.models as models



# Distillation Loss 정의
class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, targets):
        # Soft Loss: KL Divergence
        soft_loss = self.kl_div(
            torch.log_softmax(student_logits / self.temperature, dim=1),
            torch.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)
        
        # Hard Loss: CrossEntropy
        hard_loss = self.ce_loss(student_logits, targets)
        
        # Combine losses
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


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
        if self.transform:
            image = self.transform(image)
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
    
    

### Student 모델은 아주 작은 모델
class S_Model(nn.Module):
    def __init__(self):
        super(S_Model, self).__init__()
        
        # 첫 번째 블록
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding='same')  # 채널 증가
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding='same')
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x 축소
        self.dropout1 = nn.Dropout(0.2)  # Dropout 추가
        
        # 두 번째 블록
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.batchnorm = nn.BatchNorm2d(128)  # BatchNorm 추가
        self.dropout2 = nn.Dropout(0.3)  # Dropout 비율 증가

        # Global Pooling 및 FC 레이어
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global Pooling
        self.fc1 = nn.Linear(128, 64)  # 채널 크기에 따라 조정
        self.fc2 = nn.Linear(64, 7)  # 최종 클래스 수
        
    def forward(self, x):
        # 첫 번째 블록
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.pooling(x)
        x = self.dropout1(x)

        # 두 번째 블록
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.batchnorm(x)
        x = self.pooling(x)
        x = self.dropout2(x)

        # Global Pooling 및 FC 레이어
        x = self.global_pool(x)  # (batch_size, 128, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten (batch_size, 128)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 사용 가능한 gpu가 있으면 cuda로, 아니면 cpu로.
    
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지 크기 조정 (원래 256 x 256)인데 (128 x 128)로 다운샘플링도 할지말지 고민중.
    # 이 밑에는 Data Augmentation 기본 기법들 추가는 나중에 할듯.
    transforms.RandomHorizontalFlip(), # 랜덤 수평으로 뒤집는다.
    transforms.ToTensor(),  # 텐서로 변환
    ])
    
    dataset = PotatoDataset(root_dir = 'Potato_Leaf_Disease_in_uncontrolled_env', transform=transform)
    
    # 데이터셋 크기
    dataset_size = len(dataset)
    print(dataset_size)
    train_size = int(0.8 * dataset_size)  # 80%를 train으로 사용.
    val_size = int(0.1 * dataset_size) # 10%를 Validation Set으로 사용.
    test_size = dataset_size - train_size - val_size # 10%를 test set으로 사용.
    remain_size = dataset_size - train_size # random_split을 두 번 사용해서 데이터를 나눌 것임.
    
    
    # Train-Test-Validation Split
    train_dataset, remain_dataset = random_split(dataset, [train_size, remain_size])
    val_dataset, test_dataset = random_split(remain_dataset, [val_size, test_size])
    

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # batch단위로 나누어 준다.
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # 여러 번 성능을 테스트 할 것이기 때문에 Shuffle=False로 해서 항상 같은 데이터를 유지하게 함.
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) # 위와 같은 이유로 Shuffle=False임.
    
    print(dataset.class_to_idx) # Early_blight : 0, Late_blight : 1, Potato_healthy : 2 임.
    
    model = S_Model().to(device) # 모델을 GPU로 이동 
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001) # 학습률을 0.001로 설정
    criterion = nn.CrossEntropyLoss() # 손실함수는 CrossEntropy로
    kd_criterion = DistillationLoss()
    losses = []
    accuracys = []

    teacher_model = models.resnet18()
    teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 7)
    teacher_model.load_state_dict(torch.load('model/new_Resnet18_bestmodel.pth'))
    teacher_model = model.to(device)
    teacher_model.eval()

    losses = []
    accuracys = []
    
    def train_model_kd(student_model, teacher_model, train_loader, val_loader, optimizer, criterion, kd_criterion, alpha, temperature, epochs, max_patience):
        teacher_model.eval()  # Teacher 모델은 학습하지 않음
        patience = 0  # Early stopping patience 초기화
        best_model = student_model.state_dict()  # 최적 상태 저장
        best_val_loss = float('inf')  # 초기 val_loss는 무한대
        
        for epoch in range(epochs):
            student_model.train()  # Student 모델 학습 모드
            train_loss = 0  # 학습 손실 초기화
            train_correct = 0  # 학습 정확도 초기화
            train_total = 0  # 총 데이터 수 초기화
            
            # 학습 루프
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)  # 데이터를 GPU로 이동
                optimizer.zero_grad()  # 옵티마이저 초기화
                
                # Student 및 Teacher 모델 출력
                student_outputs = student_model(inputs)
                teacher_outputs = teacher_model(inputs).detach()  # Teacher 출력은 고정
                
                # CrossEntropy Loss (Student vs Ground Truth)
                ce_loss = criterion(student_outputs, targets)
                
                # Knowledge Distillation Loss (Student vs Teacher)
                kd_loss = kd_criterion(
                    torch.log_softmax(student_outputs / temperature, dim=1),
                    torch.softmax(teacher_outputs / temperature, dim=1),
                    targets = targets
                ) * (temperature ** 2)  # Temperature 조정
                
                # Total Loss
                loss = alpha * kd_loss + (1 - alpha) * ce_loss
                
                # Backward & Optimization
                loss.backward()
                optimizer.step()
                
                # Loss 및 Accuracy 계산
                train_loss += loss.item()
                _, predicted = torch.max(student_outputs, dim=1)
                train_correct += (predicted == targets).sum().item()
                train_total += targets.size(dim=0)
            
            train_accuracy = train_correct / train_total  # 학습 정확도
            train_loss /= len(train_loader)  # 평균 학습 손실
            
            # Validation Loop
            student_model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    student_outputs = student_model(inputs)
                    loss = criterion(student_outputs, targets)  # Validation은 CE Loss로 평가
                    val_loss += loss.item()
                    _, predicted = torch.max(student_outputs, dim=1)
                    val_correct += (predicted == targets).sum().item()
                    val_total += targets.size(dim=0)
            
            val_accuracy = val_correct / val_total  # 검증 정확도
            val_loss /= len(val_loader)  # 평균 검증 손실
            losses.append(val_loss)
            accuracys.append(val_accuracy)
            print(f'{epoch+1}s : train_accuracy - {train_accuracy:.3f} train_loss - {train_loss:.3f}, val_accuracy - {val_accuracy:.3f}, val_loss - {val_loss:.3f}')
            
            # Early Stopping 체크
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                best_model = student_model.state_dict()
            else:
                patience += 1
                if patience >= max_patience:
                    break
        
        # 최적 모델 저장
        torch.save(best_model, 'model/no_aug_KD_Student_a(0.5)_bestmodel.pth')
        
    train_model_kd(
    student_model=model,
    teacher_model=teacher_model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    kd_criterion=kd_criterion,
    alpha=0.5,  # KD Loss 비중
    temperature=4.0,  # Temperature 값
    epochs=50,
    max_patience=10
    )

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
    ax[0].set_title('KD_Student_t=0.5_Validation Loss')
    ax[1].plot(range(1, len(losses)+1), accuracys)
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('Validation Acc')
    ax[1].set_title('KD_Student_t=0.5_Validation Accuracy')
    plt.show()