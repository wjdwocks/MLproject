import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch
import time
from light_Student import Light_S_Model
import torchvision.models as models
import numpy as np
from Resnet18 import AddGaussianNoise
from student import S_Model

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
    

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 사용 가능한 gpu가 있으면 cuda로, 아니면 cpu로.
    image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지 크기 조정 (원래 256 x 256)인데 (128 x 128)로 다운샘플링도 할지말지 고민중.
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
    
    # Train-Test-Validation Split
    train_dataset, remain_dataset = random_split(dataset, [train_size, remain_size])
    val_dataset, test_dataset = random_split(remain_dataset, [val_size, test_size])
    

    for i in range(5):

        # DataLoader 생성
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True) # 여러 번 성능을 테스트 할 것이기 때문에 Shuffle=False로 해서 항상 같은 데이터를 유지하게 함.

        criterion = nn.CrossEntropyLoss() # 손실함수는 CrossEntropy로

        # 각 모델들의 Test 진행
        S_model = S_Model()
        S_model.load_state_dict(torch.load('model/no_aug_Student_bestmodel.pth'))
        S_model = S_model.to(device)  # 모델을 GPU 또는 CPU로 이동
        S_model.eval()


        S_model_test_time = []
        S_model_test_score = []
        S_model_val_loss = []

        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad(): # test_data에 대해서는 파라미터 가중치를 변경할 필요가 없기 때문.
            for inputs, targets in test_loader:
                start_time = time.time() # test_data predict에 걸리는 시간 계산
                inputs, targets = inputs.to(device), targets.to(device)  # 데이터를 GPU로 이동
                outputs = S_model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs, dim=1)
                test_correct += (predicted == targets).sum().item()
                test_total += targets.size(dim=0)
                end_time = time.time()
                S_model_test_time.append(end_time - start_time)
        S_model_test_score.append(test_correct/test_total)
        S_model_val_loss.append(test_loss / len(test_loader))


        KD_S_model = S_Model()
        KD_S_model.load_state_dict(torch.load('model/no_aug_KD_Student_bestmodel.pth'))
        KD_S_model = KD_S_model.to(device)  # 모델을 GPU 또는 CPU로 이동
        KD_S_model.eval()


        KD_S_model_test_time = []
        KD_S_model_test_score = []
        KD_S_model_val_loss = []

        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad(): # test_data에 대해서는 파라미터 가중치를 변경할 필요가 없기 때문.
            for inputs, targets in test_loader:
                start_time = time.time() # test_data predict에 걸리는 시간 계산
                inputs, targets = inputs.to(device), targets.to(device)  # 데이터를 GPU로 이동
                outputs = KD_S_model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs, dim=1)
                test_correct += (predicted == targets).sum().item()
                test_total += targets.size(dim=0)
                end_time = time.time()
                KD_S_model_test_time.append(end_time - start_time)
        KD_S_model_test_score.append(test_correct/test_total)
        KD_S_model_val_loss.append(test_loss / len(test_loader))

        rn18 = models.resnet18()
        rn18.fc = nn.Linear(rn18.fc.in_features, 7)
        rn18.load_state_dict(torch.load('model/new_Resnet18_bestmodel.pth'))
        rn18 = rn18.to(device)
        rn18.eval()

        rn18_test_time = []
        rn18_test_score = []
        rn18_val_loss = []

        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad(): # test_data에 대해서는 파라미터 가중치를 변경할 필요가 없기 때문.
            for inputs, targets in test_loader:
                start_time = time.time() # test_data predict에 걸리는 시간 계산
                inputs, targets = inputs.to(device), targets.to(device)  # 데이터를 GPU로 이동
                outputs = rn18(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs, dim=1)
                test_correct += (predicted == targets).sum().item()
                test_total += targets.size(dim=0)
                end_time = time.time()
                rn18_test_time.append(end_time - start_time)
        rn18_test_score.append(test_correct/test_total)
        rn18_val_loss.append(test_loss / len(test_loader))

        rn50 = models.resnet50()
        rn50.fc = nn.Linear(rn50.fc.in_features, 7)
        rn50 = rn50.to(device)
        rn50.load_state_dict(torch.load('model/new_Resnet50_bestmodel.pth'))
        rn50.eval()

        rn50_test_time = []
        rn50_test_score = []
        rn50_val_loss = []

        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad(): # test_data에 대해서는 파라미터 가중치를 변경할 필요가 없기 때문.
            for inputs, targets in test_loader:
                start_time = time.time() # test_data predict에 걸리는 시간 계산
                inputs, targets = inputs.to(device), targets.to(device)  # 데이터를 GPU로 이동
                outputs = rn50(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs, dim=1)
                test_correct += (predicted == targets).sum().item()
                test_total += targets.size(dim=0)
                end_time = time.time()
                rn50_test_time.append(end_time - start_time)
        rn50_test_score.append(test_correct/test_total)
        rn50_val_loss.append(test_loss / len(test_loader))


    print(f'Student_model : Test Set의 Accuracy(5번 평균) : {np.mean(S_model_test_score):.3f}, Test Set의 loss(5번 평균) : {np.mean(S_model_val_loss):.3f}') # 
    print(f'Student 가장 오래 걸린 Test_time : {max(S_model_test_time):.3f}, 평균 Test Time : {np.mean(S_model_test_time):.3f}')
    print(f'KD_Student_model : Test Set의 Accuracy(5번 평균) : {np.mean(KD_S_model_test_score):.3f}, Test Set의 loss(5번 평균) : {np.mean(KD_S_model_val_loss):.3f}') # 
    print(f'KD_Student 가장 오래 걸린 Test_time : {max(KD_S_model_test_time):.3f}, 평균 Test Time : {np.mean(KD_S_model_test_time):.3f}')
    print(f'Resnet18 : Test Set의 Accuracy(5회 평균) : {np.mean(rn18_test_score):.3f}, Test Set의 loss(5회 평균) : {np.mean(rn18_val_loss):.3f}')
    print(f'Resnet18 가장 오래 걸린 Test_time : {max(rn18_test_time):.3f}, 평균 Test Time : {np.mean(rn18_test_time):.3f}')      
    print(f'Resnet50 : Test Set의 Accuracy(5회 평균) : {np.mean(rn50_test_score):.3f}, Test Set의 loss(5회 평균) : {np.mean(rn50_val_loss):.3f}')
    print(f'Resnet50 가장 오래 걸린 Test_time : {max(rn50_test_time):.3f}, 평균 Test Time : {np.mean(rn50_test_time):.3f}')