import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    """
    기본적인 컨볼루션 신경망 모델
    """

    def __init__(self, num_classes):
        """
        모델의 레이어 초기화

        Args:
            num_classes (int): 출력 레이어의 뉴런 수
        """
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 입력 이미지 텐서

        Returns:
            x (torch.Tensor): num_classes 크기의 출력 텐서
        """
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x


from torchvision.models import vit_b_16


class VITB16Model(nn.Module):
    def __init__(self, num_classes):
        super(VITB16Model, self).__init__()

        # Load the pretrained ViT model
        self.vit = vit_b_16(pretrained=True)

        # Replace the classifier (Assuming 'heads' is the final classifier layer)
        num_features = self.vit.heads[0].in_features  # Adjust this index if needed
        self.vit.heads[0] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.vit(x)
        return x


from torchvision.models import vit_l_16, vit_h_14


class VITL16Model(nn.Module):
    def __init__(self, num_classes):
        super(VITL16Model, self).__init__()
        # Load the pretrained ViT large model
        self.vit = vit_l_16(pretrained=True)
        # Replace the classifier
        num_features = self.vit.heads[0].in_features
        self.vit.heads[0] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.vit(x)
        return x


import torch


class VITH14Model(nn.Module):
    def __init__(self, num_classes):
        super(VITH14Model, self).__init__()
        # Initialize the ViT huge model without pretrained weights
        self.vit = vit_h_14()

        # Manually load the pretrained weights (this is just a placeholder, replace with actual path or method)
        # pretrained_dict = torch.load('path_to_pretrained_weights.pth')
        # model_dict = self.vit.state_dict()
        # Update the model's state dict with the pretrained weights
        # model_dict.update(pretrained_dict)
        # self.vit.load_state_dict(model_dict)

        # Replace the classifier
        num_features = self.vit.heads[0].in_features
        self.vit.heads[0] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.vit(x)
        return x
