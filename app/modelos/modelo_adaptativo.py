# app/modelos/modelo_adaptativo.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.attention = SpatialAttention()
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.attention(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class ModeloAdaptativoSenas(nn.Module):
    def __init__(self, num_classes: int, tipos_senas: dict = None):
        super(ModeloAdaptativoSenas, self).__init__()
        
        if num_classes < 2:
            raise ValueError(f"Se requieren al menos 2 clases, recibidas {num_classes}")
        
        self.tipos_senas = tipos_senas or {}
        
        self.spatial_features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        self.classifier_estatico = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        
        self.classifier_dinamico = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, tipo_sena='DINAMICA'):
        batch_size, num_frames, c, h, w = x.size()
        x = x.view(batch_size * num_frames, c, h, w)
        x = self.spatial_features(x)
        x = x.view(batch_size, num_frames, -1)
        
        if tipo_sena == 'ESTATICA':
            x_frame_central = x[:, num_frames // 2, :]
            x = self.classifier_estatico(x_frame_central)
        else:
            x, _ = self.lstm(x)
            x = x[:, -1, :]
            x = self.classifier_dinamico(x)
        
        return x
    
    def forward_batch_mixto(self, x, tipos_batch):
        batch_size, num_frames, c, h, w = x.size()
        x = x.view(batch_size * num_frames, c, h, w)
        x = self.spatial_features(x)
        x = x.view(batch_size, num_frames, -1)
        
        outputs = []
        
        for i, tipo in enumerate(tipos_batch):
            x_sample = x[i:i+1]
            
            if tipo == 'ESTATICA':
                x_frame = x_sample[:, num_frames // 2, :]
                out = self.classifier_estatico(x_frame)
            else:
                x_seq, _ = self.lstm(x_sample)
                x_frame = x_seq[:, -1, :]
                out = self.classifier_dinamico(x_frame)
            
            outputs.append(out)
        
        return torch.cat(outputs, dim=0)