
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image
import os
import numpy as np
from torchvision.models import swin_t, Swin_T_Weights  # Swin Transformer
from torchvision.models import densenet121, DenseNet121_Weights  # DenseNet121
from sklearn.metrics import classification_report, roc_curve, auc
import pandas as pd
from scipy import stats
import datetime
import json
import multiprocessing
from sklearn.model_selection import StratifiedKFold, train_test_split
import cv2
# Import pydicom for DICOM file handling
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
import matplotlib.pyplot as plt
from skimage import exposure, filters, feature
import random
import torch.nn.functional as F
# 文字處理相關的庫
from sklearn.preprocessing import LabelEncoder, StandardScaler
import jieba
from gensim.models import Word2Vec
import string
from collections import Counter
import re
from transformers import BertTokenizer, BertModel
# Global normalization parameters (same as ImageNet)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
#MEAN = [0.35066684, 0.31663555, 0.26208879]
#STD = [0.03354678, 0.02275699, 0.01889922]
# DICOM processing functions
def read_dicom_file(path):
    """Read a DICOM file and convert to a format that can be processed by PIL"""
    try:
        # Read the DICOM file
        dicom = pydicom.dcmread(path)
        
        # Get pixel data
        data = dicom.pixel_array
        
        # Apply modality LUT (if needed)
        if hasattr(dicom, 'RescaleIntercept') and hasattr(dicom, 'RescaleSlope'):
            data = apply_modality_lut(data, dicom)
        
        # Apply VOI LUT (if applicable and available)
        if hasattr(dicom, 'WindowCenter') and hasattr(dicom, 'WindowWidth'):
            try:
                data = apply_voi_lut(data, dicom)
            except Exception as e:
                print(f"Warning: Could not apply VOI LUT: {e}")
        
        # Normalize to 0-255 for 8-bit depth
        data_min = data.min()
        data_max = data.max()
        
        if data_max != data_min:  # Avoid division by zero
            data = ((data - data_min) / (data_max - data_min)) * 255.0
        else:
            print(f"Warning: Uniform pixel values in DICOM file: {path}")
        
        data = data.astype(np.uint8)
        
        # Convert to RGB if grayscale
        if len(data.shape) == 2:  # Single channel (grayscale)
            data = np.stack([data, data, data], axis=2)
            
        # Convert to PIL image
        img = Image.fromarray(data)
        return img
    
    except Exception as e:
        print(f"Error reading DICOM file {path}: {e}")
        return None

class AddSpeckleNoise(object):
    def __init__(self, noise_variance=0.05):
        self.noise_variance = noise_variance
    
    def __call__(self, tensor):
        # 假設 tensor 是 [C, H, W] 格式
        noise = torch.randn_like(tensor) * self.noise_variance
        return tensor + noise
class BertTextProcessor:
    def __init__(self, df, route_col='UltrasoundRoute', report_col='REPORT_TEXT', age_col='AGE', 
                bert_model_name='emilyalsentzer/Bio_ClinicalBERT', max_length=128):
        self.df = df
        self.route_col = route_col
        self.report_col = report_col
        self.age_col = age_col
        
        # 標籤編碼器
        self.route_encoder = LabelEncoder()
        
        # 年齡標準化
        self.age_scaler = StandardScaler()
        
        # BERT 相關設定
        self.bert_model_name = bert_model_name
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.model = BertModel.from_pretrained(bert_model_name)
        self.max_length = max_length
        self.bert_dim = self.model.config.hidden_size  # 通常是 768
        
    def fit(self):
        # 處理 UltrasoundRoute
        self.route_encoder.fit(self.df[self.route_col].fillna('Unknown'))
        
        # 處理 AGE
        self.age_scaler.fit(self.df[self.age_col].values.reshape(-1, 1))
        
        # BERT 不需要 fit，預訓練模型已經處理完成
        self.model.eval()  # 設置為評估模式
        
        return self
        
    def transform(self, df):
        # 處理 UltrasoundRoute
        route_encoded = self.route_encoder.transform(df[self.route_col].fillna('Unknown'))
        route_one_hot = pd.get_dummies(route_encoded, prefix='route')
        
        # 處理 AGE
        age_scaled = self.age_scaler.transform(df[self.age_col].values.reshape(-1, 1))
        
        # 處理 REPORT_TEXT 使用 BERT
        report_embeddings = self._get_bert_embeddings(df[self.report_col].fillna(''))
        
        # 合併所有特徵
        num_route_features = len(self.route_encoder.classes_)
        features = np.zeros((len(df), num_route_features + 1 + self.bert_dim))
        
        features[:, :num_route_features] = route_one_hot
        features[:, num_route_features:num_route_features+1] = age_scaled
        features[:, num_route_features+1:] = report_embeddings
        
        return features
    
    def _get_bert_embeddings(self, reports):
        """使用 BERT 獲取文本嵌入向量"""
        embeddings = np.zeros((len(reports), self.bert_dim))
        
        with torch.no_grad():
            for i, report in enumerate(reports):
                # 編碼文本
                encoded_input = self.tokenizer(
                    report, 
                    return_tensors='pt', 
                    max_length=self.max_length, 
                    padding='max_length', 
                    truncation=True
                )
                
                # 獲取 BERT 輸出
                outputs = self.model(**encoded_input)
                
                # 使用 [CLS] token 的嵌入作為文本表示
                # 或者也可以使用所有 token 的平均值
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings[i] = cls_embedding
                
        return embeddings
    
    def get_feature_size(self):
        return len(self.route_encoder.classes_) + 1 + self.bert_dim

def extract_subject_id(filename):
    """從檔案名稱中提取受試者ID"""
    # 假設檔案格式為 P239170000001_1.dcm
    base_name = os.path.splitext(filename)[0]  # 移除 .dcm
    if '_' in base_name:
        return base_name.split('_')[0]  # 返回 P239170000001
    return base_name

def verify_no_subject_leakage(train_indices, val_indices, test_indices, dataset):
    """驗證沒有受試者洩漏"""
    def get_subject_ids_from_indices(indices):
        subject_ids = set()
        for idx in indices:
            img_path = dataset.image_files[idx]
            filename = os.path.basename(img_path)
            subject_id = extract_subject_id(filename)
            subject_ids.add(subject_id)
        return subject_ids
    
    train_subjects = get_subject_ids_from_indices(train_indices)
    val_subjects = get_subject_ids_from_indices(val_indices)
    test_subjects = get_subject_ids_from_indices(test_indices)
    
    # 檢查是否有重疊
    train_val_overlap = train_subjects.intersection(val_subjects)
    train_test_overlap = train_subjects.intersection(test_subjects)
    val_test_overlap = val_subjects.intersection(test_subjects)
    
    if train_val_overlap:
        print(f"警告：訓練集和驗證集有重疊的受試者：{train_val_overlap}")
    if train_test_overlap:
        print(f"警告：訓練集和測試集有重疊的受試者：{train_test_overlap}")
    if val_test_overlap:
        print(f"警告：驗證集和測試集有重疊的受試者：{val_test_overlap}")
    
    if not any([train_val_overlap, train_test_overlap, val_test_overlap]):
        print("✓ 驗證通過：沒有受試者洩漏問題")
    
    return len(train_subjects), len(val_subjects), len(test_subjects)

def subject_level_split(dataset, test_size=0, num_folds=5, random_state=42):
    """基於受試者層級進行資料分割"""
    
    # 提取所有唯一的受試者ID
    subject_ids = []
    subject_labels = []
    subject_to_indices = {}
    
    for idx in range(len(dataset)):
        img_path = dataset.image_files[idx]
        filename = os.path.basename(img_path)
        subject_id = extract_subject_id(filename)
        
        if subject_id not in subject_to_indices:
            subject_to_indices[subject_id] = []
            subject_ids.append(subject_id)
            # 使用該受試者的第一張影像的標籤作為受試者標籤
            subject_labels.append(dataset.labels[idx])
        
        subject_to_indices[subject_id].append(idx)
    
    print(f"總共有 {len(subject_ids)} 個唯一受試者")
    
    # 打印每個受試者的影像數量
    for subject_id in subject_ids[:5]:  # 只顯示前5個作為示例
        print(f"受試者 {subject_id} 有 {len(subject_to_indices[subject_id])} 張影像")
    
    # 基於受試者進行測試集分割
    train_val_subjects, test_subjects = train_test_split(
        subject_ids, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=subject_labels
    )
    
    # 獲取對應的影像索引
    train_val_indices = []
    test_indices = []
    
    for subject_id in train_val_subjects:
        train_val_indices.extend(subject_to_indices[subject_id])
    
    for subject_id in test_subjects:
        test_indices.extend(subject_to_indices[subject_id])
    
    # 獲取對應的標籤
    train_val_labels = [dataset.labels[idx] for idx in train_val_indices]
    
    # 對訓練驗證集中的受試者進行交叉驗證分割
    train_val_subject_labels = [subject_labels[subject_ids.index(sid)] for sid in train_val_subjects]
    
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    
    fold_splits = []
    for fold, (train_subject_indices, val_subject_indices) in enumerate(skf.split(train_val_subjects, train_val_subject_labels)):
        # 獲取訓練和驗證的受試者ID
        train_subjects_fold = [train_val_subjects[i] for i in train_subject_indices]
        val_subjects_fold = [train_val_subjects[i] for i in val_subject_indices]
        
        # 獲取對應的影像索引
        train_indices = []
        val_indices = []
        
        for subject_id in train_subjects_fold:
            train_indices.extend(subject_to_indices[subject_id])
        
        for subject_id in val_subjects_fold:
            val_indices.extend(subject_to_indices[subject_id])
        
        fold_splits.append((train_indices, val_indices))
        
        # 驗證沒有受試者洩漏
        print(f"\nFold {fold+1} 驗證：")
        train_subject_count, val_subject_count, test_subject_count = verify_no_subject_leakage(
            train_indices, val_indices, test_indices, dataset
        )
        print(f"訓練集受試者數：{train_subject_count}, 驗證集受試者數：{val_subject_count}, 測試集受試者數：{test_subject_count}")
    
    return fold_splits, test_indices, train_val_labels

# 混合模型：Swin Transformer + DenseNet
class MultimodalModel(nn.Module):
    def __init__(self, num_classes, swin_weight=0.5, text_features=None):
        super(MultimodalModel, self).__init__()
        # Swin Transformer 分支
        self.swin = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        swin_features = self.swin.head.in_features  # Swin-T has 768 features
        self.swin.head = nn.Identity()  # 移除分類頭
        
        # DenseNet121 分支
        self.densenet = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        densenet_features = self.densenet.classifier.in_features  # DenseNet121 has 1024 features
        self.densenet.classifier = nn.Identity()  # 移除分類頭
        
        # 文字特徵處理分支 - 注意: 這裡的 text_features 現在應該是 BERT 特徵的大小
        self.text_feature_size = text_features
        
        # 特徵處理層 - Swin Transformer
        self.swin_dropout = nn.Dropout(0.5)
        self.swin_bn = nn.BatchNorm1d(swin_features)
        
        # 特徵處理層 - DenseNet
        self.densenet_dropout = nn.Dropout(0.5)
        self.densenet_bn = nn.BatchNorm1d(densenet_features)
        
        # 文字特徵處理層 - 簡化 BERT 處理層，因為 BERT 已經提供了深層語義表示
        self.text_fc = nn.Sequential(
            nn.Linear(self.text_feature_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),  # 使用 GELU 激活函數，與 BERT 一致
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
        # 特徵融合層
        combined_features = swin_features + densenet_features + 128
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 分類器
        self.classifier = nn.Linear(256, num_classes)
        
        # 單模型分類器 (用於輔助訓練)
        self.swin_classifier = nn.Linear(swin_features, num_classes)
        self.densenet_classifier = nn.Linear(densenet_features, num_classes)
        self.text_classifier = nn.Linear(128, num_classes)
        
        # 融合權重 (可學習或固定)
        self.swin_weight = nn.Parameter(torch.tensor([swin_weight]), requires_grad=True)
        
    def forward(self, image, text_features):
        # Swin Transformer 分支
        swin_feat = self.swin(image)
        swin_feat = self.swin_dropout(swin_feat)
        if swin_feat.size(0) > 1:  # 僅在批次大小 > 1 時應用批次正規化
            swin_feat = self.swin_bn(swin_feat)
        swin_out = self.swin_classifier(swin_feat)
        
        # DenseNet 分支
        densenet_feat = self.densenet(image)
        densenet_feat = self.densenet_dropout(densenet_feat)
        if densenet_feat.size(0) > 1:  # 僅在批次大小 > 1 時應用批次正規化
            densenet_feat = self.densenet_bn(densenet_feat)
        densenet_out = self.densenet_classifier(densenet_feat)
        
        # 文字特徵處理 - 現在使用 BERT 嵌入
        text_feat = self.text_fc(text_features)
        text_out = self.text_classifier(text_feat)
        
        # 合併特徵
        combined_feat = torch.cat([swin_feat, densenet_feat, text_feat], dim=1)
        fused_feat = self.fusion_layer(combined_feat)
        combined_out = self.classifier(fused_feat)
        
        # 返回所有輸出以便於多任務學習
        return combined_out, swin_out, densenet_out, text_out

# 直方圖正規化
class HistogramNormalize:
    def __init__(self, num_bins=256):
        self.num_bins = num_bins

    def __call__(self, img):
        np_img = np.array(img)
        normalized_img = np.zeros_like(np_img, dtype=np.uint8)
        for i in range(3):
            channel = np_img[..., i]
            hist, bins = np.histogram(channel.flatten(), self.num_bins, density=True)
            cdf = hist.cumsum()
            cdf = (self.num_bins - 1) * cdf / cdf[-1]
            normalized_channel = np.interp(channel.flatten(), bins[:-1], cdf)
            normalized_img[..., i] = normalized_channel.reshape(channel.shape)
        return Image.fromarray(normalized_img)

class TextProcessor:
    def __init__(self, df, route_col='UltrasoundRoute', report_col='REPORT_TEXT', age_col='AGE'):
        self.df = df
        self.route_col = route_col
        self.report_col = report_col
        self.age_col = age_col
        
        # 標籤編碼器
        self.route_encoder = LabelEncoder()
        
        # 年齡標準化
        self.age_scaler = StandardScaler()
        
        # 報告文字處理
        self.word2vec_model = None
        self.max_report_length = 100  # 最大報告長度
        self.word_vector_size = 100   # 詞向量大小
        self.vocabulary = set()
        self.word_to_idx = {}
        
    def fit(self, min_word_count=2):
        # 處理 UltrasoundRoute
        self.route_encoder.fit(self.df[self.route_col].fillna('Unknown'))
        
        # 處理 AGE
        self.age_scaler.fit(self.df[self.age_col].values.reshape(-1, 1))
        
        # 處理 REPORT_TEXT
        # 移除標點符號、數字等
        processed_reports = []
        word_counts = Counter()
        
        for report in self.df[self.report_col].fillna(''):
            # 簡單的文字清理
            report = re.sub(r'[^\w\s]', '', report)
            report = re.sub(r'\d+', '', report)
            
            # 使用jieba分詞
            words = jieba.lcut(report)
            processed_reports.append(words)
            
            # 計算詞頻
            word_counts.update(words)
        
        # 建立詞彙表，過濾低頻詞
        self.vocabulary = {word for word, count in word_counts.items() if count >= min_word_count}
        
        # 建立詞到索引的映射
        self.word_to_idx = {word: idx+1 for idx, word in enumerate(self.vocabulary)}
        self.word_to_idx['<PAD>'] = 0  # 添加填充標記
        
        # 訓練 Word2Vec 模型
        self.word2vec_model = Word2Vec(
            sentences=processed_reports,
            vector_size=self.word_vector_size,
            window=5,
            min_count=min_word_count,
            workers=4
        )
        
        return self
        
    def transform(self, df):
        # 處理 UltrasoundRoute
        route_encoded = self.route_encoder.transform(df[self.route_col].fillna('Unknown'))
        route_one_hot = pd.get_dummies(route_encoded, prefix='route')
        
        # 處理 AGE
        age_scaled = self.age_scaler.transform(df[self.age_col].values.reshape(-1, 1))
        
        # 處理 REPORT_TEXT
        report_vectors = np.zeros((len(df), self.word_vector_size))
        
        for i, report in enumerate(df[self.report_col].fillna('')):
            # 簡單的文字清理
            report = re.sub(r'[^\w\s]', '', report)
            report = re.sub(r'\d+', '', report)
            
            # 使用jieba分詞
            words = jieba.lcut(report)
            
            # 計算報告詞向量 (取平均)
            word_vectors = []
            for word in words:
                if word in self.word2vec_model.wv:
                    word_vectors.append(self.word2vec_model.wv[word])
            
            if word_vectors:
                report_vectors[i] = np.mean(word_vectors, axis=0)
        
        # 合併所有特徵
        num_route_features = len(self.route_encoder.classes_)
        features = np.zeros((len(df), num_route_features + 1 + self.word_vector_size))
        
        features[:, :num_route_features] = route_one_hot
        features[:, num_route_features:num_route_features+1] = age_scaled
        features[:, num_route_features+1:] = report_vectors
        
        return features
    
    def get_feature_size(self):
        return len(self.route_encoder.classes_) + 1 + self.word_vector_size

# 直方圖標準化類，用於圖像預處理
class EnhancedHistogramNormalize:
    def __init__(self, clip_limit=2.0, grid_size=(8, 8), num_bins=256):
        self.clip_limit = clip_limit
        self.grid_size = grid_size
        self.num_bins = num_bins

    def __call__(self, img):
        np_img = np.array(img)
        
        # 將RGB轉換為LAB顏色空間 (L通道代表亮度)
        if len(np_img.shape) == 3 and np_img.shape[2] == 3:
            lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # 對亮度通道應用CLAHE
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.grid_size)
            cl = clahe.apply(l)
            
            # 合併通道
            enhanced_lab = cv2.merge((cl, a, b))
            
            # 轉回RGB
            enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        else:
            # 灰階圖像直接應用CLAHE
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.grid_size)
            enhanced_img = clahe.apply(np_img)
        
        # 再進行全域直方圖均衡化，標準化亮度分布
        normalized_img = np.zeros_like(enhanced_img, dtype=np.uint8)
        if len(enhanced_img.shape) == 3:
            for i in range(3):
                channel = enhanced_img[..., i]
                hist, bins = np.histogram(channel.flatten(), self.num_bins, density=True)
                cdf = hist.cumsum()
                cdf = (self.num_bins - 1) * cdf / cdf[-1]
                normalized_channel = np.interp(channel.flatten(), bins[:-1], cdf)
                normalized_img[..., i] = normalized_channel.reshape(channel.shape)
        else:
            hist, bins = np.histogram(enhanced_img.flatten(), self.num_bins, density=True)
            cdf = hist.cumsum()
            cdf = (self.num_bins - 1) * cdf / cdf[-1]
            normalized_img = np.interp(enhanced_img.flatten(), bins[:-1], cdf).reshape(enhanced_img.shape)
            
        return Image.fromarray(normalized_img)

# 優化3: 添加超音波影像特定的噪聲擾動函數
class UltrasoundNoiseAugmentation:
    def __init__(self, noise_intensity=0.1, speckle_ratio=0.3):
        self.noise_intensity = noise_intensity
        self.speckle_ratio = speckle_ratio
        
    def __call__(self, img):
        np_img = np.array(img).astype(np.float32) / 255.0
        
        # 添加乘性斑點噪聲 (適合超音波影像)
        speckle = np.random.randn(*np_img.shape) * self.noise_intensity
        noisy_img = np_img + np_img * speckle
        
        # 添加高斯噪聲 (模擬電子噪聲)
        if np.random.random() < self.speckle_ratio:
            gaussian_noise = np.random.randn(*np_img.shape) * (self.noise_intensity / 2)
            noisy_img += gaussian_noise
        
        # 裁剪值域到 [0, 1] 範圍
        noisy_img = np.clip(noisy_img, 0, 1)
        
        return Image.fromarray((noisy_img * 255).astype(np.uint8))

# 自訂超音波斑點噪聲擾動
class UltrasoundSpeckleNoise(object):
    def __init__(self, noise_variance=0.05):
        self.noise_variance = noise_variance
    
    def __call__(self, tensor):
        # 假設 tensor 是 [C, H, W] 格式
        if random.random() < 0.5:  # 50%的機率應用此增強
            # 生成乘性斑點噪聲 (模擬超音波的特性)
            noise = torch.randn_like(tensor) * self.noise_variance
            return tensor + tensor * noise  # 乘性噪聲
        return tensor

# 超音波陰影模擬
class UltrasoundShadowAugmentation(object):
    def __init__(self, shadow_intensity=0.5, shadow_size=0.2):
        self.shadow_intensity = shadow_intensity
        self.shadow_size = shadow_size
    
    def __call__(self, tensor):
        # 只有20%的機率會應用此增強
        if random.random() < 0.2:
            c, h, w = tensor.shape
            # 隨機選擇陰影起點
            x = int(random.uniform(0, w * 0.8))
            width = int(w * self.shadow_size)
            
            # 創建一個衰減因子，模擬超音波的陰影效果
            shadow_mask = torch.ones_like(tensor)
            for i in range(width):
                if x + i < w:
                    # 越往右陰影越深
                    attenuation = 1.0 - (self.shadow_intensity * (i / width))
                    shadow_mask[:, :, x + i] = attenuation
            
            return tensor * shadow_mask
        return tensor

# 超音波增強效果模擬
class UltrasoundEnhancementAugmentation(object):
    def __init__(self, enhancement_intensity=1.5, enhancement_size=0.2):
        self.enhancement_intensity = enhancement_intensity
        self.enhancement_size = enhancement_size
    
    def __call__(self, tensor):
        # 只有20%的機率會應用此增強
        if random.random() < 0.2:
            c, h, w = tensor.shape
            # 隨機選擇增強起點
            x = int(random.uniform(0, w * 0.8))
            y = int(random.uniform(0, h * 0.8))
            width = int(w * self.enhancement_size)
            height = int(h * self.enhancement_size)
            
            # 創建一個增強區域
            enhancement_mask = torch.ones_like(tensor)
            for i in range(width):
                for j in range(height):
                    if x + i < w and y + j < h:
                        # 高斯形狀的增強
                        dist = ((i - width/2)**2 + (j - height/2)**2) / ((width/2)**2 + (height/2)**2)
                        factor = 1.0 + (self.enhancement_intensity - 1.0) * np.exp(-dist * 4)
                        enhancement_mask[:, y + j, x + i] = factor
            
            # 應用增強，確保值在合理範圍內
            enhanced = tensor * enhancement_mask
            return torch.clamp(enhanced, 0, 1)
        return tensor

class MultimodalDataset(Dataset):
    def __init__(self, data_dir, metadata_path, image_transform=None, text_processor=None):
        self.data_dir = data_dir
        self.transform = image_transform
        
        # 讀取元數據
        self.metadata = pd.read_csv(metadata_path)
        
        # 預處理文字特徵
        if text_processor is None:
            raise ValueError("需要提供已訓練的TextProcessor實例")
        
        self.text_processor = text_processor
        self.text_features = self.text_processor.transform(self.metadata)
        
        self.image_files = []
        self.labels = []
        self.feature_indices = []
        
        valid_classes = [d for d in sorted(os.listdir(data_dir)) 
                          if os.path.isdir(os.path.join(data_dir, d)) 
                          and not d.startswith('.') 
                          and d not in ['__pycache__', '.git']]
        
        class_to_idx = {class_name: idx for idx, class_name in enumerate(valid_classes)}
        
        # 為每個有效的DICOM文件創建映射
        for idx, row in self.metadata.iterrows():
            img_id = row['ID']  # 假設這是完整ID，如 "P2391XXXXXXXX_1"
            # 尋找對應的圖像檔案
            for class_name in valid_classes:
                class_dir = os.path.join(data_dir, class_name)
                for img_name in os.listdir(class_dir):
                    # 直接匹配完整ID，不加下劃線
                    if img_name == f"{img_id}.dcm":
                        img_path = os.path.join(class_dir, img_name)
                        try:
                            # 測試是否能讀取DICOM檔案
                            img = read_dicom_file(img_path)
                            if img is not None:
                                self.image_files.append(img_path)
                                self.labels.append(class_to_idx[class_name])
                                self.feature_indices.append(idx)
                        except Exception as e:
                            print(f"警告: 跳過無效的DICOM檔案 {img_path}: {e}")
        
        print(f"總資料集大小: {len(self.image_files)} DICOM檔案")
        
        if len(self.image_files) == 0:
            print("警告: 在目錄結構中找不到有效的DICOM檔案!")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = read_dicom_file(img_path)  # 使用DICOM讀取器
        label = self.labels[idx]
        text_features = torch.FloatTensor(self.text_features[self.feature_indices[idx]])
        
        if self.transform:
            image = self.transform(image)
        
        return image, text_features, label

class MultimodalTransformSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        img_path = self.dataset.image_files[self.indices[idx]]
        # 使用 DICOM 讀取器
        image = read_dicom_file(img_path)
        label = self.dataset.labels[self.indices[idx]]
        text_features = self.dataset.text_features[self.dataset.feature_indices[self.indices[idx]]]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.FloatTensor(text_features), label

def calculate_metrics(y_true, y_pred, y_scores):
    metrics = {}
    metrics['accuracy'] = (y_true == y_pred).mean() * 100
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics['precision'] = report['weighted avg']['precision'] * 100
    metrics['recall'] = report['weighted avg']['recall'] * 100
    metrics['f1_score'] = report['weighted avg']['f1-score'] * 100
    
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 2:
        TP = ((y_pred == 1) & (y_true == 1)).sum()
        TN = ((y_pred == 0) & (y_true == 0)).sum()
        FP = ((y_pred == 1) & (y_true == 0)).sum()
        FN = ((y_pred == 0) & (y_true == 1)).sum()
        
        metrics['TP'] = TP
        metrics['FP'] = FP
        metrics['FN'] = FN
        metrics['TN'] = TN
        
        metrics['sensitivity'] = (TP / (TP + FN) if (TP + FN) > 0 else 0) * 100
        metrics['specificity'] = (TN / (TN + FP) if (TN + FP) > 0 else 0) * 100
        metrics['ppv'] = (TP / (TP + FP) if (TP + FP) > 0 else 0) * 100
        metrics['npv'] = (TN / (TN + FN) if (TN + FN) > 0 else 0) * 100
        
        try:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            metrics['auc'] = auc(fpr, tpr) * 100
        except ValueError:
            metrics['auc'] = 0
    else:
        metrics['TP'] = 0
        metrics['FP'] = 0
        metrics['FN'] = 0
        metrics['TN'] = 0
        metrics['sensitivity'] = 0
        metrics['specificity'] = 0
        metrics['ppv'] = 0
        metrics['npv'] = 0
        metrics['auc'] = 0
    
    return metrics

def calculate_confidence_interval(values, confidence=0.95):
    if len(values) < 2:
        return 0, 0
    mean = np.mean(values)
    sem = stats.sem(values)
    ci = stats.t.interval(confidence, len(values)-1, loc=mean, scale=sem)
    return ci[0], ci[1]

def evaluate_multimodal_model(model, data_loader, criterion, device):
    model.eval()
    all_labels = []
    all_predictions = []
    all_scores = []
    total_loss = 0.0
    
    with torch.no_grad():
        for images, text_features, labels in data_loader:
            images = images.to(device)
            text_features = text_features.to(device)
            labels = labels.to(device)
            
            outputs, swin_outputs, densenet_outputs, text_outputs = model(images, text_features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # 合併模型預測
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            if probabilities.shape[1] == 2:
                all_scores.extend(probabilities[:, 1].cpu().numpy())
            else:
                all_scores.extend(probabilities.max(dim=1)[0].cpu().numpy())

    metrics = calculate_metrics(np.array(all_labels), np.array(all_predictions), np.array(all_scores))
    metrics['loss'] = total_loss / len(data_loader)
    
    return metrics

def ensemble_predict_multimodal(models, data_loader, device, class_weights=None, dataset=None):
    """
    使用多個模型進行集成預測，並記錄每個樣本的預測機率
    參數:
        models: 模型列表
        data_loader: 資料載入器
        device: 計算裝置
        class_weights: 類別權重
        dataset: 用於獲取檔案路徑的資料集
    """
    all_labels = []
    all_probabilities = []
    all_image_paths = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_idx, (images, text_features, labels) in enumerate(data_loader):
            images = images.to(device)
            text_features = text_features.to(device)
            labels = labels.to(device)
            
            # 收集每個模型的預測結果
            batch_probabilities = []
            for model in models:
                model.eval()
                # 正確解包模型輸出 - 只取第一個輸出 (combined_out)
                outputs, _, _, _ = model(images, text_features)
                probabilities = torch.softmax(outputs, dim=1)
                batch_probabilities.append(probabilities)
            
            # 計算平均概率（軟投票）
            avg_probabilities = torch.mean(torch.stack(batch_probabilities), dim=0)
            _, predicted = torch.max(avg_probabilities, 1)
            
            # 計算損失
            loss = criterion(avg_probabilities.log(), labels)
            total_loss += loss.item()
            
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(avg_probabilities.cpu().numpy())
            
            # 獲取當前批次的圖像路徑
            start_idx = batch_idx * data_loader.batch_size
            end_idx = min((batch_idx + 1) * data_loader.batch_size, len(dataset))
            batch_image_paths = [dataset.dataset.image_files[dataset.indices[idx]] for idx in range(start_idx, end_idx)]
            all_image_paths.extend(batch_image_paths)
    
    # 轉換為NumPy陣列並計算評估指標
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    all_predictions = np.argmax(all_probabilities, axis=1)
    all_scores = all_probabilities[:, 1] if all_probabilities.shape[1] == 2 else np.max(all_probabilities, axis=1)
    
    metrics = calculate_metrics(all_labels, all_predictions, all_scores)
    metrics['loss'] = total_loss / len(data_loader)
    
    # 返回額外的機率和路徑資訊
    return metrics, all_labels, all_probabilities, all_image_paths

def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def save_checkpoint(model, optimizer, epoch, metrics, fold, is_best, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'fold': fold,
        'date': datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_fold{fold}_epoch{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(checkpoint_dir, f'best_model_fold{fold}_acc{metrics["accuracy"]:.2f}.pth')
        torch.save(checkpoint, best_path)
        
        serializable_metrics = convert_to_serializable(metrics)
        metrics_path = os.path.join(checkpoint_dir, f'best_metrics_fold{fold}.json')
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)

def cleanup_checkpoints(checkpoint_dir, fold, best_acc):
    for file in os.listdir(checkpoint_dir):
        if f'fold{fold}_epoch' in file and 'best' not in file:
            os.remove(os.path.join(checkpoint_dir, file))

def save_final_best_model(best_fold_metrics, checkpoint_dir):
    best_fold = max(best_fold_metrics.keys(), key=lambda k: best_fold_metrics[k]['accuracy'])
    best_metrics = best_fold_metrics[best_fold]
    best_model_path = os.path.join(checkpoint_dir, f'best_model_fold{best_fold}_acc{best_metrics["accuracy"]:.2f}.pth')
    
    if os.path.exists(best_model_path):
        final_path = os.path.join(checkpoint_dir, f'final_best_model_acc{best_metrics["accuracy"]:.2f}.pth')
        checkpoint = torch.load(best_model_path, weights_only=False)
        torch.save(checkpoint, final_path)
        
        serializable_metrics = convert_to_serializable(best_metrics)
        final_metrics_path = os.path.join(checkpoint_dir, 'final_best_metrics.json')
        with open(final_metrics_path, 'w') as f:
            json.dump({'best_fold': best_fold, 'metrics': serializable_metrics}, f, indent=4)

def create_weighted_sampler(labels):
    """
    创建加权采样器，对少数类进行过采样
    
    Args:
        labels: 训练数据集的标签
        
    Returns:
        WeightedRandomSampler: 用于过采样少数类的采样器
    """
    # 计算每个类别的样本数量
    class_counts = np.bincount(labels)
    print(f"Class counts: {class_counts}")
    
    # 计算每个类别的权重（样本越少，权重越大）
    class_weights = 1.0 / class_counts
    
    # 确保权重不是 NaN 或无穷大
    class_weights = np.nan_to_num(class_weights, nan=0.0, posinf=0.0)
    
    # 如果某个类别权重为0（可能因为没有样本），设置一个小的默认值
    if np.any(class_weights == 0):
        class_weights[class_weights == 0] = 0.0001
    
    # 规范化权重使总和为1
    class_weights = class_weights / class_weights.sum()
    
    # 为每个样本分配权重
    weights = [class_weights[label] for label in labels]
    weights = torch.DoubleTensor(weights)
    
    # 创建采样器，样本总数等于原始数据集大小
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    print(f"Created weighted sampler with weights: {class_weights}")
    
    return sampler

class AutoROICropWithAspectRatio(object):
    """超音波影像自動 ROI 擷取並保持縱橫比的轉換"""
    def __init__(self, target_size=(224, 224), margin=10):
        """
        初始化 ROI 擷取轉換
        
        參數:
            target_size (tuple): 目標影像大小 (寬, 高)
            margin (int): ROI 邊界外的額外邊距（像素）
        """
        self.target_size = target_size
        self.margin = margin
    
    def __call__(self, img):
        # 轉換 PIL Image 為 NumPy 陣列
        np_img = np.array(img)
        
        # 自動檢測 ROI
        roi = self._auto_detect_roi(np_img)
        
        # 保持縱橫比縮放
        resized_roi = self._resize_maintain_ratio(roi)
        
        # 轉回 PIL Image
        return Image.fromarray(resized_roi.astype('uint8'))
    
    def _auto_detect_roi(self, image):
        """自動檢測並裁剪超音波影像中的 ROI，使用 Otsu 演算法動態計算閾值"""
        # 轉換為灰度圖
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 1. 使用 Otsu 演算法計算最佳閾值
        otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 2. 加入安全檢查 (Sanity Check)
        # 超音波背景通常很暗，閾值不太可能超過 60，也不太可能低於 10
        final_thresh = otsu_thresh
        if otsu_thresh < 10:
            final_thresh = 10  # 避免切到太低，把純黑噪點都算進來
        elif otsu_thresh > 60:
            final_thresh = 60  # 避免切到太高，把暗部組織切掉
        
        # 3. 應用最終閾值
        _, thresh = cv2.threshold(gray, final_thresh, 255, cv2.THRESH_BINARY)
        
        # 形態學操作，移除噪點
        kernel = np.ones((5, 5), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        
        # 找出最大連通區域（假設這是超音波的主診斷區）
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 選擇最大的輪廓
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 獲取邊界矩形
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # 添加邊距
            x = max(0, x - self.margin)
            y = max(0, y - self.margin)
            w = min(image.shape[1] - x, w + 2 * self.margin)
            h = min(image.shape[0] - y, h + 2 * self.margin)
            
            # 裁剪 ROI
            roi = image[y:y+h, x:x+w]
            return roi
        
        return image  # 如果無法找到 ROI，返回原始影像
    
    def _resize_maintain_ratio(self, img):
        """縮放影像至目標大小同時保持縱橫比"""
        h, w = img.shape[:2]
        target_w, target_h = self.target_size
        
        # 計算縮放因子
        ratio = min(target_w / w, target_h / h)
        new_size = (int(w * ratio), int(h * ratio))
        
        # 縮放
        resized = cv2.resize(img, new_size)
        
        # 創建黑色畫布
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # 計算貼上位置，使影像居中
        x_offset = (target_w - new_size[0]) // 2
        y_offset = (target_h - new_size[1]) // 2
        
        # 貼上縮放後的影像
        canvas[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0]] = resized
        
        return canvas

# 添加 Focal Loss 以改善類別不平衡問題
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha = torch.tensor(self.alpha).to(inputs.device)
                batch_alpha = alpha[targets]
                loss = batch_alpha * loss
            else:
                # 簡單的權重方案: 使用固定值
                pos_mask = (targets == 1).float()
                neg_mask = 1 - pos_mask
                loss = self.alpha * pos_mask * loss + (1 - self.alpha) * neg_mask * loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# 實現 CutMix 增強方法，有助於提高模型的魯棒性和泛化能力
def cutmix(images, text_features, labels, beta=1.0, device=None):
    """
    CutMix 數據增強方法：混合圖像和標籤
    """
    # 生成混合參數
    batch_size = images.size(0)
    lam = np.random.beta(beta, beta)
    
    # 隨機選擇索引進行混合
    indices = torch.randperm(batch_size).to(device)
    shuffled_images = images[indices]
    shuffled_text = text_features[indices]
    shuffled_labels = labels[indices]
    
    # 獲取圖像尺寸
    _, h, w = images.size()[1:]
    
    # 計算剪切區域
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)
    
    # 選擇隨機中心點
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    
    # 計算邊界
    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    
    # 混合圖像
    images[:, :, y1:y2, x1:x2] = shuffled_images[:, :, y1:y2, x1:x2]
    
    # 確定混合後的真實標籤比例
    lam = 1 - ((x2 - x1) * (y2 - y1) / (h * w))
    
    # 對文本特徵也進行類似的混合
    mixed_text = text_features * lam + shuffled_text * (1 - lam)
    
    return images, mixed_text, labels, shuffled_labels, lam

def train_multimodal_model():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs: {device}")
    else:
        print(f"Using device: {device}")
        return  

    num_workers = 0
    print(f"Using {num_workers} workers for data loading (single process mode to avoid shared memory issues)")

    metadata_path = r"Clinical_data.csv"
    metadata = pd.read_csv(metadata_path)
    # 預處理文字特徵
    text_processor = BertTextProcessor(
        metadata,
        bert_model_name='emilyalsentzer/Bio_ClinicalBERT', 
        max_length=128  # 設置最大序列長度
    )
    text_processor.fit()
    text_feature_size = text_processor.get_feature_size()
    
    base_dir = r"log"
    model_name = "multimodal_ov" 
    checkpoint_dir = os.path.join(base_dir, model_name)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Model checkpoints and results will be saved to: {checkpoint_dir}")

    train_transform = transforms.Compose([
        AutoROICropWithAspectRatio(target_size=(224, 224), margin=15),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-15, 15)),  # 減少旋轉角度以保持超音波方向性
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.05, 0.05),  # 輕微平移
            scale=(0.95, 1.05),      # 輕微縮放
            fill=0
        ),
        transforms.ToTensor(),
        UltrasoundSpeckleNoise(noise_variance=0.03),       # 超音波斑點噪聲
        UltrasoundShadowAugmentation(shadow_intensity=0.4, shadow_size=0.15),  # 超音波陰影模擬
        UltrasoundEnhancementAugmentation(enhancement_intensity=1.2, enhancement_size=0.15),  # 增強模擬
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    test_transform = transforms.Compose([
        AutoROICropWithAspectRatio(target_size=(224, 224), margin=15),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    batch_size = 32
    data_dir = r"UltrasoundImage"
    if not os.path.exists(data_dir):
        print(f"Error: Data directory does not exist: {data_dir}")
        return
        
    class_names = [d for d in sorted(os.listdir(data_dir)) 
                  if os.path.isdir(os.path.join(data_dir, d)) 
                  and not d.startswith('.') 
                  and d not in ['__pycache__', '.git', '.ipynb_checkpoints']]
    print(f"Found classes: {class_names}")
    
    dataset = MultimodalDataset(
        data_dir=data_dir,
        metadata_path=metadata_path,
        image_transform=None,
        text_processor=text_processor
    )
    total_size = len(dataset)
    print(f"Total dataset size: {total_size}")
    
    for i, class_name in enumerate(class_names):
        count = sum(1 for label in dataset.labels if label == i)
        print(f"Class {class_name}: {count} photos")
    
    test_size = 0.15
    
    num_folds = 5
    
    fold_splits, test_indices, train_val_labels = subject_level_split(
        dataset, test_size=test_size, num_folds=num_folds, random_state=seed
    )
    
    # 創建測試集
    test_dataset = MultimodalTransformSubset(dataset, test_indices, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    
    # 統計測試集的類別分布
    all_labels = dataset.labels
    test_labels = [all_labels[idx] for idx in test_indices]
    test_counts = {}
    for i, class_name in enumerate(class_names):
        test_counts[class_name] = test_labels.count(i)
    
    print("\nTesting set class distribution:")
    for class_name, count in test_counts.items():
        print(f"  {class_name}: {count} photos")
    
    # 最終驗證：確保測試集沒有受試者洩漏
    print("\n=== 最終測試集驗證 ===")
    test_subject_ids = set()
    for idx in test_indices:
        img_path = dataset.image_files[idx]
        filename = os.path.basename(img_path)
        subject_id = extract_subject_id(filename)
        test_subject_ids.add(subject_id)
    
    print(f"測試集包含 {len(test_subject_ids)} 個唯一受試者")

#################################################################
    
    best_fold_metrics = {}
    train_counts_per_fold = []
    val_counts_per_fold = []
    
    patience = 30  # 提前停止耐心值
    
    for fold, (train_indices, val_indices) in enumerate(fold_splits):
        print(f"\nTraining Fold {fold+1}/{num_folds}")
        
        # 驗證當前fold沒有受試者洩漏
        print(f"Fold {fold+1} 詳細驗證：")
        train_subject_count, val_subject_count, test_subject_count = verify_no_subject_leakage(
            train_indices, val_indices, test_indices, dataset
        )
        print(f"  訓練集受試者數：{train_subject_count}")
        print(f"  驗證集受試者數：{val_subject_count}")
        print(f"  測試集受試者數：{test_subject_count}")
        
        train_subset = MultimodalTransformSubset(dataset, train_indices, transform=train_transform)
        val_subset = MultimodalTransformSubset(dataset, val_indices, transform=test_transform)
        
        print(f"Train transform type: {type(train_transform)}")
        print(f"Validation transform type: {type(test_transform)}")
        
        # 计算训练集的标签，用于创建加权采样器
        train_labels = [all_labels[idx] for idx in train_indices]
        
        # 创建加权采样器进行过采样
        sampler = create_weighted_sampler(train_labels)
        
        # 使用加权采样器创建数据加载器
        train_loader = DataLoader(
            train_subset, 
            batch_size=batch_size,
            sampler=sampler,  # 使用加权采样器而不是随机采样
            num_workers=num_workers, 
            pin_memory=False  # 設為 False 以減少共享記憶體使用
        )
        
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
        
        train_counts = {}
        for i, class_name in enumerate(class_names):
            train_counts[class_name] = train_labels.count(i)
        train_counts_per_fold.append(train_counts)
        
        val_labels = [all_labels[idx] for idx in val_indices]
        val_counts = {}
        for i, class_name in enumerate(class_names):
            val_counts[class_name] = val_labels.count(i)
        val_counts_per_fold.append(val_counts)
        
        print("Training set class distribution:")
        for class_name, count in train_counts.items():
            print(f"  {class_name}: {count} photos")
        print("Validation set class distribution:")
        for class_name, count in val_counts.items():
            print(f"  {class_name}: {count} photos")
        
        # 使用采样器后的实际批次平衡
        print("Checking batch balance with weighted sampler...")
        batch_class_counts = np.zeros(len(class_names), dtype=int)
        num_batches_to_check = min(10, len(train_loader))
        for i, (_, _, labels) in enumerate(train_loader):
            if i >= num_batches_to_check:
                break
            for cls in range(len(class_names)):
                batch_class_counts[cls] += (labels == cls).sum().item()
        
        print("Class distribution in sampled batches:")
        for i, class_name in enumerate(class_names):
            print(f"  {class_name}: {batch_class_counts[i]} samples")
        
        num_classes = len(class_names)
        model = MultimodalModel(num_classes, swin_weight=0.5, text_features=text_feature_size).to(device)


        def improved_multi_task_loss(combined_out, swin_out, densenet_out, text_out, labels, 
                           alpha=0.7, beta=0.15, gamma=0.1, delta=0.05):
            """
            改進的多任務損失函數，動態調整各分支權重
            
            參數:
                combined_out: 合併模型的輸出
                swin_out: Swin Transformer 的輸出
                densenet_out: DenseNet 的輸出
                text_out: 文本特徵的輸出
                labels: 真實標籤
                alpha, beta, gamma, delta: 各分支的權重
            """
            criterion = nn.CrossEntropyLoss(reduction='none')  # 使用 'none' 以便獲取每個樣本的損失
            
            # 計算每個分支的原始損失
            loss_combined = criterion(combined_out, labels)
            loss_swin = criterion(swin_out, labels)
            loss_densenet = criterion(densenet_out, labels)
            loss_text = criterion(text_out, labels)
            
            # 計算每個分支的預測準確性
            _, pred_combined = torch.max(combined_out.data, 1)
            _, pred_swin = torch.max(swin_out.data, 1)
            _, pred_densenet = torch.max(densenet_out.data, 1)
            _, pred_text = torch.max(text_out.data, 1)
            
            correct_combined = (pred_combined == labels).float()
            correct_swin = (pred_swin == labels).float()
            correct_densenet = (pred_densenet == labels).float()
            correct_text = (pred_text == labels).float()
            
            # 對於錯誤的預測，增強其損失權重
            # 如果主模型預測正確但輔助模型錯誤，減少輔助模型的權重
            batch_size = labels.size(0)
            
            # 動態調整權重 (僅用於示範，實際上可能需要更複雜的調整邏輯)
            dynamic_alpha = alpha
            dynamic_beta = torch.where(correct_combined > correct_swin, 
                                       torch.ones_like(correct_swin) * (beta * 0.8), 
                                       torch.ones_like(correct_swin) * (beta * 1.2))
            dynamic_gamma = torch.where(correct_combined > correct_densenet, 
                                        torch.ones_like(correct_densenet) * (gamma * 0.8), 
                                        torch.ones_like(correct_densenet) * (gamma * 1.2))
            dynamic_delta = torch.where(correct_combined > correct_text, 
                                        torch.ones_like(correct_text) * (delta * 0.8), 
                                        torch.ones_like(correct_text) * (delta * 1.2))
            
            # 應用動態權重
            weighted_loss_combined = (dynamic_alpha * loss_combined).mean()
            weighted_loss_swin = (dynamic_beta * loss_swin).mean()
            weighted_loss_densenet = (dynamic_gamma * loss_densenet).mean()
            weighted_loss_text = (dynamic_delta * loss_text).mean()

            total_loss = weighted_loss_combined + weighted_loss_swin.mean() + weighted_loss_densenet.mean() + weighted_loss_text.mean()
            
            return total_loss



        
        criterion = nn.CrossEntropyLoss()

        focal_loss = FocalLoss(gamma=2.0, alpha=0.25)
        #######################################################################################################################
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)
        
        num_epochs = 150
        best_val_acc = 0.0
        best_val_loss = float('inf')
        early_stop_counter = 0
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for i, (images, text_features, labels) in enumerate(train_loader):
                images = images.to(device)
                text_features = text_features.to(device)
                labels = labels.to(device)
                

                ###################################################################################################################
                # 應用 CutMix (有50%的機率)
                if np.random.random() < 0.5:
                    images, mixed_text, labels_a, labels_b, lam = cutmix(images, text_features, labels, beta=1.0, device=device)
                    
                    # 前向傳播
                    combined_out, swin_out, densenet_out, text_out = model(images, mixed_text)
                    
                    # 對混合標籤應用損失
                    loss_a = improved_multi_task_loss(combined_out, swin_out, densenet_out, text_out, labels_a)
                    loss_b = improved_multi_task_loss(combined_out, swin_out, densenet_out, text_out, labels_b)
                    loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    # 正常前向傳播
                    combined_out, swin_out, densenet_out, text_out = model(images, text_features)
                    
                    # 應用進階的多任務損失
                    loss = improved_multi_task_loss(combined_out, swin_out, densenet_out, text_out, labels)
                    
                    # 添加 Focal Loss 做為輔助損失 (只應用於主要輸出)
                    loss += focal_loss(combined_out, labels) * 0.2  # 權重可調整
                # 反向傳播和優化
                optimizer.zero_grad()
                loss.backward()
                #####################################################################################################################
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
                optimizer.step()
                running_loss += loss.item()
                if (i + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/10:.4f}')
                    running_loss = 0.0

            val_metrics = evaluate_multimodal_model(model, val_loader, nn.CrossEntropyLoss(), device)
            print(f'Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {val_metrics["loss"]:.4f}, '
                  f'Accuracy: {val_metrics["accuracy"]:.2f}%')
            
            # 查看模型的融合權重
            if not isinstance(model, nn.DataParallel):
                weight = torch.sigmoid(model.swin_weight.data)
                print(f'Current Swin Weight: {weight.item():.4f}, DenseNet Weight: {1-weight.item():.4f}')
            

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_val_acc = val_metrics["accuracy"]
                early_stop_counter = 0
                is_best = True
                best_fold_metrics[fold + 1] = val_metrics
                print(f'Best model saved with loss: {best_val_loss:.4f} and accuracy: {best_val_acc:.2f}%')
            else:
                early_stop_counter += 1
                is_best = False
                print(f'Early stopping counter: {early_stop_counter}/{patience}')
            
            scheduler.step(val_metrics["accuracy"])
            
            if isinstance(model, nn.DataParallel):
                save_checkpoint(model.module, optimizer, epoch + 1, val_metrics, fold + 1, is_best, checkpoint_dir)
            else:
                save_checkpoint(model, optimizer, epoch + 1, val_metrics, fold + 1, is_best, checkpoint_dir)
            
            # 檢查提前停止
            if early_stop_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        cleanup_checkpoints(checkpoint_dir, fold + 1, best_val_acc)
        print(f'Fold {fold+1} finished! Best validation accuracy: {best_val_acc:.2f}%')
    
    save_final_best_model(best_fold_metrics, checkpoint_dir)
    
    metrics_names = ['accuracy', 'sensitivity', 'specificity', 'ppv', 'npv', 'auc', 'precision', 'recall', 'f1_score']
    val_summary = {}
    for metric in metrics_names:
        values = [m[metric] for m in best_fold_metrics.values()]
        mean_value = np.mean(values)
        ci_low, ci_high = calculate_confidence_interval(values)
        val_summary[f'{metric}_mean'] = mean_value
        val_summary[f'{metric}_ci_low'] = ci_low
        val_summary[f'{metric}_ci_high'] = ci_high
    
    print("\nPerforming ensemble prediction with all fold best models...")
    #########################################################################################################################
    ensemble_models = []
    for fold in best_fold_metrics.keys():
        best_model_path = os.path.join(checkpoint_dir, f'best_model_fold{fold}_acc{best_fold_metrics[fold]["accuracy"]:.2f}.pth')
        if os.path.exists(best_model_path):
            model = MultimodalModel(num_classes, text_features=text_feature_size).to(device)  # 創建模型實例
            checkpoint = torch.load(best_model_path, weights_only=False)  # 載入檢查點
            model.load_state_dict(checkpoint['model_state_dict'])  # 載入模型狀態
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)  # 數據並行處理
            ensemble_models.append(model)
        else:
            print(f"警告: 在 {best_model_path} 未找到折疊 {fold} 的最佳模型")
    
    # 確保有模型可用於集成預測
    if not ensemble_models:
        print("錯誤: 沒有可用於集成預測的模型")
        return
    
    # 執行集成預測
    test_metrics, test_labels, test_probabilities, test_image_paths = ensemble_predict_multimodal(
        ensemble_models, test_loader, device, dataset=test_dataset
    )
    print(f"集成測試指標: 準確率: {test_metrics['accuracy']:.2f}%, 損失: {test_metrics['loss']:.4f}")
    print(f"敏感度: {test_metrics['sensitivity']:.2f}%, 特異度: {test_metrics['specificity']:.2f}%")
    
    # 新增：儲存每個測試樣本的預測機率到 CSV
    prob_df = pd.DataFrame(test_probabilities, columns=[f'Prob_Class_{i}' for i in range(num_classes)])
    prob_df['Image_Path'] = test_image_paths
    prob_df['True_Label'] = test_labels
    prob_df['Predicted_Label'] = np.argmax(test_probabilities, axis=1)
    prob_csv_path = os.path.join(checkpoint_dir, 'test_predictions_probabilities.csv')
    prob_df.to_csv(prob_csv_path, index=False)
    print(f"測試集每個樣本的預測機率已儲存至 '{prob_csv_path}'")
    #########################################################################################################################
    
    print("\n=== Dataset Class Counts ===")
    for fold in range(num_folds):
        print(f"\nFold {fold+1}:")
        print("Training set class distribution:")
        for class_name, count in train_counts_per_fold[fold].items():
            print(f"  {class_name}: {count} photos")
        print("Validation set class distribution:")
        for class_name, count in val_counts_per_fold[fold].items():
            print(f"  {class_name}: {count} photos")
    
    print("\nTesting set class distribution:")
    for class_name, count in test_counts.items():
        print(f"  {class_name}: {count} photos")

    results = []
    results.append({
        'Dataset': 'Validation',
        'TP': 0,
        'FP': 0,
        'FN': 0,
        'TN': 0,
        'Accuracy': val_summary['accuracy_mean'],
        'CI_Lower': val_summary['accuracy_ci_low'],
        'CI_Upper': val_summary['accuracy_ci_high'],
        'Sensitivity': val_summary['sensitivity_mean'],
        'CI_Lower_Sensitivity': val_summary['sensitivity_ci_low'],
        'CI_Upper_Sensitivity': val_summary['sensitivity_ci_high'],
        'Specificity': val_summary['specificity_mean'],
        'CI_Lower_Specificity': val_summary['specificity_ci_low'],
        'CI_Upper_Specificity': val_summary['specificity_ci_high'],
        'ppv': val_summary['ppv_mean'],
        'CI_Lower_ppv': val_summary['ppv_ci_low'],
        'CI_Upper_ppv': val_summary['ppv_ci_high'],
        'npv': val_summary['npv_mean'],
        'CI_Lower_npv': val_summary['npv_ci_low'],
        'CI_Upper_npv': val_summary['npv_ci_high'],
        'auc': val_summary['auc_mean'],
        'CI_Lower_auc': val_summary['auc_ci_low'],
        'CI_Upper_auc': val_summary['auc_ci_high'],
        'precision': val_summary['precision_mean'],
        'recall': val_summary['recall_mean'],
        'f1_score': val_summary['f1_score_mean'],
        'CI_Lower_f1_score': val_summary['f1_score_ci_low'],
        'CI_Upper_f1_score': val_summary['f1_score_ci_high']
    })
    
    results.append({
        'Dataset': 'Test (Ensemble)',
        'TP': test_metrics.get('TP', 0),
        'FP': test_metrics.get('FP', 0),
        'FN': test_metrics.get('FN', 0),
        'TN': test_metrics.get('TN', 0),
        'Accuracy': test_metrics['accuracy'],
        'CI_Lower': test_metrics['accuracy'],
        'CI_Upper': test_metrics['accuracy'],
        'Sensitivity': test_metrics['sensitivity'],
        'CI_Lower_Sensitivity': test_metrics['sensitivity'],
        'CI_Upper_Sensitivity': test_metrics['sensitivity'],
        'Specificity': test_metrics['specificity'],
        'CI_Lower_Specificity': test_metrics['specificity'],
        'CI_Upper_Specificity': test_metrics['specificity'],
        'ppv': test_metrics['ppv'],
        'CI_Lower_ppv': test_metrics['ppv'],
        'CI_Upper_ppv': test_metrics['ppv'],
        'npv': test_metrics['npv'],
        'CI_Lower_npv': test_metrics['npv'],
        'CI_Upper_npv': test_metrics['npv'],
        'auc': test_metrics['auc'],
        'CI_Lower_auc': test_metrics['auc'],
        'CI_Upper_auc': test_metrics['auc'],
        'precision': test_metrics['precision'],
        'recall': test_metrics['recall'],
        'f1_score': test_metrics['f1_score'],
        'CI_Lower_f1_score': test_metrics['f1_score'],
        'CI_Upper_f1_score': test_metrics['f1_score']
    })
    
    # 分析每个模型（Swin 和 DenseNet）的单独性能
    model_names = ['Combined', 'Swin', 'DenseNet']
    model_accs = ['accuracy', 'swin_accuracy', 'densenet_accuracy']
    
    # 计算所有摺疊中每個模型的平均表現
    for name, acc_key in zip(model_names, model_accs):
        if acc_key in best_fold_metrics[1]:  # 確保指標存在
            values = [m.get(acc_key, 0) for m in best_fold_metrics.values()]
            mean_value = np.mean(values)
            print(f"Average {name} accuracy across all folds: {mean_value:.2f}%")
    
    results_csv_path = os.path.join(checkpoint_dir, 'model_evaluation_results_oversampling.csv')
    df = pd.DataFrame(results)
    df.to_csv(results_csv_path, index=False)
    print(f"\nResults have been saved to '{results_csv_path}'")

    print("\n=== 受試者層級統計 ===")
    for fold in range(num_folds):
        train_indices, val_indices = fold_splits[fold]
        
        # 統計訓練集受試者
        train_subjects = set()
        for idx in train_indices:
            img_path = dataset.image_files[idx]
            filename = os.path.basename(img_path)
            subject_id = extract_subject_id(filename)
            train_subjects.add(subject_id)
        
        # 統計驗證集受試者
        val_subjects = set()
        for idx in val_indices:
            img_path = dataset.image_files[idx]
            filename = os.path.basename(img_path)
            subject_id = extract_subject_id(filename)
            val_subjects.add(subject_id)
        
        print(f"\nFold {fold+1}:")
        print(f"  訓練集：{len(train_subjects)} 受試者，{len(train_indices)} 影像")
        print(f"  驗證集：{len(val_subjects)} 受試者，{len(val_indices)} 影像")
    
    # 測試集受試者統計
    test_subjects = set()
    for idx in test_indices:
        img_path = dataset.image_files[idx]
        filename = os.path.basename(img_path)
        subject_id = extract_subject_id(filename)
        test_subjects.add(subject_id)
    
    print(f"\n測試集：{len(test_subjects)} 受試者，{len(test_indices)} 影像")

if __name__ == '__main__':
    train_multimodal_model()