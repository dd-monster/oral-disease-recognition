\# 口腔疾病多标签图像识别（Oral Disease Multi-Label Image Recognition）



\## 项目概述

本项目基于深度学习技术，构建了针对口腔病变的多标签图像识别系统，选用 InceptionV3 与 ResNet50 作为核心 backbone，实现口腔疾病的自动化筛查与辅助诊断。项目最终验证集准确率达 \*\*93%\*\*，为临床早期诊断提供高效、可落地的AI辅助方案。



\## 技术栈

\- 编程语言：Python 3.9+

\- 深度学习框架：TensorFlow 2.10+ / Keras

\- 核心模型：InceptionV3、ResNet50（迁移学习）

\- 数据处理：OpenCV、ImageDataGenerator（数据增强）、NumPy

\- 模型优化：学习率调度（LearningRateScheduler）、早停法（EarlyStopping）、自适应学习率衰减（ReduceLROnPlateau）

\- 评估与可视化：Scikit-learn、Matplotlib

\- 其他工具：Requests、OS





\## 快速开始

\### 1. 环境准备

```bash

\# 克隆项目

git clone https://github.com/dd-monster/oral-disease-recognition.git

cd oral-disease-recognition



\# 安装依赖（直接使用项目根目录的requirements.txt）

pip install -r requirements.txt

```

\### 2. 数据集准备

数据来源：Kaggle 口腔疾病公开数据集

预处理：统一图像尺寸、像素值归一化，通过 ImageDataGenerator 实现随机翻转 / 旋转 / 缩放的数据增强策略

存放：将处理后的数据按训练 / 验证 / 测试集划分，放入项目oral-disease-recognition\\data\\

\### 3. 模型训练

直接运行核心训练脚本，实现 InceptionV3/ResNet50 模型的迁移学习训练与评估：

