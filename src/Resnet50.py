# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 22:38:31 2024

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_score, recall_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.layers import Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# 定义余弦退火学习率调度器
def cosine_learning_rate_schedule(epoch, total_epochs, initial_lr):
    min_lr = 0.001 * initial_lr
    cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
    lr = min_lr + 0.5 * (initial_lr - min_lr) * cosine_decay
    return lr

# 定义早停回调
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# 定义学习率调度器回调
initial_learning_rate = 0.001
total_epochs = 100
lr_scheduler = LearningRateScheduler(lambda epoch: cosine_learning_rate_schedule(epoch, total_epochs, initial_learning_rate))

# 实例化ImageDataGenerator用于数据增强
generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=35,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)


# 下载模型
#model = ResNet50(weights='imagenet', include_top=False)
# 加载本地的ResNet50预训练权重
weights_path = "E:/R/4MOTH/TJJM/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"  #base_model = ResNet50(include_top=False, input_shape=(155, 155, 3), weights=weights_path)

# 冻结所有层
#for layer in base_model.layers:
 #   layer.trainable = False

for layer in base_model.layers[-4:]:
    layer.trainable = True


from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.regularizers import l2  # 导入 L2 正则化器

# 构建新的分类模型
x = base_model.output
x = GlobalAveragePooling2D()(x)  # 添加全局平均池化层
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)  
x = Dense(6, activation='softmax')(x)  # 假设有6个分类


model = Model(inputs=base_model.input, outputs=x)

# 编译模型
model.compile(optimizer=RMSprop(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
# 注意：训练集、验证集和测试集的路径
train_generator = generator.flow_from_directory('E:/R/4MOTH/TJJM/archive/ready_to_gen/train', target_size=(155, 155), batch_size=32)
valid_generator = generator.flow_from_directory('E:/R/4MOTH/TJJM/archive/ready_to_gen/valid', target_size=(155, 155), batch_size=32)
test_generator = generator.flow_from_directory('E:/R/4MOTH/TJJM/archive/ready_to_gen/test', target_size=(155, 155), batch_size=32)

history = model.fit(
    train_generator,
    steps_per_epoch=100,  # 根据训练集大小调整
    epochs=total_epochs,
    validation_data=valid_generator,
    validation_steps=40,  # 根据验证集大小调整
    callbacks=[lr_scheduler,early_stopping]
)

history = model.fit(
    train_generator,
    steps_per_epoch=60,  # 确保匹配训练集大小
    epochs=total_epochs,
    validation_data=valid_generator,
    validation_steps=40,  # 确保匹配验证集大小
    callbacks=[lr_scheduler, early_stopping]
)

# 评估模型
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy}")

# 绘制训练过程中的准确率和损失值图表
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# 类似地，可以绘制损失值图表
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()