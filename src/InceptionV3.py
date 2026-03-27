# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:49:19 2024

@author: Administrator
"""


# Importing needed modules.
import os
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.layers import Flatten, Dense, GlobalAveragePooling2D, Normalization
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_score, recall_score
import cv2
import requests

# Data augmentation generator
augmentation = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

def create_directories(base_path, labels):
    if 'ready_to_gen' in os.listdir(base_path):
        os.rmdir(os.path.join(base_path, 'ready_to_gen'))
    os.mkdir(os.path.join(base_path, 'ready_to_gen'))
    for subset in ['train', 'valid', 'test']:
        os.mkdir(os.path.join(base_path, 'ready_to_gen', subset))
    for label in labels:
        if label in ['ready_to_gen', 'Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset']:
            continue
        for subset in ['train', 'valid', 'test']:
            os.mkdir(os.path.join(base_path, 'ready_to_gen', subset, label))
    return os.path.join(base_path, 'ready_to_gen', 'train'), os.path.join(base_path, 'ready_to_gen', 'valid'), os.path.join(base_path, 'ready_to_gen', 'test')

def split_and_copy_data(photo_path, label_name, train_size, train_dir, valid_dir, test_dir):
    photos = os.listdir(photo_path)
    size_tr = (train_size * len(photos)) // 100
    train_photos = random.sample(photos, size_tr)
    # 从原始照片中去掉已经选为训练集的照片
    remaining_photos = [photo for photo in photos if photo not in train_photos]

# 计算剩余照片数量的一半，作为验证集的大小
    valid_size = len(remaining_photos) // 2

# 从剩余的照片中随机抽取一半作为验证集
    valid_photos = random.sample(remaining_photos, valid_size)
    for photo in photos:
        target_dir = train_dir if photo in train_photos else valid_dir if photo in valid_photos else test_dir
        src = os.path.join(photo_path, photo)
        dst = os.path.join(target_dir, label_name, photo)
        os.rename(src, dst)

# Define paths and split data
base_path = "E:/R/4MOTH/TJJM/archive"
labels = [d for d in os.listdir(base_path) if d not in ['ready_to_gen','Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset']]
train_dir, valid_dir, test_dir = create_directories(base_path, labels)
addresses = ["E:/R/4MOTH/TJJM/archive/Data caries/Data caries/caries augmented data set/preview",
            "E:/R/4MOTH/TJJM/archive/Mouth Ulcer/Mouth Ulcer/Mouth_Ulcer_augmented_DataSet/preview",
            "E:/R/4MOTH/TJJM/archive/Tooth Discoloration/Tooth Discoloration_/Tooth_discoloration_augmented_dataser/preview",
            "E:/R/4MOTH/TJJM/archive/hypodontia/hypodontia",
            "E:/R/4MOTH/TJJM/archive/Gingivitis/Gingivitis",
            "E:/R/4MOTH/TJJM/archive/Calculus/Calculus"]
for address, label in zip(addresses, labels):
    split_and_copy_data(address, label, 80, train_dir, valid_dir, test_dir)

# Load InceptionV3 weights and create the model
weights_path = "E:/R/4MOTH\TJJM/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
inception = InceptionV3(include_top=False, input_shape=(155, 155, 3), weights=weights_path)



for layer in inception.layers[-4:]:
    layer.trainable = False
    
    
output = inception.get_layer('mixed7').output

def create_model(opt, loss, metric):
    flatten = Flatten()(output)
    x = Dense(1024, activation='relu')(flatten)
    x = Normalization()(x)
    
    x = Dense(6, activation='softmax')(x)
    model = Model(inception.input, x)
    model.compile(optimizer=opt, loss=loss, metrics=[metric])
    return model

model = create_model(tf.keras.optimizers.Adam(learning_rate=0.0001), 'categorical_crossentropy', 'accuracy')

#model.compile(optimizer=RMSprop(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

#实现学习率衰减
from tensorflow.keras.callbacks import ReduceLROnPlateau

lr_reduction = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
    verbose=1
)
from tensorflow.keras.callbacks import LearningRateScheduler

# 定义余弦退火学习率调度器
def cosine_learning_rate_schedule(epoch, lr):
    min_lr = 0.001 * lr
    cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / 100))  # 假设总 epochs 为 100
    lr = min_lr + 0.5 * (lr - min_lr) * cosine_decay
    return lr

# 创建 LearningRateScheduler 回调函数
lr_scheduler = LearningRateScheduler(cosine_learning_rate_schedule)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


#########################################################

# 注意：训练集、验证集和测试集的路径
train_generator = augmentation.flow_from_directory('E:/R/4MOTH/TJJM/ready_to_gen1/train', target_size=(155, 155), batch_size=24)
valid_generator = augmentation.flow_from_directory('E:/R/4MOTH/TJJM/ready_to_gen1/valid', target_size=(155, 155), batch_size=11)
test_generator = augmentation.flow_from_directory('E:/R/4MOTH/TJJM/ready_to_gen1/test', target_size=(155, 155), batch_size=11)


history = model.fit(
    train_generator,
    steps_per_epoch=300,#需要增大，看来
    epochs=100,
    validation_data=valid_generator,
    validation_steps=60,
    callbacks=[early_stopping,lr_reduction]  # 添加新的回调函数
)


# 模型评估
test_generator = augmentation.flow_from_directory(test_dir, target_size=(155, 155), batch_size=11, class_mode='categorical')
results = model.evaluate(test_generator)
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

precision = precision_score(true_classes, predicted_classes, average='weighted')
recall = recall_score(true_classes, predicted_classes, average='weighted')
accuracy = results[1]  # Assuming accuracy is the second metric

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# 
epochs = range(1, len(history.history['accuracy']) + 1)
plt.plot(epochs, history.history['accuracy'], 'blue', label='Accuracy')
plt.plot(epochs, history.history['val_accuracy'], 'red', label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(epochs, history.history['loss'], 'black', label='Loss')
plt.plot(epochs, history.history['val_loss'], 'green', label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 验证新图
image_url = "https://assets.nhs.uk/nhsuk-cms/images/S_0118_mouth-ulcer_C0345376.width-1534.jpg"
image_data = requests.get(image_url).content
with open('image.jpg', 'wb') as handler:
    handler.write(image_data)

img = cv2.imread('image.jpg')
if img is not None:
    img = cv2.resize(img, (155, 155)) / 255.0
    img = img[..., ::-1]  # BGR to RGB

    prediction = model.predict(np.expand_dims(img, axis=0))
    predicted_class_index = np.argmax(prediction)
    print(f"Predicted class index: {predicted_class_index}")
    
    # ====================== 我帮你完整写好的分类 ======================
    class_indices = {
        0: "Calculus",
        1: "Gingivitus",
        3: "ToothDiscoloration",
        4: "Data caries",
        5: "hypodontia",
        6: "Mouth Ulcer"
    
    }
    # ================================================================

    predicted_class_name = class_indices[predicted_class_index]
    print(f"Predicted class name: {predicted_class_name}")

    # 输出 置信度从高到低 的所有分类
    print("\n置信度从高到低排序：")
    sorted_indices = np.argsort(prediction[0])[::-1]
    for index in sorted_indices:
        print(f"{class_indices[index]}: {prediction[0][index]:.4f}")

else:
    print("Failed to load image.")