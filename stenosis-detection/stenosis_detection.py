import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from tqdm import tqdm
import cv2
from keras.models import load_model
import keras
from keras import backend as K
from skimage.morphology import skeletonize
import datetime
import math
from typing import List, Tuple
import torch
import torch.nn as nn

# Global parameters and functions

WINDOW_SIZE = 20
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
MAX_VECTORS = 80


def iou_score(y_pred, y_true, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, -1) + K.sum(y_pred, -1) - intersection
    iou = (intersection + smooth)/(union + smooth)
    return iou

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
custom_objects = {"dice_coef_loss": dice_coef_loss, 'iou_score': iou_score}

with keras.saving.custom_object_scope(custom_objects):
    vessel_model = load_model("models/r2u_attention_80e.h5")
    
with keras.saving.custom_object_scope(custom_objects):
    catheter_model = load_model("models/catheter_detect.h5")

def predict_one_image(img, vessel_model):
    resized_img = cv2.resize(img, (512, 512))
    X = np.reshape(resized_img, (1, resized_img.shape[0], resized_img.shape[1], 1))
    normalized_X = X/255
    normalized_X = np.rollaxis(normalized_X, 3, 1)
    pred_y = vessel_model.predict(normalized_X, verbose=0)
    pred_y[pred_y > 0.5] = 1
    pred_y[pred_y != 1] = 0
    pred_img = np.reshape(pred_y[0]*255, (512, 512))
    match_img = pred_img*resized_img
    return pred_img, match_img

def find_largest_contour(contours):
    max_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    return max_contour

def remove_catheter(image):
    orginial_image = image.copy()
    vessel_img, _ = predict_one_image(image, vessel_model)
    catheter_img, _ = predict_one_image(image, catheter_model)
    subtract_image = vessel_img - catheter_img
    _, binary = cv2.threshold(subtract_image, 50, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = find_largest_contour(contours)
    if len(contours)>1:
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        vessel_img = cv2.bitwise_and(subtract_image, subtract_image, mask=mask)
    resized_img = cv2.resize(orginial_image , (512, 512))    
    return vessel_img/255., resized_img*vessel_img/255.

def distance(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def min_distance(x,y, vector, limit):
    if limit > 0:
        d = [distance(x, y, vector[i][0],vector[i][1]) for i in range(limit)]
        d.sort()
        return d[0]
    return 1000

def vectorize_one_image_using_center_line(img):
    vector = np.zeros((MAX_VECTORS, 3), dtype=np.float32)
    STEP = 5
    pred_img, match_img = remove_catheter(img)

    centerline = skeletonize(pred_img.astype(int))

    if np.all(pred_img == 0):
        return vector, None, None

    img_with_rectangles = cv2.cvtColor(match_img, cv2.COLOR_GRAY2BGR)
    centerline_with_rect = cv2.cvtColor(centerline.astype(np.float32) * 255, cv2.COLOR_GRAY2BGR)
    index = 0
    WS12 = WINDOW_SIZE // 2
    IMAGE_DIM = (IMAGE_HEIGHT, IMAGE_WIDTH)

    for y in range(0, IMAGE_DIM[0], STEP):
        for x in range(0, IMAGE_DIM[1], STEP):
            window = centerline[y:y + STEP, x:x + STEP]
            if np.count_nonzero(window) > 2:
                y_arr, x_arr = np.where(window == 1)
                x_w = int(x_arr.mean()) + x
                y_w = int(y_arr.mean()) + y

                if min_distance(x_w, y_w, vector, index) > WS12 * 1.5:
                    upper_left = (max(0, x_w - WS12), max(0, y_w - WS12))
                    lower_right = (min(IMAGE_WIDTH, x_w + WS12), min(IMAGE_HEIGHT, y_w + WS12))

                    if (lower_right[0] - upper_left[0]) <= 0 or (lower_right[1] - upper_left[1]) <= 0:
                        continue

                    window = pred_img[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]
                    centerline[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]] = 0

                    cv2.rectangle(img_with_rectangles, upper_left, lower_right, (0, 255, 0), 1)
                    cv2.rectangle(centerline_with_rect, upper_left, lower_right, (0, 255, 0), 1)

                    pixel_count = window.sum()
                    vector[index] = [x_w, y_w, pixel_count]
                    index += 1

                    if index >= MAX_VECTORS:
                        return vector, img_with_rectangles, centerline_with_rect

    return vector, img_with_rectangles, centerline_with_rect

## Sort vectors
def most_left_upper_point(points: List[Tuple[float, float]]) -> Tuple[float, float, float]:
    most_left_upper = points[0]

    for point in points[1:]:
        if point[0]==0 and point[1]==0:
            continue
        if point[0] < most_left_upper[0] or (point[0] == most_left_upper[0] and point[1] < most_left_upper[1]):
            most_left_upper = point

    return most_left_upper

def calculate_angle(point1, point2, point3):
    vector1 = (point1[0] - point2[0], point1[1] - point2[1])
    vector2 = (point3[0] - point2[0], point3[1] - point2[1])
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    
    magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
    magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
    
    cosine_theta = dot_product / (magnitude1 * magnitude2)
    cosine_theta = max(-1, min(1, cosine_theta))
    theta_rad = math.acos(cosine_theta)
    
    theta_deg = math.degrees(theta_rad)
    
    return theta_deg

def vector_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        x1, y1 = point1
        x2, y2 = point2
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    
def calculate_ratio(previous_point: Tuple[float, float], current_point: Tuple[float, float], next_point: Tuple[float, float], image) -> float:
    overlap = get_overlap(current_point, next_point, 20, image)
    angle = calculate_angle(previous_point, current_point, next_point)
    return 0.5*overlap + 0.5*angle

def dfs(current_point: Tuple[float, float, float], remaining_points: List[Tuple[float, float, float]], path: List[Tuple[float, float, float]], queue: List[Tuple[float, float, float]], reset_branch: Tuple[float, float], previous_point: Tuple[float, float], image) -> List[Tuple[float, float, float]]:
    if not (reset_branch[0]!=0 and reset_branch[1]!=0) or len(path)==0:
        path.append(current_point)
    if not remaining_points:
        return path
    
    remaining_points.sort(key=lambda point: vector_distance((point[0], point[1]), (current_point[0], current_point[1])))

    branches = []
    for chosen_point in remaining_points[:10]:    
        distance = vector_distance((chosen_point[0], chosen_point[1]), (current_point[0], current_point[1]))
        if not (reset_branch[0]!=0 and reset_branch[1]!=0) and (previous_point[0]!=0 and previous_point[1]!=0):
            point1 = previous_point[:2]
            point2 = current_point[:2]
            angle = calculate_angle(point1, point2, chosen_point)
        else:
            angle = -1
        if distance < 30:
            branches.append(chosen_point)
    
    if len(branches)>1:
        for time_appeared in branches[0:]:
            queue.append(current_point)
        
        if not (reset_branch[0]!=0 and reset_branch[1]!=0) and (previous_point[0]!=0 and previous_point[1]!=0):
            branches.sort(key=lambda point: calculate_ratio(previous_point, current_point, point, image), reverse=True)
        else:
            branches.sort(key=lambda point: get_overlap(point, current_point, 20, image), reverse=True)

    if len(branches)>0:
        previous_point = current_point
        next_point = branches[0]
        reset_branch = [0, 0]
        remove_index = next(i for i, point in enumerate(remaining_points) if np.array_equal(point, next_point))
        remaining_points.pop(remove_index)
    else:
        if len(queue)>0:
            next_point = queue.pop(0)
            reset_branch = next_point[:2]
            previous_point = [0,0]
        else:
            next_point = remaining_points.pop(0)
            reset_branch = [0, 0]
            previous_point = [0,0]
        
    return dfs(next_point, remaining_points, path, queue, reset_branch, previous_point, image) 

def sort_by_distance(points: List[Tuple[float, float, float]], image) -> List[Tuple[float, float, float]]:
    initial_point = most_left_upper_point(points)    
    points = [point for point in points if not np.array_equal(point, initial_point)]

    sorted_points = dfs(initial_point, points, [], [], initial_point, [0,0], image)
    return sorted_points

## Convert coordinations from bigger images to 512 x 512 images
def adjust_boxes(row):
    width_scale = 512 / row['width']
    height_scale = 512 / row['height']

    row['xmin'] = int(row['xmin'] * width_scale)
    row['ymin'] = int(row['ymin'] * height_scale)
    row['xmax'] = int(row['xmax'] * width_scale)
    row['ymax'] = int(row['ymax'] * height_scale)

    return row

def get_overlap(point1, point2, window_size, image):
    x1_min, y1_min = int(point1[0] - window_size//2), int(point1[1] - window_size//2)
    x1_max, y1_max = int(point1[0] + window_size//2), int(point1[1] + window_size//2)
    
    x2_min, y2_min = int(point2[0] - window_size//2), int(point2[1] - window_size//2)
    x2_max, y2_max = int(point2[0] + window_size//2), int(point2[1] + window_size//2)
    
    overlap_x_min = max(x1_min, x2_min)
    overlap_x_max = min(x1_max, x2_max)
    overlap_y_min = max(y1_min, y2_min)
    overlap_y_max = min(y1_max, y2_max)
    
    if overlap_x_min >= overlap_x_max or overlap_y_min >= overlap_y_max:
        return 0
    
    overlap_region = np.zeros((overlap_y_max - overlap_y_min, overlap_x_max - overlap_x_min))
    
    overlap_region1 = image[overlap_y_min:overlap_y_max, overlap_x_min:overlap_x_max]
    overlap_region2 = image[overlap_y_min:overlap_y_max, overlap_x_min:overlap_x_max]
    
    non_zero_count = np.count_nonzero(overlap_region1) + np.count_nonzero(overlap_region2)
    
    return non_zero_count


def calculate_overlap_percentage(x, y, window_size, box):
    xmin = x - window_size / 2
    ymin = y - window_size / 2
    xmax = x + window_size / 2
    ymax = y + window_size / 2

    box_xmin, box_ymin, box_xmax, box_ymax = box

    inter_xmin = max(xmin, box_xmin)
    inter_ymin = max(ymin, box_ymin)
    inter_xmax = min(xmax, box_xmax)
    inter_ymax = min(ymax, box_ymax)

    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)

    window_area = window_size * window_size

    overlap_percentage = (inter_area / window_area) * 100

    return overlap_percentage

## ViT

def smooth_labels(labels, factor=0.1):
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)
    num_classes = tf.shape(labels)[1]
    smoothed_labels = labels * (1 - factor) + (factor / tf.cast(num_classes, tf.float32))
    return smoothed_labels

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patches):
        positions = tf.expand_dims(
            tf.range(start=0, limit=self.num_patches, delta=1), axis=0
        )
        projected_patches = self.projection(patches)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config

def create_vit_classifier(input_shape, num_classes=MAX_VECTORS):
    encoded_patches = keras.Input(shape=input_shape)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = layers.Dense(num_classes, activation="sigmoid")(features)
    stenosis_model = keras.Model(inputs=encoded_patches, outputs=logits)
    return stenosis_model

## Metrics
def multi_label_accuracy(y_true, y_pred):
    y_pred = tf.round(y_pred)
    y_true = tf.round(y_true)
    predictions = tf.reduce_sum(tf.cast(tf.equal(y_true, 1), tf.float32), axis=-1)
    correct_predictions = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 1)), tf.float32), axis=-1)
    accuracy = correct_predictions / predictions
    return tf.reduce_mean(accuracy)

nINF = -100

class TwoWayLoss(tf.keras.losses.Loss):
    def __init__(self, Tp=4.0, Tn=1.0, name="two_way_loss"):
        super().__init__(name=name)
        self.Tp = Tp
        self.Tn = Tn
        self.nINF = -100.0  

    def call(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)

        class_mask = tf.reduce_any(y_true > 0, axis=0)
        sample_mask = tf.reduce_any(y_true > 0, axis=1)

        pmask = tf.where(y_true > 0, 0.0, self.nINF)
        plogit_class = tf.reduce_logsumexp(-y_pred / self.Tp + pmask, axis=0) * self.Tp
        plogit_class = tf.boolean_mask(plogit_class, class_mask)
        plogit_sample = tf.reduce_logsumexp(-y_pred / self.Tp + pmask, axis=1) * self.Tp
        plogit_sample = tf.boolean_mask(plogit_sample, sample_mask)

        nmask = tf.where(y_true == 0, 0.0, self.nINF)
        nlogit_class = tf.reduce_logsumexp(y_pred / self.Tn + nmask, axis=0) * self.Tn
        nlogit_class = tf.boolean_mask(nlogit_class, class_mask)
        nlogit_sample = tf.reduce_logsumexp(y_pred / self.Tn + nmask, axis=1) * self.Tn
        nlogit_sample = tf.boolean_mask(nlogit_sample, sample_mask)

        loss = tf.reduce_mean(tf.nn.softplus(nlogit_class + plogit_class)) + \
               tf.reduce_mean(tf.nn.softplus(nlogit_sample + plogit_sample))
        return loss

def get_criterion():
    return TwoWayLoss(Tp=1.0, Tn=1.0)

if __name__ == '__main__':
    ## Training phase
    df_train = pd.read_csv('train_labels.csv')
    df_train = df_train.apply(adjust_boxes, axis=1)
    filenames = df_train.filename.values
    images_list = np.zeros((len(filenames), MAX_VECTORS, WINDOW_SIZE, WINDOW_SIZE))

    filenames = df_train.filename.values
    vectors = np.zeros((len(filenames), MAX_VECTORS, 3))
    labels = np.zeros((len(filenames), MAX_VECTORS))
    boxes = df_train[['xmin', 'ymin', 'xmax', 'ymax']].values


    # for index, filename in tqdm(enumerate(filenames)):
    #     img = cv2.imread(os.path.join("data", filename), 0)
    #     pred_img, match_img = remove_catheter(img)
    #     vector, img_with_rectangles, centerline_with_rect = vectorize_one_image_using_center_line(img)
    #     vector = sort_by_distance(vector, pred_img)
    #     label = np.zeros((MAX_VECTORS))
    #     filtered_vector = np.zeros((MAX_VECTORS, 3))
    #     images = np.zeros((MAX_VECTORS, WINDOW_SIZE, WINDOW_SIZE))
    #     count = 0

    #     for v in vector:
    #         x, y, pixel_count = v
    #         x = int(x)
    #         y = int(y)
    #         xmin = x-WINDOW_SIZE//2
    #         xmax = x+WINDOW_SIZE//2
    #         ymin = y-WINDOW_SIZE//2
    #         ymax = y+WINDOW_SIZE//2
    #         small_image = match_img[ymin:ymax, xmin:xmax]
    #         if small_image.shape[0]==20 and small_image.shape[1]==20:
    #             filtered_vector[count] = v
    #             images[count] = small_image
    #             label[count] = int(calculate_overlap_percentage(x, y, WINDOW_SIZE//2, boxes[index])>=50)
    #             count+=1

    #     vectors[index] = filtered_vector
    #     labels[index] = label
    #     images_list[index] = images
    # np.save("processed_data/label.npy", labels)
    # np.save("processed_data/vector.npy", vectors)
    # np.save("processed_data/images_list.npy", images_list)

    

    ### Remove images cannot extract vessels
    chosen_index = []
    labels = np.load("processed_data/label.npy")
    vectors = np.load("processed_data/vector.npy")
    for index, label in enumerate(labels):
        check = len(np.where(label==1)[0])
        if check>1:
            chosen_index.append(index)
    vectors = vectors[chosen_index]
    labels = labels[chosen_index]
    images_list = np.load("processed_data/images_list.npy")
    images_list = images_list[chosen_index]
    print("N = ", len(images_list))


    ### Model config
    learning_rate = 1e-4
    weight_decay = 0.1
    batch_size = 2**5
    num_epochs = 50
    patch_size = WINDOW_SIZE
    num_patches = MAX_VECTORS
    projection_dim = 2**8
    num_heads = 8
    transformer_units = [
        projection_dim * 2,
        projection_dim
    ] 

    transformer_layers = 8
    mlp_head_units = [
        2 ** 11,
        2 ** 10,
    ] 
    N = len(images_list)
    channels = 1


    patches = images_list
    patches = tf.reshape(patches, (N, num_patches, patch_size * patch_size * channels))
    encoder = PatchEncoder(num_patches=num_patches, projection_dim=projection_dim)
    encoder_model = tf.keras.Sequential([encoder])
    encoder_model.save('patch_encoder.keras')
    # encoder_model.load_weights('patch_encoder.keras')
    encoded_patches = encoder_model(patches)
    print(encoded_patches.shape)



    # labels = smooth_labels(labels)

    train_size = len(images_list)
    transformer_units = [
        projection_dim * 2,
        projection_dim
    ]

    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    input_shape = (MAX_VECTORS, projection_dim)
    stenosis_model = create_vit_classifier(input_shape, MAX_VECTORS)
    stenosis_model.compile(
        optimizer=optimizer,
        loss=get_criterion(),
        metrics=[
            multi_label_accuracy,
        ],
    )

    weight_filename = str(datetime.datetime.now().strftime("%d %b %Y %I:%M%p")) + '__checkpoint.weights.h5'
    checkpoint_filepath = "ViT_weights/" + str(weight_filename)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="multi_label_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )
    f1_early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='multi_label_accuracy',
        patience=10,
        mode='max',
        restore_best_weights=True,
        verbose=1
    )

    history = stenosis_model.fit(
        encoded_patches,
        labels,
        batch_size=batch_size,
        epochs=num_epochs,
        callbacks=[checkpoint_callback,
                f1_early_stopping_callback],
    )

    stenosis_model.save("stenosis_detection.keras")
    # stenosis_model.load_weights("stenosis_detection.keras")


    ## Test phase

    df_test = pd.read_csv('test_labels.csv')
    df_test = df_test.apply(adjust_boxes, axis=1)
    filenames = df_test.filename.values
    images_list = np.zeros((len(filenames), MAX_VECTORS, WINDOW_SIZE, WINDOW_SIZE))

    vectors = np.zeros((len(filenames), MAX_VECTORS, 3))
    labels = np.zeros((len(filenames), MAX_VECTORS))
    boxes = df_test[['xmin', 'ymin', 'xmax', 'ymax']].values


    # for index, filename in tqdm(enumerate(filenames)):
    #     img = cv2.imread(os.path.join("data", filename), 0)
    #     pred_img, match_img = remove_catheter(img)
    #     vector, img_with_rectangles, centerline_with_rect = vectorize_one_image_using_center_line(img)
    #     vector = sort_by_distance(vector, pred_img)
    #     label = np.zeros((MAX_VECTORS))
    #     filtered_vector = np.zeros((MAX_VECTORS, 3))
    #     images = np.zeros((MAX_VECTORS, WINDOW_SIZE, WINDOW_SIZE))
    #     count = 0

    #     for v in vector:
    #         x, y, pixel_count = v
    #         x = int(x)
    #         y = int(y)
    #         xmin = x-WINDOW_SIZE//2
    #         xmax = x+WINDOW_SIZE//2
    #         ymin = y-WINDOW_SIZE//2
    #         ymax = y+WINDOW_SIZE//2
    #         small_image = pred_img[ymin:ymax, xmin:xmax]
    #         if small_image.shape[0]==20 and small_image.shape[1]==20:
    #             filtered_vector[count] = v
    #             images[count] = small_image
    #             label[count] = int(calculate_overlap_percentage(x, y, WINDOW_SIZE//2, boxes[index])>=50)
    #             count+=1

    #     vectors[index] = filtered_vector
    #     labels[index] = label
    #     images_list[index] = images
    # np.save("processed_data/test_label.npy", labels)
    # np.save("processed_data/test_vector.npy", vectors)
    # np.save("processed_data/test_images_list.npy", images_list)

    ### Remove images cannot extract vessels
    chosen_index = []
    labels = np.load("processed_data/test_label.npy")
    vectors = np.load("processed_data/test_vector.npy")
    for index, label in enumerate(labels):
        check = len(np.where(label==1)[0])
        if check>1:
            chosen_index.append(index)
    vectors = vectors[chosen_index]
    labels = labels[chosen_index]
    images_list = np.load("processed_data/test_images_list.npy")
    images_list = images_list[chosen_index]

    patches = images_list
    channels = 1
    patches = tf.reshape(patches, (len(images_list), num_patches, patch_size * patch_size * channels))
    encoded_patches = encoder_model(patches)
    pred = stenosis_model.predict(encoded_patches, verbose=0)
    print(multi_label_accuracy(labels, pred))
