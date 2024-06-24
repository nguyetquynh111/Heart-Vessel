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
    model = load_model("models/r2u_attention_80e.h5")
    
with keras.saving.custom_object_scope(custom_objects):
    catheter_model = load_model("models/catheter_detect.h5")

def predict_one_image(img, model):
    resized_img = cv2.resize(img, (512, 512))
    X = np.reshape(resized_img, (1, resized_img.shape[0], resized_img.shape[1], 1))
    normalized_X = X/255
    normalized_X = np.rollaxis(normalized_X, 3, 1)
    pred_y = model.predict(normalized_X, verbose=0)
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
    vessel_img, _ = predict_one_image(image, model)
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

# Convert coordinations from bigger images to 512 x 512 images
def adjust_boxes(row):
    width_scale = 512 / row['width']
    height_scale = 512 / row['height']

    row['xmin'] = int(row['xmin'] * width_scale)
    row['ymin'] = int(row['ymin'] * height_scale)
    row['xmax'] = int(row['xmax'] * width_scale)
    row['ymax'] = int(row['ymax'] * height_scale)

    return row

def is_window_overlap(x, y, window_size, box):
    xmin = x - window_size / 2
    ymin = y - window_size / 2
    xmax = x + window_size / 2
    ymax = y + window_size / 2

    box_xmin, box_ymin, box_xmax, box_ymax = box

    if (xmin < box_xmax and xmax > box_xmin and
        ymin < box_ymax and ymax > box_ymin):
        return 1
    else:
        return 0

# ViT

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
    model = keras.Model(inputs=encoded_patches, outputs=logits)
    return model

# Metrics
def multi_label_accuracy(y_true, y_pred):
    y_pred = tf.round(y_pred)
    y_true = tf.round(y_true)
    predictions = tf.reduce_sum(tf.cast(tf.equal(y_true, 1), tf.float32), axis=-1)
    correct_predictions = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 1)), tf.float32), axis=-1)
    accuracy = correct_predictions / predictions
    return tf.reduce_mean(accuracy)

class TwoWayLoss(tf.keras.losses.Loss):
    def __init__(self, Tp=4.0, Tn=1.0, name="two_way_loss"):
        super().__init__(name=name)
        self.Tp = Tp
        self.Tn = Tn
        self.nINF = -1000.0  

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
    return TwoWayLoss(Tp=4.0, Tn=1.0)

if __name__ == '__main__':
    df_train = pd.read_csv('train_labels.csv')
    df_train = df_train.apply(adjust_boxes, axis=1)
    filenames = df_train.filename.values
    images_list = np.zeros((len(filenames), MAX_VECTORS, WINDOW_SIZE, WINDOW_SIZE))

    filenames = df_train.filename.values
    vectors = np.zeros((len(filenames), MAX_VECTORS, 3))
    labels = np.zeros((len(filenames), MAX_VECTORS))
    boxes = df_train[['xmin', 'ymin', 'xmax', 'ymax']].values


    for index, filename in tqdm(enumerate(filenames)):
        img = cv2.imread(os.path.join("data", filename), 0)
        resized_img = cv2.resize(img, (512, 512))
        pred_img, match_img = remove_catheter(img)
        skeletion = skeletonize(pred_img.astype(int))
        vector, img_with_rectangles, centerline_with_rect = vectorize_one_image_using_center_line(img)
        label = []
        for v in vector:
            x, y, color = v
            label.append(is_window_overlap(x, y, WINDOW_SIZE//2, boxes[index]))
        vectors[index] = vector
        label = np.array(label)
        labels[index] = label
    np.save("processed_data/label.npy", labels)
    np.save("processed_data/vector.npy", vectors)

    for index, filename in tqdm(enumerate(filenames)):
        img = cv2.imread(os.path.join("data", filename), 0)
        resized_img = cv2.resize(img, (512, 512))
        pred_img, match_img = remove_catheter(img)
        vector = vectors[index]
        images = np.zeros((MAX_VECTORS, WINDOW_SIZE, WINDOW_SIZE))
        for i, v in enumerate(vector):
            x, y, pixel_count = v
            x = int(x)
            y = int(y)
            xmin = x-WINDOW_SIZE//2
            xmax = x+WINDOW_SIZE//2
            ymin = y-WINDOW_SIZE//2
            ymax = y+WINDOW_SIZE//2
            small_image = pred_img[ymin:ymax, xmin:xmax]
            images[i] = small_image
        images_list[index] = images
    np.save("processed_data/images_list.npy", images_list)

    # Remove images cannot extract vessels
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


    # Model config
    learning_rate = 1e-4
    weight_decay = 1e-4
    batch_size = 2**6
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
    encoded_patches = encoder_model(patches)
    print(encoded_patches.shape)



    labels = smooth_labels(labels)

    train_size = len(images_list)
    initial_learning_rate = 0.001
    final_learning_rate = 0.00001
    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (1 / num_epochs)
    steps_per_epoch = int(train_size/batch_size)
    transformer_units = [
        projection_dim * 2,
        projection_dim
    ]

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=initial_learning_rate,
                    decay_steps=steps_per_epoch,
                    decay_rate=learning_rate_decay_factor,
                    staircase=True)

    optimizer = keras.optimizers.Adam(
        learning_rate=lr_schedule
    )

    input_shape = (MAX_VECTORS, projection_dim)
    model = create_vit_classifier(input_shape, MAX_VECTORS)
    model.compile(
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

    history = model.fit(
        encoded_patches,
        labels,
        batch_size=batch_size,
        epochs=num_epochs,
        callbacks=[checkpoint_callback,
                f1_early_stopping_callback],
    )

    model.save("stenosis_detection.keras")