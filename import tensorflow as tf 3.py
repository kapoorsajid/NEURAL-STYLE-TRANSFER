import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
def load_and_process_image(image_path):
    img = Image.open(image_path)
    img = img.resize((512, 512))  # Resize to a manageable size
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

def deprocess_image(img):
    img = img.squeeze()
    img = img + 103.939, img + 116.779, img + 123.68  # Reverse preprocessing
    img = np.clip(img, 0, 255).astype('uint8')
    return img
def load_vgg_model():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    return vgg
def get_feature_representations(model, content_image, style_image):
    content_layers = ['block5_conv2']  # Layer to extract content from
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    
    content_outputs = [model.get_layer(name).output for name in content_layers]
    style_outputs = [model.get_layer(name).output for name in style_layers]
    
    model_outputs = content_outputs + style_outputs
    model = tf.keras.Model(inputs=model.input, outputs=model_outputs)
    
    content_features = model(content_image)
    style_features = model(style_image)
    
    return content_features, style_features
def compute_content_loss(content, target):
    return tf.reduce_mean(tf.square(content - target))

def gram_matrix(tensor):
    channels = int(tensor.shape[-1])
    a = tf.reshape(tensor, [-1, channels])
    return tf.matmul(a, a, transpose_a=True)

def compute_style_loss(style, target):
    return tf.reduce_mean(tf.square(gram_matrix(style) - gram_matrix(target)))
def total_variation_loss(image):
    a = tf.square(image[:, :-1, :-1, :] - image[:, 1:, :-1, :])
    b = tf.square(image[:, :-1, :-1, :] - image[:, :-1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 1.25))
def style_transfer(content_image_path, style_image_path, num_iterations=1000, content_weight=1e3, style_weight=1e-2):
    content_image = load_and_process_image(content_image_path)
    style_image = load_and_process_image(style_image_path)
    
    model = load_vgg_model()
    content_features, style_features = get_feature_representations(model, content_image, style_image)
    
    # Initialize the generated image as the content image
    generated_image = tf.Variable(content_image, dtype=tf.float32)
    
    optimizer = tf.optimizers.Adam(learning_rate=0.02)
    
    for i in range(num_iterations):
        with tf.GradientTape() as tape:
            generated_content_features, generated_style_features = get_feature_represent