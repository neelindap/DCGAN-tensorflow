import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

def tf_resize_images(sess, X_img_file_paths):
    X_data = []
    # tf.reset_default_graph()
    X = tf.placeholder(tf.float32, (None, None, 3))
    tf_img = tf.image.resize_images(X, (32, 32),
                                    tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    sess.run(tf.global_variables_initializer())

    # Each image is resized individually as different image may be of different size.
    for index, file_path in enumerate(X_img_file_paths):
        img = Image.open(file_path, "rb")
        resized_img = sess.run(tf_img, feed_dict = {X: img})
        X_data.append(resized_img)

    X_data = np.array(X_data, dtype = np.float32) # Convert to numpy
    return X_data

def get_mask_coord(imshape):
    vertices = np.array([[(0.09 * imshape[1], 0.99 * imshape[0]),
                          (0.43 * imshape[1], 0.32 * imshape[0]),
                          (0.56 * imshape[1], 0.32 * imshape[0]),
                          (0.85 * imshape[1], 0.99 * imshape[0])]], dtype=np.int32)
    return vertices


def get_perspective_matrices(X_img):
    offset = 15
    img_size = (X_img.shape[1], X_img.shape[0])

    # Estimate the coordinates of object of interest inside the image.
    src = np.float32(get_mask_coord(X_img.shape))
    dst = np.float32([[offset, img_size[1]], [offset, 0], [img_size[0] - offset, 0],
                      [img_size[0] - offset, img_size[1]]])

    perspective_matrix = cv2.getPerspectiveTransform(src, dst)
    return perspective_matrix


def perspective_transform(X_img):
    # Doing only for one type of example
    perspective_matrix = get_perspective_matrices(X_img)
    warped_img = cv2.warpPerspective(X_img, perspective_matrix,
                                     (X_img.shape[1], X_img.shape[0]),
                                     flags=cv2.INTER_LINEAR)
    return warped_img


def add_gaussian_noise(X_imgs):
    gaussian_noise_imgs = []
    row, col, _ = X_imgs[0].shape
    # Gaussian distribution parameters
    mean = 0
    var = 0.1
    sigma = var ** 0.5

    for X_img in X_imgs:
        gaussian = np.random.random((row, col, 1)).astype(np.float32)
        gaussian = np.concatenate((gaussian, gaussian, gaussian), axis=2)
        gaussian_img = cv2.addWeighted(X_img, 0.75, 0.25 * gaussian, 0.25, 0)
        gaussian_noise_imgs.append(gaussian_img)
    gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype=np.float32)
    return gaussian_noise_imgs