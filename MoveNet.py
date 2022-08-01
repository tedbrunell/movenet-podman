#!/usr/bin/env python
# coding: utf-8

# Import Libraries

import sys
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

class pose(object):

    def __init__(self):
        saved_path = 'model/movenet_singlepose_lightning_4'
        self.model = tf.saved_model.load(saved_path)

        threshold = .3

        self.cap = cv2.VideoCapture(-1, cv2.CAP_V4L)

        # Optional if you are using a GPU
        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # for gpu in gpus:
        #    tf.config.experimental.set_memory_growth(gpu, True)

    def get_pose(self):

        EDGES = {
            (0, 1): 'm',
            (0, 2): 'c',
            (1, 3): 'm',
            (2, 4): 'c',
            (0, 5): 'm',
            (0, 6): 'c',
            (5, 7): 'm',
            (7, 9): 'm',
            (6, 8): 'c',
            (8, 10): 'c',
            (5, 6): 'y',
            (5, 11): 'm',
            (6, 12): 'c',
            (11, 12): 'y',
            (11, 13): 'm',
            (13, 15): 'm',
            (12, 14): 'c',
            (14, 16): 'c'
            }

        movenet = self.model.signatures['serving_default']

        # Draw Joints/Points
        def draw_keypoints(frame, keypoints, confidence_threshold):
            y, x, c = frame.shape
            shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
            for kp in shaped:
                ky, kx, kp_conf = kp
                if kp_conf > confidence_threshold:
                    cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1)

        # Draw lines
        def draw_connections(frame, keypoints, edges, confidence_threshold):
            y, x, c = frame.shape
            shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
            for edge, color in edges.items():
                p1, p2 = edge
                y1, x1, c1 = shaped[p1]
                y2, x2, c2 = shaped[p2]
                if (c1 > confidence_threshold) & (c2 > confidence_threshold):
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 3)

        # Drawing Loop
        def drawing_loop(frame, keypoints_with_scores, edges, confidence_threshold):
            for person in keypoints_with_scores:
                draw_connections(frame, person, edges, confidence_threshold)
                draw_keypoints(frame, person, confidence_threshold)

        # Main Loop
        while self.cap.isOpened():
            ret, frame = self.cap.read()

            # Resize and pad the image to keep the aspect ratio and fit the expected size.
            tf_image = frame.copy()
            tf_image = tf.image.resize_with_pad(tf.expand_dims(tf_image, axis=0), 192,192)
            image = tf.cast(tf_image, dtype=tf.int32)

            # Detect image
            results = movenet(image)
            keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((1,17,3))

            # Render keypoints
            drawing_loop(frame, keypoints_with_scores, EDGES, 0.2)

            # Display the image
            ret, jpeg = cv2.imencode(".jpg", cv2.flip(frame,1))
            return jpeg.tobytes()

            if cv2.waitKey(10) & 0xFF==ord('q'):
                break

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()
