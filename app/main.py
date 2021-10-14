import sys
import os
import datetime
import numpy as np
import PIL.Image
import cv2
import tensorflow as tf

from ThreadingVideoCapture import ThreadingVideoCapture

MODEL_FLAG = 1

class ImageClassification:
    input_node = "Placeholder:0"
    output_layer = "loss:0"

    def __init__(self, current_dir):
        pb_filename = current_dir + "\\app\\ImageClassification\\model.pb"

        graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(pb_filename, 'rb') as pf:
            graph_def.ParseFromString(pf.read())
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        with tf.compat.v1.Session(graph=self.graph) as sess:
            self.network_input_size = sess.graph.get_tensor_by_name(self.input_node).shape.as_list()[1]
    
    def predict_iamge(self, image):
        with tf.compat.v1.Session(graph=self.graph) as sess:
            prob_tensor = sess.graph.get_tensor_by_name(self.output_layer)
            predictions = sess.run(prob_tensor, {self.input_node: [image]})

            return predictions


class ObjectDetection:
    input_node = "image_tensor:0"
    output_layer = ['detected_boxes:0', 'detected_scores:0', 'detected_classes:0']

    def __init__(self, current_dir):
        pb_filename = current_dir +  "\\app\\ObjectDetection\\model.pb"

        graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(pb_filename, 'rb') as pf:
            graph_def.ParseFromString(pf.read())

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        
        with tf.compat.v1.Session(graph=self.graph) as sess:
            self.network_input_size = sess.graph.get_tensor_by_name(self.input_node).shape.as_list()[1]

    def predict_iamge(self, image):
        inputs = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]

        with tf.compat.v1.Session(graph=self.graph) as sess:
            prob_tensor = [sess.graph.get_tensor_by_name(n) for n in self.output_layer]
            predictions = sess.run(prob_tensor, {self.input_node: inputs})

            return predictions


def crop_center(image, cropx, cropy):
    h, w = image.shape[:2]
    startx = w//2-(cropx//2)
    starty = h//2-(cropy//2)

    return image[starty:starty+cropy, startx:startx+cropx]


def main():
    global MODEL_FLAG

    try:
        print("\nPython %s\n" % sys.version)
        print("===")

        if MODEL_FLAG == "1":
            print("IMAGE CLASSIFICATION")
        else:
            print("OBJECT DETECTION")

        print("")

        current_dir = os.getcwd()

        labels = []
        labels_filename = current_dir + "\\app\\ImageClassification\\labels.txt"

        with open(labels_filename, 'rt') as lf:
            for l in lf:
                labels.append(l.strip())

        graph_def = tf.compat.v1.GraphDef()

        if MODEL_FLAG == "1":
            model = ImageClassification(current_dir)
        else:
            model = ObjectDetection(current_dir)

        stream = ThreadingVideoCapture(0)
        stream.start()

        while True:
            frame = stream.read()

            cropy, cropx = frame.shape[:2]
            min_dim = min(cropy, cropx)
            image = crop_center(frame, min_dim, min_dim)

            image = crop_center(image, model.network_input_size, model.network_input_size)

            try:
                predictions = model.predict_iamge(image)
            except KeyError:
                exit(1)

            if MODEL_FLAG == "1":           
                highest_probability_index = np.argmax(predictions)
                score = predictions[0][highest_probability_index]

                if highest_probability_index == 1:
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)

                if score > 0.8:
                    result = "label: {}, probability: {:.2%}".format(labels[highest_probability_index], float(score))
                    cv2.putText(frame, result, (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, thickness=2)
            else:               
                for p in zip(*predictions):
                    if float(p[1]) > .8:
                        height, width, depth = frame.shape

                        x1 = p[0][0] * width
                        y1 = p[0][1] * height
                        x2 = x1 + width * p[0][2]
                        y2 = y1 + height * p[0][3]

                        if p[2] == 1:
                            color = (0, 255, 255)
                        else:
                            color = (0, 0, 255)

                        cv2.rectangle(frame, (int(x1), int(y1), int(x2), int(y2)), color)

                        text = "{} ({:.2%})".format(labels[p[2]], float(p[1]))
                        cv2.putText(frame, text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, .5, color, thickness=2)

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        stream.stop()

if __name__ == "__main__":
    print("***")
    print("1: Image Classification / 2: Object Detection")
    print("")

    MODEL_FLAG = input()

    main()
