import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from time import time

if sys.version_info.major == 2:
    import Queue as queue
else:
    import queue

NUM_WORKERS = 1 # Don't use more than 1 worker unless you have enough memory

# Resources
pwd = os.getcwd()
VIDEO = "jabulani_vid2.mp4"
GRAPH = "frozen_inference_graph.pb"
LABEL = "annotation.pbtxt"

# Files
graph_file = os.path.join(pwd, "output_graph", GRAPH)
video_file = os.path.join(pwd, VIDEO)
label_file = os.path.join(pwd, LABEL)

def draw_boxes_and_labels(image_rgb, boxes, scores, classes, max_boxes=20, min_score_threshold=0.99, thickness=5):
    boxes_to_draw = []
    
    # Get the items and store them row-wise
    items = list(zip(boxes, scores, classes))
    items.sort(key=lambda x: x[1], reverse=True) # sort in descending order
    height, width, _ = image_rgb.shape
    for i, item in enumerate(items, start=1):
        if item[1] < min_score_threshold or i > max_boxes:
            break
        #class_name = category_index[item[2]]['name']
        #label = "{}: {:.0f}%".format(class_name, 100*item[1])
        #print(label)
        # Draw the box
        height, width, _ = image_rgb.shape
        ymin, xmin, ymax, xmax = item[0]
        p1 = tuple((int(xmin * width), int(ymin * height)))
        p2 = tuple((int(xmax * width), int(ymax * height)))
        cv2.rectangle(image_rgb, p1, p2, (0,255,0), thickness)


def detect_objects(sess, tf_graph, frame_rgb):
    image_expanded = np.expand_dims(frame_rgb, axis=0)
    image_tensor = tf_graph.get_tensor_by_name('image_tensor:0')

    # Grab the confidence levels for each class
    boxes = tf_graph.get_tensor_by_name('detection_boxes:0')
    scores = tf_graph.get_tensor_by_name('detection_scores:0')
    classes = tf_graph.get_tensor_by_name('detection_classes:0')
    num_detections = tf_graph.get_tensor_by_name('num_detections:0')

    boxes, scores, classes, num_detections = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    #visualization_utils.visualize_boxes_and_labels_on_image_array(frame_rgb, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index, use_normalized_coordinates=True, line_thickness=8)
    return(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32))

class detector(mp.Process):
    def __init__(self, input_q, output_q, exit_event):
        super(detector, self).__init__()
        self.input = input_q
        self.output = output_q
        self.exit = exit_event

    def run(self):
        # Load TF graph 
        with tf.gfile.GFile(graph_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

        self.detection_graph = tf.get_default_graph()
        self.sess = tf.Session(graph=self.detection_graph)

        while not self.exit.is_set():
            try:
                frame = self.input.get(timeout=1)
            except queue.Empty:
                continue
            frame_detect = detect_objects(self.sess, self.detection_graph, frame)
            self.output.put(frame_detect)
        self.sess.close()

    def stop(self):
        self.exit.set()


if __name__ == "__main__":
    workers = []
    in_q = []
    out_q = []

    for i in range(NUM_WORKERS):
        i, o, e = mp.Queue(), mp.Queue(), mp.Event()
        w = detector(i, o, e)
        w.start()
        workers.append(w)
        in_q.append(i)
        out_q.append(o)

    print("\n")
    print(25*"#")
    print("# Starting benchmarking #")
    print(25*"#")

    # Init Video file to read from
    cap = cv2.VideoCapture(video_file)
    counter = 0
    sum = 0
    try:
        while cap.isOpened():
            valid_frame, frame = cap.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for inq in in_q:
                inq.put(frame_rgb)
            duration = -time()
            for outq in out_q:
                outq.get()
            duration += time()
            print("[{:05d}] Detection took {:.2f} ms for {} workers".format(counter, duration * 1000 , NUM_WORKERS))
            counter += 1
            sum += duration * 1000
    except KeyboardInterrupt:
        pass
    finally:
        for worker in workers:
            worker.stop()
            worker.join(5) # wait 5 seconds
            if worker.exitcode:
                worker.terminate()
                print("Have to terminate worker")

    print(25*"#")
    print("# Finished benchmarking")
    print(25*"#")
    average = sum / counter
    print("{} Frames processed. Avg time: {:.2f} ms".format(counter, average))

