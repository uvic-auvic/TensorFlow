import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import threading

if sys.version_info.major == 2:
    import Queue as queue
else:
    import queue

# Settings
FPS = 30

# Resources
pwd = os.getcwd()
VIDEO = "SET_NAME_OF_VIDEO_HERE.mp4"
GRAPH = "frozen_inference_graph.pb"
LABEL = "annotation.pbtxt"

# Files
graph_file = os.path.join(pwd, GRAPH)
video_file = os.path.join(pwd, VIDEO)
label_file = os.path.join(pwd, LABEL)

# Label annotations
#label_map = label_map_util.load_labelmap(label_file)
#categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1, use_display_name=True)
#category_index = label_map_util.create_category_index(categories)

def draw_boxes_and_labels(image_rgb, boxes, scores, classes, max_boxes=20, min_score_threshold=0.80, thickness=3):
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


def detect_objects(sess, tf_graph, frame_rgb, bcsd):
    image_expanded = np.expand_dims(frame_rgb, axis=0)
    
    b, s, c, d, t = bcsd
    boxes, scores, classes, num_detections = sess.run([b, s, c, d], feed_dict={t: image_expanded})

    #visualization_utils.visualize_boxes_and_labels_on_image_array(frame_rgb, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index, use_normalized_coordinates=True, line_thickness=8)
    return(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32))

class detector(threading.Thread):
    def __init__(self, input_q, output_q):
        super(detector, self).__init__()
        self.input = input_q
        self.output = output_q
        self.exit = threading.Event()

        # Load TF graph 
        with tf.gfile.GFile(graph_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

        self.detection_graph = tf.get_default_graph()
        self.sess = tf.Session(graph=self.detection_graph)
        
        detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.bscdt = (detection_boxes, detection_scores, detection_classes, num_detections, image_tensor)

    def run(self):
        while not self.exit.is_set():
            try:
                frame = self.input.get(timeout=1)
            except queue.Empty:
                continue
            frame_detect = detect_objects(self.sess, self.detection_graph, frame, self.bscdt)
            self.output.put(frame_detect)
        self.sess.close()

    def stop(self):
        self.exit.set()
        


if __name__ == "__main__":
    in_q = queue.Queue()
    out_q = queue.Queue()
    tf_worker = detector(in_q, out_q)
    tf_worker.start()

    # Init Video file to read from
    cap = cv2.VideoCapture(video_file)
    delay_time_ms = int(1000 / (FPS))

    counter = 0
    # Start looping over video
    while cap.isOpened():
        valid_frame, frame = cap.read()
        if not valid_frame or frame is None:
            break

        # Apply detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        in_q.put(frame_rgb) 
        detection_parameters = out_q.get()
        draw_boxes_and_labels(frame_rgb, *detection_parameters)
        frame_bgr_out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow(VIDEO, frame_bgr_out)
        if cv2.waitKey(delay_time_ms) == ord('q'):
            break
        counter += 1

    # Kill the thread
    tf_worker.stop()
    tf_worker.join()
    cv2.destroyAllWindows()

