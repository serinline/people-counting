import cv2
import numpy as np
import tensorflow as tf

from utils import visualization


def people_counting(input_video, detection_graph, category_index, output_video_name):
    """People counting algorithm
    Args:
      input_video - video to process
      detection_graph - TensorFlow computation, represented as a dataflow graph
      category_index - dict containing the same entries as categories, but keyed  by the 'id' field of each category
      output_video_name - name of video created as the result
    """
    # input video
    cap = cv2.VideoCapture(input_video)

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_movie = cv2.VideoWriter(output_video_name, fourcc, fps, (width, height))

    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            # definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # each box represents a part of the image where a particular object was detected
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # for all the frames that are extracted from input video
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    print("end of the video")
                    break

                current_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(current_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # visualize detected objects
                counter = visualization.visualize_boxes_and_labels_on_image(current_frame,
                                                                            np.squeeze(boxes),
                                                                            np.squeeze(classes).astype(np.int32),
                                                                            np.squeeze(scores),
                                                                            category_index)

                # info text
                cv2.putText(current_frame, 'Detected ' + ': ' + str(counter), (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)

                output_movie.write(current_frame)
                print("processing frame")  # just for debugging
                cv2.imshow('object counting', current_frame)  # for real time processing

                # to stop processing press 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
