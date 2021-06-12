import cv2
import numpy as np
import tensorflow as tf

from numpy import genfromtxt
from utils import visualization


def people_counting(input_video, labels_input_video, detection_graph, category_index, output_video_name):
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
    labels = genfromtxt(labels_input_video, delimiter = ',')
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
            detected_expected = []

            # for all the frames that are extracted from input video
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    # print("end of the video")
                    detected_expected, people_expected = [sum(row[i] for row in detected_expected) for i in range(len(detected_expected[0]))]
                    err = detected_expected / people_expected
                    return round(abs(1 - err) * 100, 2)

                current_frame = frame
                frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(current_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # detect only people
                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes)

                indices = np.argwhere(classes == 1)
                boxes = np.squeeze(boxes[indices])
                scores = np.squeeze(scores[indices])
                classes = np.squeeze(classes[indices])

                # visualize detected objects
                counter = visualization.visualize_boxes_and_labels_on_image(current_frame,
                                                                            np.squeeze(boxes),
                                                                            np.squeeze(classes).astype(np.int32),
                                                                            np.squeeze(scores),
                                                                            category_index)

                expected_number_of_people = labels[int(frame_idx)][1]
                error = abs(expected_number_of_people - counter) * 100 / expected_number_of_people

                # info text
                cv2.putText(current_frame, "Detected : " + str(counter)
                            + " Expected : " + str(expected_number_of_people)
                            + " Error : " + str(round(error, 2)) + "%",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)

                detected_expected.append([counter,expected_number_of_people])

                output_movie.write(current_frame)
                cv2.imshow('object counting', current_frame)  # for real time processing

                # to stop processing press 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
