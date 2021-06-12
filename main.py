from utils import backbone
import counting_algorithm

if __name__ == '__main__':
    input_video1 = "./files/people_walking_15s.mp4"
    input_video2 = "./files/video1_800x600.mp4"
    input_video3 = "./files/pedestrian_survaillance.mp4"

    labels_input_video1 = "./labels/people_walking_15s_per_frame.csv"
    labels_input_video2 = "./labels/video1_people_per_frame.csv"
    labels_input_video3 = "./labels/pedestrian_survaillance_people_per_frame.csv"

    output_video_name1 = "result1.avi"
    output_video_name2 = "result2.avi"
    output_video_name3 = "result3.avi"

    detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28', 'mscoco_label_map.pbtxt')

    avg_error1 = counting_algorithm.people_counting(input_video1, labels_input_video1, detection_graph, category_index, output_video_name1)
    avg_error2 = counting_algorithm.people_counting(input_video2, labels_input_video2, detection_graph, category_index, output_video_name2)
    avg_error3 = counting_algorithm.people_counting(input_video3, labels_input_video3, detection_graph, category_index, output_video_name3)

    print("Average error for video {} = {}%".format(input_video1, avg_error1))
    print("Average error for video {} = {}%".format(input_video2, avg_error2))
    print("Average error for video {} = {}%".format(input_video3, avg_error3))
