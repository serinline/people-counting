from utils import backbone
import counting_algorithm

if __name__ == '__main__':
    # input_video = "./files/pedestrian_survaillance.mp4"
    input_video = "./files/people_walking_15s.mp4"
    # labels_input_video = "./labels/pedestrian_survaillance_people_per_frame.csv"
    labels_input_video = "./labels/people_walking_15s_per_frame.csv"
    output_video_name = "result.avi"

    detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28', 'mscoco_label_map.pbtxt')

    counting_algorithm.people_counting(input_video, labels_input_video, detection_graph, category_index, output_video_name)
