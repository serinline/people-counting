from utils import backbone
import counting_algorithm

if __name__ == '__main__':
    # input_video = "./files/pedestrian_survaillance.mp4"
    input_video = "./files/people_walking.mp4"

    detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28', 'mscoco_label_map.pbtxt')

    counting_algorithm.people_counting(input_video, detection_graph, category_index)
