
from utils import backbone
from api import object_counting_api

if __name__ == '__main__':
    # input_video = "./input_images_and_videos/pedestrian_survaillance.mp4"
    input_video = "./input_images_and_videos/people_walking.mp4"

    detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28', 'mscoco_label_map.pbtxt')

    is_color_recognition_enabled = False # set it to true for enabling the color prediction for the detected objects
    roi = 385 # roi line position
    deviation = 1 # the constant that represents the object counting area
    set_of_people = {}
    object_counting_api.cumulative_object_counting_x_axis(input_video, detection_graph, category_index, is_color_recognition_enabled, roi, deviation)  # counting all the objects
