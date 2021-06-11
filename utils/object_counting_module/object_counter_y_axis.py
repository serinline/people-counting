is_object_detected = [0]
bottom_position_of_previous_detected_object = [0]


def count_objects(top, bottom, deviation, height):
    if abs(((bottom + top) / 2) - (height/2)) < deviation:
        is_object_detected.insert(0, 1)

    bottom_position_of_previous_detected_object.insert(0, bottom)

    return is_object_detected
