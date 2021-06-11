is_object_detected = [0]
left_position_of_previous_detected_object = [0]


def count_objects_x_axis(right, left, deviation , width):
    if abs(((right + left) / 2) - (width/2)) < deviation:
        is_object_detected.insert(0, 1)

    left_position_of_previous_detected_object.insert(0, left)

    return is_object_detected
