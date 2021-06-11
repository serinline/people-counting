is_object_detected = [0]
left_position_of_previous_detected_object = [0]


def count_objects_x_axis(right, left, roi_position, deviation):
    if abs(((right + left) / 2) - roi_position) < deviation:
        is_object_detected.insert(0, 1)

    if left < left_position_of_previous_detected_object[0]:
        direction = "left"
    else:
        direction = "right"

    left_position_of_previous_detected_object.insert(0, left)

    return direction, is_object_detected
