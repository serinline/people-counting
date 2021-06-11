is_object_detected = [0]
bottom_position_of_previous_detected_object = [0]


def count_objects(top, bottom, roi_position, deviation):
    if abs(((bottom + top) / 2) - roi_position) < deviation:
        is_object_detected.insert(0, 1)

    if bottom > bottom_position_of_previous_detected_object[0]:
        direction = "down"
    else:
        direction = "up"

    bottom_position_of_previous_detected_object.insert(0, bottom)

    return direction, is_object_detected
