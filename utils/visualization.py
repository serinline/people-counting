import collections

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import numpy as np

from utils.statics import STANDARD_COLORS


def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color='red', thickness=4, display_str_list=()):
    """Adds a bounding box to an image.
    Args:
      image - current frame in this case
      ymin, xmin,ymax, xmax of bounding box
      color - color to draw bounding box
      thickness - line thickness
      display_str_list - list of strings to display in box
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size

    # draw bounding box around human
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)

    try:
        font = ImageFont.truetype('arial.ttf', 16)
    except IOError:
        font = ImageFont.load_default()

    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height

    # Reverse list and print from bottom to top
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)], fill=color)
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill='black',
            font=font)
        text_bottom -= text_height - 2 * margin


def visualize_boxes_and_labels_on_image(image, boxes, classes, scores, category_index, max_boxes_to_draw=None,
                                        min_score_thresh=.5, line_thickness=4):
    """Groups boxes that correspond to the same location and creates a display string for each detection and overlays
    these on the image; modifies the image in place, and returns that same image.

    Args:
      image - uint8 numpy array with shape (img_height, img_width, 3)
      boxes - a numpy array of shape [N, 4]
      classes - a numpy array of shape [N]; class indices are 1-based, and match the keys in the label map
      scores - a numpy array of shape [N] or None(if None plot all boxes as black with no classes or scores)
      category_index - a dict with category dictionaries (each holding category `id`, `name`) keyed by category indices
      max_boxes_to_draw - maximum number of boxes to visualize(if None draws all boxes)
      min_score_thresh - minimum score threshold for a box to be visualized
      line_thickness - line width of the boxes

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes
    """
    global class_name

    total_people_counter = 0

    # Create a display string (and color) for every box location, group any boxes that correspond to the same location.
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if scores is None:
                box_to_color_map[box] = 'black'
            else:
                if classes[i] in category_index.keys():
                    class_name = category_index[classes[i]]['name']
                display_str = '{}: {}%'.format(class_name, int(100 * scores[i]))

                box_to_display_str_map[box].append(display_str)
                box_to_color_map[box] = STANDARD_COLORS[classes[i] % len(STANDARD_COLORS)]

    # Draw all boxes onto image.
    for box, color in box_to_color_map.items():
        ymin, xmin, ymax, xmax = box

        total_people_counter = total_people_counter + 1

        draw_bounding_box_on_image_array(image, ymin, xmin, ymax, xmax, color=color,
                                         thickness=line_thickness, display_str_list=box_to_display_str_map[box])
    return total_people_counter


def draw_bounding_box_on_image_array(image, ymin, xmin, ymax, xmax, color='red', thickness=4, display_str_list=()):
    """Adds a bounding box to an image (numpy array).
    Args:
      image - a numpy array with shape [height, width, 3]
      ymin,xmin,ymax,xmax of bounding box
      color - default is red.
      thickness - line thickness
      display_str_list - list of strings to display in box
    """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color, thickness, display_str_list)
    np.copyto(image, np.array(image_pil))
