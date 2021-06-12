import os
import tensorflow as tf
import logging
from google.protobuf import text_format
from protos import string_int_label_map_pb2


def set_model(model_name, label_name):
    # actual model that is used for the object detection.
    path_to_ckpt = model_name + '/frozen_inference_graph.pb'

    # used to add correct label for each box
    path_to_labels = os.path.join('data', label_name)

    num_classes = 90

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.gfile.GFile(path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = load_labelmap(path_to_labels)
    categories = convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = create_category_index(categories)

    return detection_graph, category_index


def _validate_label_map(label_map):
    for item in label_map.item:
        if item.id < 1:
            raise ValueError('Label map ids should be >= 1.')


def create_category_index(categories):
    """
    Args:
    categories: a list of dictionaries with keys:
      'id' - (required)
      'name' - (required) category name
        e.g., 'cat', 'dog', 'pizza'.

      Returns:
        category_index - a dict containing the same entries as categories, but keyed  by the 'id' field of each category
    """
    category_index = {}
    for cat in categories:
        category_index[cat['id']] = cat
    return category_index


def convert_label_map_to_categories(label_map, max_num_classes, use_display_name=True):
    """Loads a label map and returns a list of dicts with following keys:
    'id' - (required)
    'name' - (required)

  Args:
    label_map - StringIntLabelMapProto or None(default categories list is created with max_num_classes categories)
    max_num_classes - maximum number of label indices to include
    use_display_name - (boolean) choose whether to load 'display_name' field as category name
  Returns:
    categories - a list of dictionaries representing all possible categories
  """
    categories = []
    list_of_ids_already_added = []
    if not label_map:
        label_id_offset = 1
        for class_id in range(max_num_classes):
            categories.append({
                'id': class_id + label_id_offset,
                'name': 'category_{}'.format(class_id + label_id_offset)
            })
        return categories
    for item in label_map.item:
        if not 0 < item.id <= max_num_classes:
            logging.info('Ignore item %d since it falls outside of requested '
                         'label range.', item.id)
            continue
        if use_display_name and item.HasField('display_name'):
            name = item.display_name
        else:
            name = item.name
        if item.id not in list_of_ids_already_added:
            list_of_ids_already_added.append(item.id)
            categories.append({'id': item.id, 'name': name})
    return categories


def load_labelmap(path):
    with tf.compat.v1.gfile.GFile(path, 'r') as fid:
        label_map_string = fid.read()
        label_map = string_int_label_map_pb2.StringIntLabelMap()
        try:
            text_format.Merge(label_map_string, label_map)
        except text_format.ParseError:
            label_map.ParseFromString(label_map_string)
    _validate_label_map(label_map)
    return label_map
