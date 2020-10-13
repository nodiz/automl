# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Convert raw COCO 2017 dataset to TFRecord.

Example usage:
    python create_coco_tf_record.py --logtostderr \
      --image_dir="${TRAIN_IMAGE_DIR}" \
      --image_info_file="${TRAIN_IMAGE_INFO_FILE}" \
      --object_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --caption_annotations_file="${CAPTION_ANNOTATIONS_FILE}" \
      --output_file_prefix="${OUTPUT_DIR/FILE_PREFIX}" \
      --num_shards=100
"""

import collections
import hashlib
import io
import json
import multiprocessing
import os
import os.path as p

from absl import app
from absl import flags
from absl import logging
import numpy as np
import PIL.Image

from pycocotools import mask
import tensorflow.compat.v1 as tf
from dataset import label_map_util
from dataset import tfrecord_util

flags.DEFINE_boolean(
    'include_masks', False, 'Whether to include instance segmentations masks '
                            '(PNG encoded) in the result. default: False.')
flags.DEFINE_string('image_dir', '', 'Directory containing images.')
flags.DEFINE_string(
    'image_info_file', '', 'File containing image information. '
                           'Tf Examples in the output files correspond to the image '
                           'info entries in this file. If this file is not provided '
                           'object_annotations_file is used if present. Otherwise, '
                           'caption_annotations_file is used to get image info.')
flags.DEFINE_string(
    'object_annotations_file', '', 'File containing object '
                                   'annotations - boxes and instance masks.')
flags.DEFINE_string('caption_annotations_file', '', 'File containing image '
                                                    'captions.')
flags.DEFINE_string('output_file_prefix', '/tmp/train', 'Path to output file')
flags.DEFINE_integer('num_shards', 32, 'Number of shards for output file.')
flags.DEFINE_integer('num_threads', None, 'Number of threads to run.')
FLAGS = flags.FLAGS


def create_tf_example(image,
                      image_dir,
                      bbox_annotations=None,
                      category_index=None,
                      caption_annotations=None,
                      include_masks=False):
    """Converts image and annotations to a tf.Example proto.
  
    Args:
      image: dict with keys: [u'license', u'file_name', u'coco_url', u'height',
        u'width', u'date_captured', u'flickr_url', u'id']
      image_dir: directory containing the image files.
      bbox_annotations:
        list of dicts with keys: [u'segmentation', u'area', u'iscrowd',
          u'image_id', u'bbox', u'category_id', u'id'] Notice that bounding box
          coordinates in the official COCO dataset are given as [x, y, width,
          height] tuples using absolute coordinates where x, y represent the
          top-left (0-indexed) corner.  This function converts to the format
          expected by the Tensorflow Object Detection API (which is which is
          [ymin, xmin, ymax, xmax] with coordinates normalized relative to image
          size).
      category_index: a dict containing COCO category information keyed by the
        'id' field of each category.  See the label_map_util.create_category_index
        function.
      caption_annotations:
        list of dict with keys: [u'id', u'image_id', u'str'].
      include_masks: Whether to include instance segmentations masks
        (PNG encoded) in the result. default: False.
  
    Returns:
      example: The converted tf.Example
      num_annotations_skipped: Number of (invalid) annotations that were ignored.
  
    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    img = cv2.imread(img_path)
    height, width, channel = img.shape
    image_height = height
    image_width = width
    filename = p.basename(img_path)
    image_id = _get_img_id(img_path)
    
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    key = hashlib.sha256(encoded_jpg).hexdigest()
    feature_dict = {
        'image/height':
            tfrecord_util.int64_feature(image_height),
        'image/width':
            tfrecord_util.int64_feature(image_width),
        'image/filename':
            tfrecord_util.bytes_feature(filename.encode('utf8')),
        'image/source_id':
            tfrecord_util.bytes_feature(str(image_id).encode('utf8')),
        'image/key/sha256':
            tfrecord_util.bytes_feature(key.encode('utf8')),
        'image/encoded':
            tfrecord_util.bytes_feature(encoded_jpg),
        'image/format':
            tfrecord_util.bytes_feature('jpeg'.encode('utf8')),
    }
    
    num_annotations_skipped = 0
    if bbox_annotations:
        xmin = []
        xmax = []
        ymin = []
        ymax = []
        is_crowd = []
        category_names = []
        category_ids = []
        area = []
        encoded_mask_png = []
        for object_annotations in bbox_annotations:
            xmin.append(object_annotations["XMin"])
            xmax.append(object_annotations["XMax"])
            ymin.append(object_annotations["YMin"])
            ymax.append(object_annotations["YMax"])
            is_crowd.append(False)
            category_id = int(object_annotations['LabelName'])
            category_ids.append(category_id)
            category_names.append(invClassDict[category_id].encode('utf8'))
            area.append(height * width * (object_annotations["XMax"]-object_annotations["XMin"])
                        *(object_annotations["YMax"]-object_annotations["YMin"]))
            
           if include_masks:
                run_len_encoding = mask.frPyObjects(object_annotations['segmentation'],
                                                    image_height, image_width)
                binary_mask = mask.decode(run_len_encoding)
                if not object_annotations['iscrowd']:
                    binary_mask = np.amax(binary_mask, axis=2)
                pil_image = PIL.Image.fromarray(binary_mask)
                output_io = io.BytesIO()
                pil_image.save(output_io, format='PNG')
                encoded_mask_png.append(output_io.getvalue())
                
        
        feature_dict.update({
            'image/object/bbox/xmin':
                tfrecord_util.float_list_feature(xmin),
            'image/object/bbox/xmax':
                tfrecord_util.float_list_feature(xmax),
            'image/object/bbox/ymin':
                tfrecord_util.float_list_feature(ymin),
            'image/object/bbox/ymax':
                tfrecord_util.float_list_feature(ymax),
            'image/object/class/text':
                tfrecord_util.bytes_list_feature(category_names),
            'image/object/class/label':
                tfrecord_util.int64_list_feature(category_ids),
            'image/object/is_crowd':
                tfrecord_util.int64_list_feature(is_crowd),
            'image/object/area':
                tfrecord_util.float_list_feature(area),
        })
        if include_masks:
            feature_dict['image/object/mask'] = (
                tfrecord_util.bytes_list_feature(encoded_mask_png))
    
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return key, example, num_annotations_skipped


def _pool_create_tf_example(args):
    return create_tf_example(*args)


def _load_object_annotations(annotations_bbox_fname, category_index):
    """Loads object annotation JSON file."""
    
    annotations_bbox = pd.read_csv(annotations_bbox_fname,
                                   usecols=["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"])
    
    annotations_bbox = annotations_bbox[annotations_bbox["LabelName"].isin(category_index)]
    images = annotations_bbox["ImageID"].unique()
    
    img_to_obj_annotation = defaultdict(list)
    logging.info('Building bounding box index.')
    for annotation in annotations_bbox.iterrows():
        image_id = annotation[1]["ImageID"]
        
        img_to_obj_annotation[image_id].append(annotation[1])

    missing_annotation_count = 0
    for image in images:
        if image not in img_to_obj_annotation:
            missing_annotation_count += 1
    
    logging.info('%d images are missing bboxes.', missing_annotation_count)
    
    return img_to_obj_annotation


def _load_classes_info(classes_fname, mask_fname, base_path):
    #classes_list = pd.read_csv(p.join(base_path, classes_fname), header=None)
    #classes_list = classes_list.dropna()
    
    def parse_str(stringa):
        return stringa.replace(" ", "_").replace("_(Animal)", "")
    
    def parse_id(stringa):
        return stringa.replace("/", "")
    
    reader = csv.reader(open(p.join(base_path, classes_fname)))
    classDict = {}
    invClassDict = {}
    for line in reader:
        idc, name = line
        name = parse_str(name)
        classDict[name] = idc
        invClassDict[idc] = name
        
    classes = [parse_str(x) for x in classDict.keys()]
    category_index = invClassDict.keys()

    reader = csv.reader(open(p.join(base_path, mask_fname)))
    
    has_mask_dict = {}
    for line in reader:
        class_id, hasmask = line
        has_mask_dict[class_id] = hasmask
    return classes, category_index, classDict, invClassDict, has_mask_dict


def _get_img_id(image_path):
    return os.path.basename(image_path.replace(".jpg", ""))
    
    
def _create_tf_record_from_animal_annotations(base_dir,
                                            image_dir,
                                            output_path,
                                            num_shards,
                                            object_annotations_file,
                                            classes_csv, mask_csv,
                                            segmentation_fname,
                                            include_masks=False,
                                            ):
    """Loads COCO annotation json files and converts to tf.Record format.
  
    Args:
      image_info_file: JSON file containing image info. The number of tf.Examples
        in the output tf Record files is exactly equal to the number of image info
        entries in this file. This can be any of train/val/test annotation json
        files Eg. 'image_info_test-dev2017.json',
        'instance_annotations_train2017.json',
        'caption_annotations_train2017.json', etc.
      image_dir: Directory containing the image files.
      output_path: Path to output tf.Record file.
      num_shards: Number of output files to create.
      object_annotations_file: JSON file containing bounding box annotations.
      caption_annotations_file: JSON file containing caption annotations.
      include_masks: Whether to include instance segmentations masks
        (PNG encoded) in the result. default: False.
    """
    
    logging.info('writing to output path: %s', output_path)
    writers = [
        tf.python_io.TFRecordWriter(output_path + '-%05d-of-%05d.tfrecord' %
                                    (i, num_shards)) for i in range(num_shards)
    ]
    images = glob.glob(p.join(image_dir, "Centipode", "train", "*.jpg"))  # example

    classes, category_index, classDict, invClassDict, has_mask_dict = \
        _load_classes_info(classes_fname, mask_fname, base_dir)

    img_to_obj_annotation = None
    
    img_to_obj_annotation = (
        _load_object_annotations(object_annotations_file, category_index))
    
    def _get_object_annotation(image_id):
        return img_to_obj_annotation[image_id]
        
    pool = multiprocessing.Pool(FLAGS.num_threads)
    total_num_annotations_skipped = 0
    for idx, (_, tf_example, num_annotations_skipped) in enumerate(
            pool.imap(
                _pool_create_tf_example,
                [(image, _get_object_annotation(_get_img_id(image)),
                  category_index, include_masks)
                 for image in images])):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(images))
        
        total_num_annotations_skipped += num_annotations_skipped
        writers[idx % num_shards].write(tf_example.SerializeToString())
    
    pool.close()
    pool.join()
    
    for writer in writers:
        writer.close()
    
    logging.info('Finished writing, skipped %d annotations.',
                 total_num_annotations_skipped)


def main(_):
    assert FLAGS.base_dir, '`base dir` missing.'
    base_dir = FLAGS.base_dir
    image_dir = p.join(base_dir, "images")

    classes_csv = "classes.csv"
    mask_csv = "hasmask.csv"
    annotations_bbox_fname = 'oidv6-train-annotations-bbox.csv'
    class_descriptions_fname = 'class-descriptions-boxable.csv'
    train_segmentation_fname = 'train-annotations-object-segmentation.csv'
    train_boxable = "train-images-boxable-with-rotation.csv"
    
    directory = os.path.dirname(FLAGS.output_file_prefix)
    if not tf.gfile.IsDirectory(directory):
        tf.gfile.MakeDirs(directory)
    
    _create_tf_record_from_animal_annotations(base_dir, FLAGS.image_dir,
                                            FLAGS.output_file_prefix,
                                            FLAGS.num_shards,
                                            annotations_bbox_fname,
                                            classes_csv, mask_csv,
                                            train_segmentation_fname,
                                            FLAGS.include_masks
                                            )


if __name__ == '__main__':
    app.run(main)
