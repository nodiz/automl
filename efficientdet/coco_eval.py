#  {imageID, x1, y1, w, h, score, class }
import json
import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def cocoBox(megaBox, shape):
    cocobox = megaBox.copy()
    cocobox[0] *= shape[0]
    cocobox[1] *= shape[1]
    cocobox[2] *= shape[0]
    cocobox[3] *= shape[1]
    return [int(x) for x in cocobox]


def oidBox(_oidBox):
    ## predictions are [y,x,y,x] !!!!!
    return list(map(int, [*_oidBox[:2][::-1], _oidBox[3] - _oidBox[1],  _oidBox[2] - _oidBox[0]]))


def coco_eval(results, file, save=True, boxTransform='mega'):
    imageAnnotations = json.load(open(file))
    img_dict = {img['file_name']: img for img in imageAnnotations['images']}
    detections = []
    for res in results:
        id = img_dict[os.path.basename(res['file'])]['id']
        if 'detections' not in res:
            continue
        for ann in res['detections']:
            coco_ann = [id]
            if boxTransform == 'mega':
                h = img_dict[os.path.basename(res['file'])]['height']
                w = img_dict[os.path.basename(res['file'])]['width']
                coco_ann.extend(cocoBox(ann['bbox'], (w, h)))
            elif boxTransform == 'oid':
                coco_ann.extend(oidBox(ann['bbox']))
            coco_ann.append(ann['conf'])
            coco_ann.append(int(ann['category']))
            detections.append(coco_ann)

    coco_gt = COCO(file)
    detections = np.array(detections)
    image_ids = list(set(detections[:, 0]))
    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.params.catIds = [1]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_metrics = coco_eval.stats

    if save:
        f = open(f'out/{os.path.basename(file[:-5])}', 'w')
        f.write(str(coco_metrics))

    return coco_metrics