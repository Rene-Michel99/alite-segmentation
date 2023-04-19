import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import pathlib

from inference_utils.visualization import show_inference


def get_area_of_segmentation(detections, min_confidence=0.5):
    boxes = np.asarray(detections["detection_boxes"][0])
    scores = np.asarray(detections["detection_scores"][0])
    mask = np.asarray(detections["detection_masks_reframed"])
    area = []
    masks_to_return = []
    for i in range(boxes.shape[0]):
        if scores[i] is None or scores[i] > min_confidence:
            box = tuple(boxes[i].tolist())
            if mask[i] is not None and int(np.sum(mask[i].astype(bool))) > 0:
                # decoded_mask = np.uint8(255.0*alpha*(mask[i] > 0))
                area.append(int(np.sum(mask[i].astype(bool))))
                masks_to_return.append(mask[i])
    return area, masks_to_return


def IOU(mask_truth, mask_predicted, debugging=False):
    if mask_truth is None or mask_predicted is None:
        return 0
    tp = cv.bitwise_and(mask_truth, mask_truth.copy(), mask=mask_predicted)
    fp = cv.bitwise_or(mask_truth, mask_truth.copy(), mask=mask_predicted) - mask_truth
    fn = cv.bitwise_or(mask_truth, mask_truth.copy(), mask=mask_predicted) - mask_predicted

    if debugging:
        fig, axr = plt.subplots(1, 2, figsize=(20, 20))
        axr[0].imshow(mask_truth)
        axr[1].imshow(mask_predicted)

    tp = np.sum(tp > 0)
    fp = np.sum(fp > 0)
    fn = np.sum(fn > 0)
    return tp / (tp + fp + fn)


def get_validation_mask(width, height, segmentation):
    thresh = np.zeros(shape=(height, width))

    contours = []
    for item in segmentation:
        points = []
        for i in range(0, len(item), 2):
            points.append((item[i], item[i + 1]))
        contours.append([points])
    contours = np.array(contours, dtype=np.int32)

    cv.drawContours(thresh, contours, -1, (255), 2)
    cv.fillPoly(thresh, pts=contours, color=(255))
    return thresh.astype(np.uint8)


def pad_data(
        masks_predicted,
        masks_validated,
        class_validated,
        class_model,
        penalty
):
    if len(masks_validated) < len(masks_predicted):
        dif = len(masks_predicted) - len(masks_validated)
        class_validated.extend(
            [penalty for _ in range(dif)]
        )
        masks_validated.extend(
            [[0, 0, 0] for _ in range(dif)]
        )
    elif len(masks_predicted) < len(masks_validated):
        dif = len(masks_validated) - len(masks_predicted)
        class_model.extend(
            [penalty for _ in range(dif)]
        )
        masks_predicted.extend(
           [[penalty] for _ in range(dif)]
        )

    return masks_predicted, masks_validated, class_validated, class_model


def save_metrics(
        detections,
        img_info,
        metadata_imgs,
        use_iou=False,
        penalty=-1
):
    _, masks_predicted = get_area_of_segmentation(detections)

    masks_validated = []
    for item in img_info:
        width = metadata_imgs[item['image_id']]['width']
        height = metadata_imgs[item['image_id']]['height']
        masks_validated.append(
            get_validation_mask(width, height, item['segmentation'])
        )

    class_validated = [item['category_id'] for item in img_info]
    class_model = np.asarray(detections['detection_classes'][0]).astype(int).tolist()[:len(masks_predicted)]

    masks_predicted, masks_validated, class_validated, class_model = pad_data(
        masks_predicted, masks_validated,
        class_validated, class_model, penalty
    )

    areas_errors = []
    if len(masks_validated) > 0 and len(masks_predicted) > 0:
        if use_iou:
            areas_errors = [
                IOU(mask_truth, mask_predicted)
                for mask_truth, mask_predicted in zip(masks_validated, masks_predicted)]
        else:
            masks_validated = np.array(masks_validated)
            masks_predicted = np.array(masks_predicted)

            for mask_truth, mask_predicted in zip(masks_validated, masks_predicted):
                mask_truth = np.array(mask_truth)
                mask_predicted = np.array(mask_predicted)
                area_mask_truth = np.sum(mask_truth.astype(bool))
                area_mask_predicted = np.sum(mask_predicted.astype(bool)) if len(mask_predicted) > 1 else -1

                if area_mask_truth > 0 and area_mask_predicted > 0:
                    areas_errors.append(area_mask_truth / area_mask_predicted)
                elif area_mask_predicted < 0:
                    areas_errors.append(penalty)
                else:
                    areas_errors.append(0)
                areas_errors.append(0)
    else:
        areas_errors.append(0)

    arr_precision_class = [int(cl_model == cl_validated)
                           for cl_model, cl_validated in zip(class_model, class_validated)]
    return areas_errors, arr_precision_class


def calc_metrics(data, masking_model):
    metrics_img = []
    class_metrics_imgs = []
    path_to_test_images_dir = pathlib.Path('maskrcnn/data/test')
    test_images_path = sorted(list(path_to_test_images_dir.glob("*.jpg")))
    for i, image_path in enumerate(test_images_path):
        img_id = [
            img['id'] for img in data['images']
            if img['file_name'] == str(image_path).replace('maskrcnn/data/test/', '')
        ][0]
        img_info = [item for item in data['annotations'] if item['image_id'] == img_id]

        print("Calculating metrics...")
        detections, img = show_inference(masking_model, image_path, show_mask=True)
        mask_metrics, class_metrics = save_metrics(detections, img_info, data['images'])
        print("Metrics: ", mask_metrics)
        metrics_img.extend(mask_metrics)
        class_metrics_imgs.extend(class_metrics)
    return np.array(metrics_img), np.array(class_metrics_imgs)
