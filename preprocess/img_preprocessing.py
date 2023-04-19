import os
import json
import cv2 as cv


def generate_labelmap(path_json, output_filepath):
  with open(path_json, 'rb') as f:
    data = json.load(f)

    labelmap = data['categories']
    for i, item in enumerate(labelmap):
      if item['id'] == 0:
        labelmap.pop(i)
        break

    with open(os.path.join(output_filepath, 'labelmap.pbtxt'), 'w') as f:
      for item in labelmap:
        st_item = str(item).replace('{', '{\n')
        st_item = st_item.replace('}', '\n}\n')
        st_item = st_item.replace(',', ',\n')
        st_item = st_item.replace("'id'", '    id')
        st_item = st_item.replace("'name'", '   name')
        st_item = st_item.replace(
            "'supercategory': '{}'".format(item['supercategory']),
            '   display_name: "{}"'.format(item['name']))
        st_item = st_item.replace("'", '"')
        f.write('item '+str(st_item))


def create_dirs(path_preprocessed):
    os.system("mkdir " + path_preprocessed.split("/")[0])
    os.system("mkdir " + path_preprocessed)
    os.system("mkdir {}/{}".format(path_preprocessed, 'train'))
    os.system("mkdir {}/{}".format(path_preprocessed, 'test'))
    os.system("mkdir {}/{}".format(path_preprocessed, 'valid'))
    print("Directories created")


def clahe_preprocess(img, params):
    # TODO: Make clahe preprocess function
    return img


def preprocess_dataset(
        params,
        current_dataset,
        annotations='_annotations.coco.json',
        use_clahe=False,
        verbose=False
):
    dataset_paths = [
        os.path.join(current_dataset, 'train'),
        os.path.join(current_dataset, 'test'),
        os.path.join(current_dataset, 'valid')
    ]
    path_preprocessed = 'maskrcnn/data'

    if os.path.exists(path_preprocessed):
        os.system('rm -rf' + path_preprocessed)

    print("Preprocessing dataset")

    create_dirs(path_preprocessed)
    valid_img_extensions = ['jpg', 'png', 'jpeg', 'bmp', 'tif']
    for dataset in dataset_paths:
        type_dir = dataset.split('/')[1]

        for filename in os.listdir(dataset):
            if verbose:
                print("Preprocessing {}".format(dataset, filename))

            try:
                extension = filename.split('.')[len(filename.split('.')) - 1]
                if extension in valid_img_extensions:
                    img = cv.imread('{}/{}'.format(dataset, filename))

                    if use_clahe:
                        img = clahe_preprocess(img, params)
                    else:
                        img = cv.bilateralFilter(img, params[0], params[1], params[2])

                    cv.imwrite('{}/{}/{}'.format(path_preprocessed, type_dir, filename), img)
            except Exception as e:
                print("Error trying to process {}: {}".format(os.path.join(dataset, filename), e))

        os.system('cp {} {}'.format(
            os.path.join(dataset, annotations),
            os.path.join(path_preprocessed, type_dir, annotations)
        ))
    generate_labelmap('unprocessed_data/train/{}'.format(annotations), path_preprocessed)
