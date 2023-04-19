# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

r"""Creates and runs TF2 object detection models.

For local training/evaluation run:
PIPELINE_CONFIG_PATH=path/to/pipeline.config
MODEL_DIR=/tmp/model_outputs
NUM_TRAIN_STEPS=10000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python model_main_tf2.py -- \
  --model_dir=$MODEL_DIR --num_train_steps=$NUM_TRAIN_STEPS \
  --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
  --pipeline_config_path=$PIPELINE_CONFIG_PATH \
  --alsologtostderr
"""
import tensorflow.compat.v2 as tf
import pathlib
import os

from google.protobuf import text_format
from models.research.object_detection import model_lib_v2
from models.research.object_detection.utils import config_util
from models.research.object_detection.builders import model_builder
from models.research.object_detection.utils import ops as utils_ops
from models.research.object_detection.utils import label_map_util
from models.research.object_detection import exporter_lib_v2
from models.research.object_detection.protos import pipeline_pb2
from configs.create_coco_tf_record import generate


def convert_data_to_tfrecords(
        include_masks,
        train_image_dir,
        test_image_dir,
        train_annotations_file,
        test_annotations_file,
        output_dir
):
    if not os.path.exists(output_dir):
        os.system("mkdir " + output_dir)

    generate(
        include_masks,
        train_image_dir,
        test_image_dir,
        train_annotations_file,
        test_annotations_file,
        output_dir
    )


def conversion(
        pipeline_config_path,
        trained_checkpoint_dir,
        output_directory,
        input_type='image_tensor',
        config_override='',
        use_side_inputs=False,
        side_input_shapes='',
        side_input_types='',
        side_input_names=''
):
    tf.enable_v2_behavior()

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    text_format.Merge(config_override, pipeline_config)
    exporter_lib_v2.export_inference_graph(
        input_type, pipeline_config, trained_checkpoint_dir,
        output_directory, use_side_inputs, side_input_shapes,
        side_input_types, side_input_names)


def train_model(
        model_dir,
        pipeline_config_path,
        num_train_steps=None,
        eval_on_train_data=False,
        eval_timeout=3600,
        use_tpu=False,
        tpu_name=None,
        num_workers=1,
        sample_1_of_n_eval_examples=None,
        sample_1_of_n_eval_on_train_examples=5,
        checkpoint_dir=None,
        checkpoint_every_n=1000,
        record_summaries=True
):
    """
    Args:
        model_dir: Path to output model directory
                   where event and checkpoint files will be written. 
        pipeline_config_path: Path to pipeline config file.
        num_train_steps: Number of training steps to run.
        eval_on_train_data: Enable evaluating on train
                            data (only supported in distributed training).
        eval_timeout: Number of seconds to wait for an
                      evaluation checkpoint before exiting.
        use_tpu: Whether to use the tpu or not.
        tpu_name: Name of the Cloud TPU for Cluster Resolvers.
        num_workers: When num_workers > 1, training uses
                     MultiWorkerMirroredStrategy. When num_workers = 1 it uses
                     MirroredStrategy.
        sample_1_of_n_eval_examples: Will sample one of every n
                                     eval input examples, where n is provided.
        sample_1_of_n_eval_on_train_examples: Will sample one of
                                              every n train input examples for evaluation,
                                              where n is provided. This is only used if
                                              `eval_on_train_data` is True.
        checkpoint_dir: Path to directory holding a checkpoint.  If
                        checkpoint_dir` is provided, this binary operates in eval-only mode,
                        writing resulting metrics to `model_dir`.
        checkpoint_every_n: Integer defining how often we checkpoint.
        record_summaries: Whether to record summaries defined by the model
                          or the training pipeline. This does not impact the'
                          summaries of the loss values which are always'
                          recorded.
    """
    if not os.path.exists(model_dir):
        os.system("mkdir " + model_dir)
    if checkpoint_dir:
        model_lib_v2.eval_continuously(
            pipeline_config_path=pipeline_config_path,
            model_dir=model_dir,
            train_steps=num_train_steps,
            sample_1_of_n_eval_examples=sample_1_of_n_eval_examples,
            sample_1_of_n_eval_on_train_examples=(
                sample_1_of_n_eval_on_train_examples),
            checkpoint_dir=checkpoint_dir,
            wait_interval=300, timeout=eval_timeout)
    else:
        if use_tpu:
            # TPU is automatically inferred if tpu_name is None and
            # we are running under cloud ai-platform.
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
                tpu_name)
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.experimental.TPUStrategy(resolver)
        elif num_workers > 1:
            strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        else:
            strategy = tf.compat.v2.distribute.MirroredStrategy()

        with strategy.scope():
            model_lib_v2.train_loop(
                pipeline_config_path=pipeline_config_path,
                model_dir=model_dir,
                train_steps=num_train_steps,
                use_tpu=use_tpu,
                checkpoint_every_n=checkpoint_every_n,
                record_summaries=record_summaries)


def get_inference_model(training_path, pipeline_file, model_dir, path_to_labels):
    filenames = list(pathlib.Path(training_path).glob('*.index'))

    filenames.sort()

    #pipeline_file = 'maskrcnn/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8-colab.config'

    # generally you want to put the last ckpt from training in here
    configs = config_util.get_configs_from_pipeline_file(pipeline_file)
    model_config = configs['model']
    detection_model = model_builder.build(
        model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(
        model=detection_model)
    ckpt.restore(os.path.join(str(filenames[-1]).replace('.index', '')))

    def get_model_detection_function(model):
        """Get a tf.function for detection."""

        @tf.function
        def detect_fn(image):
            """Detect objects in image."""

            image, shapes = model.preprocess(image)
            prediction_dict = model.predict(image, shapes)
            detections = model.postprocess(prediction_dict, shapes)

            return detections, prediction_dict, tf.reshape(shapes, [-1])

        return detect_fn

    detect_fn = get_model_detection_function(detection_model)

    # patch tf1 into `utils.ops`
    utils_ops.tf = tf.compat.v1
    # Patch the location of gfile
    tf.gfile = tf.io.gfile

    def load_model(model_dir):
        model = tf.saved_model.load(str(model_dir))
        return model

    # model_dir = 'finetuned-maskrcnn/saved_model'
    masking_model = load_model(model_dir)

    category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)

    # map labels for inference decoding
    label_map_path = configs['eval_input_config'].label_map_path
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=label_map_util.get_max_label_map_index(label_map),
        use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

    return category_index, label_map_dict, masking_model
