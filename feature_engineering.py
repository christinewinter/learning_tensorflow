import os

import tensorflow as tf

from tfx.components import CsvExampleGen
from tfx.components import ExampleValidator
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Transform

import tensorflow_transform.beam as tft_beam
from google.protobuf.json_format import MessageToDict
from tensorflow_transform.tf_metadata import dataset_metadata, schema_utils
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext

import tempfile

# location of the pipeline metadata store
_pipeline_root = './pipeline'

# directory of the raw data files
_data_root = './data'

# path to the raw training data
_data_filepath = os.path.join(_data_root, 'metro_traffic_volume.csv')

# Declare the InteractiveContext and use a local sqlite file as the metadata store.
# You can ignore the warning about the missing metadata config file
context = InteractiveContext(pipeline_root=_pipeline_root)

# Instantiate ExampleGen with the input CSV dataset
example_gen = CsvExampleGen(input_base=_data_root)

# Run the component using the InteractiveContext instance
context.run(example_gen)

# get the artifact object
artifact = example_gen.outputs['examples'].get()[0]
artifact_uri = artifact.uri

# Get the URI of the output artifact representing the training examples, which is a directory
train_uri = os.path.join(artifact_uri, 'Split-train')

# Get the list of files in this directory (all compressed TFRecord files)
tfrecord_filenames = [os.path.join(train_uri, name)
                      for name in os.listdir(train_uri)]

# Create a `TFRecordDataset` to read these files
dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type="GZIP")

# Instantiate StatisticsGen with the ExampleGen ingested dataset
statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

# Run the component
context.run(statistics_gen)

# Plot the statistics generated
context.show(statistics_gen.outputs['statistics'])

# Instantiate SchemaGen with the output statistics from the StatisticsGen
schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])

# Run the component
context.run(schema_gen)

# Visualize the output
context.show(schema_gen.outputs['schema'])

# Instantiate ExampleValidator with the statistics and schema from the previous steps
example_validator = ExampleValidator(statistics=statistics_gen.outputs['statistics'],
                                     schema=schema_gen.outputs['schema'])

# Run the component
context.run(example_validator)

# Visualize the output
context.show(example_validator.outputs['anomalies'])

### For transformations
# Save preprocessing constants in a separate module/config
# Save preprocessing code in a python module with a function called preprocessing_fn that updates the features using
# tensorflow_transform tft
# Load metadata data with schema

import transform_module
from testing_values import feature_description, raw_data

# NOTE: These next two lines are for reloading your traffic_transform module in case you need to
# update your initial solution and re-run this cell. Please do not remove them especially if you
# have revised your solution. Else, your changes will not be detected.
import importlib
importlib.reload(transform_module)

raw_data_metadata = dataset_metadata.DatasetMetadata(schema_utils.schema_from_feature_spec(feature_description))

with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
    transformed_dataset, _ = (
        (raw_data, raw_data_metadata) | tft_beam.AnalyzeAndTransformDataset(transform_module.preprocessing_fn))

transformed_data, transformed_metadata = transformed_dataset

# Instantiate the Transform component
# transform_module_file contains a python module where the transformations are defined in
transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=os.path.abspath(_transform_module_file))

# Run the component.
# The `enable_cache` flag is disabled in case you need to update your transform module file.
context.run(transform, enable_cache=False)

# Get the uri of the transform graph
transform_graph_uri = transform.outputs['transform_graph'].get()[0].uri

# Get the URI of the output artifact representing the transformed examples
train_uri = os.path.join(transform.outputs['transformed_examples'].get()[0].uri, 'Split-train')

# Get the list of files in this directory (all compressed TFRecord files)
tfrecord_filenames = [os.path.join(train_uri, name)
                      for name in os.listdir(train_uri)]

# Create a `TFRecordDataset` to read these files
transformed_dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type="GZIP")