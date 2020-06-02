# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Fine-tunes an ELECTRA model on a downstream task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json

import tensorflow.compat.v1 as tf

import configure_finetuning
from model import modeling
from model import optimization
from util import training_utils
from util import utils

from model import tokenization

import abc
import csv
import os

import numpy as np
import scipy
import sklearn

from typing import List, Tuple
from model import modeling
import random

def get_shared_feature_specs(config: configure_finetuning.FinetuningConfig):
  """Non-task-specific model inputs."""
  return [
      FeatureSpec("input_ids", [config.max_seq_length]),
      FeatureSpec("input_mask", [config.max_seq_length]),
      FeatureSpec("segment_ids", [config.max_seq_length]),
      FeatureSpec("task_id", []),
  ]


class FeatureSpec(object):
  """Defines a feature passed as input to the model."""

  def __init__(self, name, shape, default_value_fn=None, is_int_feature=True):
    self.name = name
    self.shape = shape
    self.default_value_fn = default_value_fn
    self.is_int_feature = is_int_feature

  def get_parsing_spec(self):
    return tf.io.FixedLenFeature(
        self.shape, tf.int64 if self.is_int_feature else tf.float32)

  def get_default_values(self):
    if self.default_value_fn:
      return self.default_value_fn(self.shape)
    else:
      return np.zeros(
          self.shape, np.int64 if self.is_int_feature else np.float32)



class Preprocessor(object):
  """Class for loading, preprocessing, and serializing fine-tuning datasets."""

  def __init__(self, config: configure_finetuning.FinetuningConfig, tasks):
    self._config = config
    self._tasks = tasks
    self._name_to_task = {task.name: task for task in tasks}

    self._feature_specs = get_shared_feature_specs(config)
    for task in tasks:
      self._feature_specs += task.get_feature_specs()
    self._name_to_feature_config = {
        spec.name: spec.get_parsing_spec()
        for spec in self._feature_specs
    }
    assert len(self._name_to_feature_config) == len(self._feature_specs)

  def prepare_train(self):
    return self._serialize_dataset(self._tasks, True, "train")

  def prepare_predict(self, tasks, split):
    return self._serialize_dataset(tasks, False, split)

  def _serialize_dataset(self, tasks, is_training, split):
    """Write out the dataset as tfrecords."""
    dataset_name = "_".join(sorted([task.name for task in tasks]))
    dataset_name += "_" + split
    dataset_prefix = os.path.join(
        self._config.preprocessed_data_dir, dataset_name)
    tfrecords_path = dataset_prefix + ".tfrecord"
    metadata_path = dataset_prefix + ".metadata"
    batch_size = (self._config.train_batch_size if is_training else
                  self._config.eval_batch_size)

    utils.log("Loading dataset", dataset_name)
    n_examples = None
    if (self._config.use_tfrecords_if_existing and
        tf.io.gfile.exists(metadata_path)):
      n_examples = utils.load_json(metadata_path)["n_examples"]

    if n_examples is None:
      utils.log("Existing tfrecords not found so creating")
      examples = []
      for task in tasks:
        task_examples = task.get_examples(split)
        examples += task_examples
      if is_training:
        random.shuffle(examples)
      utils.mkdir(tfrecords_path.rsplit("/", 1)[0])
      n_examples = self.serialize_examples(
          examples, is_training, tfrecords_path, batch_size)
      utils.write_json({"n_examples": n_examples}, metadata_path)

    input_fn = self._input_fn_builder(tfrecords_path, is_training)
    if is_training:
      steps = int(n_examples // batch_size * self._config.num_train_epochs)
    else:
      steps = n_examples // batch_size

    return input_fn, steps

  def serialize_examples(self, examples, is_training, output_file, batch_size):
    """Convert a set of `InputExample`s to a TFRecord file."""
    n_examples = 0
    with tf.io.TFRecordWriter(output_file) as writer:
      for (ex_index, example) in enumerate(examples):
        if ex_index % 2000 == 0:
          utils.log("Writing example {:} of {:}".format(
              ex_index, len(examples)))
        for tf_example in self._example_to_tf_example(
            example, is_training,
            log=self._config.log_examples and ex_index < 1):
          writer.write(tf_example.SerializeToString())
          n_examples += 1
      # add padding so the dataset is a multiple of batch_size
      while n_examples % batch_size != 0:
        writer.write(self._make_tf_example(task_id=len(self._config.task_names))
                     .SerializeToString())
        n_examples += 1
    return n_examples

  def _example_to_tf_example(self, example, is_training, log=False):
    examples = self._name_to_task[example.task_name].featurize(
        example, is_training, log)
    if not isinstance(examples, list):
      examples = [examples]
    for example in examples:
      yield self._make_tf_example(**example)

  def _make_tf_example(self, **kwargs):
    """Make a tf.train.Example from the provided features."""
    for k in kwargs:
      if k not in self._name_to_feature_config:
        raise ValueError("Unknown feature", k)
    features = collections.OrderedDict()
    for spec in self._feature_specs:
      if spec.name in kwargs:
        values = kwargs[spec.name]
      else:
        values = spec.get_default_values()
      if (isinstance(values, int) or isinstance(values, bool) or
          isinstance(values, float) or isinstance(values, np.float32) or
          (isinstance(values, np.ndarray) and values.size == 1)):
        values = [values]
      if spec.is_int_feature:
        feature = tf.train.Feature(int64_list=tf.train.Int64List(
            value=list(values)))
      else:
        feature = tf.train.Feature(float_list=tf.train.FloatList(
            value=list(values)))
      features[spec.name] = feature
    return tf.train.Example(features=tf.train.Features(feature=features))

  def _input_fn_builder(self, input_file, is_training):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
      """The actual input function."""
      d = tf.data.TFRecordDataset(input_file)
      if is_training:
        d = d.repeat()
        d = d.shuffle(buffer_size=100)
      return d.apply(
          tf.data.experimental.map_and_batch(
              self._decode_tfrecord,
              batch_size=params["batch_size"],
              drop_remainder=True))

    return input_fn

  def _decode_tfrecord(self, record):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(record, self._name_to_feature_config)
    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name, tensor in example.items():
      if tensor.dtype == tf.int64:
        example[name] = tf.cast(tensor, tf.int32)
      else:
        example[name] = tensor
    return example

class Scorer(object):
  """Abstract base class for computing evaluation metrics."""

  __metaclass__ = abc.ABCMeta

  def __init__(self):
    self._updated = False
    self._cached_results = {}

  @abc.abstractmethod
  def update(self, results):
    self._updated = True

  @abc.abstractmethod
  def get_loss(self):
    pass

  @abc.abstractmethod
  def _get_results(self):
    return []

  def get_results(self, prefix=""):
    results = self._get_results() if self._updated else self._cached_results
    self._cached_results = results
    self._updated = False
    return [(prefix + k, v) for k, v in results]

  def results_str(self):
    return " - ".join(["{:}: {:.2f}".format(k, v)
                       for k, v in self.get_results()])


class SentenceLevelScorer(Scorer):
  """Abstract scorer for classification/regression tasks."""

  __metaclass__ = abc.ABCMeta

  def __init__(self):
    super(SentenceLevelScorer, self).__init__()
    self._total_loss = 0
    self._true_labels = []
    self._preds = []

  def update(self, results):
    super(SentenceLevelScorer, self).update(results)
    self._total_loss += results['loss']
    self._true_labels.append(results['label_ids'] if 'label_ids' in results
                             else results['targets'])
    self._preds.append(results['predictions'])

  def get_loss(self):
    return self._total_loss / len(self._true_labels)


class AccuracyScorer(SentenceLevelScorer):

  def _get_results(self):
    correct, count = 0, 0
    for y_true, pred in zip(self._true_labels, self._preds):
      count += 1
      correct += (1 if y_true == pred else 0)
    return [
        ('accuracy', 100.0 * correct / count),
        ('loss', self.get_loss()),
    ]



class Example(object):
  __metaclass__ = abc.ABCMeta

  def __init__(self, task_name):
    self.task_name = task_name


class Task(object):
  """Override this class to add a new fine-tuning task."""

  __metaclass__ = abc.ABCMeta

  def __init__(self, config: configure_finetuning.FinetuningConfig, name):
    self.config = config
    self.name = name

  def get_test_splits(self):
    return ["test"]

  @abc.abstractmethod
  def get_examples(self, split):
    pass

  @abc.abstractmethod
  def get_scorer(self) -> Scorer:
    pass

  @abc.abstractmethod
  def get_feature_specs(self) -> List[FeatureSpec]:
    pass

  @abc.abstractmethod
  def featurize(self, example: Example, is_training: bool,
                log: bool=False):
    pass

  @abc.abstractmethod
  def get_prediction_module(
      self, bert_model: modeling.BertModel, features: dict, is_training: bool,
      percent_done: float) -> Tuple:
    pass

  def __repr__(self):
    return "Task(" + self.name + ")"




class InputExample(Example):
  """A single training/test example for simple sequence classification."""

  def __init__(self, eid, task_name, text_a, text_b=None, label=None):
    super(InputExample, self).__init__(task_name)
    self.eid = eid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class SingleOutputTask(Task):
  """Task with a single prediction per example (e.g., text classification)."""

  __metaclass__ = abc.ABCMeta

  def __init__(self, config: configure_finetuning.FinetuningConfig, name,
               tokenizer):
    super(SingleOutputTask, self).__init__(config, name)
    self._tokenizer = tokenizer

  def get_examples(self, split):
    return self._create_examples(read_tsv(
        os.path.join(self.config.raw_data_dir(self.name), split + ".tsv"),
        max_lines=100 if self.config.debug else None), split)

  @abc.abstractmethod
  def _create_examples(self, lines, split):
    pass

  def featurize(self, example: InputExample, is_training, log=False):
    """Turn an InputExample into a dict of features."""
    tokens_a = self._tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
      tokens_b = self._tokenizer.tokenize(example.text_b)

    if tokens_b:
      # Modifies `tokens_a` and `tokens_b` in place so that the total
      # length is less than the specified length.
      # Account for [CLS], [SEP], [SEP] with "- 3"
      _truncate_seq_pair(tokens_a, tokens_b, self.config.max_seq_length - 3)
    else:
      # Account for [CLS] and [SEP] with "- 2"
      if len(tokens_a) > self.config.max_seq_length - 2:
        tokens_a = tokens_a[0:(self.config.max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it
    # makes it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
      tokens.append(token)
      segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
      for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
      tokens.append("[SEP]")
      segment_ids.append(1)

    input_ids = self._tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < self.config.max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == self.config.max_seq_length
    assert len(input_mask) == self.config.max_seq_length
    assert len(segment_ids) == self.config.max_seq_length

    if log:
      utils.log("  Example {:}".format(example.eid))
      utils.log("    tokens: {:}".format(" ".join(
          [tokenization.printable_text(x) for x in tokens])))
      utils.log("    input_ids: {:}".format(" ".join(map(str, input_ids))))
      utils.log("    input_mask: {:}".format(" ".join(map(str, input_mask))))
      utils.log("    segment_ids: {:}".format(" ".join(map(str, segment_ids))))

    eid = example.eid
    features = {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids,
        "task_id": self.config.task_names.index(self.name),
        self.name + "_eid": eid,
    }
    self._add_features(features, example, log)
    return features

  def _load_glue(self, lines, split, text_a_loc, text_b_loc, label_loc,
                 skip_first_line=False, eid_offset=0, swap=False):
    examples = []
    for (i, line) in enumerate(lines):
      try:
        if i == 0 and skip_first_line:
          continue
        eid = i - (1 if skip_first_line else 0) + eid_offset
        text_a = tokenization.convert_to_unicode(line[text_a_loc])
        if text_b_loc is None:
          text_b = None
        else:
          text_b = tokenization.convert_to_unicode(line[text_b_loc])
        if "test" in split or "diagnostic" in split:
          label = self._get_dummy_label()
        else:
          label = tokenization.convert_to_unicode(line[label_loc])
        if swap:
          text_a, text_b = text_b, text_a
        examples.append(InputExample(eid=eid, task_name=self.name,
                                     text_a=text_a, text_b=text_b, label=label))
      except Exception as ex:
        utils.log("Error constructing example from line", i,
                  "for task", self.name + ":", ex)
        utils.log("Input causing the error:", line)
    return examples

  @abc.abstractmethod
  def _get_dummy_label(self):
    pass

  @abc.abstractmethod
  def _add_features(self, features, example, log):
    pass



class ClassificationTask(SingleOutputTask):
  """Task where the output is a single categorical label for the input text."""
  __metaclass__ = abc.ABCMeta

  def __init__(self, config: configure_finetuning.FinetuningConfig, name,
               tokenizer, label_list):
    super(ClassificationTask, self).__init__(config, name, tokenizer)
    self._tokenizer = tokenizer
    self._label_list = label_list

  def _get_dummy_label(self):
    return self._label_list[0]

  def get_feature_specs(self):
    return [FeatureSpec(self.name + "_eid", []),
            FeatureSpec(self.name + "_label_ids", [])]

  def _add_features(self, features, example, log):
    label_map = {}
    for (i, label) in enumerate(self._label_list):
      label_map[label] = i
    label_id = label_map[example.label]
    if log:
      utils.log("    label: {:} (id = {:})".format(example.label, label_id))
    features[example.task_name + "_label_ids"] = label_id

  def get_prediction_module(self, bert_model, features, is_training,
                            percent_done):
    num_labels = len(self._label_list)
    reprs = bert_model.get_pooled_output()

    if is_training:
      reprs = tf.nn.dropout(reprs, keep_prob=0.9)

    logits = tf.layers.dense(reprs, num_labels)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = features[self.name + "_label_ids"]
    labels = tf.one_hot(label_ids, depth=num_labels, dtype=tf.float32)

    losses = -tf.reduce_sum(labels * log_probs, axis=-1)

    outputs = dict(
        loss=losses,
        logits=logits,
        predictions=tf.argmax(logits, axis=-1),
        label_ids=label_ids,
        eid=features[self.name + "_eid"],
    )
    return losses, outputs

  def get_scorer(self):
    return AccuracyScorer()


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def read_tsv(input_file, quotechar=None, max_lines=None):
  """Reads a tab separated value file."""
  with tf.io.gfile.GFile(input_file, "r") as f:
    reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
    lines = []
    for i, line in enumerate(reader):
      if max_lines and i >= max_lines:
        break
      lines.append(line)
    return lines


  
class StandardTSV(ClassificationTask):
  def __init__(self, config: configure_finetuning.FinetuningConfig,
               task_name: str, task_config: dict, tokenizer):
    self.task_config = task_config
    labels = len(self.task_config.get("labels", None))
    labels_list = []
    for i in range(0, labels):
      labels_list.append(str(i))
    super(StandardTSV, self).__init__(config, task_name, tokenizer,
                               labels_list)
    #self.task_config = task_config

  def get_examples(self, split):
    return self._create_examples(read_tsv(
        os.path.join(self.config.raw_data_dir(self.name), split + ".tsv"),
        quotechar="\"",
        max_lines=100 if self.config.debug else None), split)

  def _create_examples(self, lines, split):
    text_column_2 = self.task_config.get("text_column_2", None)
    header = self.task_config.get("header", False)
    return self._load_glue(lines, split, self.task_config["text_column"],
                           text_column_2, self.task_config["label_column"],
                           skip_first_line=header)


def get_tasks(config: configure_finetuning.FinetuningConfig):
  tokenizer = tokenization.FullTokenizer(vocab_file=config.vocab_file,
                                         do_lower_case=config.do_lower_case)
  return [get_task(config, task_name, tokenizer)
          for task_name in config.task_names]


def get_task(config: configure_finetuning.FinetuningConfig, task_name,
             tokenizer):
  """Get an instance of a task based on its name."""
  if (task_name in config.tasks):
    if config.tasks[task_name]["type"] == "classification":
      return StandardTSV(config, task_name, config.tasks[task_name], tokenizer)
    else:
      raise ValueError("Unknown task type: " + config.tasks[task_name]["type"])
  else:
    raise ValueError("Unknown task " + task_name)



class FinetuningModel(object):
  """Finetuning model with support for multi-task training."""

  def __init__(self, config: configure_finetuning.FinetuningConfig, tasks,
               is_training, features, num_train_steps):
    # Create a shared transformer encoder
    bert_config = training_utils.get_bert_config(config)
    self.bert_config = bert_config
    if config.debug:
      bert_config.num_hidden_layers = 3
      bert_config.hidden_size = 144
      bert_config.intermediate_size = 144 * 4
      bert_config.num_attention_heads = 4
    assert config.max_seq_length <= bert_config.max_position_embeddings
    bert_model = modeling.BertModel(
        bert_config=bert_config,
        is_training=is_training,
        input_ids=features["input_ids"],
        input_mask=features["input_mask"],
        token_type_ids=features["segment_ids"],
        use_one_hot_embeddings=config.use_tpu,
        embedding_size=config.embedding_size)
    percent_done = (tf.cast(tf.train.get_or_create_global_step(), tf.float32) /
                    tf.cast(num_train_steps, tf.float32))

    # Add specific tasks
    self.outputs = {"task_id": features["task_id"]}
    losses = []
    for task in tasks:
      with tf.variable_scope("task_specific/" + task.name):
        task_losses, task_outputs = task.get_prediction_module(
            bert_model, features, is_training, percent_done)
        losses.append(task_losses)
        self.outputs[task.name] = task_outputs
    self.loss = tf.reduce_sum(
        tf.stack(losses, -1) *
        tf.one_hot(features["task_id"], len(config.task_names)))


def model_fn_builder(config: configure_finetuning.FinetuningConfig, tasks,
                     num_train_steps, pretraining_config=None):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):
    """The `model_fn` for TPUEstimator."""
    utils.log("Building model...")
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    model = FinetuningModel(
        config, tasks, is_training, features, num_train_steps)

    # Load pre-trained weights from checkpoint
    init_checkpoint = config.init_checkpoint
    if pretraining_config is not None:
      init_checkpoint = tf.train.latest_checkpoint(pretraining_config.model_dir)
      utils.log("Using checkpoint", init_checkpoint)
    tvars = tf.trainable_variables()
    scaffold_fn = None
    if init_checkpoint:
      assignment_map, _ = modeling.get_assignment_map_from_checkpoint(
          tvars, init_checkpoint)
      if config.use_tpu:
        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()
        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # Build model for training or prediction
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          model.loss, config.learning_rate, num_train_steps,
          weight_decay_rate=config.weight_decay_rate,
          use_tpu=config.use_tpu,
          warmup_proportion=config.warmup_proportion,
          layerwise_lr_decay_power=config.layerwise_lr_decay,
          n_transformer_layers=model.bert_config.num_hidden_layers
      )
      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=model.loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn,
          training_hooks=[training_utils.ETAHook(
              {} if config.use_tpu else dict(loss=model.loss),
              num_train_steps, config.iterations_per_loop, config.use_tpu, 10)])
    else:
      assert mode == tf.estimator.ModeKeys.PREDICT
      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions=utils.flatten_dict(model.outputs),
          scaffold_fn=scaffold_fn)

    utils.log("Building complete")
    return output_spec

  return model_fn


class ModelRunner(object):
  """Fine-tunes a model on a supervised task."""

  def __init__(self, config: configure_finetuning.FinetuningConfig, tasks,
               pretraining_config=None):
    self._config = config
    self._tasks = tasks
    self._preprocessor = Preprocessor(config, self._tasks)

    is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
    tpu_cluster_resolver = None
    if config.use_tpu and config.tpu_name:
      tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
          config.tpu_name, zone=config.tpu_zone, project=config.gcp_project)
    tpu_config = tf.estimator.tpu.TPUConfig(
        iterations_per_loop=config.iterations_per_loop,
        num_shards=config.num_tpu_cores,
        per_host_input_for_training=is_per_host,
        tpu_job_name=config.tpu_job_name)
    run_config = tf.estimator.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=config.model_dir,
        save_checkpoints_steps=config.save_checkpoints_steps,
        save_checkpoints_secs=None,
        tpu_config=tpu_config)

    if self._config.do_train:
      (self._train_input_fn,
       self.train_steps) = self._preprocessor.prepare_train()
    else:
      self._train_input_fn, self.train_steps = None, 0
    model_fn = model_fn_builder(
        config=config,
        tasks=self._tasks,
        num_train_steps=self.train_steps,
        pretraining_config=pretraining_config)
    self._estimator = tf.estimator.tpu.TPUEstimator(
        use_tpu=config.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        predict_batch_size=config.predict_batch_size)

  def train(self):
    utils.log("Training for {:} steps".format(self.train_steps))
    self._estimator.train(
        input_fn=self._train_input_fn, max_steps=self.train_steps)

  def evaluate(self):
    return {task.name: self.evaluate_task(task) for task in self._tasks}
  
  def test(self):
    tasks = get_tasks(self._config)
    for task in tasks:
      for split in task.get_test_splits():
        self.write_classification_outputs([task], 1, split)

  def evaluate_task(self, task, split="dev", return_results=True):
    """Evaluate the current model."""
    utils.log("Evaluating", task.name)
    eval_input_fn, _ = self._preprocessor.prepare_predict([task], split)
    results = self._estimator.predict(input_fn=eval_input_fn,
                                      yield_single_examples=True)
    scorer = task.get_scorer()
    for r in results:
      if r["task_id"] != len(self._tasks):  # ignore padding examples
        r = utils.nest_dict(r, self._config.task_names)
        scorer.update(r[task.name])
    if return_results:
      utils.log(task.name + ": " + scorer.results_str())
      utils.log()
      return dict(scorer.get_results())
    else:
      return scorer

  def write_classification_outputs(self, tasks, trial, split):
    """Write classification predictions to disk."""
    utils.log("Writing out predictions for", tasks, split)
    predict_input_fn, _ = self._preprocessor.prepare_predict(tasks, split)
    results = self._estimator.predict(input_fn=predict_input_fn,
                                      yield_single_examples=True)
    # task name -> eid -> model-logits
    logits = collections.defaultdict(dict)
    for r in results:
      if r["task_id"] != len(self._tasks):
        r = utils.nest_dict(r, self._config.task_names)
        task_name = self._config.task_names[r["task_id"]]
        logits[task_name][r[task_name]["eid"]] = (
            r[task_name]["logits"] if "logits" in r[task_name]
            else r[task_name]["predictions"])
    for task_name in logits:
      utils.log("Pickling predictions for {:} {:} examples ({:})".format(
          len(logits[task_name]), task_name, split))
      if trial <= self._config.n_writes_test:
        utils.write_pickle(logits[task_name], self._config.test_predictions(
            task_name, split, trial))


def write_results(config: configure_finetuning.FinetuningConfig, results):
  """Write evaluation metrics to disk."""
  utils.log("Writing results to", config.results_txt)
  utils.mkdir(config.results_txt.rsplit("/", 1)[0])
  utils.write_pickle(results, config.results_pkl)
  with tf.io.gfile.GFile(config.results_txt, "w") as f:
    results_str = ""
    for trial_results in results:
      for task_name, task_results in trial_results.items():
        if task_name == "time" or task_name == "global_step":
          continue
        results_str += task_name + ": " + " - ".join(
            ["{:}: {:.2f}".format(k, v)
             for k, v in task_results.items()]) + "\n"
    f.write(results_str)
  utils.write_pickle(results, config.results_pkl)


def electra_finetuning(configs):
  data_dir = configs["data_dir"]
  model_name = configs["model_name"]
  hparams = configs["hparams"]
  tf.logging.set_verbosity(tf.logging.ERROR)
  config = configure_finetuning.FinetuningConfig(
      model_name, data_dir, **hparams)
  
  trial = 1
  heading_info = "model={:}, trial {:}/{:}".format(
      config.model_name, trial, config.num_trials)
  heading = lambda msg: utils.heading(msg + ": " + heading_info)
  heading("Config")
  utils.log_config(config)
  generic_model_dir = config.model_dir
  tasks = get_tasks(config)
  # Train and evaluate num_trials models with different random seeds
  config.model_dir = generic_model_dir + "_" + str(trial)
  if config.do_train:
    utils.rmkdir(config.model_dir)

  model_runner = ModelRunner(config, tasks)
  return model_runner
  #run_finetuning(test_obj)


if __name__ == "__main__":
  main()
