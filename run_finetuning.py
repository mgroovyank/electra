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

import pickle
import sys

import tensorflow.compat.v1 as tf


from finetune import preprocessing
from finetune import task_builder
from model import modeling
from model import optimization
from util import training_utils

import os

import tensorflow.compat.v1 as tf

def load_json(path):
  with tf.io.gfile.GFile(path, "r") as f:
    return json.load(f)


def write_json(o, path):
  if "/" in path:
    tf.io.gfile.makedirs(path.rsplit("/", 1)[0])
  with tf.io.gfile.GFile(path, "w") as f:
    json.dump(o, f)


def load_pickle(path):
  with tf.io.gfile.GFile(path, "rb") as f:
    return pickle.load(f)


def write_pickle(o, path):
  if "/" in path:
    tf.io.gfile.makedirs(path.rsplit("/", 1)[0])
  with tf.io.gfile.GFile(path, "wb") as f:
    pickle.dump(o, f, -1)


def mkdir(path):
  if not tf.io.gfile.exists(path):
    tf.io.gfile.makedirs(path)


def rmrf(path):
  if tf.io.gfile.exists(path):
    tf.io.gfile.rmtree(path)


def rmkdir(path):
  rmrf(path)
  mkdir(path)


def log(*args):
  msg = " ".join(map(str, args))
  sys.stdout.write(msg + "\n")
  sys.stdout.flush()


def log_config(config):
  for key, value in sorted(config.__dict__.items()):
    log(key, value)
  log()


def heading(*args):
  log(80 * "=")
  log(*args)
  log(80 * "=")


def nest_dict(d, prefixes, delim="_"):
  """Go from {prefix_key: value} to {prefix: {key: value}}."""
  nested = {}
  for k, v in d.items():
    for prefix in prefixes:
      if k.startswith(prefix + delim):
        if prefix not in nested:
          nested[prefix] = {}
        nested[prefix][k.split(delim, 1)[1]] = v
      else:
        nested[k] = v
  return nested


def flatten_dict(d, delim="_"):
  """Go from {prefix: {key: value}} to {prefix_key: value}."""
  flattened = {}
  for k, v in d.items():
    if isinstance(v, dict):
      for k2, v2 in v.items():
        flattened[k + delim + k2] = v2
    else:
      flattened[k] = v
  return flattened


class FinetuningConfig(object):
  """Fine-tuning hyperparameters."""

  def __init__(self, model_name, data_dir, **kwargs):
    # general
    self.model_name = model_name
    self.debug = False  # debug mode for quickly running things
    self.log_examples = False  # print out some train examples for debugging
    self.num_trials = 1  # how many train+eval runs to perform
    self.do_train = True  # train a model
    self.do_eval = True  # evaluate the model
    self.keep_all_models = True  # if False, only keep the last trial's ckpt

    # model
    self.model_size = "small"  # one of "small", "base", or "large"
    self.task_names = ["chunk"]  # which tasks to learn
    # override the default transformer hparams for the provided model size; see
    # modeling.BertConfig for the possible hparams and util.training_utils for
    # the defaults
    self.model_hparam_overrides = (
        kwargs["model_hparam_overrides"]
        if "model_hparam_overrides" in kwargs else {})
    self.embedding_size = None  # bert hidden size by default
    self.vocab_size = 30522  # number of tokens in the vocabulary
    self.do_lower_case = True

    # training
    self.learning_rate = 5e-5
    self.weight_decay_rate = 0.01
    self.layerwise_lr_decay = 0.8  # if > 0, the learning rate for a layer is
                                   # lr * lr_decay^(depth - max_depth) i.e.,
                                   # shallower layers have lower learning rates
    self.num_train_epochs = 3.0  # passes over the dataset during training
    self.warmup_proportion = 0.1  # how much of training to warm up the LR for
    self.save_checkpoints_steps = 1000000
    self.iterations_per_loop = 1000
    self.use_tfrecords_if_existing = True  # don't make tfrecords and write them
                                           # to disc if existing ones are found

    # writing model outputs to disc
    self.write_test_outputs = True  # whether to write test set outputs,
                                     # currently supported for GLUE + SQuAD 2.0
    self.n_writes_test = 5  # write test set predictions for the first n trials

    # sizing
    self.max_seq_length = 64
    self.train_batch_size = 16
    self.eval_batch_size = 8
    self.predict_batch_size = 8
    self.double_unordered = True  # for tasks like paraphrase where sentence
                                  # order doesn't matter, train the model on
                                  # on both sentence orderings for each example
    # for qa tasks
    self.max_query_length = 64   # max tokens in q as opposed to context
    self.doc_stride = 128  # stride when splitting doc into multiple examples
    self.n_best_size = 20  # number of predictions per example to save
    self.max_answer_length = 30  # filter out answers longer than this length
    self.answerable_classifier = True  # answerable classifier for SQuAD 2.0
    self.answerable_uses_start_logits = True  # more advanced answerable
                                              # classifier using predicted start
    self.answerable_weight = 0.5  # weight for answerability loss
    self.joint_prediction = True  # jointly predict the start and end positions
                                  # of the answer span
    self.beam_size = 20  # beam size when doing joint predictions
    self.qa_na_threshold = -2.75  # threshold for "no answer" when writing SQuAD
                                  # 2.0 test outputs

    # TPU settings
    self.use_tpu = False
    self.num_tpu_cores = 1
    self.tpu_job_name = None
    self.tpu_name = None  # cloud TPU to use for training
    self.tpu_zone = None  # GCE zone where the Cloud TPU is located in
    self.gcp_project = None  # project name for the Cloud TPU-enabled project

    # default locations of data files
    self.data_dir = data_dir
    pretrained_model_dir = os.path.join(data_dir, "models", model_name)
    self.raw_data_dir = os.path.join(data_dir, "finetuning_data", "{:}").format
    self.vocab_file = os.path.join(pretrained_model_dir, "vocab.txt")
    if not tf.io.gfile.exists(self.vocab_file):
      self.vocab_file = os.path.join(self.data_dir, "vocab.txt")
    task_names_str = ",".join(
        kwargs["task_names"] if "task_names" in kwargs else self.task_names)
    self.init_checkpoint = None if self.debug else pretrained_model_dir
    self.model_dir = os.path.join(pretrained_model_dir, "finetuning_models",
                                  task_names_str + "_model")
    results_dir = os.path.join(pretrained_model_dir, "results")
    self.results_txt = os.path.join(results_dir,
                                    task_names_str + "_results.txt")
    self.results_pkl = os.path.join(results_dir,
                                    task_names_str + "_results.pkl")
    qa_topdir = os.path.join(results_dir, task_names_str + "_qa")
    self.qa_eval_file = os.path.join(qa_topdir, "{:}_eval.json").format
    self.qa_preds_file = os.path.join(qa_topdir, "{:}_preds.json").format
    self.qa_na_file = os.path.join(qa_topdir, "{:}_null_odds.json").format
    self.preprocessed_data_dir = os.path.join(
        pretrained_model_dir, "finetuning_tfrecords",
        task_names_str + "_tfrecords" + ("-debug" if self.debug else ""))
    self.test_predictions = os.path.join(
        pretrained_model_dir, "test_predictions",
        "{:}_{:}_{:}_predictions.pkl").format

    # update defaults with passed-in hyperparameters
    self.tasks = {}
    self.update(kwargs)

    # default hyperparameters for different model sizes
    if self.model_size == "large":
      self.learning_rate = 5e-5
      self.layerwise_lr_decay = 0.9
    elif self.model_size == "small":
      self.embedding_size = 128

    # debug-mode settings
    if self.debug:
      self.save_checkpoints_steps = 1000000
      self.use_tfrecords_if_existing = False
      self.num_trials = 1
      self.iterations_per_loop = 1
      self.train_batch_size = 32
      self.num_train_epochs = 3.0
      self.log_examples = True

    # passed-in-arguments override (for example) debug-mode defaults
    self.update(kwargs)

  def update(self, kwargs):
    for k, v in kwargs.items():
      if k not in self.__dict__:
        raise ValueError("Unknown hparam " + k)
      self.__dict__[k] = v


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
    log("Building model...")
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    model = FinetuningModel(
        config, tasks, is_training, features, num_train_steps)

    # Load pre-trained weights from checkpoint
    init_checkpoint = config.init_checkpoint
    if pretraining_config is not None:
      init_checkpoint = tf.train.latest_checkpoint(pretraining_config.model_dir)
      log("Using checkpoint", init_checkpoint)
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
          predictions=flatten_dict(model.outputs),
          scaffold_fn=scaffold_fn)

    log("Building complete")
    return output_spec

  return model_fn


class ModelRunner(object):
  """Fine-tunes a model on a supervised task."""

  def __init__(self, config: configure_finetuning.FinetuningConfig, tasks,
               pretraining_config=None):
    self._config = config
    self._tasks = tasks
    self._preprocessor = preprocessing.Preprocessor(config, self._tasks)

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
    log("Training for {:} steps".format(self.train_steps))
    self._estimator.train(
        input_fn=self._train_input_fn, max_steps=self.train_steps)

  def evaluate(self):
    return {task.name: self.evaluate_task(task) for task in self._tasks}
  
  def test(self):
    tasks = task_builder.get_tasks(self._config)
    for task in tasks:
      for split in task.get_test_splits():
        self.write_classification_outputs([task], 1, split)

  def evaluate_task(self, task, split="dev", return_results=True):
    """Evaluate the current model."""
    log("Evaluating", task.name)
    eval_input_fn, _ = self._preprocessor.prepare_predict([task], split)
    results = self._estimator.predict(input_fn=eval_input_fn,
                                      yield_single_examples=True)
    scorer = task.get_scorer()
    for r in results:
      if r["task_id"] != len(self._tasks):  # ignore padding examples
        r = nest_dict(r, self._config.task_names)
        scorer.update(r[task.name])
    if return_results:
      log(task.name + ": " + scorer.results_str())
      log()
      return dict(scorer.get_results())
    else:
      return scorer

  def write_classification_outputs(self, tasks, trial, split):
    """Write classification predictions to disk."""
    log("Writing out predictions for", tasks, split)
    predict_input_fn, _ = self._preprocessor.prepare_predict(tasks, split)
    results = self._estimator.predict(input_fn=predict_input_fn,
                                      yield_single_examples=True)
    # task name -> eid -> model-logits
    logits = collections.defaultdict(dict)
    for r in results:
      if r["task_id"] != len(self._tasks):
        r = nest_dict(r, self._config.task_names)
        task_name = self._config.task_names[r["task_id"]]
        logits[task_name][r[task_name]["eid"]] = (
            r[task_name]["logits"] if "logits" in r[task_name]
            else r[task_name]["predictions"])
    for task_name in logits:
      log("Pickling predictions for {:} {:} examples ({:})".format(
          len(logits[task_name]), task_name, split))
      if trial <= self._config.n_writes_test:
        write_pickle(logits[task_name], self._config.test_predictions(
            task_name, split, trial))


def write_results(config: configure_finetuning.FinetuningConfig, results):
  """Write evaluation metrics to disk."""
  log("Writing results to", config.results_txt)
  mkdir(config.results_txt.rsplit("/", 1)[0])
  write_pickle(results, config.results_pkl)
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
  write_pickle(results, config.results_pkl)


def electra_finetuning(configs):
  data_dir = configs["data_dir"]
  model_name = configs["model_name"]
  hparams = configs["hparams"]
  tf.logging.set_verbosity(tf.logging.ERROR)
  config = FinetuningConfig(
      model_name, data_dir, **hparams)
  
  trial = 1
  heading_info = "model={:}, trial {:}/{:}".format(
      config.model_name, trial, config.num_trials)
  heading = lambda msg: utils.heading(msg + ": " + heading_info)
  heading("Config")
  utils.log_config(config)
  generic_model_dir = config.model_dir
  tasks = task_builder.get_tasks(config)
  # Train and evaluate num_trials models with different random seeds
  config.model_dir = generic_model_dir + "_" + str(trial)
  if config.do_train:
    rmkdir(config.model_dir)

  model_runner = ModelRunner(config, tasks)
  return model_runner
  #run_finetuning(test_obj)


if __name__ == "__main__":
  main()
