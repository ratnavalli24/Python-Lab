from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import numpy as np
import pandas
from sklearn import metrics
import tensorflow as tf
FLAGS = None
MAX_DOCUMENT_LENGTH = 10
learning_rate = 0.09
training_iters = 10000
display_steps = 1000
EMBEDDING_SIZE = 10
n_words = 50
MAX_LABEL = 15
WORDS_FEATURE = 'words'
def estimator_spec_for_softmax_classification(logits, labels, mode):
  predicted_classes = tf.argmax(logits, 1)
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={
            'class': predicted_classes,
            'prob': tf.nn.softmax(logits)
        })
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
  eval_metric_ops = {
      'accuracy':
          tf.metrics.accuracy(labels=labels, predictions=predicted_classes)
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
def bag_of_words_model(features, labels, mode):
  bow_column = tf.feature_column.categorical_column_with_identity(
      WORDS_FEATURE, num_buckets=n_words)
  bow_embedding_column = tf.feature_column.embedding_column(
      bow_column, dimension=EMBEDDING_SIZE)
  bow = tf.feature_column.input_layer(
      features, feature_columns=[bow_embedding_column])
  logits = tf.layers.dense(bow, MAX_LABEL, activation=None)
  return estimator_spec_for_softmax_classification(
      logits=logits, labels=labels, mode=mode)
def cnn_model(features, labels, mode):
  word_vectors = tf.contrib.layers.embed_sequence(
      features[WORDS_FEATURE], vocab_size=n_words, embed_dim=EMBEDDING_SIZE)
  word_list = tf.unstack(word_vectors, axis=1)
  cell = tf.nn.rnn_cell.GRUCell(EMBEDDING_SIZE)
  _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)
  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)
  return estimator_spec_for_softmax_classification(
      logits=logits, labels=labels, mode=mode)
def main(unused_argv):
  global n_words
  tf.logging.set_verbosity(tf.logging.INFO)
  dbpedia = tf.contrib.learn.datasets.load_dataset(
      'dbpedia', test_with_fake_data=FLAGS.test_with_fake_data)
  x_train = pandas.Series(dbpedia.train.data[:, 1])
  y_train = pandas.Series(dbpedia.train.target)
  x_test = pandas.Series(dbpedia.test.data[:, 1])
  y_test = pandas.Series(dbpedia.test.target)
  vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
      MAX_DOCUMENT_LENGTH)
  x_transform_train = vocab_processor.fit_transform(x_train)
  x_transform_test = vocab_processor.transform(x_test)
  x_train = np.array(list(x_transform_train))
  x_test = np.array(list(x_transform_test))
  n_words = len(vocab_processor.vocabulary_)
  print('Total words: %d' % n_words)
  model_fn = cnn_model
  if FLAGS.bow_model:
    x_train -= 1
    x_test -= 1
    model_fn = bag_of_words_model
  classifier = tf.estimator.Estimator(model_fn=model_fn)
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={WORDS_FEATURE: x_train},
      y=y_train,
      batch_size=len(x_train),
      num_epochs=None,
      shuffle=True)
  classifier.train(input_fn=train_input_fn, steps=100)
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={WORDS_FEATURE: x_test}, y=y_test, num_epochs=1, shuffle=False)
  predictions = classifier.predict(input_fn=test_input_fn)
  y_predicted = np.array(list(p['class'] for p in predictions))
  y_predicted = y_predicted.reshape(np.array(y_test).shape)

  # Score with sklearn.
  score = metrics.accuracy_score(y_test, y_predicted)
  print('Accuracy (sklearn): {0:f}'.format(score))

  # Score with tensorflow.
  scores = classifier.evaluate(input_fn=test_input_fn)
  print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--test_with_fake_data',
      default=False,
      help='Test the example code with fake data.',
      action='store_true')
  parser.add_argument(
      '--bow_model',
      default=False,
      help='Run with BOW model instead of CNN.',
      action='store_true')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)