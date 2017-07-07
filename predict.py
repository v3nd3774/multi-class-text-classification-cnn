import os
import sys
import json
import logging
import itertools
import data_helper
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import learn
from sklearn.metrics import confusion_matrix

logging.getLogger().setLevel(logging.INFO)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Referenced from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title+".pdf", format="pdf")

def predict_unseen_data():
        """
        Modified to return labels and array of true and predicted
        labels respectively
        Step 0: load trained model and parameters
        """
        params = json.loads(open('./parameters.json').read())
        checkpoint_dir = sys.argv[1]
        if not checkpoint_dir.endswith('/'):
                checkpoint_dir += '/'
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir + 'checkpoints')
        logging.critical('Loaded the trained model: {}'.format(checkpoint_file))

        """Step 1: load data for prediction"""
        test_file = sys.argv[2]
        test_examples = pd.read_csv(test_file)

        # labels.json was saved during training, and it has to be loaded during prediction
        labels = json.loads(open('./labels.json').read())
        one_hot = np.zeros((len(labels), len(labels)), int)
        np.fill_diagonal(one_hot, 1)
        label_dict = dict(zip(labels, one_hot))

        x_raw = [example['text'] for i, example in test_examples.iterrows()]
        x_test = [data_helper.clean_str(x) for x in x_raw]
        logging.info('The number of x_test: {}'.format(len(x_test)))

        y_test = None
        if 'label' in test_examples:
                y_raw = [example['label'] for i, example in test_examples.iterrows()]
                y_test = [label_dict[y] for y in y_raw]
                logging.info('The number of y_test: {}'.format(len(y_test)))

        vocab_path = os.path.join(checkpoint_dir, "vocab.pickle")
        vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
        x_test = np.array(list(vocab_processor.transform(x_test)))

        """Step 2: compute the predictions"""
        graph = tf.Graph()
        with graph.as_default():
                session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
                sess = tf.Session(config=session_conf)

                with sess.as_default():
                        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                        saver.restore(sess, checkpoint_file)

                        input_x = graph.get_operation_by_name("input_x").outputs[0]
                        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
                        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

                        batches = data_helper.batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)
                        all_predictions = []
                        for x_test_batch in batches:
                                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                                all_predictions = np.concatenate([all_predictions, batch_predictions])

        if y_test is not None:
                y_test = np.argmax(y_test, axis=1)
                correct_predictions = sum(all_predictions == y_test)
                logging.critical('The accuracy is: {}'.format(correct_predictions / float(len(y_test))))
                return labels, [ labels[y] for y in y_test], [ labels[int(y)] for y in all_predictions]

if __name__ == '__main__':
        # python3 predict.py ./trained_model_1478649295/ ./data/small_samples.json
        labels, true, predicted = predict_unseen_data()
        cm = confusion_matrix(true, predicted, labels=labels)
        plot_confusion_matrix(cm, labels, normalize=True, title="NCNF")
        plot_confusion_matrix(cm, labels, title="CNF")
