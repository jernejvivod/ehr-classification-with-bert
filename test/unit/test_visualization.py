import os
import unittest

import numpy as np
from sklearn import metrics

from classification_with_embeddings.evaluation.visualization import write_classification_report, plot_confusion_matrix, plot_roc
from test.test_utils import get_relative_path


class TestVisualization(unittest.TestCase):
    _y_true = [0, 1, 1]
    _y_pred = [0, 1, 0]
    _y_proba = [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]]

    def test_write_classification_report(self):
        cr_out_path = 'word2vec_cr.txt'
        cr = metrics.classification_report(self._y_true, self._y_pred)
        write_classification_report(cr, get_relative_path(__file__, '.'), 'word2vec')
        self.assertTrue(os.path.exists(get_relative_path(__file__, cr_out_path)))
        with open(get_relative_path(__file__, cr_out_path), 'r') as f:
            self.assertTrue(len(f.readlines()) > 0)
        os.remove(get_relative_path(__file__, cr_out_path))

    def test_plot_confusion_matrix(self):
        cm_out_path = "word2vec_cm.png"
        plot_confusion_matrix(self._y_pred, self._y_true, [0, 1], ['zero', 'one'], get_relative_path(__file__, '.'), 'word2vec')
        self.assertTrue(os.path.exists(get_relative_path(__file__, cm_out_path)))
        os.remove(get_relative_path(__file__, cm_out_path))

    def test_plot_roc(self):
        roc_out_path = "word2vec_roc.png"
        plot_roc(np.array(self._y_proba), self._y_true, 1, get_relative_path(__file__, '.'), 'word2vec')
        self.assertTrue(os.path.exists(get_relative_path(__file__, roc_out_path)))
        os.remove(get_relative_path(__file__, roc_out_path))
