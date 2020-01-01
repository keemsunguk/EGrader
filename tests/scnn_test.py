import unittest
import sys
import logging
sys.path.append('/Users/keemsunguk/Projects/EssayGrader/')
from egrader.spacy_cnn import SpacyCNN


scnn = SpacyCNN('SAT')
logger = logging.getLogger('SpacyCNN')


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        scnn.add_text_category(['1', '2', '3', '4', '5', '6'])
        scnn.load_df('SAT')
        scnn.train_text()

    def test_scnn(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
