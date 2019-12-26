import unittest
import sys
import logging
sys.path.append('/Users/keemsunguk/Projects/EssayGrader/')
from egrader.db_util import DBUtil

logger = logging.getLogger('DBTest')


class TestDBUtil(unittest.TestCase):
    def setUp(self):
        self.db_util = DBUtil()
        self.expected = {'RemoteSAT': 3263, 'RemoteTOEFL': 4684, 'RemoteGRE': 6651, 'LocalTotal': 32246}

    def test_remote(self):
        sat_count = self.db_util.get_labeled_essays('SAT', merge_0_1=True).shape[0]
        toelf_count = self.db_util.get_labeled_essays('TOEFL', merge_0_1=True).shape[0]
        gre_count = self.db_util.get_labeled_essays('GRE', merge_0_1=True).shape[0]
        logger.info(sat_count, toelf_count, gre_count)
        self.assertEqual(sat_count, self.expected['RemoteSAT'])
        self.assertEqual(toelf_count, self.expected['RemoteTOEFL'])
        self.assertEqual(gre_count, self.expected['RemoteGRE'])

    def test_local(self):
        total_local_essays = self.db_util.local_ec.count_documents({})
        self.assertEqual(total_local_essays, self.expected['LocalTotal'])


if __name__ == '__main__':
    unittest.main()
