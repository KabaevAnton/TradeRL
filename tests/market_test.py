import unittest

from common.market import *


class MarketTest(unittest.TestCase):


    def test_buy_shares(self):
        shares = buy_shares(10.0, 5.0)
        self.assertEqual(shares, 2)
        shares = buy_shares(10.0, 6.2)
        self.assertEqual(shares, 1)
        shares = buy_shares(13.0, 6.2)
        self.assertEqual(shares, 2)


