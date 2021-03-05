import unittest

from common.downloader import YahooDownloader


class YahooDownloaderTest(unittest.TestCase):

    def test_fetch_data(self):
        downloader = YahooDownloader('2017-01-01','2017-01-30', ['MSFT'])
        dataframe = downloader.fetch_data()
        self.assertGreater(dataframe.shape[0], 0)
        

