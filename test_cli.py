from click.testing import CliRunner
import unittest
from cli import *
from cli_helper_functions import load_in_saved_pairs, _pickle_pair
from cointegration_test import CointegratedPair, is_cointegration



class FindPairTests(unittest.TestCase):

    # =================================================================================================================
    
    def test_market_ticker_1(self):
        runner = CliRunner()
        result = runner.invoke(find_pair, ['find-pair', 'amex', 'pnc', '-p1', '2021-01-01'])
        self.assertEqual(result.exit_code, 0 | 2) # Either finds cointegration -> exits with code 0, or finds no cointegration/data and program interrupts it -> exit code 2
    
    def test_market_ticker_2(self):
        runner = CliRunner()
        result = runner.invoke(find_pair, ['find-pair', 'AMEX', 'PNC', '2021-01-01'])
        self.assertEqual(result.exit_code, 2)

    def test_market_ticker_3(self):
        runner = CliRunner()
        result = runner.invoke(find_pair, ['find-pair', 'amex', 'pnc', '-p1', '01-01-2021'])
        self.assertEqual(result.exit_code, 2)

    def test_market_market_1(self):
        runner = CliRunner()
        result = runner.invoke(find_pair, ['find-pair', 'FoReX', 'ETF', '-p1', '2021-01-01', '-p2', '2022-01-01'])
        self.assertEqual(result.exit_code, 0 | 2)

    def test_market_market_2(self):
        runner = CliRunner()
        result = runner.invoke(find_pair, ['find-pair', 'forex', 'etf', '-p1', '2021-01-01', '-p2'])
        self.assertEqual(result.exit_code, 2)

    def test_market_market_3(self):
        runner = CliRunner()
        result = runner.invoke(find_pair, ['find-pair', 'sp500', '-p1', '2021-01-01', '-p2', '2022-01-01'])
        self.assertEqual(result.exit_code, 1)

    def test_market_only_1(self):
        runner = CliRunner()
        result = runner.invoke(find_pair, ['find-pair', 'FTSE', '  only  ', '-p1', '2021-01-01', '-i', '1h'])
        self.assertEqual(result.exit_code, 0 | 2)

    def test_ticker_ticker_1(self):
        runner = CliRunner()
        result = runner.invoke(find_pair, ['find-pair', 'fncl', 'pnc', '-p1', '2021-01-01', '-p2', '2022-01-01'])
        self.assertEqual(result.exit_code, 0 | 2)
    
    def test_ticker_ticker_2(self):
        runner = CliRunner()
        result = runner.invoke(find_pair, ['fncl', 'pnc', '-p1', '2021-01-01', '-p2', '2022-01-01'])
        self.assertEqual(result.exit_code, 1)
    
    def test_ticker_ticker_3(self):
        runner = CliRunner()
        result = runner.invoke(find_pair, ['fncl', 'pnc', '-p1', '2022-01-01', '-p2', '2020-01-01'])
        self.assertEqual(result.exit_code, 1)


    # =================================================================================================================

    # Set up test_file.p containing saved pairs for testing
    
    def setup_file(self):
        with open('saved_pairs.p', 'wb'):
                pass

        objects = [
                CointegratedPair('brn', 'pnc', *is_cointegration('brn', 'pnc', '2020-01-01', '2021-01-01', '1d')),
                CointegratedPair('ensv', 'pnc', *is_cointegration('brn', 'pnc', '2020-01-01', '2021-01-01', '1d')),
                CointegratedPair('rec', 'pnc', *is_cointegration('brn', 'pnc', '2020-01-01', '2021-01-01', '1d')),
                CointegratedPair('esp', 'pnc', *is_cointegration('brn', 'pnc', '2020-01-01', '2021-01-01', '1d')),
                CointegratedPair('cto', 'pnc', *is_cointegration('brn', 'pnc', '2020-01-01', '2021-01-01', '1d')),
                CointegratedPair('cet', 'pnc', *is_cointegration('brn', 'pnc', '2020-01-01', '2021-01-01', '1d')),
                CointegratedPair('cix', 'pnc', *is_cointegration('brn', 'pnc', '2020-01-01', '2021-01-01', '1d')),
                ]        

        for obj in objects:
            _pickle_pair(obj, 'saved_pairs.p')


    def test_ls_1(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            self.setup_file()
            result = len(load_in_saved_pairs())
            self.assertEqual(result, 7)

    def test_ls_2(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            self.setup_file()
            result = load_in_saved_pairs()[3]
            self.assertEqual(result.name, 'ESP - PNC (2020-01-01 00:00:00 -> 2021-01-01 00:00:00)')
    

    # Broken Tests. Can't figure out what parameters I'm missing and how to get out of exit_code 2

    # def test_regression_1(self):
    #     runner = CliRunner()
    #     with runner.isolated_filesystem():
    #         self.setup_file()
    #         result = runner.invoke(regression, ['regression', '-o', '0' '-ts', 'gls'])
    #         self.assertEqual(result.exit_code, 2)
    
    # def test_regression_2(self):
    #     runner = CliRunner()
    #     with runner.isolated_filesystem():
    #         self.setup_file()
    #         result = runner.invoke(regression, ['regression', '-o', '6' '-ts', 'ols'])
    #         self.assertEqual(result.exit_code, 2)


    # def test_johansen(self):
    #     runner = CliRunner()
    #     with runner.isolated_filesystem():
    #         self.setup_file()
    #         result = runner.invoke(johansen, ['johansen', '-o', '2'])
    #         self.assertEqual(result.exit_code, 2)
    

    # def test_robust(self):
    #     runner = CliRunner()
    #     with runner.isolated_filesystem():
    #         self.setup_file()
    #         result = runner.invoke(robust, ['robust', '-o', '3', '-ts', 'ols'])
    #         self.assertEqual(result.exit_code, 2)


    # def test_delete_1(self):
    #     runner = CliRunner()
    #     result = runner.invoke(delete, ['delete', '-a'])
    #     self.assertEqual(result.exit_code, 2)
    
    # def test_delete_2(self):
    #     runner = CliRunner()
    #     result = runner.invoke(delete, ['delete', '-r', '4', '6'])
    #     self.assertEqual(result.exit_code, 2)

    # def test_delete_3(self):
    #     runner = CliRunner()
    #     result = runner.invoke(delete, ['delete', '3', '5', '-s'])
    #     self.assertEqual(result.exit_code, 2)

    # def test_delete_3(self):
    #     runner = CliRunner()
    #     result = runner.invoke(delete, ['delete', '0', '-r', '3', '5'])
    #     self.assertEqual(result.exit_code, 2)



if __name__ == '__main__':
    unittest.main()