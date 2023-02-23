import yfinance as yf
import json # For reading in .json files as python
import click
import sys
import pickle # To save pairs to file
from cointegration_test import *
from bug_fix import *



def process_ticker_search(inputs: tuple) -> tuple:
    """Handles user's search inputs for 'find-pair' command."""
    # Click only allows fixed user inputs or indefinited, I wanted variable but fixed number: 1|2
    # Also wanted different types of input to be passed => need to do this manually

    processed_input = _user_input_dict(inputs)

    # There are 4 possible input scenarios, each indicated by an integer
    i = 2
    for _ in processed_input:
        if processed_input[_] == 'market':
            i -= 0.5
        elif processed_input[_] == 'ticker':
            i += 0.5
        elif processed_input[_] == 'only':
            return 0, processed_input
    
    return int(i), processed_input


def _user_input_dict(inputs: tuple) -> dict:
    """Returns a dict containing both user inputs formatted for next step as keys, and their types as values."""
    output = {}

    for item in inputs:
        # Ticker symbols formatted as uppercase, markets formatted as their file path to be passed to next step in control flow
        formatted_item, item_type = _get_input_types(item)
        output[formatted_item] = item_type
    
    return output


def _get_input_types(item: str) -> tuple:
    """Determine user input is valid in 'process_ticker_search' function."""
    
    match item.lower().strip():
        case 'amex' | 'nyse' | 'etf' | 'forex' | 'lse' | 'nasdaq' | 'ftse' | 'sp500':
            return f'{item.lower().strip()}.json', 'market'
        case 'only':
            return 'only', 'only'
        case other:
            if is_ticker_active(item.lower().strip()):
                return f'{item.upper().strip()}', 'ticker'
            else:
                # Raise SystemExit to avoid printing entire traceback error message.
                # Most built-in python exceptions inherit from Exception class, SystemExit inherits from BaseException
                # and causes interpreter to exit without handling error and printing stack traceback.
                raise SystemExit(f'\n\'{item}\' is not a valid security ticker symbol or market accessible to the program')


def is_ticker_active(ticker: str) -> bool:
    """Checks that ticker symbol is currently active."""
    
    history = yf.Ticker(ticker).history(period='1y', progress=False)
    
    return False if history.empty else True


def read_in_scraped_tickers(file: str, folder: str='stock_scraper/') -> list[int]:
    """
    Read in ticker symbols from JSON in 'cointegration_apirs_finder/stock_scraper' folder.
    Currently, program is set to only find cointegration between securities in the same markets, 
    so only work within one json file for now.
    """
    # Open a list of dicts containing basic security info (including ticker symbols) from /stock_scapers
    filepath = f'{folder}{file}'

    with open(filepath, 'r') as f:
        security_dicts = json.load(f) # Use .load() (not .loads() which opens file contents as string)
    
    ticker_list = [security_dict['ticker'] for security_dict in security_dicts]
    
    return ticker_list


def market_only(filepath: str, start: str, end: str, freq: str, find_all: bool=False) -> tuple:
    """Handle control flow for 'market' - 'only' and return needed data to instantiate CointegrationPair object."""
    market_tickers = read_in_scraped_tickers(filepath)
    visited_tickers = set()
    
    for ticker1 in market_tickers:
        if is_ticker_active(ticker1):
            for ticker2 in market_tickers:
                if is_ticker_active(ticker2):
                    try:
                        output = are_securities_cointegrated(ticker1, ticker2, start, end, freq, visited_tickers, find_all)
                        if output:
                            return output
                    except ZeroDivisionError: # Occurs when same ticker in different markets is tested
                        pass

        visited_tickers.add(ticker1)

    click.secho('FINISHED'.center(80, '='), fg='bright_blue', bold=True)


def market_market(filepath1: str, filepath2: str, start: str, end: str, freq: str, find_all: bool=False) -> tuple:
    """Handle control flow for 'market' - 'market' and return needed data to instantiate CointegrationPair object."""
    market_ticker1 = read_in_scraped_tickers(filepath1)
    market_ticker2 = read_in_scraped_tickers(filepath2)
    
    # Normally this wouldn't be necessary but I have included the option of specifying the FTSE 350 or S&P 500
    # at the user's discretion, so there may be overlapping tickers, e.g. if user inputs FTSE and LSE
    visited_tickers = set()

    for ticker1 in market_ticker1:
        if is_ticker_active(ticker1):
            for ticker2 in market_ticker2:
                if is_ticker_active(ticker2):
                    try:
                        output = are_securities_cointegrated(ticker1, ticker2, start, end, freq, visited_tickers, find_all)
                        if output:
                            return output
                    except ZeroDivisionError: # Occurs when same ticker in different markets is tested
                        pass

        visited_tickers.add(ticker1)

    click.secho('FINISHED'.center(80, '='), fg='bright_blue', bold=True)


def market_ticker(filepath: str, ticker: str, start: str, end: str, freq: str, find_all: bool=False) -> tuple:
    """Handle control flow for 'market' - 'ticker' and return needed data to instantiate CointegrationPair object."""
    market_tickers = read_in_scraped_tickers(filepath)
    visited_tickers = set()
    
    for market_ticker in market_tickers:
        if is_ticker_active(market_ticker):
            try:
                output = are_securities_cointegrated(market_ticker, ticker, start, end, freq, visited_tickers, find_all)
                if output:
                    return output
            except ZeroDivisionError: # Occurs when same ticker in different markets is tested
                pass

        visited_tickers.add(market_ticker)
    
    click.secho('FINISHED'.center(80, '='), fg='bright_blue', bold=True)


def are_securities_cointegrated(ticker1: str, ticker2: str, start: str, end: str, freq: str, visited_ticker_set: set, find_all: bool) -> tuple:
    """Handles the cointegration checking loop, returns parameters for CointegratedPair instantiation if cointegration is found."""
    try:    
        if ticker1 not in visited_ticker_set and ticker2 != ticker1:
            if not is_cointegration(ticker1, ticker2, start, end, freq):
                click.secho(f'{ticker1.upper()} - {ticker2.upper()} for {start} -> {end}| NO COINTEGRATION', fg='bright_red', bold=True)
            else:
                click.secho(f'{ticker1} - {ticker2} for {start} -> {end} | COINTEGRATION', fg='bright_green', bold=True, blink=True)

                test_dataframe, t_stat, p_val, crit_val, param = is_cointegration(ticker1, ticker2, start, end, freq)
                
                if _is_already_saved(ticker1, ticker2):
                    click.secho(f'{ticker1.upper()} - {ticker2.upper()} | ALREADY EXISTS IN FILE', fg='bright_white', bold=True)
                
                else:
                    if find_all: # If find_all=True, then automatically save the pair to file
                        current = CointegratedPair(ticker1, ticker2, test_dataframe, t_stat, p_val, crit_val, param)
                        _pickle_pair(current)
                        click.secho('SAVED TO FILE', fg='bright_white', bold=True)

                    else:
                        view_pair = back_to_cli()
                        if view_pair:        
                            return (ticker1, ticker2, test_dataframe, t_stat, p_val, crit_val, param)
                        else:
                            pass
    
    except ValueError:
        click.secho(f'NO DATA for {start} -> {end}', fg='bright_yellow', bold=True)
        pass


def back_to_cli() -> bool:
    """Asks user if they want to save cointegrated pair, discard it or exit program."""
    running = True
   
    while running:
        save = input('\nView cointegrated security pair? [y/n] (press \'e\' to exit)\n')
        if save.lower().strip() == 'y':
            return True
        elif save.lower().strip() == 'n':
            return False
        elif save.lower().strip() == 'e':
            sys.exit()


def save_pair(object):
    """Saves cointegrated pair to file using pickle."""
    running = True
    
    while running:
        save = input('Save cointegrated pair to file? [y/n]\n')
        if save.lower().strip() == 'y':
            _pickle_pair(object)
            running = False
        elif save.lower().strip() == 'n':
            sys.exit()


def _pickle_pair(object, filename: str='pairs.p'):
    """Appends new pair to list of saved pairs in 'pairs.p'."""
    
    try:
        updated = pickle.load(open(filename, 'rb'))
        updated += [object]
        pickle.dump(updated, open(filename, 'wb'))
    except EOFError:
        pickle.dump(list(object), open(filename, 'wb'))


def load_in_saved_pairs(filename: str='pairs.p'):
    """Loads in all saved cointegration pairs from pickle file and displays to terminal."""
    
    for i, pair in enumerate(pickle.load(open(filename, 'rb'))):
        click.secho(f'\n{i} {pair.name}', fg='magenta')
    
    # This specific return layout is a bug work around
    if pickle.load(open(filename, 'rb')):
        return pickle.load(open(filename, 'rb'))
    else:
        click.secho('NO PAIRS SAVED', fg='magenta', bold=True)


def _is_already_saved(ticker1: str, ticker2: str, filename: str='pairs.p') -> bool:
    """Checks if a cointegrated pair is already saved to file."""
    
    for pair in pickle.load(open(filename, 'rb')):
        if ticker1.lower() == pair.security_1.lower() and ticker2.lower() == pair.security_2.lower():
            return True
        elif ticker1.lower() == pair.security_2.lower() and ticker2.lower() == pair.security_1.lower():
            return True

    return False


def plot_pair(selected_pair, selected_plots: list, test_spec: str):
    """Outputs visual plot of securities, spread and/or z-score."""
    # Check if selected_plots is empty
    if not selected_plots:
        selected_plots = ['p']
    
    # For simplicity, just instantiate all optional plots, then print the user specified
    selected_pair.gen_pairs_trade_params(test_spec) # Needed to obtain self.hedge_ratio
    selected_pair.get_spread()
    selected_pair.get_z()

    selected_pair.graph(test_spec, *selected_plots)


def view_egranger(current_object):
    """Prints Engle-Granger test results to terminal."""
    
    current_object.get_visualisation_parameters()
    terminal_output = TestResults(current_object.test_name, current_object.visualisation_parameters)
    terminal_output.specify_test()
    terminal_output.construct_visualisation()
    
    return terminal_output.output


def view_johansen(current_object):
    """Prints Johansen test results to terminal."""

    terminal_output = TestResults(current_object.test_name, current_object.visualisation_parameters, current_object.johansen_data)
    terminal_output.specify_test()
    terminal_output.construct_visualisation()

    return terminal_output.output


def check_selection(objects: list, deleted_objects: dict, filename: str='pairs.p'):
    """Saves edited pair list to file after user calls delete."""
    for index, pair in deleted_objects.items():
        click.secho(f'\n{index} {pair.name}', fg='bright_red', bold=True)    
    confirm = input(f'\nCONFIRM: Do you want to delete these {len(deleted_objects)} CointegrationPair objects? [y/n]\n')
    
    running = True
    while running:
        if confirm.lower().strip() == 'y':
            pickle.dump(objects, open(filename, 'wb'))
            running = False
        elif confirm.lower().strip() == 'n':
            sys.exit()
