import click
from datetime import datetime
from cointegration_test import *
from cli_helper_functions import *



INTERVALS = ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
MODELS = ('ols', 'gls')
ROBUST = ('adf', 'kpss')


@click.group()
def main():
    pass

@click.command()
def show_markets():
    """List available markets which program can search through"""
    click.secho("Markets available:\n\n AMEX | LSE | NYSE | NASDAQ | FOREX | ETF | FTSE | SP500 \n", bold=True)

@click.command()
@click.argument(
    "securities",
    type=str,
    nargs=2,
    )
@click.option(
    '-p1',
    '--period1',
    'p1',
    type=str,
    required=True,
    )
@click.option(
    '-p2',
    '--period2',
    'p2',
    default='current',
    type=str,
    required=False,
    )
@click.option(
    '-i',
    '--interval',
    'interval',
    default='1d',
    type=click.Choice(INTERVALS),
    required=True,
    )
@click.option(
    '-fa',
    '--find-all',
    'find_all',
    is_flag=True,
)

def find_pair(securities, p1, p2, interval, find_all):
    """
    Find Cointegrated Security Pairs\n
    'find-pair {market} {only}': searches for pairs within the specified market\n
    'find-pair {market} {market}': searches for pairs between the specified markets\n
    'find-pair {security_ticker} {market}': takes the given security and looks for a cointegration match within the specified market    
    """
    search_type, search_info = process_ticker_search(securities) # search_info -> dict{'filepath | ticker symbol': '"market" | "ticker"'}

    # After finding a cointegrating pair, instantiate CointegrationPair object, print output to terminal and ask user to save
    match search_type:
        
        case 0: # 1 market passed
            click.secho('CASE 0', fg='bright_white', bold=True)
            filepath = list(search_info.keys())[0]
            parameters = market_only(filepath, p1, p2, interval, find_all)
            if not find_all:
                current_obj = CointegratedPair(*parameters)
            
                click.secho('\nCOINTEGRATION\n', fg='bright_green', bold=True, blink=True)
                click.secho(current_obj, fg='bright_cyan', bold=True)
                terminal = view_egranger(current_obj)
                print(f'\n{terminal}\n')

                save_pair(current_obj)

        
        case 1: # 2 markets passed
            click.secho('CASE 1', fg='bright_white', bold=True)
            filepath1, filepath2 = list(search_info.keys())[0], list(search_info.keys())[1]
            parameters = market_market(filepath1, filepath2, p1, p2, interval, find_all)
            if not find_all:
                current_obj = CointegratedPair(*parameters)

                click.secho('\nCOINTEGRATION\n', fg='bright_green', bold=True, blink=True)
                click.secho(current_obj, fg='bright_cyan', bold=True)
                terminal = view_egranger(current_obj)
                print(f'\n{terminal}\n')

                save_pair(current_obj)

        
        case 2: # 1 market, 1 valid ticker passed
            click.secho('CASE 2', fg='bright_white', bold=True)
            for i in search_info.keys():
                if search_info[i] == 'market':
                    filepath = i
                else:
                    ticker = i

            parameters = market_ticker(filepath, ticker, p1, p2, interval, find_all)
            if not find_all:
                current_obj = CointegratedPair(*parameters)

                click.secho('\nCOINTEGRATION\n', fg='bright_green', bold=True, blink=True)
                click.secho(current_obj, fg='bright_cyan', bold=True)
                terminal = view_egranger(current_obj)
                print(f'\n{terminal}\n')

                save_pair(current_obj)

        
        case 3: # 2 valid tickers passed
            click.secho('CASE 3', fg='bright_white', bold=True)
            ticker1, ticker2 = search_info.keys()
            
            if is_cointegration(ticker1, ticker2, p1, p2, interval):
                test_dataframe, t_stat, p_val, crit_val, param = is_cointegration(ticker1, ticker2, p1, p2, interval)
                try:
                    current_obj = CointegratedPair(ticker1, ticker2, test_dataframe, t_stat, p_val, crit_val, param)
                except ZeroDivisionError:
                    click.secho('SAME TICKER or SAME COMPANY, DIFFERENT EXCHANGE', fg='bright_red', bold=True)

                click.secho('\nCOINTEGRATION\n', fg='bright_green', bold=True, blink=True)
                click.secho(current_obj, fg='bright_cyan', bold=True)
                
                terminal = view_egranger(current_obj)
                print(f'\n{terminal}\n')

                save_pair(current_obj)
            
            else:
                click.secho(f"{ticker1.upper()} - {ticker2.upper()} NOT COINTEGRATED BETWEEN {p1} AND {datetime.now().strftime('%Y-%m-%d' if p2 == 'current' else p2)}", fg='red', bold=True)
    

@click.command()
def ls():
    """List all saved cointegration pairs in 'pairs.p' as: {index} {CointegrationPair}"""
    load_in_saved_pairs()


@click.command()
@click.argument(
    'pair_index',
    type=int,
    nargs=-1,
)
@click.option(
    '-r',
    '--range',
    'r',
    type=int,
    nargs=2,
)
@click.option(
    '-s',
    '--save',
    'save',
    default=False,
    is_flag=True,
)
@click.option(
    '-a',
    '--all',
    'delete_all',
    is_flag=True,
)

def delete(r, save, delete_all, pair_index):
    """
    Deletes a saved CointegratioPair object/range of objects from 'pairs.p'\n
    Enter saved pair's index to delete\n
    '-r {index 1} {index 2}' specifies a range to be deleted\n
    '-s' flag which deletes all pairs but the input selection/range
    """
    all_pairs = load_in_saved_pairs()
    
    try:
        indexes = [int(i) for i, _ in enumerate(all_pairs)]

        if delete_all:
            indexes = []

        if r:
            try:
                del indexes[r[0]:r[1]+1]
            except ValueError:
                del indexes[r[0]:]
        
        if pair_index:
            try:
                for n in pair_index:
                    if int(n) in indexes:
                        indexes.remove(int(n))
            except ValueError:
                pass

        delete_list = {}
        
        if not save:
            for i in range(len(all_pairs)-1, -1, -1):
                if i not in indexes:
                    delete_list[i] = all_pairs[i]
                    del all_pairs[i]
        else:
            for i in range(len(all_pairs)-1, -1, -1):
                if i in indexes:
                    delete_list[i] = all_pairs[i]
                    del all_pairs[i]

        check_selection(all_pairs, delete_list)

        # Print updated pairs list
        for i, pair in enumerate(pickle.load(open('pairs.p', 'rb'))):
            click.secho(f'\n{i} {pair.name}', fg='bright_green')
    
    except TypeError:
        pass    


@click.command()
@click.option(
    '-o',
    '--object',
    'pair_index',
    type=int,
    required=True,
)
@click.option(
    '-ts',
    '--test-spec',
    'test_spec',
    type=click.Choice(MODELS),
    default='gls',
)
def regression(pair_index, test_spec):
    """Show regression output and statistics for given CointegrationPair and estimation technique"""
    selected_pair = load_in_saved_pairs()[int(pair_index)]
    click.secho(selected_pair, fg='bright_green', bold=True)
    
    selected_pair.gen_pairs_trade_params(test_spec)
    selected_pair.show_parameter_estimation(test_spec)


@click.command()
@click.option(
    '-o',
    '--object',
    'pair_index',
    type=int,
    required=True,
)
@click.option(
    '-ts',
    '--test-spec',
    'test_spec',
    type=click.Choice(MODELS),
    required=False,
    default='gls',
)
@click.option(
    '-s',
    '--spread',
    'spread',
    is_flag=True,
)
@click.option(
    '-z',
    '--z-score',
    'z_score',
    is_flag=True,
)
def plot(pair_index, test_spec, spread, z_score):
    """Plots a selected CointegrationPair, and/or its corresponding Spread (-s) and/or z-Score (-z)"""
    selected_pair = load_in_saved_pairs()[int(pair_index)]
    click.secho(selected_pair, fg='bright_green', bold=True)
    
    selected_plots = []
    if spread:
        selected_plots.append('s')
    if z_score:
        selected_plots.append('z')
    
    #print(selected_pair.coint_pair_dataframe, selected_pair.security_1, selected_pair.security_2)
    plot_pair(selected_pair, selected_plots, test_spec)


@click.command()
@click.option(
    '-o',
    '--object',
    'pair_index',
    type=int,
    required=True,
)
def johansen(pair_index):
    """Generate Johansen test data for robustness"""
    selected_pair = load_in_saved_pairs()[int(pair_index)]
    click.secho(selected_pair, fg='bright_green', bold=True)

    j_pair = Johansen(selected_pair.security_1, selected_pair.security_2, selected_pair.coint_pair_dataframe, selected_pair.param)
    j_pair.get_johansen()
    j_pair.format_results()
    j_pair.get_visualisation_parameters()
    
    click.echo(f'\n{view_johansen(j_pair)}\n')
    
    # For Unit Tests
    return 'complete'


@click.command()
@click.option(
    '-o',
    '--object',
    'pair_index',
    required=True,
)
@click.option(
    '-ts',
    '--test-spec',
    'test_spec',
    type=click.Choice(ROBUST),
    required=True,
)
def robust(pair_index, test_spec):
    """Displays ADF / KPSS tests for stationarity of individual ticker data"""
    selected_pair = load_in_saved_pairs()[int(pair_index)]
    r_test = RobustnessTest(selected_pair.security_1, selected_pair.security_2, selected_pair.coint_pair_dataframe, selected_pair.param[0], selected_pair.param[1])
    click.secho(selected_pair, fg='bright_green', bold=True)

    if test_spec.lower().strip() == 'adf':
        r_test.adf_test()
        # Check attribute exists
        if not r_test.adf:
            raise ValueError('No ADF test data detected, check object dataframe')
    elif test_spec.lower().strip() == 'kpss':
        r_test.kpss_test()
        if not r_test.kpss:
            raise ValueError('No KPSS test data detected, check object dataframe')
    
    r_test.visualise_test(test_spec)



main.add_command(show_markets)
main.add_command(find_pair)
main.add_command(ls)
main.add_command(delete)
main.add_command(regression)
main.add_command(plot)
main.add_command(johansen)
main.add_command(robust)
