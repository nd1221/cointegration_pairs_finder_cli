import yfinance as yf
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import coint, adfuller, kpss
from datetime import datetime, timedelta
import statsmodels.regression.linear_model as reg
import statsmodels.tools.tools as stt
import matplotlib.pyplot as plt



def is_cointegration(security_1: int, security_2: int, start: str, end: str='current', freq: int='1d') -> int | bool:
    
    """
    Searches for a cointegrating relationship between two securities using Engle-Granger test.
    If relationship is present, function returns t-stat, p-val, crit-val, and relevant DataFrame.
    Return values will be used to initiate a 'Cointegration' object.
    """
    # Check if end parameter is passed
    if end == 'current':
        end_ = datetime.now().strftime('%Y-%m-%d')
        # Bug fix -> without this, run into 'str has no attribute.date()' error
        end_ = datetime.strptime(end_, '%Y-%m-%d')
    else:
        end_ = datetime.strptime(end, '%Y-%m-%d')
        

    # Set parameter list
    param = [datetime.strptime(start, '%Y-%m-%d'), end_, freq]

    # Download historical price data from yfinance package as pandas.DataFrame
    historical_info = yf.download(f"{security_1} {security_2}", start=param[0], end=param[1], interval=param[2])

    # Remove unnecessary columns from DataFrame (only keep adjusted-close for securities)
    test_data = historical_info.iloc[:, 0:2]

    # Loop through DataFrame to remove NaN values, bc statsmodels does not remove missing values
    test_data = test_data.dropna()

    # Remove DataFrame column header level 0
    test_data = test_data.droplevel(0, axis=1)

    
    # Implement Engle-Granger test for cointegration

    # Parameters specified: constant, AIC lag selection criterion for ADF test, maxlag chosen according to Schwert (1989)  
    maxlag = int( 12 * ( ( len(test_data[security_1.upper()] ) / 100) ** 0.25 ) )
    param.append(maxlag)

    # Engle-Granger test
    t_stat, p_val, crit_val = coint(test_data[security_1.upper()].values, test_data[security_2.upper()].values, maxlag=maxlag, return_results=True)

    # Return EG test results
    if p_val <= 0.1:
        if t_stat <= crit_val[0]:
            sig = 1
        elif t_stat <= crit_val[1]:
            sig = 5
        elif t_stat <= crit_val[2]:
            sig = 10
        else:
            raise ValueError('P-value cannot indicate significance if t-statistic greater than all critical values.')
        
        param.append(sig)

        return test_data, t_stat, p_val, crit_val, param
    
    else: # p-value > 0.1 -> no statistically significant cointegration present
        return False


class CointegratedPair:
    """
    Creates base initialisation of a cointegrated security pair for use in other objects/visualisations.
    Class stores the attributes of an Engle-Granger cointegration test between two securities,
    which is used as the default method of finding trading pairs.
    """
    test_name = 'egranger'


    def __init__(self, security_1: str, security_2: str, coint_pair_dataframe, t_statistic: int, p_value: float, critical_values, param: list):
        self.security_1 = security_1.upper()
        self.security_2 = security_2.upper()
        
        self.coint_pair_dataframe = coint_pair_dataframe
        self.t_stat = t_statistic
        self.p_value = p_value
        self.critical_values = critical_values.tolist() # .tolist() is a Pandas method (taken from Numpy)
        self.param = param

        # self.name is so each object has a unique identifier for loading in from pickle files
        self.name = f'{self.security_1} - {self.security_2} ({self.param[0]} -> {self.param[1]})'


    def __repr__(self):
        """Print Info Summary, but make it slow af :/ Surely there's a faster way to do this..."""
        
        return f"\nCointegration Pair: {self.security_1} {self.security_2}\n\n\
Cointegration test: Engle-Granger\n\
Significance: {self.param[4]}%\n\n"


    def _format_egranger_tstat(self):
        """Adds marker to denote significance at 1%, 5% and 10% levels by *, **, ***, respectively."""
        if self.p_value <= 0.01:
            self.t_stat = f'{self.t_stat:.4f}***'
        elif self.p_value <= 0.05:
            self.t_stat = f'{self.t_stat:.4f}**'
        elif self.p_value <= 0.1:
            self.t_stat = f'{self.t_stat:.4f}*'
        else:
            self.t_stat = f'{self.t_stat:.4f}'


    def get_visualisation_parameters(self):
        """Returns an attribute in a form which can be passed to TestResults class to generate output."""
        
        # Format t-statistic by calling helper method
        CointegratedPair._format_egranger_tstat(self)
        
        # Need start and end dates, number of observations, data frequency, and security names
        # Finally, need to pass hardcoded layout substrings which contain various items to be 
        # assembled to fit the 80 char output line limit

        # Equation layout substrings
        self.equation = '{:<40}'.format(f'Series: {self.security_1} {self.security_2}')
        self.equation_spec = '{:>40}'.format(f'Maxlags (SIC): {self.param[3]} | Obs: {len(self.coint_pair_dataframe[self.security_1])}')
        # Critical values and test statistic header layout substrings
        self.cv_header = '{:^39}'.format('Critical Values')
        self.cv_values = '| {:<37} | {:^10} | {:^10} | {:^10} |'.format('Test Statistic (p-value)', '1%', '5%', '10%')
        # Descriptive information
        self.test_info = f'Results show Engle-Granger cointegration test between {self.security_1} and {self.security_2}.\n\
Null hypothesis: No cointegration present.\n\
*, **, *** denotes significance at the 10%, 5% and 1% levels, respectively.\n\
Critical values obtained from MacKinnon (1990, 2010).'

        # Create visualisation parameters list to pass to TestResults
        self.visualisation_parameters = [
            self.t_stat, self.p_value, self.critical_values, self.param,
            self.equation, self.equation_spec, self.cv_header, self.cv_values,
            self.test_info
        ]

    
    def gen_pairs_trade_params(self, test_specification: int='gls'):
        """
        Generates the hedge ratio and constant needed to create a pairs trading equation.
        Allows user to specify parameter estimation method: OLS, GLS.
        """
        supported_tests = ['ols', 'gls']

        # Check test is supported
        if test_specification.lower() not in supported_tests:
            raise ValueError('Program does not support this model specification.')

        # Set up regressor/regressand dataframe series for parameter estimation 
        # Manually add constant to dataframe bc OLS() and GLS() won't do it automatically
        regressand = self.coint_pair_dataframe[self.security_1]
        regressor = stt.add_constant(self.coint_pair_dataframe[self.security_2], prepend=True)
        
        # Generate parameter estimates using OLS specification
        self.ols = reg.OLS(regressand, regressor).fit()

        # Generate parameter estimates using GLS specification
        self.gls = reg.GLS(regressand, regressor).fit()

        # Get hedge ratio and constant using match case block
        match test_specification.lower():
            case 'ols':
                reg_coefficients = self.ols.params.tolist()
                self.pair_constant = reg_coefficients[0]
                self.hedge_ratio = reg_coefficients[1]
            case 'gls':
                reg_coefficients = self.gls.params.tolist()
                self.pair_constant = reg_coefficients[0]
                self.hedge_ratio = reg_coefficients[1]
        

    def show_parameter_estimation(self, test_specification: int='gls'):
        """Outputs results + summary statistics from OLS/GLS estimations if user requires them."""
        supported_tests = ['ols', 'gls']

        # Check test is supported
        if test_specification.lower() not in supported_tests:
            print(f'Cannot show estimation summary statistics\nProgram does not support model specification: {test_specification}')
        
        # Print summaries to terminal using match case block
        match test_specification.lower():
            case 'ols':
                print(f"{'='*78}\n{self.ols.summary()}\n{'='*78}")
            case 'gls':
                print(f"{'='*78}\n{self.gls.summary()}\n{'='*78}")


    def get_spread(self):
        """Calculate spread and append to coint_pair_dataframe."""

        # Calculate spread series and add to dataframe using apply() function

        # Define function to calculate spread and pass it to each dataframe row through df.apply()
        def calculate_spread(row, hedge, cons):
            """Applies formula: spread = {security_1} - {hedge_ratio} * {security_2} - {constant}, to each row in dataframe."""
            return row[0] - (hedge * row[1]) - cons

        # Create new dataframe column for spread
        self.coint_pair_dataframe['spread'] = self.coint_pair_dataframe.apply(calculate_spread, args=[self.hedge_ratio, self.pair_constant], axis=1)

        self.spread = self.coint_pair_dataframe['spread']


    def _add_datapoint(self, start: str, end: str):
        """Appends 1 period yfinance data for the pair to the existing pair dataframe."""
        # Download and format dataframe
        new_df = yf.download(f'{self.security_1} {self.security_2}', start=start, end=end).iloc[:, 0:2].dropna().droplevel(0, axis=1)

        # Check that data is not missing or a duplicate (weird yfinance bug creates duplicates based on trading volume, I think...)
        if new_df.empty:
            # Is missing?
            pass
        elif start in self.coint_pair_dataframe.index:
            # Is duplicate?
            pass
        else:
            # Calculate spread colum + concatenate dataframes
            new_df['spread'] = new_df.apply(lambda x: new_df[self.security_1].iloc[0] - self.hedge_ratio * new_df[self.security_2].iloc[0] - self.pair_constant, axis=1)

            # Pass list to concat() rather than individual dataframes for speed
            frames = [new_df, self.coint_pair_dataframe]
            self.coint_pair_dataframe = pd.concat(frames)
        
        # Need this return statement, o/w if/elif block return NoneType object which raises error when passed through len() in get_z()
        # _add_datapoint() returns None, then while loop in get_z() sets self.coint_pair_dataframe == None
        return self.coint_pair_dataframe


    def get_z(self):
        """Calculate z_score of spread using a 7-day moving average."""
        # Get initial number of observations
        init_len = len(self.coint_pair_dataframe)
        
        # Set start and end dates to cycle through bc yfinance needs both start and end to return 1 day's data
        # To get specific date specify start as that date, and end as the next day
        
        # Start from date preceding  in dataframe
        new_period_start = self.param[0] - timedelta(days=1)
        new_period_end = self.param[0]

        # Add 6 extra datapoints to create rolling window (moving average) parameter estimates for first date in original sample
        while len(self.coint_pair_dataframe) < init_len + 6:
            # Use '<' over '<=' bc loop o/w appends additional unwanted datapoint
            self.coint_pair_dataframe = self._add_datapoint(new_period_start, new_period_end)
            new_period_start -= timedelta(days=1)
            new_period_end -= timedelta(days=1)

        # Create moving average for mean + stdev of spread and append as columns
        # Can now drop the datapoints added as were only needed to get mean+stdev for first few sample datapoints
        self.coint_pair_dataframe['m_avg'] = self.coint_pair_dataframe.spread.rolling(7).mean()
        self.coint_pair_dataframe['m_avg_std'] = self.coint_pair_dataframe.spread.rolling(7).std()
        self.coint_pair_dataframe.dropna(how='any', inplace=True)
        
        # Create z-Score
        # z-Score = ( spread - (7-day moving average) ) / stdev(spread)
        self.coint_pair_dataframe['z_score'] = self.coint_pair_dataframe.apply(lambda row: ( row.spread - row.m_avg ) / row.m_avg_std, axis=1)
        self.z = self.coint_pair_dataframe['z_score']


    def _plot_pair(self, test_spec: str='ols'):
        """Creates matplotlib.pyplot plot of cointegration securities pair for visual inspection."""
        # Plot securities on ax
        self.pair_plot.plot(self.coint_pair_dataframe[self.security_1], color='#0388fc', label=f'{self.security_1}')
        self.pair_plot.plot(self.coint_pair_dataframe[self.security_2], color='#f0352b', label=f'{self.security_2}')
        
        # Set ax_pair axis labels and title and their position on screen using x,y coordinates
        self.pair_plot.set_ylabel('Price (USD)', fontsize=12)
        self.pair_plot.set_xlabel('Date', fontsize=12)
        self.pair_plot.set_title(f'Visual Plot: {self.security_1} - {self.security_2} Pair ({test_spec.upper().strip()})', fontsize=20, y=1.03)
        
        # Make output more aesthetically pleasing
        self.pair_plot.fill_between(self.coint_pair_dataframe.index, self.coint_pair_dataframe[self.security_1], self.coint_pair_dataframe[self.security_2], color='#dee0df')
        self.pair_plot.legend(loc='best')
        self.pair_plot.grid()        


    def _plot_spread(self):
        """Creates matplotlib.pyplot plot of pair spread for visual inspection."""
        # Plot spread on ax
        self.spread_plot.plot(self.spread, color='#0388fc', label='Spread')
        self.spread_plot.axhline(y=0, color='#f0352b', linestyle='--')

        # Set ax_spread axis labels and title and their position on screen using x,y coordinates
        self.spread_plot.set_xlabel('Date', fontsize=12)
        self.spread_plot.set_title('Estimated Sample Spread', fontsize=20, y=1.03)

        # Make output more aesthetically pleasing
        self.spread_plot.fill_between(self.spread.index, self.spread, 0, color='#dee0df')
        self.spread_plot.legend(loc='best')
        self.spread_plot.grid()


    def _plot_z(self):
        """Creates matplotlib.pyplot plot of pair 7-day moving average z-Score for visual inspection."""
        # Plot z-Score on ax as bar chart
        self.z_plot.bar(self.z.index, self.z, color='#0388fc', edgecolor='#014375', linewidth=0.15, label='z-Score')
        
        # Plot stdev markers
        stdev_z = self.z.std()
        self.z_plot.axhline(y=stdev_z, color='red', linewidth=0.5, linestyle='--', label='1 STDEV')
        self.z_plot.axhline(y=-stdev_z, color='red', linewidth=0.5, linestyle='--')
        self.z_plot.axhline(y=2*stdev_z, color='black', linewidth=0.5, linestyle='--', label='2 STDEV')
        self.z_plot.axhline(y=-2*stdev_z, color='black', linewidth=0.5, linestyle='--',)

        # Set ax_spread axis labels and title and their position on screen using x,y coordinates
        self.z_plot.set_xlabel('Date', fontsize=12)
        self.z_plot.set_title('Sample z-Score Trading Signal', fontsize=20, y=1.03)

        # Make output more aesthetically pleasing
        self.z_plot.legend(loc='upper left') # Legend very likely to cover some data, position over oldest data on plot
        self.z_plot.grid()


    def graph(self, test_spec: str, *args: str):
        """Creates figure and allows user to specify plots (combination of pair, spread or z-Score) to be drawn to it."""
        # 'p' = pairs plot, 's' = spread plot, 'z' = z-core plot
        accepted_args = ['p', 's', 'z']  
        
        # Prepare args variable
        # Positional arguments stored as tuple, convert to list to .insert()/.remove() incorrect parameters
        if not args:
            args = list('p') # 'p' is the default parameter of self.graph()
        else:
            args = [n.lower() for i, n in enumerate(args) if n not in args[:i] and n.lower() != 'p' and n.lower() in accepted_args]
            # 'p' is removed from list comprehension for easy insertion into index 0
            args.insert(0, 'p')

        
        # Print error if user inputs non-accepted graph argument
        for item in args:
            if item not in accepted_args:
                print(f'Arg: {item} is not valid')
                args.remove(item)

        # Initialise figure and spaces for plots
        fig_column = len(args)
        fig, ax = plt.subplots(fig_column, 1)
        # h_pad kwarg is defined as a fraciton of fontsize


        # Set space between plots
        if len(args) == 1:
            pad = 4
        elif len(args) == 2:
            pad = 3
        elif len(args) == 3:
            pad = 2
        
        fig.tight_layout(pad=pad)

        # Set plot size
        height_parameter = len(args) * 3.5
        fig.set_figwidth(10)
        fig.set_figheight(height_parameter)

        # Initialise inputted plots
        for item in args:
            i = args.index(item)

            match item:
                case 'p':
                    # Work around for AxesSubplot object not subscriptable error when figure specified is 1x1 (in this case, AxesSubplot has no __getitem__()??)
                    if len(args) == 1:
                        self.pair_plot = ax
                    # If multiple plot arguments passed:
                    else:
                        self.pair_plot = ax[i]
                    self._plot_pair(test_spec)
                case 's':
                    self.spread_plot = ax[i]
                    self._plot_spread()
                case 'z':
                    self.z_plot = ax[i]
                    self._plot_z()
        
        # If initialised, show plot
        plt.show()



class Johansen:
    """Creates Johansen cointegration object for robustness testing using statsmodels module."""
    test_name = 'johansen'

    def __init__(self, security_1: str, security_2: str, coint_pair_dataframe, param: list):
        self.security_1 = security_1
        self.security_2 = security_2
        self.coint_pair_dataframe = coint_pair_dataframe
        self.param = param
    

    def get_johansen(self):
        """Run Johansen cointegration test using statsmodels.tsa.vector_ar.vecm"""

        # Johansen test assuming 1 cointegrating relationship, a constant and 1 lagged difference
        test = coint_johansen(self.coint_pair_dataframe, 0, 1)
        # Statistics of Trace test
        trace = test.trace_stat.tolist()
        trace_cv = test.trace_stat_crit_vals.tolist()
        # Statistics of Maximum Eigenvalue test
        me = test.max_eig_stat.tolist()
        me_cv = test.max_eig_stat_crit_vals.tolist()
        # With a pair there will only ever be at most 1 cointegrating equation
        coint_eq = ['None', 'At most 1']
        
        self.johansen_data = [coint_eq, trace, trace_cv, me, me_cv]


    def format_results(self):
        """
        Checks if Johansen test shows evidence of cointegration.
        Denotes a rejection of the null of no cointegration by '*' next to relevant critical value.
        """
        # Trace test
        for i, cv in enumerate(self.johansen_data[2][0]):
            if self.johansen_data[1][0] < cv:
                self.johansen_data[2][0][i] = f'{cv}--'
            else:
                self.johansen_data[2][0][i] = f'{cv}*'
        for i, cv in enumerate(self.johansen_data[2][1]):
            if self.johansen_data[1][1] < cv:
                self.johansen_data[2][1][i] = f'{cv}*'

                
        # Max Eigenvalue test
        for i, cv in enumerate(self.johansen_data[4][0]):
            if self.johansen_data[3][0] < cv:
                self.johansen_data[4][0][i] = f'{cv}--'
            else:
                self.johansen_data[4][0][i] = f'{cv}*'
        for i, cv in enumerate(self.johansen_data[4][1]):
            if self.johansen_data[3][1] < cv:
                self.johansen_data[4][1][i] = f'{cv}*'


    def get_visualisation_parameters(self):
        """Returns list of variables to pass to TestResults to construct an output visualisation."""
        
        # Need start and end dates, number of observations, data frequency, and security names
        # Also need johansen test types and descriptive info
        # Finally, need to pass hardcoded layout substrings which contain various items to be 
        # assembled to fit the 80 char output line limit

        # Critical value layout substring (same for all instances)
        cv_header = '{:^39}'.format('Critical Values')
        # Cointegrating equation form layout substring
        self.equation = '{:<34}'.format(f'VECRANK {self.security_1} {self.security_2}')
        # Observations to pass to equation spec substring
        self.obs = len(self.coint_pair_dataframe[self.security_1])
        # Equation spec layout substring
        self.equation_spec = '{:>40}'.format(f'Trend: constant term | Lags: 1 | Obs: {self.obs}')
        # Johansen test descriptive information
        self.test_info = f'Results show unrestricted Johansen cointegration test between {self.security_1} and {self.security_2}.\n\
Null hypothesis: no cointegrating relationship.\n\
Cointegration present if test statistic falls below critical value.\n\
* denotes rejection of null at specified significance level.\n\
-- denotes non-rejection of null hypothesis of zero cointegrating pairs; i.e.,\n\
no cointegration present.'


        self.visualisation_parameters = [
            self.security_1,
            self.security_2,
            self.param,
            cv_header,
            self.equation,
            self.equation_spec,
            self.test_info,
        ]



class RobustnessTest:
    """
    Robustness check for given pair in the form of user-specified unit root procedure + graphs and summary output.
    For statistical significance, most cointegration tests bar one/two (e.g., Bounds) assume non-stationarity of time series,
    i.e., both securities which make up a pair must be integrated of order I(1).
    """

    def __init__(self, security_1: str, security_2: str, coint_pair_dataframe, start: str, end: str):
        self.security_1 = security_1
        self.security_2 = security_2
        self.coint_pair_dataframe = coint_pair_dataframe
        self.start = start.date()
        self.end = end.date()

    
    def adf_test(self):
        """Returns a list of dicts of useful statistics from ADF test for stationarity."""
        self.adf = []
        
        # No need to specify maxlag equation, adfuller() default maxlag value is Schwert (1989) formula (default information criterion used is AIC)
        adf_security_1 = adfuller(self.coint_pair_dataframe[self.security_1], regression='c', store=True, regresults=True) # adfuller() function automatically differences series according to source code, no need to do it manually
        adf_security_2 = adfuller(self.coint_pair_dataframe[self.security_2], regression='c', store=True, regresults=True)
        
        # Create list containing dicts of statistics for each security tested
        for security in [adf_security_1, adf_security_2]:
            _ = {
                't_stat': security[-1].adfstat,
                'p_val': security[1],
                'crit_vals': security[-1].critvalues,
                'obs': security[-1].nobs,
                'lag': security[-1].maxlag,
            }

            self.adf.append(_)


    def kpss_test(self):
        """Returns a list of dicts of useful statistics from KPSS test for stationarity."""
        self.kpss = []

        # nlags='auto' sets lags according to Hobijn, et al. (1998)
        kpss_security_1 = kpss(self.coint_pair_dataframe[self.security_1], regression='c', nlags='auto', store=True)
        kpss_security_2 = kpss(self.coint_pair_dataframe[self.security_2], regression='c', nlags='auto', store=True)

        # Create list containing dicts of statistics for each security tested
        for security in [kpss_security_1, kpss_security_2]:
            _ = {
                't_stat': security[0],
                'p_val': security[1],
                'crit_vals': security[2],
                'obs': security[-1].nobs,
                'lag': security[-1].lags,
            }

            self.kpss.append(_)


    def visualise_test(self, test: str):
        """Returns a string representation of unit root test results."""
        accepted_tests = ['adf', 'kpss']
        if test not in accepted_tests:
            print(f'ValueError: {test} is not a supported unit root test')
            # force stop function
            return None
        
        # Common format variables (should be capitalised by convention but I forgot)
        end_break = '='*80
        line = '-'*80
        cv_title = '{:^39}'.format('Critical Values')
        statistic_headings = '| {:<37} | {:^10} | {:^10} | {:^10} |'.format('Test Statistic ( p-value )', '10%', '5%', '1%')

        # Print desired output using match-case block
        match test:
            
            case 'adf':
                # Format test data for header, body and footer
                series1 = f'Series: {self.security_1}, {self.start} - {self.end}'.ljust(40)
                series1_info = f"Lags (AIC): {self.adf[0]['lag']} | No. of Obs: {self.adf[0]['obs']}".rjust(40)
                series2 = f"Series: {self.security_2}, {self.start} - {self.end}".ljust(40)
                series2_info = f"Lags (AIC): {self.adf[1]['lag']} | No. of Obs: {self.adf[1]['obs']}".rjust(40)
                info = 'ADF null hypothesis: unit root present\n\
Significance using MacKinnon (1990, 2010) critical values is denoted by *, **,\nand *** at the 10%, 5% and 1% levels, respectively.\n\
Series is stationary if t-statistic is larger than 10% significance level.'

                l1 = f'{series1}{series1_info}'
                l2 = cv_title.rjust(80)
                l3 = statistic_headings
                l4 = '| {:<37} | {:^10.4f} | {:^10.4f} | {:^10.4f} |'.format(f"{self.adf[0]['t_stat']:.4f} ({self.adf[0]['p_val']:.4f})", self.adf[0]['crit_vals']['10%'], self.adf[0]['crit_vals']['5%'], self.adf[0]['crit_vals']['1%'])
                l5 = f'{series2}{series2_info}'
                l6 = cv_title.rjust(80)
                l7 = statistic_headings
                l8 = '| {:<37} | {:^10.4f} | {:^10.4f} | {:^10.4f} |'.format(f"{self.adf[1]['t_stat']:.4f} ({self.adf[1]['p_val']:.4f})", self.adf[1]['crit_vals']['10%'], self.adf[1]['crit_vals']['5%'], self.adf[1]['crit_vals']['1%'])
                
                body = f'{l1}\n\n{line}\n\n{l2}\n{l3}\n\n{l4}\n\n{end_break}\n\n{l5}\n\n{line}\n\n{l6}\n{l7}\n\n{l8}'
                header = f"{end_break}\n{'Augmented Dickey-Fuller Unit Root Test Results'.center(80)}\n{end_break}"
                footer = f"{line}\n{info}\n{end_break}"

            case 'kpss':
                # Format test data for header, body and footer
                series1 = f'Series: {self.security_1}, {self.start} - {self.end}'.ljust(40)
                series1_info = f"Lags (AIC): {self.kpss[0]['lag']} | No. of Obs: {self.kpss[0]['obs']}".rjust(40)
                series2 = f"Series: {self.security_2}, {self.start} - {self.end}".ljust(40)
                series2_info = f"Lags (AIC): {self.kpss[1]['lag']} | No. of Obs: {self.kpss[1]['obs']}".rjust(40)                
                info = 'KPSS null hypothesis: no unit root present\n\
Significance using Kwiatowski, et al. (1992) critical values is denoted by *,\n**, and *** at the 10%, 5% and 1% levels, respectively.\n\
Series is stationary if t-statistic is larger than a given significance level.'

                l1 = f'{series1}{series1_info}'
                l2 = cv_title.rjust(80)
                l3 = statistic_headings
                l4 = '| {:<37} | {:^10.4f} | {:^10.4f} | {:^10.4f} |'.format(f"{self.kpss[0]['t_stat']:.4f} ({self.kpss[0]['p_val']:.4f})", self.kpss[0]['crit_vals']['10%'], self.kpss[0]['crit_vals']['5%'], self.kpss[0]['crit_vals']['1%'])
                l5 = f'{series2}{series2_info}'
                l6 = cv_title.rjust(80)
                l7 = statistic_headings
                l8 = '| {:<37} | {:^10.4f} | {:^10.4f} | {:^10.4f} |'.format(f"{self.kpss[1]['t_stat']:.4f} ({self.kpss[1]['p_val']:.4f})", self.kpss[1]['crit_vals']['10%'], self.kpss[1]['crit_vals']['5%'], self.kpss[1]['crit_vals']['1%'])

                body = f'{l1}\n\n{line}\n\n{l2}\n{l3}\n\n{l4}\n\n{end_break}\n\n{l5}\n\n{line}\n\n{l6}\n{l7}\n\n{l8}'
                header = f"{end_break}\n{'KPSS Unit Root Test Results'.center(80)}\n{end_break}"
                footer = f"{line}\n{info}\n{end_break}"
        
        # Print formatted output results
        print(f'{header}\n\n{body}\n\n{footer}')



class TestResults:
    """Class creates a visualisation of cointegration test output in string format for printing in the terminal."""
    
    # Formatting variables
    line = '-'*80
    d_line = '='*80


    def __init__(self, test_name: str, test_visualisation_parameters: list, test_data: list[int]=None):
        self.test_name = test_name
        self.td = test_data
        self.tvp = test_visualisation_parameters


    def specify_test(self):
        """Match-case block to generate the desired test output string."""
        
        match self.test_name:
            
            case 'egranger':
                # Format test data for body of results
                l1 = f'{self.tvp[4]}{self.tvp[5]}'
                l2 = '{:>80}'.format(f'Period: {self.tvp[3][0]} - {self.tvp[3][1]} | Frequency: {self.tvp[3][2]}')
                l3 = '{:>80}'.format(self.tvp[6])
                l4 = self.tvp[7]
                l5 = '| {:<37} | {:^10.4f} | {:^10.4f} | {:^10.4f} |'.format(f'{self.tvp[0]} ({self.tvp[1]:.4f})', self.tvp[2][0], self.tvp[2][1], self.tvp[2][2])

                # Define header, body and footer attributes
                self.out_title = 'Engle Granger Cointegration Test'.center(80)
                self.out_body = f'{l1}\n{l2}\n\n{self.line}\n\n{l3}\n{l4}\n\n{l5}'
                self.out_footer = self.tvp[8]

            case 'johansen':
                # Format test data for body of results
                l1 = f'{self.tvp[4]}{self.tvp[5]}'
                l2 = '{:>80}'.format(f'Period: {self.tvp[2][0]} - {self.tvp[2][1]} | Frequency: {self.tvp[2][2]}')
                l3 = 'Trace Test'
                l4 = self.tvp[3].rjust(80)
                l5 = '| {:<37} | {:^10} | {:^10} | {:^10} |'.format('Test Statistic', '10%', '5%', '1%')
                l6 = '| {:<37} | {:^10} | {:^10} | {:^10} |'.format(f'{self.td[1][0]:.4f} ({self.td[0][0]})', self.td[2][0][0], self.td[2][0][1], self.td[2][0][2])
                l7 = '| {:<37} | {:^10} | {:^10} | {:^10} |'.format(f'{self.td[1][1]:.4f} ({self.td[0][1]})', self.td[2][1][0], self.td[2][1][1], self.td[2][1][2])
                l8 = 'Maximum Eigenvalue Test'
                l9 = self.tvp[3].rjust(80)
                l10 = '| {:<37} | {:^10} | {:^10} | {:^10} |'.format('Test Statistic', '10%', '5%', '1%')
                l11 = '| {:<37} | {:^10} | {:^10} | {:^10} |'.format(f'{self.td[3][0]:.4f} ({self.td[0][0]})', self.td[4][0][0], self.td[4][0][1], self.td[4][0][2])
                l12 = '| {:<37} | {:^10} | {:^10} | {:^10} |'.format(f'{self.td[3][1]:.4f} ({self.td[0][1]})', self.td[4][1][0], self.td[4][1][1], self.td[4][1][2])
                l13 = self.tvp[6]

                # Define header, body and footer attributes
                self.out_title = 'Johansen Cointegration Test'.center(80)
                self.out_body = f'{l1}\n{l2}\n\n{self.line}\n{l3}\n{self.line}\n{l4}\n{l5}\n\n{l6}\n{l7}\n\n\n{self.line}\n{l8}\n{self.line}\n{l9}\n{l10}\n\n{l11}\n{l12}'
                self.out_footer = l13

                return self.out_title, self.out_body, self.out_footer
    

    def construct_visualisation(self):
        """Formats title, body and footer attributes."""
        self.output = f'{self.d_line}\n{self.out_title}\n{self.d_line}\n\n{self.out_body}\n\n\n{self.line}\n{self.out_footer}\n{self.d_line}'
        return self.output
