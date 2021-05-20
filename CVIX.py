# --------------------------------
# Chinese VIX
# @Author: Bowen  @Date:2021/05/20
# --------------------------------


#In[]
import pandas as pd
import numpy as np
import datetime as dt
import statsmodels.api as sm
import matplotlib.pyplot as plt
from WindPy import w
w.start()


# In[]
class Option():
    '''
    A class describing the data structure of an option contract, 
    including its basic info and daily price.
    '''
    def __init__(self, option_code, us_code, us_price, exe_mode, close_price, exe_price, ptm_day):
        self.code = option_code
        self.us_code = us_code
        self.us_price = us_price
        self.exe_mode = exe_mode
        self.price = close_price
        self.strike = exe_price
        self.ptm_day = ptm_day
        self.tau = self.ptm_day/365
        pass


# In[]
class WindPyOptionDataReader():
    '''
    A class reading, handling data of all traded options from WindPy 
    given the underlier's code and a specific date.
    '''
    def __init__(self, us_code, date):
        self.us_code = us_code
        self.date = date
        pass

    def us_price(self):
        date_str = self.date.strftime("%Y-%m-%d")
        us_price = w.wss(self.us_code,
                         fields='close',
                         tradeDate=date_str,
                         usedf=True)[1].iloc[0, 0]
        return us_price

    def option_set(self):
        option_set = w.wset("optionchain",
                            date=self.date.strftime("%Y-%m-%d"),
                            us_code=self.us_code,
                            option_var="全部",
                            call_put="全部",
                            field='option_code',
                            usedf=True)[1]
        return list(option_set.option_code)

    def raw_option_data(self):
        fields = "sec_name,exe_mode,exe_type,exe_price,exe_ratio," +\
            "startdate,lasttradingdate,totaltm,ptmtradeday,ptmday,settlementmethod," +\
            "pre_close,open,high,low,close,vwap,settle,pct_chg,volume,amt," +\
            "theoryvalue,intrinctvalue,timevalue,delta,gamma,vega,theta,rho," +\
            "underlyinghisvol_30d,us_hisvol,underlyinghisvol_90d,us_impliedvol"
        raw_option_data = w.wss(self.option_set(),
                                fields=fields,
                                tradeDate=self.date.strftime("%Y-%m-%d"),
                                usedf=True)[1]
        return raw_option_data

    def create_option_list(self):
        data_df =self.raw_option_data()
        us_price = self.us_price()
        option_list = ([Option(option_code, self.us_code, us_price, exe_mode, close_price, exe_price, ptm_day) 
            for option_code, exe_mode, close_price, exe_price, ptm_day in zip(data_df.index, 
                                                                              data_df['EXE_MODE'],            
                                                                              data_df['CLOSE'],
                                                                              data_df['EXE_PRICE'],
                                                                              data_df['PTMDAY'])])
        return option_list 

    @staticmethod
    def split_option_list_by_term(option_list):
        '''
        Return a dictionary, whose keys are days to matuirty and values are option objects.
        '''
        day_to_maturities = sorted(list(set([option.ptm_day for option in option_list])))
        option_dict = {}
        for dtm in day_to_maturities:
            option_dict[dtm] = [option for option in option_list if option.ptm_day == dtm]
        return option_dict
    

# In[]
class TermOptionData():
    '''
    A class describing data structure of all options on a single term.
    The input is an option list obtained from instance method WindPyOptionDataReader.create_option_list()
    '''

    def __init__(self, option_list):
        self.us_code = option_list[0].us_code
        self.S0 = option_list[0].us_price
        self.ptm_day = option_list[0].ptm_day
        self.tau = option_list[0].tau
        self.calls = [option for option in option_list if option.exe_mode=='认购']
        self.puts = [option for option in option_list if option.exe_mode=='认沽']
        self.strikes = [call.strike for call in self.calls]
        self.call_prices = [call.price for call in self.calls]
        self.put_prices = [put.price for put in self.puts]
        self.diff = [np.abs(call_price - put_price) for call_price, put_price in zip(self.call_prices, self.put_prices)]
        pass

    def tabulate(self):
        data_frame = pd.DataFrame({'Strikes': self.strikes,
                                'Calls': self.call_prices,
                                'Puts': self.put_prices,
                                'Difference': self.diff})
        return data_frame
    
    
# In[]
class VarSwapRateCalculator():
    '''
    A class used to compute the fair variance swap rate 
    given data structure of all options on a term.
    The input is an instance from class TermOptionData.
    '''

    def __init__(self, term_option_data):
        self.data_table = term_option_data.tabulate()
        self.tau = term_option_data.tau
        self.rf = self.implied_discount_rate()
        pass

    def implied_discount_rate(self):
        y = self.data_table['Puts'] - self.data_table['Calls'] 
        x = self.data_table['Strikes'].values
        regression_res = sm.OLS(y, sm.add_constant(x)).fit()
        implied_rf = np.log(regression_res.params[1]) / (-self.tau)
        return implied_rf

    def F(self):
        id_diffmin = self.data_table['Difference'].idxmin()
        CK = self.data_table.loc[id_diffmin, 'Calls']
        PK = self.data_table.loc[id_diffmin, 'Puts']
        K = self.data_table.loc[id_diffmin, 'Strikes']
        return np.exp(self.rf*self.tau) * (CK-PK) + K

    def K0(self):
        return self.data_table['Strikes'][self.data_table['Strikes']<self.F()].max()

    def Q(self):
        selected_puts = self.data_table.Puts[self.data_table['Strikes']<self.K0()]
        atm_option = (self.data_table.Puts[self.data_table['Strikes']==self.K0()] +\
             self.data_table.Calls[self.data_table['Strikes']==self.K0()]) / 2
        selected_calls = self.data_table.Calls[self.data_table['Strikes']>self.K0()]
        return selected_puts.append(atm_option).append(selected_calls)

    def K(self):
        return self.data_table.Strikes

    def deltaK(self):
        deltaK = []
        K = self.data_table.Strikes
        for i in range(len(K)):
            if i == 0:
                deltaK.append(K[i+1]-K[i])
            elif i == len(K)-1:
                deltaK.append(K[i]-K[i-1])
            else:
                deltaK.append((K[i+1]-K[i-1])/2)
        return pd.Series(deltaK)
    
    def value(self):
        first_term = (self.deltaK()*self.Q()/self.K()**2 ).sum() * 2/self.tau * np.exp(self.rf*self.tau)
        second_term = -(self.F()/self.K0()-1)**2 / self.tau
        return first_term + second_term


# In[]
class VIX():
    '''
    A class ensembling all previous steps and used to compute VIX.
    The inputs are the underlier's code and the calculation date.
    '''
    def __init__(self, us_code, date):
        self.us_code = us_code
        self.date = date
        self.option_prep = WindPyOptionDataReader.split_option_list_by_term(
            WindPyOptionDataReader(us_code, date).create_option_list())
        self.left_days = list(self.option_prep.keys())
        self.tau = [d/365 for d in self.left_days]
        self.N1 = self.left_days[self.two_term_id()[0]]
        self.N2 = self.left_days[self.two_term_id()[1]]
        self.T1 = self.tau[self.two_term_id()[0]]
        self.T2 = self.tau[self.two_term_id()[1]]
        self.near_term_options = self.option_prep[self.N1]
        self.next_term_options = self.option_prep[self.N2]
        self.VT1 = VarSwapRateCalculator(TermOptionData(self.near_term_options)).value()
        self.VT2 = VarSwapRateCalculator(TermOptionData(self.next_term_options)).value()

    def two_term_id(self):
        left_days_tmp = sorted(self.left_days + [30])
        if left_days_tmp.index(30) == 0:
            near_term_id = 0
        else:
            if left_days_tmp[left_days_tmp.index(30)-1] > 8:
                near_term_id = left_days_tmp.index(30) - 1
            else:
                near_term_id = left_days_tmp.index(30)
        next_term_id = near_term_id + 1
        return near_term_id, next_term_id

    @staticmethod
    def linear_interp(x1, x2, y1, y2, x_star):
        k = (y2-y1)/(x2-x1)
        dx = x_star - x1
        dy = k * dx
        y_star = y1 + dy
        return y_star
    
    def value(self):
        vol = VIX.linear_interp(self.N1, self.N2, self.T1*self.VT1, self.T2*self.VT2, 30)
        return np.sqrt(365/30 * vol) * 100


# In[]
if __name__ == '__main__':
    
    # params
    us_code = '510050.SH'
    start = dt.date(2017, 1, 1)
    end = dt.date(2017, 12, 31)
    time_stamps = pd.date_range(start, end)
    
    # compute Chinses VIX
    vix_dict = {time_stamp.date(): VIX(us_code, time_stamp.date()).value()
                for time_stamp in time_stamps}
    vix_ts = pd.Series(vix_dict)

    # obtain iVIX data from WindPy
    ivix =  w.wsd("IVIX.SH", "close", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
                  "Days=Alldays;Fill=Previous", 
                  usedf=True)[1]['CLOSE']

    # series correlation coefficient
    corr = vix_ts.corr(ivix)
    
    # plot C-VIX and iVIX
    plt.plot(ivix, '--', color='slategrey', label='iVIX')
    plt.plot(vix_ts, '-', color='teal', label='C-VIX')
    plt.ylabel('Vol (%)')
    plt.xlabel('Date')
    plt.legend()
    plt.xticks(rotation=45)