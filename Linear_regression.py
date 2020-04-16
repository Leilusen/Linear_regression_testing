import pandas as pd
import numpy as np

class Regression:

    def __init__(self, data = None, x_cols = None, y_col = None, cutoff = None):

        """ Regression class for calculating linear regression estimators and associated test statistics

        Attributes:
            data (pandas dataframe)
            y_col (str): name of the dependent variable
            x_cols (list of str): list of names of explanatory variables
            cutoff (int): row number for division between train and test datasets
        """
        self.data = data
        if cutoff == None:
            self.cutoff = int(0.8 * data.shape[0])
        else:
            self.cutoff = cutoff
        self.train_data = data.loc[:self.cutoff]
        self.test_data = data.loc[self.cutoff+1:]
        self.x_cols = x_cols
        self.y_col = y_col

        self.n, self.result, self.y_fitted, self.b, self.e, self.s,\
        self.R_sq, self.AIC, self.BIC, self.DF = self.get_regression_result(self.train_data, self.x_cols, self.y_col)


    def get_regression_result(self, data, x_cols, y_col):

        index = ['Const'] + x_cols
        result = pd.DataFrame(0, index=index, columns=['b', 'se_b', 't_b'])
        n = len(data)
        k = len(x_cols) + 1
        DF = n - k
        x = np.c_[np.ones(n), data.loc[:,x_cols]]
        y = np.array(data[y_col])
        y_mean = y.mean()
        y_demean_sq_sum = sum([(i - y_mean) ** 2 for i in y])
        b = self.get_estimates(x, y)
        y_fitted = self.get_fitted(x, b)
        e = self.get_residuals(y, y_fitted)
        sum_e_sq = self.get_sum_squared_residuals(e)
        s = self.get_standard_error(e, DF)
        R_sq = self.get_R_squared(e, y_demean_sq_sum)
        A = np.linalg.solve(np.dot(x.T, x), np.identity(k))
        AIC = round(np.log(s ** 2) + 2 * k / n, 3)
        BIC = round(np.log(s ** 2) + k * np.log(n) / n, 3)

        for i, feat in enumerate(index):
            s_b = s * np.sqrt(A[i,i])
            t_b = b[i] / s_b
            result.loc[feat,'b'] = round(b[i], 3)
            result.loc[feat,'se_b'] = round(s_b, 3)
            result.loc[feat,'t_b'] = round(t_b, 3)

        return n, result, y_fitted, b, e, s, R_sq, AIC, BIC, DF

    def fit(self, test_data, x_cols = None):

        if x_cols == None: x_cols = self.x_cols
        n = test_data.shape[0]
        x = np.c_[np.ones(n), test_data.loc[:,x_cols]]
        return np.dot(x, self.b)

    def get_estimates(self, x, y):
        b = np.linalg.solve(np.dot(x.T,x), np.dot(x.T, y))
        return b

    def get_fitted(self, x, b):
        return np.dot(x, b)

    def get_residuals(self, y, y_fitted):
        return y - y_fitted

    def get_standard_error(self, e, DF):
        return np.sqrt(np.dot(e.T, e) / DF)

    def get_R_squared(self, e, y_demean_sq_sum):
        return 1 - np.dot(e.T, e) / y_demean_sq_sum

    def get_sum_squared_residuals(self, e):
        return sum(e ** 2)

    def get_skewness_kurtosis(self, e, n):

        e_mean = e.mean()
        e_demean_quad_sum = sum([(i - e_mean) ** 2 for i in e])
        e_demean_cube_sum = sum([(i - e_mean) ** 3 for i in e])
        e_demean_cuar_sum = sum([(i - e_mean) ** 4 for i in e])
        S = e_demean_cube_sum / n / (e_demean_quad_sum / n) ** (3 / 2)
        K = e_demean_cuar_sum / n / (e_demean_quad_sum / n) ** 2

        return S, K

    def Jarque_Bera(self):

        n = self.n
        S, K = self.get_skewness_kurtosis(self.e, n) # skewness and kurtosis

        JB = round((np.sqrt(n / 6) * S) ** 2 + (np.sqrt(n / 24) * (K-3)) ** 2, 3)

        return "Jarque-Bera test statistic: {}".format(JB)

    def RESET_statistic(self):

        p=1
        n = self.n
        x_data = np.c_[np.ones(n), self.train_data.loc[:, self.x_cols]]
        y_data = np.array(self.train_data[self.y_col])
        y_demean_sq_sum = sum([(i - y_data.mean()) ** 2 for i in y_data])
        x = np.c_[x_data, self.y_fitted ** 2]
        DF = n - x.shape[1]
        b = self.get_estimates(x, y_data)
        y_fitted = self.get_fitted(x, b)
        e = self.get_residuals(y_data, y_fitted)
        s = self.get_standard_error(e, DF)
        R_sq = self.get_R_squared(e, y_demean_sq_sum)
        RESET = round((R_sq - self.R_sq) / p / ((1 - R_sq) / DF), 3)

        return "RESET statistic: {} ({}, {})".format(RESET, p, DF)


    def F_test(self, restricted_cols = None):

        restr_model = Regression(self.data, x_cols = restricted_cols,
                                     y_col = self.y_col, cutoff = self.cutoff)

        g = len(self.x_cols) - len(restr_model.x_cols)
        n = self.n
        k = len(self.x_cols) + 1

        R1_sq = self.R_sq # R-squared value of the unrestricted model
        R0_sq = restr_model.R_sq # R-squared value of the restricted model
        F = round((R1_sq - R0_sq) / g / ((1 - R1_sq) / (n - k)), 3)

        return 'F-value: {} ({}, {})'.format(F, g, n-k)

    def Chow_tests(self, cutoff = None):

        df1 = self.train_data[:cutoff]
        df2 = self.train_data[cutoff:]

        k = len(self.x_cols) + 1
        n = self.n
        n1 = df1.shape[0]
        n2 = df2.shape[0]

        e_df1 = self.get_regression_result(df1, self.x_cols, self.y_col)[4]
        e_df2 = self.get_regression_result(df2, self.x_cols, self.y_col)[4]

        # Calculating Chow_break test statistic

        sum_e_sq_full = self.get_sum_squared_residuals(self.e)
        sum_e_sq_df1 = self.get_sum_squared_residuals(e_df1)
        sum_e_sq_df2 = self.get_sum_squared_residuals(e_df2)

        Chow_break = round((sum_e_sq_full - sum_e_sq_df1 - sum_e_sq_df2) / k / \
                     ((sum_e_sq_df1 + sum_e_sq_df2) / (n - 2 * k)), 3)


        Chow_forecast = round((sum_e_sq_full - sum_e_sq_df1) / n2 / (sum_e_sq_df1 / (n1 - k)), 3)

        return "Chow break: {} ({}, {}), Chow forecast: {} ({}, {})".format(Chow_break, k, n-2*k, Chow_forecast, n2, n1-k)

    def out_of_sample_forecast_errors(self):

        df_test = self.test_data
        forecasts = self.fit(df_test)

        columns = ['Error value']
        index = ['RMSE (x100)', 'MAE (x100)', 'SUM (x100)']
        errors = pd.DataFrame(0, index=index, columns=columns)
        n = df_test.shape[0]

        errors.loc['RMSE (x100)', 'Error value'] = round(100 * np.sqrt(sum((df_test[self.y_col] -\
                                                           forecasts) ** 2) / n), 3)
        errors.loc['MAE (x100)', 'Error value'] = round(100 * sum(abs(df_test[self.y_col] - forecasts)) / n, 3)
        errors.loc['SUM (x100)', 'Error value'] = round(100 * sum(df_test[self.y_col] - forecasts), 3)

        return errors


    def __repr__(self):
        return "Regression results:\n\n{}\n\n s: {},\n R_sq: {}".\
                format(self.result, round(self.s,3), round(self.R_sq,3))
