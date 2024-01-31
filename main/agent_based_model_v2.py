import datetime
import numpy as np
import pandas as pd

seed = 5
np.random.seed(seed)


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_distances(df1, df2, x_col1, y_col1, x_col2, y_col2):
    x1, y1 = df1[x_col1].values, df1[y_col1].values
    x2, y2 = df2[x_col2].values, df2[y_col2].values

    # Create meshgrid for vectorized computation
    X1, X2 = np.meshgrid(x1, x2)
    Y1, Y2 = np.meshgrid(y1, y2)

    # Calculate distances using vectorized Euclidean distance formula
    distances = np.sqrt((X2 - X1)**2 + (Y2 - Y1)**2)

    # Convert the result to a DataFrame
    distances_df = pd.DataFrame({
        'index1': np.repeat(np.arange(len(df1)), len(df2)),
        'index2': np.tile(np.arange(len(df2)), len(df1)),
        'distance': distances.flatten()
    })

    return distances_df

class FoodEnvironment:
    def __init__(self):
        # set a dummy clone
        self.clone_dimensions = (10, 20)
        self.initialize_clone()

    def initialize_clone(self):
        # Framework requires a clone
        # set a dummy clone
        pass

    ##########################
    # differential equations #
    ##########################

    # first term, differential equation internal effects
    @staticmethod
    def diff_eq_term_one(x, a, betaH, gammaH):
        return -((betaH / (1.0 + np.exp(-gammaH * (x - a)))) - (betaH / 2.0))

    # second term, differential equation food outlet effects
    @staticmethod
    def diff_eq_term_two(y, a, betaS, gammaS):
        return ((betaS / (1.0 + np.exp(-gammaS * (y - a)))) - (betaS / 2.0))

    def initial(self):
        init_start = datetime.datetime.now()

        ##############
        # Households #
        ##############

        # create households phenomenon
        self.hh = pd.read_csv('data/households_frontdoor.csv', names=['x', 'y'])
        self.hh['x'] = self.hh['x'].astype(float)
        self.hh['y'] = self.hh['y'].astype(float)
        self.hh['utility'] = 0
        self.hh['max_fs_prop'] = 0
        self.hh['healthy_choice'] = 0
        self.hh['unhealthy_choice'] = 0
        self.hh['index1'] = self.hh.index

        # set income for each household
        number_of_households = len(self.hh)  # Assuming a number for illustration
        mean_income = 37.9
        sd = 10

        self.hh['income'] = np.random.normal(mean_income, sd, number_of_households)
        self.hh['high_income'] = self.hh['income'] >= 37.9

        # set initial propensity of households from -2 to 2
        self.hh['propensity_high_income'] = -2 + np.random.beta(4, 1.55, number_of_households) * 4
        self.hh['propensity_low_income'] = -2 + np.random.beta(4, 2.06, number_of_households) * 4
        self.hh['hh_propensity'] = np.where(self.hh['high_income'], self.hh['propensity_high_income'], self.hh['propensity_low_income'])
        self.hh['high_income'] = self.hh['high_income'].astype(int)

        # set default propensity parameter
        self.hh['a'] = np.random.uniform(-0.0001, 0.0001, number_of_households)

        # set betaH parameter
        self.betaH = 8.0

        # set gammaH parameter
        self.gammaH = 0.8
        self.resultingSlopeAtZero = (self.gammaH * self.betaH) / 4.0

        # set betaS parameter
        proportion_one = 0.7
        self.betaS = proportion_one * self.betaH

        # set gammaS parameter
        proportion_two = 4.0
        self.gammaS = ((4 * self.resultingSlopeAtZero) / self.betaS) * proportion_two


        ##############
        # Foodstores #
        ##############

        # create foodstores phenomenon
        self.fs = pd.read_csv('data/foodstores_frontdoor.csv', names=['x', 'y'])
        self.fs['x'] = self.fs['x'].astype(float)
        self.fs['y'] = self.fs['y'].astype(float)
        self.fs['index2'] = self.fs.index

        number_of_foodstores = len(self.fs)

        self.fs['expensive_price'] = np.random.choice([1, 0], number_of_foodstores, p=[0.5, 0.5])
        self.fs['fs_propensity'] = np.random.uniform(-2, 2, number_of_foodstores)

        self.distance_hh_fs = calculate_distances(self.hh, self.fs, 'x', 'y', 'x', 'y')

        self.timestep = 0.333333

        # matrix for price score (hh_hing_income, fs_expensive_price)
        self.price_score_matrix = {
            (1, 1) : 2.43,
            (1, 0) : 2.21,
            (0, 1) : 1.84,
            (0, 0) : 2.84
        }

        # Technical details
        self.price_weight = 0.35
        self.propensity_weight = 0.45
        self.distance_weight = 0.2

    
        self.distance_hh_fs['d_scaled'] = (self.distance_hh_fs['distance'] - self.distance_hh_fs['distance'].min()) / (self.distance_hh_fs['distance'].max() - self.distance_hh_fs['distance'].min())
        self.distance_hh_fs['w_dist'] = self.distance_weight * (1 / self.distance_hh_fs['d_scaled'])

        self.distance_hh_fs = self.distance_hh_fs.merge(self.hh[['index1', 'high_income']], on='index1', how='left')
        self.distance_hh_fs = self.distance_hh_fs.merge(self.fs[['index2', 'expensive_price']], on='index2', how='left')

        self.distance_hh_fs['price_score'] = self.distance_hh_fs.apply(lambda x: self.price_score_matrix[(x['high_income'], x['expensive_price'])], axis=1)
        self.distance_hh_fs['w_price'] = self.price_weight * self.distance_hh_fs['price_score']

        self.hh = self.hh[['x','y','index1','income','high_income', 'hh_propensity','a','utility','max_fs_prop','healthy_choice','unhealthy_choice']]

        self.hh.to_csv('output/households_data_init.csv', index=False)
        self.fs.to_csv('output/foodstores_data_init.csv', index=False)
        self.distance_hh_fs.to_csv('output/distances_fs_hh_init.csv', index=False)

        end = datetime.datetime.now() - init_start

        print(f'init: {end}')

    def dynamic(self, i):
        start = datetime.datetime.now()

        # calculate utility for each hh
        self.distance_hh_fs['pref_score'] = self.distance_hh_fs.apply(lambda x: (self.hh['hh_propensity'][x['index1']] + 2) / 4 if self.fs['fs_propensity'][x['index2']] >= 0 else 1 - (self.hh['hh_propensity'][x['index1']] + 2) / 4, axis=1)
        self.distance_hh_fs['w_pref'] = self.propensity_weight * self.distance_hh_fs['pref_score']

        self.distance_hh_fs['utility'] = self.distance_hh_fs['w_dist'] + self.distance_hh_fs['w_price'] + self.distance_hh_fs['w_pref']

        idx = self.distance_hh_fs.groupby('index1')['utility'].idxmax().values
        hh_utility = self.distance_hh_fs.iloc[idx]
        hh_utility = hh_utility.merge(self.fs, how='inner', right_on='index2', left_on='index2')
        hh_utility = hh_utility[['index1', 'index2', 'utility', 'fs_propensity']]
        
        hh_utility.sort_values('index1')
        self.hh.sort_values('index1')

        self.hh['utility'] = hh_utility['utility']
        self.hh['max_fs_prop'] = hh_utility['fs_propensity']
        self.hh['healthy_choice'] = self.hh.apply(lambda x: x['healthy_choice'] + 1 if x['max_fs_prop'] >= 0 else x['healthy_choice'], axis=1)
        self.hh['unhealthy_choice'] = self.hh.apply(lambda x: x['unhealthy_choice'] + 1 if x['max_fs_prop'] < 0 else x['unhealthy_choice'], axis=1)
        

        # Update household propensity
        self.hh['hh_propensity'] = self.hh.apply(
                    lambda x: x['hh_propensity'] + self.timestep \
                        * (
                            self.diff_eq_term_one(x['hh_propensity'], x['a'], self.betaH, self.gammaH)
                            + self.diff_eq_term_two(x['max_fs_prop'], x['a'], self.betaS, self.gammaS)
                        ),
                        axis=1
        )

        self.hh.to_csv('output/' + str(i) + '_housholds_data_dynamic.csv', index=False)

        # Print run duration info
        end = datetime.datetime.now() - start
        print(f'ts:  {end}  write')

    def run(self, timesteps):
        self.initial()

        for i in range(timesteps):
            self.dynamic(i)


if __name__ == "__main__":
    timesteps = 12
    myModel = FoodEnvironment()
    myModel.run(timesteps)
