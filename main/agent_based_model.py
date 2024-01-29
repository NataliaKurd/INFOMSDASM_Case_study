import datetime
import numpy as np

import pcraster as pcr
import pcraster.framework as pcrfw

import campo

seed = 5
pcr.setrandomseed(seed)


class FoodEnvironment(pcrfw.DynamicModel):
    def __init__(self):

        pcrfw.DynamicModel.__init__(self)
        # Framework requires a clone
        # set a dummy clone
        pcr.setclone(10, 20, 10, 0, 0)


    ##########################
    # differential equations #
    ##########################
    # first term, differential equation internal effects
    def diffEqTermOne(self, x, a, betaH, gammaH):
        return -((betaH / (1.0 + campo.exp(-gammaH * (x - a)))) - (betaH / 2.0))

    # second term, differential equation food outlet effects
    def diffEqTermTwo(self, y, a, betaS, gammaS):
        return ((betaS / (1.0 + campo.exp(-gammaS * (y - a)))) - (betaS / 2.0))

    def initial(self):
        init_start = datetime.datetime.now()
        self.foodenv = campo.Campo()
        

        ##############
        # Households #
        ##############

        # create households phenomenon
        self.hh = self.foodenv.add_phenomenon('hh')
        self.hh.add_property_set('fd', 'data/households_frontdoor.csv')

        # constants 
        self.hh.fd.utility = 0
        self.hh.fd.price_weight = 0.45
        self.hh.fd.propensity_weight = 0.55

        # set income for each household
        number_of_households = self.hh.nr_agents
        mean_income = 37.9 # https://opendata.cbs.nl/statline/#/CBS/en/dataset/83739ENG/table?ts=1706440130257
        sd = 10

        self.hh.fd.true = 1
        self.hh.fd.false = 0.000001
        self.hh.fd.income = np.random.normal(mean_income, sd, number_of_households)
        self.hh.fd.high_income = self.hh.fd.income >= 37.9
        self.hh.fd.high_income_int = campo.where(self.hh.fd.high_income, self.hh.fd.true, self.hh.fd.false)

        # set initial propensity of households from 0 to 1
        # mean 0.739 and 0.661 for high- and low-income households, respectively http://dx.doi.org/10.1046/j.1365-277X.1998.00084.x

        self.hh.fd.propensity_low_income = np.random.beta(4, 2.06, number_of_households)
        self.hh.fd.propensity_high_income = np.random.beta(4, 1.55, number_of_households)
        self.hh.fd.x = campo.where(self.hh.fd.high_income, self.hh.fd.propensity_high_income, self.hh.fd.propensity_low_income)


        # set default propensity parameter
        self.hh.fd.lower = 0.45
        self.hh.fd.upper = 0.55
        self.hh.fd.a = campo.uniform(self.hh.fd.lower, self.hh.fd.upper, seed)

        # set betaH parameter
        self.hh.fd.betaH = 8.0

        # set gammaH parameter
        self.hh.fd.gammaH = 0.8
        self.hh.fd.resultingSlopeAtZero = (self.hh.fd.gammaH * self.hh.fd.betaH) / 4.0

        # set betaS parameter
        proportionOne = 0.7
        self.hh.fd.betaS = proportionOne * self.hh.fd.betaH

        # set gammaS parameter
        proportionTwo = 4.0
        self.hh.fd.gammaS = ((4 * self.hh.fd.resultingSlopeAtZero) / self.hh.fd.betaS) * proportionTwo

        # add the surroundings property set
        self.hh.add_property_set('sur', 'data/households_surrounding.csv')

        # calculate distance away from center
        # assign location of shop to property in surroundings property set
        self.hh.sur.start_locations = campo.feature_to_raster(self.hh.sur, self.hh.fd)
        # set some parameters for distance calculation
        self.hh.sur.initial_friction = 0
        self.hh.sur.friction = 1
        # calculate the distance
        self.hh.sur.distance = campo.spread(self.hh.sur.start_locations, self.hh.sur.initial_friction, self.hh.sur.friction)

        # calculate the weight for averaging propensity of households in surroundings
        # calculate a zone of less than maxdistance (m) away from foodstore
        maxdistance = 500
        self.hh.sur.area = self.hh.sur.distance <= maxdistance
        # set value to assign outside zone and inside zone
        low = 0.000001
        high = 1.0
        self.hh.sur.low = low
        self.hh.sur.high = high
        # calculate the weight for distance
        self.hh.sur.weight = campo.where(self.hh.sur.area, self.hh.sur.high, self.hh.sur.low)

        # technical detail
        self.hh.set_epsg(28992)


        ##############
        # Foodstores #
        ##############

        # create foodstores phenomenon
        self.fs = self.foodenv.add_phenomenon('fs')

        # add the frontdoor property set
        self.fs.add_property_set('fd', 'data/foodstores_frontdoor.csv')

        # add the surroundings property set
        self.fs.add_property_set('sur', 'data/foodstores_surrounding.csv')

        # calculate distance away from center
        # assign location of shop to property in surroundings property set
        self.fs.sur.start_locations = campo.feature_to_raster(self.fs.sur, self.fs.fd)
        # set some parameters for distance calculation
        self.fs.sur.initial_friction = 0
        self.fs.sur.friction = 1
        # calculate the distance
        self.fs.sur.distance = campo.spread(self.fs.sur.start_locations, self.fs.sur.initial_friction, self.fs.sur.friction)

        # calculate the weight for averaging propensity of households in surroundings
        # calculate a zone of less than 250 m away from foodstore
        self.fs.sur.area = self.fs.sur.distance <= 100
        # set value to assign outside zone and inside zone
        self.fs.sur.high = high
        self.fs.sur.low = low
        # calculate the weight
        self.fs.sur.weight = campo.where(self.fs.sur.area, self.fs.sur.high, self.fs.sur.low)

        # technical detail
        self.fs.set_epsg(28992)

        # calculate propensity of surrounding households for each foodstore as starting point for dynamic part
        self.fs.fd.y = campo.focal_agents(self.fs.fd, self.fs.sur.weight, self.hh.fd.x, fail=True)
        
        # add price level feature, expensive if propensity is higher then average, cheap if lower
        self.fs.fd.true = 1
        self.fs.fd.false = 0.000001
        self.fs.fd.expensive = self.fs.fd.y >= 0.5
        self.fs.fd.expensive_int = campo.where(self.fs.fd.expensive, self.fs.fd.true, self.fs.fd.false)

        # set the duration (years) of one time step
        self.timestep = 0.333333

        # create the output lue data set
        self.foodenv.create_dataset("food_environment.lue")

        # create real time settings for lue
        date = datetime.date(2000, 1, 2)
        time = datetime.time(12, 34)
        start = datetime.datetime.combine(date, time)
        unit = campo.TimeUnit.month
        stepsize = 4
        self.foodenv.set_time(start, unit, stepsize, self.nrTimeSteps())

        # technical detail
        self.hh.fd.x.is_dynamic = True
        self.hh.fd.y = 0.0 # temporary value
        self.hh.fd.y.is_dynamic = True
        self.fs.fd.y.is_dynamic = True

        # write the lue dataset
        self.foodenv.write()

        # print the run duration
        end = datetime.datetime.now() - init_start
        print(f'init: {end}')

        ##############
        # Households #
        ##############


    def dynamic(self):
        start = datetime.datetime.now()

        self.hh.fd.average_fs_price = campo.focal_agents(self.hh.fd, self.hh.sur.weight, self.fs.fd.expensive_int, fail=True)
        self.hh.fd.y = campo.focal_agents(self.hh.fd, self.hh.sur.weight, self.fs.fd.y, fail=True)
        
        self.hh.fd.utility = self.hh.fd.price_weight * (self.hh.fd.high_income_int - self.hh.fd.average_fs_price) \
            + self.hh.fd.propensity_weight * (self.hh.fd.y - self.hh.fd.x)

        # update household propensity
        self.hh.fd.x = self.hh.fd.x + self.timestep \
                       * (self.diffEqTermOne(self.hh.fd.x, self.hh.fd.a, self.hh.fd.betaH, self.hh.fd.gammaH)
                       + self.diffEqTermTwo(self.hh.fd.y, self.hh.fd.a, self.hh.fd.betaS, self.hh.fd.gammaS))

        # average house propensity in neighbourhood of stores
        self.fs.fd.y = campo.focal_agents(self.fs.fd, self.fs.sur.weight, self.hh.fd.x, fail=True)

        # update stores prices
        self.fs.fd.expensive = self.fs.fd.y >= 0.5
        self.fs.fd.expensive_int = campo.where(self.fs.fd.expensive, self.fs.fd.true, self.fs.fd.false)

        # print run duration info
        self.foodenv.write(self.currentTimeStep())
        end = datetime.datetime.now() - start
        print(f'ts:  {end}  write')

if __name__ == "__main__":
    timesteps = 12
    myModel = FoodEnvironment()
    dynFrw = pcrfw.DynamicFramework(myModel, timesteps)
    dynFrw.run()
