import datetime

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
        self.hh.add_property_set('fd', 'households_frontdoor.csv')

        # set default propensity parameter
        self.hh.fd.lower = -0.0001
        self.hh.fd.upper = 0.0001
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

        # set initial propensity of households
        self.hh.fd.lower = -2
        self.hh.fd.upper = 2
        self.hh.fd.x = campo.uniform(self.hh.fd.lower, self.hh.fd.upper, seed)

        # add the surroundings property set
        self.hh.add_property_set('sur', 'households_surrounding.csv')

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
        low = 0.0000001
        high = 1.0
        self.hh.sur.low = low
        self.hh.sur.high = high
        # calculate the weight
        self.hh.sur.weight = campo.where(self.hh.sur.area, self.hh.sur.high, self.hh.sur.low)

        # technical detail
        self.hh.set_epsg(28992)

        ##############
        # Foodstores #
        ##############

        # create foodstores phenomenon
        self.fs = self.foodenv.add_phenomenon('fs')

        # add the frontdoor property set
        self.fs.add_property_set('fd', 'foodstores_frontdoor.csv')

        # create food store propensities either 0.2 or -0.2 (random selection)
        # create random values from a uniform distribution
        self.fs.fd.lower = 0.0
        self.fs.fd.upper = 1.0
        self.fs.fd.tmp = campo.uniform(self.fs.fd.lower, self.fs.fd.upper, seed)
        # select households that will have a low propensity (True) or high (False)
        self.fs.fd.isStandard = self.fs.fd.tmp < 0.7
        # assign the propensities
        self.fs.fd.low_y = 0.00001
        self.fs.fd.high_y = 0.2
        self.fs.fd.y = campo.where(self.fs.fd.isStandard, self.fs.fd.low_y, self.fs.fd.high_y)

        ##############
        # Households #
        ##############

        # technical detail
        self.fs.set_epsg(28992)

        # average store propensity in neighbourhood of houses
        self.hh.fd.y = campo.focal_agents(self.hh.fd, self.hh.sur.weight, self.fs.fd.y, fail=True)

        ########
        # Misc #
        ########

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

        # technical detail to inform lue properties may change over time
        self.hh.fd.x.is_dynamic = True

        # write the lue dataset
        self.foodenv.write()

        # print the run duration
        end = datetime.datetime.now() - init_start
        print(f'init: {end}')

    def dynamic(self):
        start = datetime.datetime.now()

        # household propensity
        self.hh.fd.x = self.hh.fd.x + self.timestep \
                       * (self.diffEqTermOne(self.hh.fd.x, self.hh.fd.a, self.hh.fd.betaH, self.hh.fd.gammaH)
                       + self.diffEqTermTwo(self.hh.fd.y, self.hh.fd.a, self.hh.fd.betaS, self.hh.fd.gammaS))

        self.foodenv.write(self.currentTimeStep())
        end = datetime.datetime.now() - start
        print(f'ts:  {end}  write')


if __name__ == "__main__":
    timesteps = 12
    myModel = FoodEnvironment()
    dynFrw = pcrfw.DynamicFramework(myModel, timesteps)
    dynFrw.run()
