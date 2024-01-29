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

    def initial(self):
        init_start = datetime.datetime.now()
        self.foodenv = campo.Campo()

        ##############
        # Households #
        ##############

        # create households phenomenon
        self.hh = self.foodenv.add_phenomenon('hh')
        self.hh.add_property_set('fd', 'data/households_frontdoor.csv')

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
        # proportionOne = 0.7
        proportionOne = 0.00001
        self.hh.fd.betaS = proportionOne * self.hh.fd.betaH

        # set gammaS parameter
        proportionTwo = 4.0
        self.hh.fd.gammaS = ((4 * self.hh.fd.resultingSlopeAtZero) / self.hh.fd.betaS) * proportionTwo

        # set initial propensity of households
        self.hh.fd.lower = -2
        self.hh.fd.upper = 2
        self.hh.fd.x = campo.uniform(self.hh.fd.lower, self.hh.fd.upper, seed)

        # technical detail
        self.hh.set_epsg(28992)

        # set the duration (years) of one time step
        self.timestep = 0.333333

        # create the output lue data set
        self.foodenv.create_dataset("output/food_environment.lue")

        # create real time settings for lue
        date = datetime.date(2000, 1, 2)
        time = datetime.time(12, 34)
        start = datetime.datetime.combine(date, time)
        unit = campo.TimeUnit.month
        stepsize = 4
        self.foodenv.set_time(start, unit, stepsize, self.nrTimeSteps())

        # technical detail
        self.hh.fd.x.is_dynamic = True

        # write the lue dataset
        self.foodenv.write()

        # print the run duration
        end = datetime.datetime.now() - init_start
        print(f'init: {end}')

    # differential equation internal effects
    def diffEqTermOne(self, x, a, betaH, gammaH):
        return -((betaH / (1.0 + campo.exp(-gammaH * (x - a)))) - (betaH / 2.0))

    def dynamic(self):
        start = datetime.datetime.now()

        ## update household propensity
        self.hh.fd.x = self.hh.fd.x + self.timestep \
                                     * self.diffEqTermOne(self.hh.fd.x, self.hh.fd.a, self.hh.fd.betaH, self.hh.fd.gammaH)

        # print run duration info
        self.foodenv.write(self.currentTimeStep())
        end = datetime.datetime.now() - start
        print(f'ts:  {end}  write')


if __name__ == "__main__":
    timesteps = 12
    myModel = FoodEnvironment()
    dynFrw = pcrfw.DynamicFramework(myModel, timesteps)
    dynFrw.run()
