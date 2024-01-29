import pcraster as pcr
import pcraster.framework as pcrfw

import campo

# set the seed such that each time the program is run, random number generators
# will return the same result
seed = 5
pcr.setrandomseed(seed)


class FoodEnvironment(pcrfw.StaticModel):
    def __init__(self):
        pcrfw.StaticModel.__init__(self)

    def initial(self):
        foodenv = campo.Campo()

        # foodstores

        foodstores = foodenv.add_phenomenon("foodstores")
        foodstores.add_property_set("frontdoor", "foodstores_frontdoor.csv")

        foodstores.add_property_set("surrounding", "foodstores_surrounding.csv")
        foodstores.surrounding.start_locations = campo.feature_to_raster(foodstores.surrounding, foodstores.frontdoor)
        foodstores.surrounding.initial_friction = 0
        foodstores.surrounding.friction = 1
        foodstores.surrounding.distance = campo.spread(foodstores.surrounding.start_locations,
                                          foodstores.surrounding.initial_friction, foodstores.surrounding.friction)
        foodstores.surrounding.area = foodstores.surrounding.distance <= 250

        foodstores.surrounding.high = 1.0
        foodstores.surrounding.low = 0.0000001
        foodstores.surrounding.weight = campo.where(foodstores.surrounding.area, foodstores.surrounding.high,
                                        foodstores.surrounding.low)

        # households
        households = foodenv.add_phenomenon("households")
        households.add_property_set("frontdoor", "households_frontdoor.csv")

        # assign propensity for healthy food as a random value between lower and upper
        households.frontdoor.lower = -2.0
        households.frontdoor.upper = 2.0
        households.frontdoor.x = campo.uniform(households.frontdoor.lower, households.frontdoor.upper)

        # required technical settings
        foodstores.set_epsg(28992)
        households.set_epsg(28992)

        # foodstores updated with information from households

        # calculate for each foodstore the average of propensity of households in the surrounding,
        # weighted by foodstores.surrounding.weight and assign to y
        foodstores.frontdoor.y = campo.focal_agents(foodstores.frontdoor, foodstores.surrounding.weight,
                                          households.frontdoor.x)

        # write data set
        foodenv.create_dataset("food_environment.lue")
        foodenv.write()


if __name__ == "__main__":
    myModel = FoodEnvironment()
    staticFrw = pcrfw.StaticFramework(myModel)
    staticFrw.run()
