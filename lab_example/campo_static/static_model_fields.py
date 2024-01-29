import pcraster.framework as pcrfw

import campo


class FoodEnvironment(pcrfw.StaticModel):
    def __init__(self):
        pcrfw.StaticModel.__init__(self)

    def initial(self):
        foodenv = campo.Campo()
        foodstores = foodenv.add_phenomenon("foodstores")
        foodstores.add_property_set("frontdoor", "data/foodstores_frontdoor.csv")

        foodstores.frontdoor.postal_code = 1234

        foodstores.frontdoor.lower = -0.5
        foodstores.frontdoor.upper = 0.5
        foodstores.frontdoor.x_initial = campo.uniform(foodstores.frontdoor.lower, foodstores.frontdoor.upper)

        foodstores.add_property_set('surrounding', 'data/foodstores_surrounding.csv')

        foodstores.surrounding.lower = 3
        foodstores.surrounding.upper = 12
        foodstores.surrounding.c = campo.uniform(foodstores.surrounding.lower, foodstores.surrounding.upper)

        foodstores.surrounding.start_locations = campo.feature_to_raster(foodstores.surrounding, foodstores.frontdoor)

        foodenv.create_dataset("food_environment.lue")
        foodenv.write()


if __name__ == "__main__":
    myModel = FoodEnvironment()
    staticFrw = pcrfw.StaticFramework(myModel)
    staticFrw.run()
