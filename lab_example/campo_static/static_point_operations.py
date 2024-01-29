import pcraster.framework as pcrfw

import campo


class FoodEnvironment(pcrfw.StaticModel):
    def __init__(self):
        pcrfw.StaticModel.__init__(self)

    def initial(self):
        foodenv = campo.Campo()
        foodstores = foodenv.add_phenomenon("foodstores")
        foodstores.add_property_set("frontdoor", "foodstores_frontdoor.csv")

        foodstores.frontdoor.postal_code = 1234

        foodstores.frontdoor.lower = -0.5
        foodstores.frontdoor.upper = 0.5
        foodstores.frontdoor.x_initial = campo.uniform(foodstores.frontdoor.lower, foodstores.frontdoor.upper)

        foodenv.create_dataset("food_environment.lue")
        foodenv.write()


if __name__ == "__main__":
    myModel = FoodEnvironment()
    staticFrw = pcrfw.StaticFramework(myModel)
    staticFrw.run()
