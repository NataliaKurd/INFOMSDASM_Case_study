import pcraster.framework as pcrfw

import campo


class FoodEnvironment(pcrfw.StaticModel):
    def __init__(self):
        pcrfw.StaticModel.__init__(self)

    def initial(self):
        pass


if __name__ == "__main__":
    myModel = FoodEnvironment()
    staticFrw = pcrfw.StaticFramework(myModel)
    staticFrw.run()
