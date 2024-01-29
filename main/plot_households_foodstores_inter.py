import lue.data_model as ldm
import campo
import pandas
import matplotlib.pyplot as plt

dataset = ldm.open_dataset('food_environment.lue')

for i in range(1, 13):
    dataframe = campo.dataframe.select(dataset.hh, property_names=['x', 'y', 'utility', 'high_income'])
    campo.to_gpkg(dataframe, 'output/households.gpkg', 'EPSG:28992', i)
    dataframe = campo.dataframe.select(dataset.fs, property_names=['y', 'expensive'])
    campo.to_gpkg(dataframe, 'output/foodstores.gpkg', 'EPSG:28992', i)


dataframe = campo.dataframe.select(dataset.hh, property_names=['x'])
campo.to_csv(dataframe, "output/households.csv")
propFrame = pandas.read_csv("output/households_x.csv")
propFrame.plot(legend=False, xlabel="time steps (1 step = 4 months)", ylabel="household propensity")
plt.savefig("output/households_x.pdf")

dataframe = campo.dataframe.select(dataset.fs, property_names=['y'])
campo.to_csv(dataframe, "output/foodstores.csv")
propFrame = pandas.read_csv("output/foodstores_y.csv")
propFrame.plot(legend=False, xlabel="time steps (1 step = 4 months)", ylabel="food store propensity")
plt.savefig("output/foodstores_y.pdf")
