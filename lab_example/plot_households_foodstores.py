import lue.data_model as ldm
import campo
import pandas
import matplotlib.pyplot as plt

dataset = ldm.open_dataset('food_environment.lue')

dataframe = campo.dataframe.select(dataset.fs, property_names=['y'])
campo.to_gpkg(dataframe, 'foodstores.gpkg', 'EPSG:28992')

dataframe = campo.dataframe.select(dataset.hh, property_names=['y'])
campo.to_gpkg(dataframe, 'households.gpkg', 'EPSG:28992')

for i in range(1, 13):
    dataframe = campo.dataframe.select(dataset.hh, property_names=['x'])
    campo.to_gpkg(dataframe, 'households.gpkg', 'EPSG:28992', i)

dataframe = campo.dataframe.select(dataset.hh, property_names=['weight'])
campo.to_tiff(dataframe, 'EPSG:28992', '.')

dataframe = campo.dataframe.select(dataset.hh, property_names=['x'])
campo.to_csv(dataframe, "households.csv")
propFrame = pandas.read_csv("households_x.csv")
propFrame.plot(legend=False, xlabel="time steps (1 step = 4 months)", ylabel="household propensity")
plt.savefig("households_x.pdf")
