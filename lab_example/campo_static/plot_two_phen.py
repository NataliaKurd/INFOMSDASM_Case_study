import lue.data_model as ldm
import campo

dataset = ldm.open_dataset('food_environment.lue')

# select point property of households
dataframe = campo.dataframe.select(dataset.households, property_names=['x'])
# convert to gpkg file
campo.to_gpkg(dataframe, 'households.gpkg', 'EPSG:28992')

# select point property of foodstores
dataframe = campo.dataframe.select(dataset.foodstores, property_names=['y'])
# convert to gpkg file
campo.to_gpkg(dataframe, 'foodstores.gpkg', 'EPSG:28992')

# select field properties of foodstores
dataframe = campo.dataframe.select(dataset.foodstores, property_names=['weight', 'area'])
# convert selction to tiff
campo.to_tiff(dataframe, 'EPSG:28992', '.')
