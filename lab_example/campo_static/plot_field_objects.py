import lue.data_model as ldm
import campo


dataset = ldm.open_dataset('food_environment.lue')

dataframe = campo.dataframe.select(dataset.foodstores, property_names=['c'])
print(dataframe)

campo.to_pdf(dataframe, 'EPSG:28992', '.')
