# 
# example copypaste down, see the models directory for existence of model .h5 
# python PlotModel.py postim_weights.h5
# Programming marko.rantala@pvoodoo.com
# v1.0.0.1 20190322 


import sys
import keras
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import plot_model


if len(sys.argv) != 2:
	print("Usage: python PlotModel.py  model ")
	exit()

model_name = sys.argv[1]
model = load_model("Models/" + model_name)  # 

print(model.summary())

plot_model(model, to_file='Models/' + model_name + '.png', show_shapes=True)


print("See the file: models/"+model_name + '.png')

#feature_count = model.layers[0].input.shape.as_list()[1]
#if Debug:
#    print(model.layers[0].input.shape.as_list())


##############################
# Own ad: For NinjaTrader related stuff: check https://pvoodoo.com or blog: https://pvoodoo.blogspot.com/?view=flipcard
##############################