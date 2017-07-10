try:
    from ggplot import *
except ImportError:
    print("You do not have ggplot installed\n copy the following to install\npip install ggplot")
else: 

	import numpy as np
	import pandas as pd


	def plot_loss(loss_array, batch_size, num_epochs, labelType): 
	    loss_table = pd.DataFrame(loss_array)
	    loss_table.columns = ['loss']
	    loss_table['index'] = loss_table.index
	    p = ggplot(aes(x='index', y='loss'), data=loss_table) + geom_point() + geom_line()
	    p.save(labelType +'.tiff', width=12, height=8, dpi=144)


