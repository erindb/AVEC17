try:
	from ggplot import *
except ImportError:
	print("You do not have ggplot installed\n copy the following to install\npip install ggplot")
else: 

	import numpy as np
	import pandas as pd
	import datetime

	#add seed
	#hidden size
	#h2 size
	#non-linearity (GRU)


	def plot_loss(loss_array, labelType, hidden_size, h2_size, batch_size, num_epochs): 
		time = datetime.datetime.today().strftime('%Y-%m-%d--%H-%M-%S')
		#Plot Table
		loss_table = pd.DataFrame(loss_array)
		loss_table.columns = ['epoch', 'batch', 'loss']
		p = ggplot(aes('epoch', 'loss'), data=loss_table) + geom_point() + geom_line() + facet_wrap('batch')

		#Save plot in HTML file
		p.save('./plots/' + time + "-" + labelType + '_train.tiff', width=4, height=4, dpi=144)
		f1=open('./plots/loss_stats.html', 'a+')
		f1=open('./plots/loss_stats.html', 'a+')
		f1.write('<br><h3>' + time + ' - Training </h3>'
		'<br>Batch Size:	   ' + str(batch_size) + 
		'<br>Number of Epochs: ' + str(num_epochs) + "<br>--" +
		'<br>Hidden Size:	   ' + str(hidden_size) + 
		'<br>H2 Size:          ' + str(h2_size) + "<br>--" +
		'<br>Label :<b>	       ' + labelType +
		'</b><br><img src="' + time + "-" + labelType + '_train.tiff">')


	def plot_test(predicted_labels, true_labels, labelType, hidden_size, h2_size, batch_size, num_epochs):
		time = datetime.datetime.today().strftime('%Y-%m-%d--%H-%M-%S')
		test_data = pd.DataFrame({'true_labels':np.squeeze(true_labels), 'predicted_labels':np.squeeze(predicted_labels)})
		test_data['time'] = test_data.index + 1
		test_data = pd.melt(test_data, id_vars=['time'], value_vars=['true_labels', 'predicted_labels'],
		var_name='label_type', value_name='value')
		p = ggplot(aes(x='time', y='value', colour='label_type'), data=test_data) + geom_line(size=2)
		#Save plot in HTML file
		p.save('./plots/' + time + "-" + labelType + '_test.tiff', width=9, height=4, dpi=288)
		f1=open('./plots/loss_stats.html', 'a+')
		f1.write('<br><h3>' + time + ' - Testing </h3>'
		'<br>Batch Size:	   ' + str(batch_size) + 
		'<br>Number of Epochs: ' + str(num_epochs) + "<br>--" +
		'<br>Hidden Size:      ' + str(hidden_size) + 
		'<br>H2 Size:	       ' + str(h2_size) + "<br>--" +
		'<br>Label :<b>	       ' + labelType +
		'</b><br><img src="' + time + "-" + labelType + '_test.tiff">')

	def end_html(): 
		f1=open('./plots/loss_stats.html', 'a+')
		f1.write('<br><h1>---------------------------</h1>')


	