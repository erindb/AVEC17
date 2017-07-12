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


	def plot_loss(loss_array, labelType, rnn_parameter_dict): 
		#Plot Table
		time = datetime.datetime.today().strftime('%Y-%m-%d--%H-%M-%S')
		loss_table = pd.DataFrame(loss_array)
		loss_table.columns = ['epoch', 'batch', 'loss']
		p = ggplot(aes('epoch', 'loss'), data=loss_table) + geom_point() + geom_line() + facet_wrap('batch')
		p.save('./plots/' + time + "-" + labelType + '_train.tiff', width=4, height=4, dpi=144)
		
		#Open and write files
		f1=open('./plots/loss_stats.html', 'a+')
		f1.write('<br><h3>' + time + ' - Training </h3>')
		for key, value in rnn_parameter_dict.items():
				f1.write('<br>' + key + ": " + str(value))
		f1.write('</b><br><img src="' + time + "-" + labelType + '_train.tiff">')


	def plot_test(predicted_labels, true_labels, labelType, rnn_parameter_dict):
		#Plot Table
		time = datetime.datetime.today().strftime('%Y-%m-%d--%H-%M-%S')
		test_data = pd.DataFrame({'true_labels':np.squeeze(true_labels), 'predicted_labels':np.squeeze(predicted_labels)})
		test_data['time'] = test_data.index + 1
		test_data = pd.melt(test_data, id_vars=['time'], value_vars=['true_labels', 'predicted_labels'],
		var_name='label_type', value_name='value')
		p = ggplot(aes(x='time', y='value', colour='label_type'), data=test_data) + geom_line(size=2)
		p.save('./plots/' + time + "-" + labelType + '_test.tiff', width=9, height=4, dpi=288)

		#Open and write files
		f1=open('./plots/loss_stats.html', 'a+')
		f1.write('<br><h3>' + time + ' - Training </h3>')
		for key, value in rnn_parameter_dict.items():
				f1.write('<br>' + key + ": " + str(value))
		f1.write('</b><br><img src="' + time + "-" + labelType + '_test.tiff">')

	def end_html(): 
		f1=open('./plots/loss_stats.html', 'a+')
		f1.write('<br><h1>---------------------------</h1>')


	