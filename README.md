# Dissolved-Oxygen-MLPrediction
This repository is for the code and data used in the paper "Dissolved oxygen and hypolimnetic hypoxia predictions via machine learning models in various lakes".

It contains LSTM shuffle training years scripts for 5 lakes: 

	2009-2019 Ekoln shuffle training years.py
	2010-2015 ME shuffle training years.py
	2010-2018 Fureso shuffle training years.py
	2017-2020 Erken shuffle training years.py
	2017-2020 Mueggelsee shuffle training years.py
	
and the main model code in jupyter notebook script 

	Models for O2 prediction.ipynb
	
and python script 

	Models for two-step nutrient prediction.py 
	
which can be used with specific inputs for each lake. The example provided is for Lake Erken.

The data used are formatted and stored in the 'Training data' folder. Each lake has two corresponding files, 'lakename_Daily_Observation_df_nowinter.csv' and 'lakename_Observation_df_nowinter.csv'.

The model results are stored in 'Nutrient prediction' folder under the folders with lake names.
