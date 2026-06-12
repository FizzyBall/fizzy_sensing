Alle recorded data. 
Data acqui spreadsheet
CNN: model 36, 37, 38, 39. Data 16, 17, 18, 21


Short summary how the repo is ingedeeld:
Directories: ./Data -> Training data, Testing data -> Each training data/ testing data file with a description. Also paste the data log sheet in here.
./Random Forest/ 

To make use of the written code, a set of modules are needed. These are:
Numpy
PyQT 
.....


A short summary of the project and the code written will follow;
The main goal was to add senses to Fizzy so it can react to actions that are done on the ball. For this a dashboard was created, which can now show live data readouts, orientation of the IMU, can record the data for training and testing purposes.
It can be used to Analyse (label) data, Featurize the data by extracting given features (like mean, max and so on) for the Random Forest training, it can train the Random Forest with a lot of changable options. 
Furthermore, a trained model can be used for live classification. This can be done for both random forest models and CNN models. Finally, a classification analysis tab is present. This is used to evaluate the trained models on a new, unseen, testset.
Start the dashboard by running 'fizzy_imu_dashboard.py', this can be found in the IMU dashboard subdirectory, if you want only the dashboard with the above described functionalities. Or run 'fizzy_main_combined.py' if you also want to be able to control fizzy by random input or use a controller to send inputs to either fizzy or set markers while recording.
If you want to change anything, mostly files in the utilities folders can be used to adjust behavior or change settings. I.e. changing the way the imu data is extracted can be done in imu_data_extractor.py. Adding classes can be done in fizzy_config.py. 


In Python_Test_Code subdirectory a lot of testing files can be found, these mostly speek for themselves and are somewhat modified files from the Fizzy software repository. 

In the DataAndModels the best performing models are included for both CNN and RandomForest. Also there data can be found to train your own models. See the RF folder or the CNN folder for their respective training data. A more comprehensive guide to training the RF and CNN models can be found there. 

Disclaimer: most of the code for the interface was written by Claude; so it might be a bit unreadable. 