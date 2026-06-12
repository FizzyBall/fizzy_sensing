The trained models and data can be found here.

How to train a Random Forest using the IMU dashboard:
1. start by recording data using Fizzy of your to be classified movements. Make sure to keep some spacing between your interactions, so it is easier to label your data afterwards. 
    The toggle buttons can be used to include or not include the data in the final .csv file. 
    If you are using an XBox controller during recording (use fizzy_main_combined.py) you can use the D-Pad to place markers in your data, these can be found in the analyse tab to be used while labeling. The Xbox controller can also be used to start a recording or stop a recording. 
    If you are running fizzy_main_combined.py you can also click on the action buttons to make Fizzy do movements while recording. 
    The default setting is to transform the data to 'Relative to World'. It uses the Earths magnetic field to find the Z-axis and from there it transforms the data into that reference frame. This can be changed to 'Relative to Fizzy' if that is what you want. 
    Templates can be created and used as well.
2. Analyse tab.
    The analyse tab can be used to label your recorded data, to be used in classification. Start by selecting your CSV file and making a directory to save your labeled data. The data is saved inside that directory as subfolders with the name of the class, and in those folders the CSV files named ClassName1.csv, ClassName2.csv...... 
    Each CSV corresponds to one window, the window size can be set in the control panel part, in either seconds or samples. Seconds is what is used for RF, since Fizzy in downlink mode does not have a constant sampling rate. The overlap can also be set here. It defaults to 50%. 
    Make sure to label your labeled data per recording and give them a name like ClassName#ClassName-GroupName. This can later be used in training so the model won't get validated on data it is trained on. 
    Metadata .json files are also generated when the save labeled data button is pressed. These are used for final testing of your model, if you decide to record a testing set. 
    These can also be used to stop halfway through labeling a recording, by reloading the json when you want to continue
3. Featurize tab
    The data for RF has to be featurized. To featurize, add your labeled data folders, set a save location and file name and pick your features you want to be extracted. If there is a big file the application might become unresponsive, solution is to get a cup of coffee and it will be done when you return!
    To add your own features, you can use data_windower.py. There are examples of how to add custom features or add features that are applied to every input data stream. 
    The accelaration data and gyro data in the X, Y direction are combined to acc_XY and gyro_xy. This is because this makes models more robust.
4. Training tab. 
    To train your RF model, select your featurized CSV files, select an output folder and file name.
    Then you can change the trianing settings. The test size variable is used to evaluate the model at the end, returning the final score.
    Random state = 42 for reproducibility. Amount of iterations is for how many iterations of a RandomizedSearch for hyperparameters will be done. 
    The Random Forest can be changed to Balanced Random Forest for an unbalanced dataset. 
    You can choose between GridSearchCV and RandomizedSearchCV, RandomizedSearchCV often gives a very clear indication if it will work and is way faster. 
    The KFold methods can be changed to your liking, and number of splits can be adjusted as well as the scoring metric for the CV algorithm
    The search space can be set as either, for RandomizedSearchCV as a space to search using randomintegers and for GridSearchCV a list of integers to iterate through.
    For any further explanation about the functions, refer to the SciKit-Learn (or IMBLearn) libraries and their user manuals. 
    Dont forget to hit train model!
    The Dashboard will return a confusion matrix, macro F1-score, accuracy, macro Recall and macro Precision. 
5. Live classification.
    Speaks for itself, connect Fizzy to your PC, select a model (DO NOT FORGET to set your window size and overlap to the correct values) and click load.
    The Dashboard should return the probabilities and will choose a class. 
6. Classification Analysis. 
    This can be used to analyse your model on a different test set. Load a model or multiple models. Select a recording, that you have labeled. Make sure it is the original unprocessed version. Also select the corresponding .json file and hit 'Load'. 
    The Dashboard can then be used to navigate through each classification and compare your labeling to the classifiers labels. You can also adjust your labels here if you find out that you have mislabeled something. 
    The final confusion matrix, macro F1 score, macro Recall, macro Precision, Accuracy and inference time is printed in the window in the bottom right.

Known issues:
Sometimes fizzy will disconnect and not be able to reconnect. Turn your wifi on and off and connection should resume
The scaling of the windows of the software sometimes changes and drops of your screen. To avoid this use either a bigger monitor or re size your window (go out of fullscreen and back into it).
Do not click quit app if you do not want to quit. The app will come with a warning if you want to save or not, it shuts down no matter what you click. 

