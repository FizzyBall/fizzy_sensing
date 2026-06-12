"""
Here define the data version, what classes need to be classified and the input channels used.
"""

# Define here what data folder should be used.
version = '21'


#Here define the class names to be classified.

#Labels for v7
# LABELS = {'idle' : 0, 'tap': 1 , 'lift': 2, 'kick': 3}

#Labels for v8, v10, v14, v15, 
# LABELS = {'idle' : 0, 'tap': 1, 'shake': 2 , 'drop' : 3}

# Labels for v11, 18
# LABELS = {'idle' : 0, 'lift': 1, 'kick': 2}

#Labels for v12, v13, 16, 20, 21
LABELS = {'idle' : 0, 'shake': 1, 'tap': 2 , 'drop' : 3, 'lift': 4, 'kick': 5}



#Here define the amount of classes to be classified
CLASSES_AMOUNT = 6


#Here define the channels names to be used.

#

#
FEATURE_COLUMNS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x','gyro_y' ,'gyro_z']


# FEATURE_COLUMNS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x','gyro_y' ,'gyro_z', 'motor_input']


#Here define the amount of input channels.
INPUT_AMOUNT = 6

