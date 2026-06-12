# How to Train the CNN

## 1. Data Acquisition

First, data needs to be recorded using Fizzy. This is done by connecting Fizzy to your device and running the Python file `fizzy_main_combined.py`. This should automatically put you in the **Record** tab.

Next, in the recording control panel, either press the **Browse** button to select an existing file to save the recorded CSVs to, or let the application create a new file by giving the file a name in the text box next to **Browse**.

Above the text box of **Browse**, the name of the CSV file that will be recorded should also be defined. This name should be defined in the following way:

- The CSV name should contain the name of the interaction that is going to be recorded, and it shouldn't contain another name of an interaction that will also be classified. For example, if `drop` is being classified, another interaction can't be named `droplift`, since it also contains the name `drop`. The name of this interaction should be changed.
- There is also an **auto-increment** button that makes it so every time a new recording is started the name gets incremented. It's needed to increment, otherwise the files will be overwritten.

In **Recording Controls**, select all the input channels and select an amount of seconds to record.

Next, press **Start Recording** and start recording the CSV files.

The way the CSV files should be recorded to suit the model well is the following:

- Firstly, see how long the interaction takes and choose that amount of seconds to record with about 0.2-0.5 seconds of buffer to account for reaction time.
- Then record each interaction individually. Using the autoincrement button makes it easier to do. 

## 2. Training the Model
All the files and folders mentioned will be located in the folder Model Training, unless stated otherwise.

Now, in the folder `all_recordings`, create a folder called `recordings_v{version}`, where `{version}` should be the version number of that recording. Note that there are already existing data sets, so a non-existing number should be chosen. Move all the CSVs that are going to be trained into this `recordings_v{version}` folder.

In `Trainsettings.py`, the class names and the amount of classes should be defined. The amount of inputs and the input names should also be defined. Lastly, the version number of the folder with the CSVs in it should also be defined.

There are already pre-existing names for classes and input channels defined. These are defined for each different version of data, which can also be found in the data acquisition log.

If all the input channels are selected, these are the names that should be used when defining the input names:

```
roll, pitch, yaw, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, motor_input
```

Next, save all the changes made and run `Trainer_CNN.py`. This will create a mean and std saved in the `Mean, std` folder, which is located one folder above Model Training in the CNN folder, with the names `imu_std_v{version}` and `imu_mean_v{version}`, with `{version}` being the data version. The model weights get saved in the same folder as the `Trainer_CNN.py` file is in, as `Trained_xx.pth`. This `xx` should be renamed to a number, and then moved into the `Models` folder, which is located one folder above Model Training in the CNN folder. Note that this number cannot be an already-existing number.

## 3. Running the Model
All the files and folders mentioned will be in the Model Running folder, unless stated otherwise.

There are three dashboards that can be run, the first one is `LiveClassifier.py`. This one runs one model, where the settings can be changed in `settingsSingleModel.py`.

The following settings can be changed:

- The model number and the data version number. The data version number also needs to be defined so the right mean and std are chosen.

- The input and output amounts of the model should also be defined along with the class names. Note that the class names should be defined in the same order as they are in the `datac.py` file.

- A confidence threshold per class can be defined. If a class doesn't exist this class can be added or it uses the general confidence threshold if it isn't added.

- A consecutive confident count can be changed per class, which means that it needs an amount of confident classifications in a row to actually classify that class.

- After each class a cooldown can be added where the model doesn't classify anything anymore. This can also be set to 0 so there is no cooldown.

All of these settings can also be changed while in the UI, and they are able to be saved 

The second dashboard is the `LiveClassifyDashboard.py` which is to run two models that switch between modes. This works as following, the ball is initially on the ground model, which means the motor is actuating randomly. When the ball is lifted, it switches to in hand mode, turning the model off. Dropping the ball again puts the model in ground mode.



This classifier has its own settings file called `settingsHandGroundModel.py`. Here the same parameters can be changed just as with the `LiveClassifier.py`. The only extra parameters it has are these:

- Since it has 2 modes, there are two models that need their weights, version, input amounts, output amounts etc. to be changed

- Secondly, this classifier switches between models, so there are two extra parameters called LIFT_COUNT_TO_SWITCH and DROP_COUNT_TO_SWITCH, which are the amount of times the models needs to confidently classify a lift or a drop to switch between models.


The last dashboard lives inside of another folder called IMU dashboard. The dashboard is called `fizzy_main_combined.py`. This dashboard will classify the model live without any post processing, or will classify pre recorded data. When this dashboard is opened, navigate to either the Live Classification tab or the Classification Analysis tab. This is done by pressing the red Record button which opens down a dropdown menu.

- To classify live, press the Live Classification option and at the top left set the window size to 64. Make sure the setting at the left is on samples and that there is 50% overlap. Then select all the IMU data thats wanted to see.


Select the option CNN in the box Live Classification and choose the appropriate model, weights and version. Then press load model. If connected to Fizzy, it's going to start classifying live.

- To classify pre recorded data, select the Classification Analysis tab and at the top left set the window size to 64. Make sure the setting at the left is on samples and that there is 50% overlap. Then select all the IMU data thats wanted to see.

To select the model scroll all the way down in the box inside of clasification analysis control and select the weights of the model. Next in CNN options, select whether you want to analyse the singular model or one of the two Hand or Ground models. Next to that also choose the appropriate version number.

Then below that select browse recording, choose a CSV file and then press load model. The model will then be loaded showing what it classified per window.


Note that for both the live classification and classification of pre recorded data the models themselves, so the input, output and the classnames, need to be changed in the settings files living inside of Model Running.

Normal corresponds to the file `settingsSingleModel.py`, Hand and ground both are in the `settingsHandGroundModel.py`.
Only the input amount, output amount and the class names need to be changed in those files to be able to use `fizzy_main_combined.py` . 