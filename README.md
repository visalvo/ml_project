# A self driving car with reinforcement learning

### Machine Learning Project
##### A.A. 2018/2019
###### Antonio Lategano
###### Salvatore Visaggi

## Usage

#### Donkey simulator
If you have a Mac the DonkeyCar simulator starts automatically 
when you run the python file `ddqn.py`. Instead, if you have a Windows PC 
you can find the simulator in the `donkey_windows` folder. 
You have to start the executable `DonkeySim.exe` before running the Python file.

#### Setup
You need Python 3.6 or higher and you have to install all the requirements.
```bash
pip install -r requirements.txt
``` 

Second, you have to install `donkey_gym ` python package, which extends the 
OpenAI gym class to allow RL developers to interact with Donkey environment 
using the familiar OpenAI gym like interface.

To install the package, navigate to `ml_project/donkey_gym` folder and type the following command
```
$ cd ml_project/donkey_gym
$ pip install -e .
```

#### Run

In the `config.py` file you can find lots of variables you can modify. The most important are:
* `EPISODES` number of epochs
* `TRAIN` if set to `True` you can train your model, otherwise you can use the pretrained model
* `LANE_DETECTION_TYPE` 1 = bw raw images | 2 = lane detection | 3 = points detection
* `MODEL_TYPE` 1 = atari | 2 = custom_nn

To start the execution
```
$ python ddqn.py
```
