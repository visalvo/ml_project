ENV_NAME = "donkey-generated-roads-v0"
EPISODES = 100
TRAIN = False
HEADLESS = False
STATS_FILE = "./stats/test-4.csv"
MODEL_PATH = "./save_model/"
MODEL_NAME = "model_test-4.h5"
MODEL_TYPE = 1  # 1 = atari | 2 = custom_nn

ACTION_SPACE = 15
LANE_DETECTION_TYPE = 1  # 1 = bw raw images | 2 = lane detection | 3 = points detection

IMG_ROWS, IMG_COLS = 80, 80
IMG_STACK = 4

THROTTLE_MIN = 0.3
THROTTLE_MAX = 0.8
# Cross track error max - donkey_sim.py
CTE_MAX_ERR = 2.0
# Cross track error limit for penalty - donkey_sim.py
CTE_LIMIT = 1.0

# These are hyper parameters for the DDQN
DISCOUNT = 0.99
LEARNING_RATE = 1e-4
EPSILON_MIN = 0.02
BATCH_SIZE = 64
TRAIN_START = 100
EXPLORE = 10000
MAX_REPLAY = 10000
