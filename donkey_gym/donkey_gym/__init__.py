from gym.envs.registration import register

register(
    id='donkey-generated-roads-v0',
    entry_point='donkey_gym.envs:GeneratedRoadsEnv',
    max_episode_steps=2000
)
