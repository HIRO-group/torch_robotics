from torch_robotics.environments import EnvYaml, EnvSpheres3D
import pdb
import torch
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes
import matplotlib.pyplot as plt
import os

# Specify the path to your YAML file
yaml_file = os.path.expanduser("~") + "/mpd-public/deps/torch_robotics/torch_robotics/environments/env_descriptions/env_anuj.yaml"

# Instantiate the EnvYaml environment
tensor_args = {"device": "cpu", "dtype": torch.float32}
env = EnvYaml(tensor_args=tensor_args, yaml_file=yaml_file)
fig, ax = create_fig_and_axes(env.dim)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
env.render(ax)
plt.show()
