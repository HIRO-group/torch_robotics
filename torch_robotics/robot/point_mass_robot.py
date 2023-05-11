from torch_robotics.robot.robot_base import RobotBase
from torch_robotics.torch_utils.torch_utils import to_numpy


class PointMassRobot(RobotBase):

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def fk_map_impl(self, q):
        # There is no forward kinematics. Assume it's the identity.
        # Add task space dimension
        return q.unsqueeze(-2)

    def render(self, ax, q=None, color='blue', **kwargs):
        ax.scatter(q[0], q[1], color=color)

    def render_trajectory(self, ax, traj_q=None, start_state=None, goal_state=None, **kwargs):
        if traj_q is not None:
            traj = to_numpy(traj_q)
            ax.plot(traj[:, 0], traj[:, 1], 'b-', markersize=3)
        if start_state is not None:
            ax.plot(to_numpy(start_state[0]), to_numpy(start_state[1]), 'go', markersize=7)
        if goal_state is not None:
            ax.plot(to_numpy(goal_state[0]), to_numpy(goal_state[1]), 'ro', markersize=7)
