import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pdb

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import (
    ObjectField,
    MultiSphereField,
    MultiBoxField,
    MultiCylinderField
)
from torch_robotics.robots import RobotPointMass, RobotPanda
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes
import random
import scipy.spatial.transform.rotation as R


class EnvYaml(EnvBase):
    def __init__(self, name="EnvDense2D", tensor_args=None, **kwargs):
        self.yaml_file = kwargs.get("yaml_file")
        self.data = None
        self.load_yaml()
        self.box_colors = []
        self.box_centers = []
        self.box_sizes = []
        self.cylinder_centers = []
        self.cylinder_radii = []
        self.cylinder_heights = []
        self.sphere_centers = []
        self.sphere_radii = []
        self.box_oris = []
        self.define_obstacles()
        # pdb.set_trace()
        # # Select the first 50 items for each type
        # self.box_centers = self.box_centers[500:600]
        # self.box_sizes = self.box_sizes[500:600]
        # self.cylinder_centers = self.cylinder_centers[500:600]
        # self.cylinder_radii = self.cylinder_radii[500:600]
        # self.cylinder_heights = self.cylinder_heights[500:600]
        # self.sphere_centers = self.sphere_centers[500:600]
        # self.sphere_radii = self.sphere_radii[500:600]
        # visualize the boxes (as spheres) using matplotlib
        # fig, ax = create_fig_and_axes(dim=3)
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # for i in range(len(self.box_centers)):
        #     self.plot_sphere(
        #         ax, self.box_centers[i], 
        #         [0, 0, 0], 
        #         self.box_sizes[i],
        #         'viridis'
        #     )
        # plt.show()
        # pdb.set_trace()
        fields = []
        if self.box_centers:
            boxes = MultiBoxField(
                centers=torch.tensor(self.box_centers),
                sizes=torch.tensor(self.box_sizes),
                oris=torch.tensor(self.box_oris),
                tensor_args=tensor_args,
            )
            fields.append(boxes)
        if self.cylinder_centers:
            cylinders = MultiCylinderField(
                centers=torch.tensor(self.cylinder_centers),
                radii=torch.tensor(self.cylinder_radii),
                heights=torch.tensor(self.cylinder_heights),
                tensor_args=tensor_args,
            )
            fields.append(cylinders)
        # if self.sphere_centers:
        #     spheres = MultiSphereField(
        #         centers=torch.tensor(self.sphere_centers),
        #         radii=torch.tensor(self.sphere_radii),
        #         tensor_args=tensor_args,
        #     )
        #     fields.append(spheres)
        # add one sphere at 1, 1, 1
        # sphere = MultiSphereField(
        #     centers=torch.tensor([[1, 1, 1]]),
        #     radii=torch.tensor([0.1]),
        #     tensor_args=tensor_args,
        # )
        # fields.append(sphere)
        obj_field = ObjectField(fields, "obstacles")
        obj_list = [obj_field]

        super().__init__(
            name=name,
            limits=torch.tensor(
                [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], **tensor_args
            ),  # environments limits
            obj_fixed_list=obj_list,
            tensor_args=tensor_args,
            precompute_sdf_obj_fixed = False,
            **kwargs
        )

    def plot_sphere(self, ax, center, pos, radius, cmap):
        u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 20j]
        x = radius * (np.cos(u) * np.sin(v))
        y = radius * (np.sin(u) * np.sin(v))
        z = radius * np.cos(v)
        ax.plot_surface(
            x + center[0] + pos[0],
            y + center[1] + pos[1],
            z + center[2] + pos[2],
            cmap=cmap,
            alpha=1,
        )

    def cuboid_data(self, o, size=(1, 1, 1)):
        X = [
            [[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
            [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
            [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
            [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
            [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
            [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]],
        ]
        X = np.array(X).astype(float)
        for i in range(3):
            X[:, :, i] *= size[i]
        X += np.array(o)
        return X

    def plot_rectangle(self, ax, center, pos, dims, cmap):
        g = []
        for (
            p,
            s,
        ) in zip([center], [dims]):
            g.append(self.cuboid_data(p, size=(s[0], s[1], s[2])))
        collection = Poly3DCollection(np.concatenate(g), facecolors="blue")
        ax.add_collection3d(collection)

    def load_yaml(self):
        with open(self.yaml_file, "r") as stream:
            try:
                self.data = yaml.load(
                    stream, Loader=yaml.Loader
                )  # DO NOT USE UNTRUSTED YAMLS
            except yaml.YAMLError as exc:
                print(exc)

    def all_same(self, items):
        return all(x == items[0] for x in items)

    def define_rectangular_obstacles_as_cubes(self):
        return NotImplementedError

    def define_obstacles(self):
        collision_object_list = self.data["world"]["collision_objects"]
        box_objects = [
            obj
            for obj in collision_object_list
            if obj["primitives"][0]["type"] == "box"
        ]
        cylinder_objects = [
            obj
            for obj in collision_object_list
            if obj["primitives"][0]["type"] == "cylinder"
        ]
        sphere_objects = [
            obj
            for obj in collision_object_list
            if obj["primitives"][0]["type"] == "sphere"
        ]

        # fig, ax = create_fig_and_axes(dim=3)
        if len(box_objects) > 0:
            for box in box_objects:
                if 'Can' in box['id']:
                    print('doing can')
                # if box["id"] == "table_top":
                #     continue
                # if 'table_leg' in box["id"]:
                #     continue
                # get intended dimensions of box (blh)
                box_size = box["primitives"][0]["dimensions"]
                # get position
                box_center = box["primitive_poses"][0]["position"]

                box_orientation = box["primitive_poses"][0]["orientation"]
                # if box size is not square, break up into multiple boxes

                # visualize the boxes using matplotlib

                # self.plot_rectangle(ax, box_center,
                #                     [0, 0, 0],
                #                     box_size,
                #                     'Blues')

                frac_table_top = 0.15
                frac_shelf = 0.15
                frac_else = 0.15
                if False: #not self.all_same(box_size):

                    # get length of shortest side
                    min_side_arg = np.argmin(box_size)
                    min_side = box_size[min_side_arg]
                    # Calculate the number of cubes needed in each dimension
                    num_cubes_x = int(np.ceil(box_size[0] / min_side))
                    num_cubes_y = int(np.ceil(box_size[1] / min_side))
                    num_cubes_z = int(np.ceil(box_size[2] / min_side))

                    # Calculate the step size between cubes in each dimension
                    # step_x = box_size[0] / num_cubes_x
                    # step_y = box_size[1] / num_cubes_y
                    # step_z = box_size[2] / num_cubes_z

                    # Calculate the number of cubes to keep in each dimension
                    if box['id'] == "table_top":
                        num_cubes_x_to_keep = int(num_cubes_x * frac_table_top)
                        num_cubes_y_to_keep = int(num_cubes_y * frac_table_top)
                        num_cubes_z_to_keep = int(num_cubes_z * frac_table_top)
                    elif 'shelf' in box['id']:
                        num_cubes_x_to_keep = int(num_cubes_x * frac_shelf)
                        num_cubes_y_to_keep = int(num_cubes_y * frac_shelf)
                        num_cubes_z_to_keep = int(num_cubes_z * frac_shelf)   
                    else:
                        num_cubes_x_to_keep = int(num_cubes_x * frac_else)
                        num_cubes_y_to_keep = int(num_cubes_y * frac_else)
                        num_cubes_z_to_keep = int(num_cubes_z * frac_else)
                    if num_cubes_x_to_keep == 0:
                        num_cubes_x_to_keep = 1
                    if num_cubes_y_to_keep == 0:
                        num_cubes_y_to_keep = 1
                    if num_cubes_z_to_keep == 0:
                        num_cubes_z_to_keep = 1

                    # Calculate the step size between cubes to keep in each dimension
                    step_x_to_keep = box_size[0] / num_cubes_x_to_keep
                    step_y_to_keep = box_size[1] / num_cubes_y_to_keep
                    step_z_to_keep = box_size[2] / num_cubes_z_to_keep

                    # Generate the cube centers to keep
                    for i in range(num_cubes_x_to_keep):
                        for j in range(num_cubes_y_to_keep):
                            for k in range(num_cubes_z_to_keep):
                                if box['id'] == 'table_top' or 'shelf' in box['id']:
                                    # we need to do an affine transform of all the box 
                                    # centers about the center of the table top
                                    # get the center of the table top
                                    table_center = box_center
                                    # get the rotation quaternion
                                    orientation = box["primitive_poses"][0]["orientation"]
                                    rot = R.Rotation.from_quat(orientation)
                                    # add 180 deg in z axis
                                    # rot = rot * R.Rotation.from_euler('y', -np.pi)
                                    # get the rotation matrix
                                    rot_matrix = rot.as_matrix()
                                    # find homogenous transformation matrix
                                    homogenous_matrix = np.zeros((4, 4))
                                    homogenous_matrix[:3, :3] = rot_matrix
                                    # homogenous_matrix[:3, 3] = table_center
                                    homogenous_matrix[3, 3] = 1
                                    x = box_center[0] + i * step_x_to_keep
                                    y = box_center[1] - j * step_y_to_keep
                                    z = box_center[2] + k * step_z_to_keep
                                    if box['id'] == 'table_top':
                                        x += 0.3
                                    elif 'shelf' in box['id']:
                                        x -= 0.22
                                        y += 0.22
                                    # apply the transformation
                                    new_center = np.dot(homogenous_matrix, np.array([x, y, z, 1]))
                                    if box['id'] == 'table_top':
                                        new_center[3] += 0.02
                                    self.box_centers.append(new_center[:3].tolist())
                                    self.box_sizes.append(min_side)
                                else:
                                    self.box_centers.append(
                                        [
                                            box_center[0] + i * step_x_to_keep,
                                            box_center[1] + j * step_y_to_keep,
                                            box_center[2] + k * step_z_to_keep,
                                        ]
                                    )
                                    self.box_sizes.append(min_side)
                                # if 'table_top' in box['id']:
                                #     self.box_colors.append('Blues')
                                # elif 'shelf' in box['id']:
                                #     self.box_colors.append('Blues')
                                # elif 'leg' in box['id']:
                                #     self.box_colors.append('Reds')
                                # else:
                                #     print('setting to green')
                                #     self.box_colors.append('Greens')
                    # # Generate the cube centers
                    # for i in range(num_cubes_x):
                    #     for j in range(num_cubes_y):
                    #         for k in range(num_cubes_z):
                    #             self.box_centers.append(
                    #                 [
                    #                     box_center[0] + i * step_x,
                    #                     box_center[1] + j * step_y,
                    #                     box_center[2] + k * step_z,
                    #                 ]
                    #             )
                    #             self.box_sizes.append(min_side)
                    
                        
                else:
                    self.box_centers.append(
                        [box_center[0], box_center[1], box_center[2]]
                    )
                    self.box_sizes.append([box_size[0], box_size[1], box_size[2]])
                    self.box_oris.append([box_orientation[3], box_orientation[0], box_orientation[1], box_orientation[2]])
        # Select a random uniform 10% of the boxes
        # num_boxes = len(self.box_centers)
        # num_boxes_to_keep = int(num_boxes * 0.1)
        # random_indices = random.sample(range(num_boxes), num_boxes_to_keep)
        # self.box_centers = [self.box_centers[i] for i in random_indices]
        # self.box_sizes = [self.box_sizes[i] for i in random_indices]
        # plt.show()

        if len(cylinder_objects) > 0:
            for cylinder in cylinder_objects:
                # get intended dimensions of cylinder (r,h)
                cylinder_radius = cylinder["primitives"][0]["dimensions"][1]
                cylinder_height = cylinder["primitives"][0]["dimensions"][0]
                # get position
                cylinder_center = cylinder["primitive_poses"][0]["position"]
                self.cylinder_centers.append(
                    [cylinder_center[0], cylinder_center[1], cylinder_center[2]]
                )
                self.cylinder_radii.append(cylinder_radius)
                self.cylinder_heights.append(cylinder_height)

        if len(sphere_objects) > 0:
            for sphere in sphere_objects:
                # get intended dimensions of sphere (r)
                sphere_radius = sphere["primitives"][0]["dimensions"][0]
                # get position
                sphere_center = sphere["primitive_poses"][0]["position"]
                self.sphere_centers.append(
                    [sphere_center[0], sphere_center[1], sphere_center[2]]
                )
                self.sphere_radii.append(sphere_radius)