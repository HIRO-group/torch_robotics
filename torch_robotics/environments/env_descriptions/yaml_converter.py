import yaml

def convert_yaml_to_yml(yaml_file):
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)

    yml_data = {'cuboid': {}, 'cylinder': {}}
    for obj in data['world']['collision_objects']:
        if obj['primitives'][0]['type'] == 'cylinder':
            yml_data['cylinder'][obj['id']] = {
                'radius': obj['primitives'][0]['dimensions'][1],
                'height': obj['primitives'][0]['dimensions'][1],
                'pose': obj['primitive_poses'][0]['position'] + obj['primitive_poses'][0]['orientation']
            }
        elif obj['primitives'][0]['type'] == 'box':
            yml_data['cuboid'][obj['id']] = {
                'dims': obj['primitives'][0]['dimensions'],
                'pose': obj['primitive_poses'][0]['position'] + obj['primitive_poses'][0]['orientation']
            }
        # Add other types if needed

    with open('output.yml', 'w') as f:
        yaml.safe_dump(yml_data, f, default_flow_style=None)

convert_yaml_to_yml('/home/gilberto/mpd-public/deps/torch_robotics/torch_robotics/environments/env_descriptions/env_shelf/env_shelf.yaml')
