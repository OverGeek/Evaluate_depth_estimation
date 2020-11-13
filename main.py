import pandas as pd
import json
import numpy as np
import math

cam_frame_output = []
with open('./prediction_files/camera_frame_set_1_hypoth2.txt') as f:
    for line in f:
        cam_frame_output.append(json.loads(line))

cam_frame_output = cam_frame_output[:-3]  # exclude 000065_rgb.png

robot_poses = {}
with open('./prediction_files/robot_poses.txt') as f:
    for line in f:
        pose = json.loads(line)
        for key, val in pose.items():
            robot_poses[key] = np.array(val)

ground_truth_corner_coords = pd.read_csv('./ground_truth_data/corner_ground_truth.csv')

errors = []

cam_frame_corner_mat = []
for row in cam_frame_output:
    img_name = row['ID']
    mat = row['Mat']
    obj_name = row['Name']

    if not ground_truth_corner_coords['label'].isin([obj_name]).any().any():
        continue

    ground_truth_coords = ground_truth_corner_coords[ground_truth_corner_coords['label']==obj_name]

    robot_pose = robot_poses[img_name]
    corner_ordering = {0: 3, 1: 2, 2: 0, 3: 1}

    corner_id = 0
    for corner_coords in mat:
        x = corner_coords[1]
        y = corner_coords[2]
        z = corner_coords[3]

        corner_mat = np.array([[1., 0., 0., x],
                               [0., 1., 0., y],
                               [0., 0., 1., z],
                               [0., 0., 0., 1.]])

        updated_corner_coords = np.dot(robot_pose, corner_mat)

        updated_x = updated_corner_coords[0][-1]
        updated_y = updated_corner_coords[1][-1]
        updated_z = updated_corner_coords[2][-1]

        gt_corner_id = corner_ordering[corner_id]
        gt_corner_depth = float(ground_truth_coords[f'C{gt_corner_id+1}_Z'])

        errors.append(((updated_z-gt_corner_depth)**2))

print(math.sqrt(sum(errors)/len(errors)))
