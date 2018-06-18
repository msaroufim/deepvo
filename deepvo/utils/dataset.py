import math
import os

import numpy as np
import cv2

from tensorpack.dataflow.base import RNGDataFlow

from deepvo.utils.config import ConfigManager as CM


class DatasetKitti:
    """ Class for manipulating Dataset"""
    def __init__(self, is_training=True):
        self._current_initial_frame = 0
        self._current_trajectory_index = 0
        self._prev_trajectory_index = 0
        self._current_train_epoch = 0
        self._current_test_epoch = 0
        self._img_height, self._img_width = 384, 1280
        self._data_dir = CM.common()['dataset']['kitti']['path']
        self._is_training = is_training
        if is_training:
            self._current_trajectories = CM.model()['train']['kitti']['train_set']
        else:
            self._current_trajectories = CM.model()['train']['kitti']['test_set']

    def get_image(self, trajectory, frame_index):
        # TODO : path?
        img_path = os.path.join(self._data_dir, 'sequences/%02d/image_2/%06d.png' % (trajectory, frame_index))

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            return None, None   # TODO
        img = cv2.resize(img, (self._img_width, self._img_height), fx=0, fy=0)
        return img, img_path

    def get_poses(self, trajectory):
        with open(os.path.join(self._data_dir, 'poses/%02d.txt' % trajectory)) as f:
            poses = np.array([[float(x) for x in line.split()] for line in f])
        return poses

    def _set_next_trajectory(self):
        print 'in _set_next_trajectory, current_trj_index is %d' % self._current_trajectory_index
        if self._current_trajectory_index < len(self._current_trajectories) - 1:
            self._prev_trajectory_index = self._current_trajectory_index
            self._current_trajectory_index += 1
            self._current_initial_frame = 0
        else:
            print 'New Epoch Started'
            if self._is_training:
                self._current_train_epoch += 1
            else:
                self._current_test_epoch += 1
            self._prev_trajectory_index = self._current_trajectory_index
            self._current_trajectory_index = 0
            self._current_initial_frame = 0

    def get_next_batch(self):
        """ Function that returns the batch for dataset
        """
        img_batch = []
        motion_batch = []
        pose_batch = []
        img_path_batch = []

        poses = self.get_poses(self._current_trajectories[self._current_trajectory_index])

        for j in range(CM.model()['train']['batch_size']):
            img_stacked_series = []
            motion_series = []
            pose_series = []
            img_path_series = []
            print('Current Trajectory is : %d' % self._current_trajectories[self._current_trajectory_index])

            read_img, read_path = self.get_image(self._current_trajectories[self._current_trajectory_index],
                                                 self._current_initial_frame + CM.model()['train']['time_step'])

            if read_img is None:
                self._set_next_trajectory()   # TODO : possible bug
            for i in range(self._current_initial_frame, self._current_initial_frame + CM.model()['train']['time_step']):
                img1, img1_path = self.get_image(self._current_trajectories[self._current_trajectory_index], i)
                img2, img2_path = self.get_image(self._current_trajectories[self._current_trajectory_index], i + 1)
                img_aug = np.concatenate([img1, img2], -1)
                img_stacked_series.append(img_aug)
                img_path_series.append(img1_path)

                cf = self._current_initial_frame
                pose = get_ground_7d_poses(poses[cf, :], poses[i + 1, :])     # TODO : change the starting point.
                pose_series.append(pose)

                motion = get_ground_6d_poses(poses[i, :], poses[i + 1, :])
                motion_series.append(motion)
            img_batch.append(img_stacked_series)
            img_path_batch.append(img_path_series)
            motion_batch.append(motion_series)
            pose_batch.append(pose_series)
            self._current_initial_frame += CM.model()['train']['time_step']
        motion_batch = np.array(motion_batch)
        pose_batch = np.array(pose_batch)
        img_batch = np.array(img_batch)
        # print label_batch.shape
        # print img_batch.shape
        # print("Label_batch")
        # print label_batch[0,:,:]
        return img_batch, motion_batch, pose_batch, img_path_batch


# def get_ground_6d_poses(p):
#     """ For 6dof pose representaion """
#     pos = np.array([p[3], p[7], p[11]])
#     R = np.array([[p[0], p[1], p[2]], [p[4], p[5], p[6]], [p[8], p[9], p[10]]])
#     angles = rotation_matrix_to_euler_angles(R)
#     return np.concatenate((angles, pos))    # rpyxyz

def get_ground_6d_poses(p, p2):
    """ For 6dof pose representaion """
    pos1 = np.array([p[3], p[7], p[11]])
    pos2 = np.array([p2[3], p2[7], p2[11]])
    pos = pos2 - pos1

    R1 = np.array([[p[0], p[1], p[2]], [p[4], p[5], p[6]], [p[8], p[9], p[10]]])
    R2 = np.array([[p2[0], p2[1], p2[2]], [p2[4], p2[5], p2[6]], [p2[8], p2[9], p2[10]]])
    angles = rotation_matrix_to_euler_angles(np.matmul(np.linalg.inv(R1), R2))
    return np.concatenate((angles, pos))    # rpyxyz


def get_ground_7d_poses(p, p2):
    """ For 6dof pose representaion """
    pos1 = np.array([p[3], p[7], p[11]])
    pos2 = np.array([p2[3], p2[7], p2[11]])
    pos = pos2 - pos1

    R1 = np.array([[p[0], p[1], p[2]], [p[4], p[5], p[6]], [p[8], p[9], p[10]]])
    R2 = np.array([[p2[0], p2[1], p2[2]], [p2[4], p2[5], p2[6]], [p2[8], p2[9], p2[10]]])
    quat = rotation_matrix_to_quaternion(np.matmul(np.linalg.inv(R1), R2))
    return np.concatenate((quat, pos))    # qxyz


def is_rotation_matrix(R):
    """ Checks if a matrix is a valid rotation matrix
        referred from https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotation_matrix_to_euler_angles(R):
    """ calculates rotation matrix to euler angles
        referred from https://www.learnopencv.com/rotation-matrix-to-euler-angles
    """
    assert(is_rotation_matrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def rotation_matrix_to_quaternion(R):
    assert (is_rotation_matrix(R))

    qw = np.sqrt(1 + np.sum(np.diag(R))) / 2.0
    qx = (R[2, 1] - R[1, 2]) / (4 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw)

    return np.array([qw, qx, qy, qz])


if __name__ == '__main__':
    CM.set_cfg('deepvo')
    CM.model()['train']['time_step'] = 1

    d = DatasetKitti()
    batch = d.get_next_batch()
    pass