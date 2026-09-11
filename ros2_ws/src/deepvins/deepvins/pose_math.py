"""Pose operations shared by DeepVINS inference and tests."""

import numpy as np


def normalize_quaternion(quaternion):
    """Normalize a quaternion stored in wxyz order."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError('quaternion norm is zero or non-finite')
    return quaternion / norm


def multiply_quaternions(left, right):
    """Return the Hamilton product of two wxyz quaternions."""
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array([
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ])


def rotate_vector(quaternion, vector):
    """Rotate a 3-vector by a normalized wxyz quaternion."""
    _, qx, qy, qz = quaternion
    q_vector = np.array([qx, qy, qz])
    return (vector + 2.0 * np.cross(
        q_vector,
        np.cross(q_vector, vector) + quaternion[0] * vector))


def compose_pose(absolute_pose, relative_pose):
    """Compose two [x, y, z, qw, qx, qy, qz] poses."""
    absolute_pose = np.asarray(absolute_pose, dtype=np.float64)
    relative_pose = np.asarray(relative_pose, dtype=np.float64)
    if absolute_pose.shape != (7,) or relative_pose.shape != (7,):
        raise ValueError('poses must each contain seven values')

    absolute_q = normalize_quaternion(absolute_pose[3:])
    relative_q = normalize_quaternion(relative_pose[3:])
    translation = absolute_pose[:3] + rotate_vector(
        absolute_q, relative_pose[:3])
    quaternion = normalize_quaternion(
        multiply_quaternions(absolute_q, relative_q))
    return np.concatenate((translation, quaternion))
