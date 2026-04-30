import torch
import torch.nn as nn

EPS = 1e-6

def normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    return quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)

def quaternion_to_euler(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = normalize_quaternion(quaternion)
    qw, qx, qy, qz = quaternion.unbind(dim=-1)

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = torch.asin(sinp.clamp(-1.0 + EPS, 1.0 - EPS))

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)

def euler_to_rotation_matrix(euler: torch.Tensor) -> torch.Tensor:
    roll, pitch, yaw = euler.unbind(dim=-1)

    cr = torch.cos(roll)
    sr = torch.sin(roll)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)

    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr

    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr

    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr

    row0 = torch.stack((r00, r01, r02), dim=-1)
    row1 = torch.stack((r10, r11, r12), dim=-1)
    row2 = torch.stack((r20, r21, r22), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)

def rotation_matrix_to_euler(rotation: torch.Tensor) -> torch.Tensor:
    sy = torch.sqrt((rotation[..., 0, 0] ** 2 + rotation[..., 1, 0] ** 2).clamp_min(EPS))
    singular = sy < 1e-6

    roll = torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2])
    pitch = torch.atan2(-rotation[..., 2, 0], sy)
    yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])

    singular_roll = torch.atan2(-rotation[..., 1, 2], rotation[..., 1, 1])
    singular_pitch = torch.atan2(-rotation[..., 2, 0], sy.clamp_min(EPS))
    singular_yaw = torch.zeros_like(yaw)

    roll = torch.where(singular, singular_roll, roll)
    pitch = torch.where(singular, singular_pitch, pitch)
    yaw = torch.where(singular, singular_yaw, yaw)

    return torch.stack((roll, pitch, yaw), dim=-1)

def euler_to_quaternion(euler: torch.Tensor) -> torch.Tensor:
    roll, pitch, yaw = euler.unbind(dim=-1)

    half_roll = roll * 0.5
    half_pitch = pitch * 0.5
    half_yaw = yaw * 0.5

    cr = torch.cos(half_roll)
    sr = torch.sin(half_roll)
    cp = torch.cos(half_pitch)
    sp = torch.sin(half_pitch)
    cy = torch.cos(half_yaw)
    sy = torch.sin(half_yaw)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    quaternion = torch.stack((qw, qx, qy, qz), dim=-1)
    return normalize_quaternion(quaternion)

def ensure_sequence_dim(pose: torch.Tensor) -> torch.Tensor:
    if pose.ndim == 2:
        return pose.unsqueeze(1)
    if pose.ndim != 3:
        raise ValueError(
            f"Expected pose tensor with shape [B, 7] or [B, T, 7], got {tuple(pose.shape)}"
        )
    return pose

def split_pose(pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pose.size(-1) != 7:
        raise ValueError(
            f"Expected pose tensor with last dimension 7 for [x, y, z, qw, qx, qy, qz], got {pose.size(-1)}"
        )
    translation = pose[..., :3]
    quaternion = pose[..., 3:]
    return translation, quaternion

def split_prediction(prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.size(-1) == 7:
        return split_pose(prediction)
    if prediction.size(-1) == 6:
        translation = prediction[..., :3]
        euler = prediction[..., 3:]
        quaternion = euler_to_quaternion(euler)
        return translation, quaternion
    raise ValueError(
        f"Expected prediction last dimension 6 ([x, y, z, roll, pitch, yaw]) "
        f"or 7 ([x, y, z, qw, qx, qy, qz]), got {prediction.size(-1)}"
    )

def prediction_to_transform(prediction: torch.Tensor) -> torch.Tensor:
    prediction = ensure_sequence_dim(prediction)
    translation, quaternion = split_prediction(prediction)
    euler = quaternion_to_euler(quaternion)
    rotation = euler_to_rotation_matrix(euler)

    batch, steps = translation.shape[:2]
    transform = torch.eye(4, device=prediction.device, dtype=prediction.dtype).repeat(batch, steps, 1, 1)
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = translation
    return transform

def pose_to_transform(pose: torch.Tensor) -> torch.Tensor:
    pose = ensure_sequence_dim(pose)
    translation, quaternion = split_pose(pose)
    euler = quaternion_to_euler(quaternion)
    rotation = euler_to_rotation_matrix(euler)

    batch, steps = translation.shape[:2]
    transform = torch.eye(4, device=pose.device, dtype=pose.dtype).repeat(batch, steps, 1, 1)
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = translation
    return transform

def compose_sequence_transforms(transform: torch.Tensor) -> torch.Tensor:
    transform = transform.clone()
    cumulative = []
    running = torch.eye(4, device=transform.device, dtype=transform.dtype).repeat(transform.size(0), 1, 1)

    for step in range(transform.size(1)):
        running = running @ transform[:, step]
        cumulative.append(running)

    return torch.stack(cumulative, dim=1)

def rmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((prediction - target) ** 2) + 1e-8)

class PoseSequenceLoss(nn.Module):
    def __init__(
        self,
        frame_weight: float = 0.75,
        sequence_weight: float = 0.25,
        translation_weight: float = 1.0,
        rotation_weight: float = 1.0,
    ):
        super().__init__()
        self.frame_weight = frame_weight
        self.sequence_weight = sequence_weight
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction = ensure_sequence_dim(prediction)
        target = ensure_sequence_dim(target)

        pred_translation, pred_quaternion = split_prediction(prediction)
        target_translation, target_quaternion = split_pose(target)

        pred_frame_euler = quaternion_to_euler(pred_quaternion)
        target_frame_euler = quaternion_to_euler(target_quaternion)

        lframe_xyz = rmse(pred_translation, target_translation)
        lframe_euler = rmse(pred_frame_euler, target_frame_euler)
        lframe = (
            self.translation_weight * lframe_xyz
            + self.rotation_weight * lframe_euler
        )

        pred_frame_transform = prediction_to_transform(prediction)
        target_frame_transform = pose_to_transform(target)

        pred_sequence_transform = compose_sequence_transforms(pred_frame_transform)
        target_sequence_transform = compose_sequence_transforms(target_frame_transform)

        pred_sequence_translation = pred_sequence_transform[..., :3, 3]
        target_sequence_translation = target_sequence_transform[..., :3, 3]

        pred_sequence_euler = rotation_matrix_to_euler(pred_sequence_transform[..., :3, :3])
        target_sequence_euler = rotation_matrix_to_euler(target_sequence_transform[..., :3, :3])

        lsequence_xyz = rmse(pred_sequence_translation, target_sequence_translation)
        lsequence_euler = rmse(pred_sequence_euler, target_sequence_euler)
        lsequence = (
            self.translation_weight * lsequence_xyz
            + self.rotation_weight * lsequence_euler
        )

        loss = self.frame_weight * lframe + self.sequence_weight * lsequence

        metrics = {
            "loss": loss.detach(),
            "lframe": lframe.detach(),
            "lframe_xyz": lframe_xyz.detach(),
            "lframe_euler": lframe_euler.detach(),
            "lsequence": lsequence.detach(),
            "lsequence_xyz": lsequence_xyz.detach(),
            "lsequence_euler": lsequence_euler.detach(),
        }
        return loss, metrics
