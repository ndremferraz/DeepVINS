import torch 
import torch.nn as nn

EPS = 1e-6

def quaternion_to_rot(quaternion: torch.Tensor) -> torch.Tensor:

    qw, qx, qy, qz = quaternion.unbind(dim=-1)

    # First row of the rotation matrix
    r00 = 2 * (qw * qw + qx * qx) - 1
    r01 = 2 * (qx * qy - qw * qz)
    r02 = 2 * (qx * qz + qw * qy)
     
    # Second row of the rotation matrix
    r10 = 2 * (qx * qy + qw * qz)
    r11 = 2 * (qw * qw + qy * qy) - 1
    r12 = 2 * (qy * qz - qw * qx)
     
    # Third row of the rotation matrix
    r20 = 2 * (qx * qz - qw * qy)
    r21 = 2 * (qy * qz + qw * qx)
    r22 = 2 * (qw * qw + qz * qz) - 1

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

def pose_to_transform(pose: torch.Tensor) -> torch.Tensor:

    translation, quaternion = split_pose(pose)
    rotation = quaternion_to_rot(quaternion)

    batch, context = pose.shape[:2]

    transform = torch.eye(4, device=pose.device, dtype=pose.dtype).repeat(batch, context, 1, 1)

    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = translation

    return transform

def transform_to_components(transform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    translation = transform[..., :3, 3]
    euler = rotation_matrix_to_euler(transform[..., :3, :3])
    return translation, euler

def split_pose(pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

    translation = pose[..., :3]
    quaternion = pose[..., 3:]

    return translation, quaternion

def normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    return quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)

def rmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((prediction - target) ** 2) + 1e-8)

class PoseSequenceLoss(nn.Module):

  def __init__(
      self,
      translation_weight: float = 0.5,
      rotation_weight: float = 0.5
  ):

    super().__init__()

    self.translation_weight = translation_weight
    self.rotation_weight = rotation_weight

  def forward(
    self,
    pred_pose: torch.Tensor,
    input_pose: torch.Tensor,
    target_pose: torch.Tensor
  ) -> torch.Tensor:
     
    pred_transform = pose_to_transform(pred_pose)
    input_transform = pose_to_transform(input_pose)

    pred_abs_transform = input_transform @ pred_transform
    pred_translation, pred_euler = transform_to_components(pred_abs_transform)

    target_transform = pose_to_transform(target_pose)
    target_translation, target_euler = transform_to_components(target_transform)

    translation_loss = rmse(pred_translation, target_translation)
    rotation_loss = rmse(pred_euler, target_euler)

    total_loss = self.translation_weight * translation_loss + self.rotation_weight * rotation_loss

    metrics = {
       "loss": total_loss.detach(),
       "translation_loss": rotation_loss.detach(),
       "rotation_loss": translation_loss.detach()
    }

    return total_loss, metrics







     
     


      
    



