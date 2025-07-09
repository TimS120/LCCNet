import torch
import mathutils

def invert_pose(R, T):
    """
    Given the 'sampled pose' (aka H_init), we want CMRNet to predict inv(H_init).
    inv(T*R) will be used as ground truth for the network.
    Args:
        R (mathutils.Euler): Rotation of 'sampled pose'
        T (mathutils.Vector): Translation of 'sampled pose'

    Returns:
        (R_GT, T_GT) = (mathutils.Quaternion, mathutils.Vector)
    """
    R = R.to_matrix().to_4x4()
    T = mathutils.Matrix.Translation(T)
    RT = T @ R
    RT = RT.inverted()
    T_GT, R_GT, _ = RT.decompose()
    return R_GT.normalized(), T_GT

def quat_mul(q1, q2):
    # Hamilton product of two torch quaternions [w,x,y,z]
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return torch.tensor([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

# --- example input ---
rotx, roty, rotz = -0.1041676079234717, -0.1535481542591603, 0.2137340899813528
transl = mathutils.Vector((-0.23616647575217797,
                           -0.15872444944968733,
                           -0.21246866159984545))

# build original quaternion + translation
orig_euler = mathutils.Euler((rotx, roty, rotz))
orig_quat  = orig_euler.to_quaternion()     # mathutils.Quaternion
q_orig     = torch.tensor((orig_quat.w,
                           orig_quat.x,
                           orig_quat.y,
                           orig_quat.z))
T_orig     = torch.tensor((transl.x, transl.y, transl.z))

# invert via your function
R_inv_raw, T_inv_vec = invert_pose(orig_euler, transl)

# determine whether R_inv_raw is Euler or Quaternion
if isinstance(R_inv_raw, mathutils.Euler):
    R_inv_quat = R_inv_raw.to_quaternion()
else:
    R_inv_quat = R_inv_raw

q_inv = torch.tensor((R_inv_quat.w,
                      R_inv_quat.x,
                      R_inv_quat.y,
                      R_inv_quat.z))
T_inv = torch.tensor((T_inv_vec.x,
                      T_inv_vec.y,
                      T_inv_vec.z))

# compose: rotation then translation
q_comp = quat_mul(q_inv, q_orig)

# rotate the original translation by the inverse quaternion, then add inverse translation
# (R_inv_quat is a mathutils.Quaternion)
rotated_vec = R_inv_quat @ transl         # rotates the mathutils.Vector 'transl'
rotated = torch.tensor((rotated_vec.x,
                        rotated_vec.y,
                        rotated_vec.z))
T_comp = rotated + T_inv

print("composed quaternion:", q_comp)     # ≈ [1,0,0,0]
print("composed translation:", T_comp)    # ≈ [0,0,0]
