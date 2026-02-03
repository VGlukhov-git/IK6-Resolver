import numpy as np
from scipy.optimize import fsolve
import math
last_guess = np.array([0.1, 0.0, 0.0]) 

def get_angle(v1, v2):
    dot = np.clip(np.dot(v1/np.linalg.norm(v1), v2/np.linalg.norm(v2)), -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def get_opposite_point(A, T, L):
    # Unpack point and angles
    x, y, z = A
    azimuth_deg, elevation_deg = T
    elevation_deg = elevation_deg
    # Convert degrees to radians
    # azimuth_deg += 90
    # elevation_deg += 90
    
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    # Calculate the unit vector components for direction T
    dx = math.cos(elevation) * math.cos(azimuth)
    dy = math.cos(elevation) * math.sin(azimuth)
    dz = math.sin(elevation)
    # New point is A MINUS (Direction * Distance) to go in the opposite direction
    x_new = x - L * dx
    y_new = y - L * dy
    z_new = z + L * dz
    
    return [x_new, y_new, z_new]

def solve_kinematics(A, T_angles, L, R, H, dL):
    global last_guess
    
    th_t, ph_t = np.radians(T_angles[0]), np.radians(T_angles[1])
    T_vec = np.array([np.sin(th_t)*np.cos(ph_t), np.sin(th_t)*np.sin(ph_t), np.cos(th_t)])
    
    def equations(p):
        v_th, v_ph, alpha = p
        
        V = np.array([np.sin(v_th)*np.cos(v_ph), np.sin(v_th)*np.sin(v_ph), np.cos(v_th)])
        dCA = np.cross(-V, T_vec)
        if np.linalg.norm(dCA) < 1e-9: return [1e6, 1e6, 1e6]
        dCA /= np.linalg.norm(dCA)
        C, S = A - dCA * R, A - dCA * R + V * H
        
        dL_vec = np.array([np.cos(alpha), np.sin(alpha), 0.0]) * dL
        
        ref = A - S                      # or S - C depending on your convention
        if np.dot(dL_vec, ref) < 0 and alpha < 0:
            dL_vec = -dL_vec
            alpha += np.pi
        L_proj = (S - dL_vec)[:2]
        eq1 = np.dot(L_proj, dL_vec[:2])
        eq2 = np.linalg.norm(S - dL_vec) - L
        Z = np.array([0.0, 0.0, 1.0])
        n = np.cross(C, Z)
        nn = np.linalg.norm(n)
        if nn < 1e-8:
            eq3 = 0.0
        else:
            n /= nn
            eq3 = np.dot(S, n)  
        return [eq1, eq2, eq3]

    res, info, ier, msg = fsolve(equations, last_guess, full_output=True)
    if ier == 1:
        last_guess = res
        v_t, v_p, alpha = res
        V = np.array([np.sin(v_t)*np.cos(v_p), np.sin(v_t)*np.sin(v_p), np.cos(v_t)])
        dCA = np.cross(-V, T_vec); dCA /= np.linalg.norm(dCA)
        C, S = A - dCA * R, A - dCA * R + V * H
        dL_v = np.array([np.cos(alpha), np.sin(alpha), 0.0]) * dL
        ref = A - S                      # or S - C depending on your convention
        if np.dot(dL_v, ref) < 0 and alpha < 0:
            dL_v = -dL_v
            alpha += np.pi
        
        a1 = np.degrees(alpha)
        vec_L = S - dL_v
        a2 = np.degrees(np.arctan2(vec_L[2], np.linalg.norm(vec_L[:2])))
        vec_H = C - S
        a3 = 180.0 - get_angle(-vec_L, vec_H)
        plane_norm = np.cross(vec_L, [0, 0, 1])
        ref_dir = np.cross(vec_H, plane_norm)
        a4 = np.degrees(np.arctan2(np.dot(dCA, plane_norm), np.dot(dCA, ref_dir)))
        a5 = get_angle(T_vec, vec_H)
        return C, S, V, dCA, T_vec, dL_v, (a1, a2, a3, a4, a5)
    return None