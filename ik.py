import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import fsolve

# --- Параметры ---
L_const, R_const, H_const = 315.0, 65.0, 220.0
dL_val = -20.0 
T_angles = [0.0, -0.0]

# Параметры КУБА
cube_center = np.array([300.0, 0.0, 250.0])
cube_size = 150.0 
half = cube_size / 2

# Вершины куба
v = [
    cube_center + np.array([-half, -half, -half]), # 0
    cube_center + np.array([ half, -half, -half]), # 1
    cube_center + np.array([ half,  half, -half]), # 2
    cube_center + np.array([-half,  half, -half]), # 3
    cube_center + np.array([-half, -half,  half]), # 4
    cube_center + np.array([ half, -half,  half]), # 5
    cube_center + np.array([ half,  half,  half]), # 6
    cube_center + np.array([-half,  half,  half])  # 7
]
path_indices = [0, 1, 2, 3, 0, 4, 5, 1, 5, 6, 2, 6, 7, 3, 7, 4]

num_frames = 480
last_guess = np.array([0.1, 0.0, 0.0]) 

def get_angle(v1, v2):
    dot = np.clip(np.dot(v1/np.linalg.norm(v1), v2/np.linalg.norm(v2)), -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def solve_kinematics(A, T_angles, L, R, H, dL):
    global last_guess
    th_t, ph_t = np.radians(T_angles[0]), np.radians(T_angles[1])
    T_vec = np.array([np.sin(th_t)*np.cos(ph_t), np.sin(th_t)*np.sin(ph_t), np.cos(th_t)])
    
    def equations(p):
        v_th, v_ph, alpha = p
        V = np.array([np.sin(v_th)*np.cos(v_ph), np.sin(v_th)*np.sin(v_ph), np.cos(v_th)])
        dCA = np.cross(V, T_vec)
        if np.linalg.norm(dCA) < 1e-9: return [1e6, 1e6, 1e6]
        dCA /= np.linalg.norm(dCA)
        C, S = A - dCA * R, A - dCA * R + V * H
        dL_vec = np.array([np.cos(alpha), np.sin(alpha), 0.0]) * dL
        L_proj = (S - dL_vec)[:2]
        eq1 = np.dot(L_proj, dL_vec[:2])
        eq2 = np.linalg.norm(S - dL_vec) - L
        eq3 = L_proj[0]*(C-dL_vec)[1] - L_proj[1]*(C-dL_vec)[0]
        
        penalty = 0
        if S[2] < A[2]: penalty = (A[2] - S[2]) ** 2
        return [eq1, eq2, eq3 + penalty]

    res, info, ier, msg = fsolve(equations, last_guess, full_output=True)
    if ier == 1:
        last_guess = res
        v_t, v_p, alpha = res
        V = np.array([np.sin(v_t)*np.cos(v_p), np.sin(v_t)*np.sin(v_p), np.cos(v_t)])
        dCA = np.cross(V, T_vec); dCA /= np.linalg.norm(dCA)
        C, S = A - dCA * R, A - dCA * R + V * H
        dL_v = np.array([np.cos(alpha), np.sin(alpha), 0.0]) * dL
        
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

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.clear()
    
    total_segments = len(path_indices) - 1
    segment_f = (frame / num_frames) * total_segments
    idx = int(segment_f) % total_segments
    t_interp = segment_f % 1.0
    A_curr = v[path_indices[idx]] + (v[path_indices[idx+1]] - v[path_indices[idx]]) * t_interp
    
    for i in range(4):
        ax.plot([v[i][0], v[(i+1)%4][0]], [v[i][1], v[(i+1)%4][1]], [v[i][2], v[(i+1)%4][2]], 'r:', alpha=0.1)
        ax.plot([v[i+4][0], v[((i+1)%4)+4][0]], [v[i+4][1], v[((i+1)%4)+4][1]], [v[i+4][2], v[((i+1)%4)+4][2]], 'r:', alpha=0.1)
        ax.plot([v[i][0], v[i+4][0]], [v[i][1], v[i+4][1]], [v[i][2], v[i+4][2]], 'r:', alpha=0.1)

    res = solve_kinematics(A_curr, T_angles, L_const, R_const, H_const, dL_val)
    if res:
        C, S, V, dCA, T_v, dL_v, angs = res
        
        # Отрисовка узловых точек (Joints)
        joints = [dL_v, S, C, A_curr]
        colors = ['purple', 'orange', 'blue', 'red']
        labels = ['Base', 'S', 'C', 'A']
        
        for pt, clr, lbl in zip(joints, colors, labels):
            ax.scatter(pt[0], pt[1], pt[2], color=clr, s=50, edgecolors='white', zorder=5)
            ax.text(pt[0], pt[1], pt[2] - 20, lbl, color=clr, fontsize=9, fontweight='bold', ha='center')

        # Геометрия
        ax.plot([A_curr[0], A_curr[0]+T_v[0]*100], [A_curr[1], A_curr[1]+T_v[1]*100], [A_curr[2], A_curr[2]+T_v[2]*100], color='gold', lw=3)
        L_p = (C - dL_v); L_p[2] = 0
        gp, gz = np.meshgrid([0, 1.2], [-50, 650])
        ax.plot_surface(dL_v[0]+gp*L_p[0], dL_v[1]+gp*L_p[1], gz, alpha=0.1, color='orange')

        # Конус
        u, h = np.meshgrid(np.linspace(0, 2*np.pi, 20), np.linspace(0, 1, 3))
        bn = np.cross(V, dCA); bn /= np.linalg.norm(bn)
        X = S[0] - V[0]*h*H_const + R_const*h*(dCA[0]*np.cos(u) + bn[0]*np.sin(u))
        Y = S[1] - V[1]*h*H_const + R_const*h*(dCA[1]*np.cos(u) + bn[1]*np.sin(u))
        Z = S[2] - V[2]*h*H_const + R_const*h*(dCA[2]*np.cos(u) + bn[2]*np.sin(u))
        ax.plot_surface(X, Y, Z, alpha=0.2, color='cyan', edgecolor='none')

        # Скелет
        ax.plot([0, dL_v[0]], [0, dL_v[1]], [0, 0], 'purple', lw=3)
        ax.plot([dL_v[0], S[0]], [dL_v[1], S[1]], [dL_v[2], S[2]], 'orange', lw=2)
        ax.plot([S[0], C[0]], [S[1], C[1]], [S[2], C[2]], 'blue', lw=2)
        ax.plot([C[0], A_curr[0]], [C[1], A_curr[1]], [C[2], A_curr[2]], 'red', lw=3)
        ax.quiver(A_curr[0], A_curr[1], A_curr[2], T_v[0]*70, T_v[1]*70, T_v[2]*70, color='green')

        # Текстовые метки суставов (A1-A5)
        ax.text(dL_v[0]/2, dL_v[1]/2, 10, f"A1:{angs[0]:.1f}°", color='purple', fontweight='bold')
        ax.text(dL_v[0], dL_v[1], dL_v[2]+30, f"A2:{angs[1]:.1f}°", color='darkorange', fontweight='bold')
        ax.text(S[0], S[1], S[2]+30, f"A3:{angs[2]:.1f}°", color='blue', fontweight='bold')
        ax.text(C[0], C[1], C[2]+30, f"A4:{angs[3]:.1f}°", color='red', fontweight='bold')
        ax.text(A_curr[0], A_curr[1], A_curr[2]+50, f"A5:{angs[4]:.1f}°", color='green', fontweight='bold')
        
        # Текстовые метки в углу
        ax.text2D(0.02, 0.91, f"L4: {np.linalg.norm(S-C):.2f}", transform=ax.transAxes)
        ax.text2D(0.02, 0.83, f"L2: {np.linalg.norm(S-dL_v):.2f}", transform=ax.transAxes)
        ax.text2D(0.02, 0.79, f"dL: {np.linalg.norm(0-dL_v):.2f}", transform=ax.transAxes)
        ax.text2D(0.02, 0.87, f"L5: {np.linalg.norm(C-A_curr):.2f}", transform=ax.transAxes)
        
        angle_ABC = get_angle(A_curr - C, C - S)
        ax.text2D(0.02, 0.70, f"Угол AB ^ BC: {angle_ABC:.2f}°", 
                  transform=ax.transAxes, fontsize=12, 
                  color='red' if not np.isclose(angle_ABC, 90, atol=0.1) else 'green')

    ax.set_xlim(0, 600); ax.set_ylim(-300, 300); ax.set_zlim(0, 600)
    ax.set_box_aspect([1, 1, 1])

ani = FuncAnimation(fig, update, frames=num_frames, interval=1)
plt.show()