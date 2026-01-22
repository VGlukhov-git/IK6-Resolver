import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utility import solve_kinematics, get_opposite_point
# --- Параметры ---
L_const, R_const, H_const = 315.0, 65.0, 220.0
L6_length = 70.0
dL_val = -20.0 
T_angles = [0.0, 0.0]

# Параметры КУБА
cube_center = np.array([300.0, 200.0, 100.0])
cube_size = 200.0 
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

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.clear()
    
    total_segments = len(path_indices) - 1
    segment_f = (frame / num_frames) * total_segments
    idx = int(segment_f) % total_segments
    t_interp = segment_f % 1.0
    A_curr = v[path_indices[idx]] + (v[path_indices[idx+1]] - v[path_indices[idx]]) * t_interp
    A_curr = get_opposite_point(A_curr, T_angles, L6_length)

    for i in range(4):
        ax.plot([v[i][0], v[(i+1)%4][0]], [v[i][1], v[(i+1)%4][1]], [v[i][2], v[(i+1)%4][2]], 'r:', alpha=0.1)
        ax.plot([v[i+4][0], v[((i+1)%4)+4][0]], [v[i+4][1], v[((i+1)%4)+4][1]], [v[i+4][2], v[((i+1)%4)+4][2]], 'r:', alpha=0.1)
        ax.plot([v[i][0], v[i+4][0]], [v[i][1], v[i+4][1]], [v[i][2], v[i+4][2]], 'r:', alpha=0.1)

    res = solve_kinematics(A_curr, [T_angles[1] + 90, T_angles[0]], L_const, R_const, H_const, dL_val)
    if res:
        C, S, V, dCA, T_v, dL_v, angs = res
        
        E = A_curr + T_v * L6_length
        # Отрисовка узловых точек (Joints)
        joints = [dL_v, S, C, A_curr, E]
        colors = ['purple', 'orange', 'blue', 'red', 'black']
        labels = ['Base', 'S', 'C', 'A', 'E']
        
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
        ax.plot([A_curr[0], E[0]], [A_curr[1], E[1]], [A_curr[2], E[2]], 
                color='black', lw=4, label='Axis 6')
        ax.quiver(A_curr[0], A_curr[1], A_curr[2], 
                  T_v[0]*L6_length, T_v[1]*L6_length, T_v[2]*L6_length, 
                  color='green', arrow_length_ratio=0.3)
        if abs(T_v[2]) > 0.9:
            up = np.array([1, 0, 0])
        else:
            up = np.array([0, 0, 1])
            
        # 2. Строим перпендикуляр через двойное векторное произведение
        # Вектор 'side' будет перпендикулярен T_v
        side = np.cross(T_v, up)
        side /= np.linalg.norm(side) # Нормируем
        
        # 3. Конечная точка перпендикулярного вектора (длиной 40)
        L_perp = 40.0
        P_perp = E + side * L_perp
        
        # 4. Отрисовка
        ax.plot([E[0], P_perp[0]], [E[1], P_perp[1]], [E[2], P_perp[2]], 
                color='magenta', lw=2, label='Orientation')
        ax.quiver(E[0], E[1], E[2], 
                  side[0]*L_perp, side[1]*L_perp, side[2]*L_perp, 
                  color='magenta', arrow_length_ratio=0.2)
        
        # Текстовые метки суставов (A1-A5)
        ax.text(dL_v[0]/2, dL_v[1]/2, 10, f"A1:{angs[0]:.1f}°", color='purple', fontweight='bold')
        ax.text(dL_v[0], dL_v[1], dL_v[2]+30, f"A2:{angs[1]:.1f}°", color='darkorange', fontweight='bold')
        ax.text(S[0], S[1], S[2]+30, f"A3:{angs[2]:.1f}°", color='blue', fontweight='bold')
        ax.text(C[0], C[1], C[2]+30, f"A4:{angs[3]:.1f}°", color='red', fontweight='bold')
        ax.text(A_curr[0], A_curr[1], A_curr[2]+50, f"A5:{angs[4]:.1f}°", color='green', fontweight='bold')
        
        cone_axis = C - S
        v1 = cone_axis / np.linalg.norm(cone_axis)
        v2 = T_v / np.linalg.norm(T_v)
        dot_product = np.dot(v1, v2)
        angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)

        ax.text2D(0.02, 0.60, f"Угол Axis5-6 (Конус): {angle_deg:.2f}°", 
                transform=ax.transAxes, color='blue', fontweight='bold')
        ax.text2D(0.02, 0.91, f"L4: {np.linalg.norm(S-C):.2f}", transform=ax.transAxes)
        ax.text2D(0.02, 0.83, f"L2: {np.linalg.norm(S-dL_v):.2f}", transform=ax.transAxes)
        ax.text2D(0.02, 0.79, f"dL: {np.linalg.norm(0-dL_v):.2f}", transform=ax.transAxes)
        ax.text2D(0.02, 0.87, f"L5: {np.linalg.norm(C-A_curr):.2f}", transform=ax.transAxes)
        
        angle_ACS = get_angle(A_curr - C, C - S)
        angle_E = get_angle(E - A_curr, A_curr - C)
        ax.text2D(0.02, 0.70, f"Угол AC ^ CS: {angle_ACS:.2f}°", 
                  transform=ax.transAxes, fontsize=12, 
                  color='red' if not np.isclose(angle_ACS, 90, atol=0.1) else 'green')
        ax.text2D(0.02, 0.67, f"Угол AE ^ AC: {angle_E:.2f}°", 
                  transform=ax.transAxes, fontsize=12, 
                  color='red' if not np.isclose(angle_E, 90, atol=0.1) else 'green')

    ax.set_xlim(0, 600); ax.set_ylim(-300, 300); ax.set_zlim(0, 600)
    ax.set_box_aspect([1, 1, 1])

ani = FuncAnimation(fig, update, frames=num_frames, interval=1)
plt.show()