import matplotlib
matplotlib.use("Agg")  # GUI 창 없이 백엔드 설정
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm
import os

value_dict = {1: 62, 2:28, 3:27, 4:12, 5:26}
full_data_dict = {1:(7,7), 2:(7,3), 3:(3,7), 4:(3,3), 5:(4,4)}
L_set = {0,3,4,6,7}
W_set = {0,3,4,6,7}
L0 = 10; W0 = 10; m=5

g = {(r,s):0 for r in L_set for s in W_set}


a = {}
for i in range(1,m+1):
    for p in {x for x in L_set if (x+full_data_dict[i][0] <= L0)}:
        for q in {y for y in W_set if (y+full_data_dict[i][1] <= W0)}:
            for r in L_set:
                for s in W_set: 
                    a[i,p,q,r,s] = 1 if (p <= r <= p + full_data_dict[i][0] - 1) and \
                    (q <= s <= q + full_data_dict[i][1] - 1) else 0


def solve_subproblems_for_type_i(i: int):
    tmp = float('-inf')
    coord_tuple = (None,None)
    for p in {x for x in L_set if x+full_data_dict[i][0]<=L0}:
        for q in {y for y in W_set if y+full_data_dict[i][1]<=W0}:
            cand = value_dict[i] - sum(g[r,s]*a[i,p,q,r,s] for r in L_set for s in W_set) 
            if tmp < cand:
                tmp = cand
                coord_tuple = (p,q)
    if tmp < 0:
        return None
    return (i,coord_tuple[0], coord_tuple[1])


def save_cutting_image(directory, X: set[tuple[int,int,int]], file_name):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, L0)
    ax.set_ylim(0, W0)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Bottom')
    ax.set_ylabel('Left')
    ax.set_xticks(sorted(L_set))
    ax.set_yticks(sorted(W_set))
    ax.set_title(file_name)
    
    for x in sorted(L_set):
        ax.axvline(x=x, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    # 가로선
    for y in sorted(W_set):
        ax.axhline(y=y, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    cmap = plt.get_cmap("tab10")  

    for (i, p, q) in X:
        w, h = full_data_dict[i]
        color = cmap((i-1) % 5)  
        rect = Rectangle((p, q), w, h, facecolor=color, edgecolor='black',alpha=0.5)
        ax.add_patch(rect)
        ax.text(p + w / 2, q + h / 2, str(i),
                ha='center', va='center', fontsize=9, color='white')

    os.makedirs(directory, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"{file_name}.png"), dpi=150)
    plt.close(fig)

def one_step_subgradient_procedure(iter: int):
    X = set()
    z_lb = 0
    G = dict()
    for i in range(1,m+1):
        sol = solve_subproblems_for_type_i(i)
        if sol is not None:
            X.add(sol)
    z_ub = sum(value_dict[i] for i,p,q in X) - sum(g[r,s]*a[i,p,q,r,s] for i,p,q in X for r in L_set for s in W_set) + sum(g[r,s] for r in L_set for s in W_set)    
    save_cutting_image("C:\\Users\\UserK\\Desktop\\Read_and_Understand\\Coding-Implementation\\[Beasly]Lagrangian_Relaxation\\cutting_images",X,file_name=f"{iter}th_cutting")
    is_overlapped = False
    for r in L_set:
        for s in W_set: 
            tmp = 0
            for i,p,q in X:
                tmp += a[i,p,q,r,s]
                if tmp <= 1:
                    continue
                is_overlapped = True
                break
            if is_overlapped:
                break
        if is_overlapped:
            break  
    if is_overlapped == False:
        print(f"[iter{iter}]Primal Feasibility reached!")
        z_lb = sum(value_dict[i] for i,_,_ in X)
    if z_lb == z_ub:
        print("stop! Optimality reached!")
        return True, z_ub
    for r in L_set:
        for s in W_set:
            G[r,s] = 1 - sum(a[i,p,q,r,s] for i,p,q in X)
    if iter <= 20:
        f = 2
    elif 21 <= iter <= 30:
        f = 1
    elif 31 <= iter <= 35:
        f = 0.5
    elif 36 <= iter <= 40:
        f = 0.25
    elif 41 <= iter <= 45:
        f = 0.125
    elif 46 <= iter <= 50:
        f = 0.0625
    elif 51 <= iter <= 55:
        f = 0.03125
    elif 55 <= iter <= 60:
        f = 0.015625
    elif 61 <= iter <= 65:
        f = 0.007815
    else:
        f = 0
    denom = sum(G[r,s]**2 for r in L_set for s in W_set)
    t = f*(z_ub - z_lb)/denom if denom > 0 else 0
    for r in L_set:
        for s in W_set:
            g[r,s] = max(0, g[r,s] - t*G[r,s]) 
    return False, z_ub


def subgradient_optimize():
    for iter in range(1,66):
        is_stop, z_ub = one_step_subgradient_procedure(iter)
        if is_stop:
            with open("g_log.txt", "a", encoding="utf-8") as f:
                formatted_g = {k:round(v,3) for k,v in g.items()}
                f.write(f"[iter{iter}]z_ub = {z_ub:.2f} | g = {formatted_g}\n")
            break
        with open("g_log.txt", "a", encoding="utf-8") as f:
            formatted_g = {k:round(v,3) for k,v in g.items()}
            f.write(f"[iter{iter}]z_ub = {z_ub:.2f} | g = {formatted_g}\n")
            
if __name__ == "__main__":
    subgradient_optimize()
