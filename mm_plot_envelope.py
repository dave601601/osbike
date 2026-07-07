import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.colors import LinearSegmentedColormap
delays=[0,5,10,15,20]
sev=["s0.25","s0.5","s0.75","s1.0"]
sub=["1° / μ1.1 / 1cm","2° / μ0.9 / 2cm","3° / μ0.6 / 4cm","4° / μ0.4 / 5cm"]
data=np.array([[100,100,100,33,0],[100,100,100,0,0],[66,66,33,0,0],[33,0,33,0,0]],float)
green=LinearSegmentedColormap.from_list("s",["#f2f4f3","#d7ece0","#8fd0ab","#3f9d6b","#1c6b43"])
fig,ax=plt.subplots(figsize=(8.8,5.8),dpi=150)
fig.subplots_adjust(top=0.86,bottom=0.16)
im=ax.imshow(data,cmap=green,vmin=0,vmax=100,aspect="auto")
for i in range(4):
    for j in range(5):
        v=data[i,j]
        ax.text(j,i,f"{v:.0f}%",ha="center",va="center",color="#ffffff" if v>=55 else "#2b2f2c",fontsize=14,fontweight="bold")
ax.set_xticks(range(5)); ax.set_xticklabels(delays,fontsize=11)
ax.set_yticks(range(4)); ax.set_yticklabels([f"{s}\n{u}" for s,u in zip(sev,sub)],fontsize=9.5)
ax.set_xlabel("Actuator delay [ms]  (50 Hz control)",fontsize=11.5)
ax.set_ylabel("Terrain severity  (slope / μ / bump)",fontsize=11.5)
fig.suptitle("LQR combined survival rate:  terrain × delay",fontsize=14,fontweight="bold",y=0.965)
ax.set_title("moving-mass, v=1.5, 30° turn, 30 s hold, 3 seeds",fontsize=10.5,color="#555",pad=8)
ax.set_xticks(np.arange(-.5,5,1),minor=True); ax.set_yticks(np.arange(-.5,4,1),minor=True)
ax.grid(which="minor",color="white",linewidth=3); ax.tick_params(which="minor",length=0)
ax.axvline(2.5,color="#c0392b",lw=1.6,ls=(0,(4,2)))
fig.text(0.30,0.045,"◀ LQR robust",color="#1c6b43",fontsize=11,fontweight="bold",ha="center")
fig.text(0.66,0.045,"realistic HW delay ▶  LQR collapses = RL / delay-aware target",color="#c0392b",fontsize=10.5,fontweight="bold",ha="center")
cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.03); cb.set_label("30 s survival [%]",fontsize=10)
fig.savefig("/home/bike/bike/lqr_envelope_map.png",bbox_inches="tight",facecolor="white")
print("saved")
