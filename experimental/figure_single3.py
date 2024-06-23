import matplotlib.pyplot as plt
import numpy as np

format = "pdf"
file_pre = "/home/zpp/Projects/CognisancePlus/experimental/"
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 43  # 设置字体大小
plt.rc('text', usetex=True)
plt.figure(figsize=(14, 10))

# 创建示例数据
x_rn = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
x_rd = [2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
y_rn_clt = [0.5576, 0.5576, 0.5553, 0.5574, 0.5592, 0.5581]
y_rd_clt = [0.5597, 0.5558, 0.5553, 0.5560, 0.5578, 0.5581]
y_rn_glt = [0.4745, 0.4745, 0.4751, 0.4753, 0.4754, 0.4757]
y_rd_glt = [0.4756, 0.4755, 0.4751, 0.4747, 0.4768, 0.4740]

ax = plt.gca()
ax.set_ylim(0.4, 0.7)


# plt.plot(x_rn,y_rn_clt,'s-',color = 'royalblue',label="CLT Protocol", linewidth=2, markersize=10)#s-:方形
# plt.plot(x_rn,y_rn_glt,'o-',color = 'deeppink',label="GLT Protocol", linewidth=2, markersize=10)#o-:圆形
# plt.xlabel('$d_{rn}$')
# plt.ylabel("F1-Score$^*$")
# plt.legend(loc = "best")#图例
# plt.savefig(file_pre + "for_rn.pdf", format=format, bbox_inches="tight")
# plt.show()

plt.plot(x_rn,y_rd_clt,'s-',color = 'royalblue',label="CLT Protocol", linewidth=2, markersize=10)#s-:方形
plt.plot(x_rn,y_rd_glt,'o-',color = 'deeppink',label="GLT Protocol", linewidth=2, markersize=10)#o-:圆形
plt.xlabel('$d_{rd}$')
plt.ylabel("F1-Score$^*$")
plt.legend(loc = "best")#图例
plt.savefig(file_pre + "for_rd.pdf", format=format, bbox_inches="tight")
plt.show()


