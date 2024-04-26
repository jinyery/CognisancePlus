import matplotlib.pyplot as plt
import numpy as np

format = "pdf"
file_pre = "/mnt/c/Users/yjy/OneDrive/桌面/"
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 38  # 设置字体大小
plt.rc('text', usetex=True)
plt.figure(figsize=(15, 10))
# 创建示例数据
categories = ["Baseline", "RandAug", "BLSoftmax", "Logit-Adj", "TADE", "RIDE"]
data1 = [
    [0.7505, 0.8083, 0.7655, 0.7515, 0.8043, 0.8202],
    [0.6014, 0.6827, 0.6015, 0.6040, 0.6403, 0.7026],
    [0.4296, 0.5055, 0.4170, 0.4298, 0.4954, 0.5346],
    [0.2188, 0.2987, 0.2104, 0.2175, 0.2814, 0.2935],
]  # 第一组数据
data2 = [
    [0.8227, 0.8227, 0.8262, 0.8221, 0.8448, 0.8506],
    [0.7052, 0.7052, 0.7117, 0.7126, 0.7514, 0.7677],
    [0.5300, 0.5300, 0.5280, 0.5307, 0.5654, 0.5686],
    [0.3809, 0.3809, 0.4003, 0.3814, 0.3924, 0.4674],
]  # 第二组数据
tmp = 3
ax = plt.gca()
if tmp == 0:
    ax.set_ylim(0.65, 0.95)
    filename = "improve_animal10n."+format
elif tmp == 1:
    ax.set_ylim(0.55, 0.85)
    filename = "improve_animal10nlt."+format
elif tmp == 2:
    ax.set_ylim(0.25, 0.65)
    filename = "improve_food101n."+format
elif tmp == 3:
    ax.set_ylim(0.15, 0.55)
    filename = "improve_food101nlt."+format
ax.xaxis.set_tick_params(rotation=18)

# 设置柱形宽度
bar_width = 0.35
bar_interval = 0.03

# 生成 x 坐标位置
x = np.arange(len(categories))

# 创建第一组柱形图
plt.bar(
    x, data1[tmp], width=bar_width, label="Original", color="royalblue", align="center"
)

# 创建第二组柱形图，将 x 坐标右移 bar_width 以实现双排效果
plt.bar(
    x + bar_width + bar_interval,
    data2[tmp],
    width=bar_width,
    label=r'+\textsc{Cognisance}$^+$',
    color="deeppink",
    align="center",
)

# 设置 x 轴标签和标题
# plt.xlabel('Categories')
plt.ylabel("F1-Score$^*$")
# plt.title('Double Bar Chart')

# 设置 x 轴刻度标签
plt.xticks(x + bar_width / 2 + bar_interval / 2, categories)

# 添加图例
plt.legend()
plt.savefig(file_pre + filename, format=format, bbox_inches="tight")
# 显示图形
plt.show()
