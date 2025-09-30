from datasets import load_dataset

# 下载 V2X-Sim-2.0-mini 数据集
dataset = load_dataset("ai4ce/V2X-Sim-2.0-mini", split="train")

# 查看第一个样本
#print(dataset[0])
dataset[0]["image"].show()
