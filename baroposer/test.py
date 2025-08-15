import re
import pandas as pd

# 定义模型名称和日志文件路径
model_name = 'heightposer_noise+'
dataset_name = 'imuposer'
log_path = 'data/eval/' + model_name + '/' +'lw_rp' + '/' + dataset_name + '/log.txt'

# 用于存放提取到的数值
results = []

# 打开文件逐行读取
with open(log_path, 'r') as f:
    for line in f:
        # 使用正则表达式匹配行中的所有数字（包括浮点数）
        numbers = re.findall(r'[-+]?\d*\.\d+|\d+', line)
        if numbers:
            # 提取最后一个数字，并转换为浮点型
            results.append(float(numbers[-1]))

# 将结果转换成DataFrame的一列
df = pd.DataFrame(results, columns=['Mesh_Error'])
# save to txt
df.to_csv('data/eval/' + model_name + '/lw_rp/' + dataset_name + '/mesh_error.txt', index=False)