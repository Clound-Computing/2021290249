import json
import csv

# 打开并加载JSON文件
with open('gossipcop_v3-1_style_based_fake.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 创建CSV文件，并定义CSV的头部（即字段名）
fieldnames = ['origin_label', 'origin_text']

# 创建并写入CSV文件
with open('output.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    # 遍历JSON中的每个键（即每个记录）
    for key, entry in data.items():
        # 清理origin_text字段中的换行符和逗号
        cleaned_text = entry['origin_text'].replace('\n', ' ').replace(',', ';')
        # 检查并修改标签
        label = 1 if entry['origin_label'] == 'legitimate' else 4
        # 准备要写入的数据
        row = {
            'origin_label': label,
            'origin_text': cleaned_text
        }
        # 写入处理后的数据到CSV
        writer.writerow(row)





