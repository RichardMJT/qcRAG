# import spacy


# nlp = spacy.load("zh_core_web_trf")


# text = '广东省广州市'
# doc = nlp(text)

# for token in doc:
#     print(token.text)
# from pathlib import Path
# import os
# temp_dir = os.path.join(Path('/home/opt').parent, 'temp')
# print(temp_dir)

import spacy
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# 加载 spaCy 模型
nlp = spacy.load("zh_core_web_lg")

# 示例文本列表
all_text = [
    "不必说碧绿的菜畦，光滑的石井栏，高大的皂荚树，紫红的桑葚；也不必说鸣蝉在树叶里长吟，肥胖的黄蜂伏在菜花上，轻捷的叫天子（云雀）忽然从草间直窜向云霄里去了。单是周围的短短的泥墙根一带，就有无限趣味。油蛉在这里低唱， 蟋蟀们在这里弹琴。翻开断砖来，有时会遇见蜈蚣；还有斑蝥，倘若用手指按住它的脊梁，便会“啪”的一声，从后窍喷出一阵烟雾。何首乌藤和木莲藤缠络着，木莲有莲房一般的果实，何首乌有臃肿的根。有人说，何首乌根是有像人形的，吃了便可以成仙，我于是常常拔它起来，牵连不断地拔起来，也曾因此弄坏了泥墙，却从来没有见过有一块根像人样。如果不怕刺，还可以摘到覆盆子，像小珊瑚珠攒成的小球，又酸又甜，色味都比桑葚要好得远。",
    "不必说碧绿的菜畦，光滑的石井栏，高大的皂荚树，紫红的桑葚；也不必说鸣蝉在树叶里长吟，肥胖的黄蜂伏在菜花上，轻捷的叫天子（云雀）忽然从草间直窜向云霄里去了。单是周围的短短的泥墙根一带，就有无限趣味。油蛉在这里低唱， 蟋蟀们在这里弹琴。翻开断砖来，有时会遇见蜈蚣；还有斑蝥，倘若用手指按住它的脊梁，便会“啪”的一声，从后窍喷出一阵烟雾。何首乌藤和木莲藤缠络着，木莲有莲房一般的果实，何首乌有臃肿的根。有人说，何首乌根是有像人形的，吃了便可以成仙，我于是常常拔它起来，牵连不断地拔起来，也曾因此弄坏了泥墙，却从来没有见过有一块根像人样。如果不怕刺，还可以摘到覆盆子，像小珊瑚珠攒成的小球，又酸又甜，色味都比桑葚要好得远。 "
]

# 定义处理单个文本的函数
def process_text(text):
    # doc = nlp(text)
    # return [token.text for token in doc]  # 返回分词结果
    return ['111']

if __name__ == "__main__":
    # 设置进程池的大小
    num_workers = 2
    # num_workers = max(cpu_count(), len(all_text))  # 使用 CPU 核心数或文本数量的较小值
    print(f"Using {num_workers} worker processes.")

    # 创建进程池
    with Pool(processes=num_workers) as pool:
        # 使用 tqdm 显示进度条
        results = list(tqdm(pool.imap(process_text, all_text), total=len(all_text)))

    # 打印结果
    for result in results:
        print(result)

# import multiprocessing
# import time

# # 定义一个简单的任务函数
# def worker(name):
#     print(f"进程 {name} 开始执行")
#     time.sleep(2)  # 模拟耗时任务
#     print(f"进程 {name} 执行完毕")

# # 主程序
# if __name__ == "__main__":
#     # 创建一个进程池，指定最大进程数为 3
#     with multiprocessing.Pool(processes=3) as pool:
#         # 启动多个进程，执行 worker 函数
#         for i in range(5):
#             pool.apply_async(worker, args=(i,))  # 异步执行任务

#         # 关闭进程池，等待所有进程完成
#         pool.close()
#         pool.join()

#     print("所有进程执行完成")