# import nltk
# nltk.download('averaged_perceptron_tagger_eng')
# from unstructured.partition.auto import partition

# elements = partition("/home/ezdx/文档/code/crag/data/parse/data/1706.03762v7.pdf")

# print("\n\n".join([str(el) for el in elements]))

from unstructured.partition.pdf import partition_pdf

# 提取图像
elements = partition_pdf(
    filename="/home/ezdx/文档/code/crag/data/parse/data/1706.03762v7.pdf",  # PDF 文件路径
    strategy="hi_res",  # 使用高分辨率策略
    extract_element_types=["Image"],  # 指定提取图像
    extract_images_in_pdf = True,
    image_output_dir_path="/home/ezdx/文档/code/crag/data/extracted_images"  # 图像保存路径
)
print(elements)