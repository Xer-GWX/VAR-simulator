from PIL import Image
Image.MAX_IMAGE_PIXELS = 933120000

def merge_images(image_paths, output_path):
    # 打开所有图片
    images = [Image.open(img) for img in image_paths]

    # 确定合并后图片的总宽度和最大高度
    widths, heights = zip(*(img.size for img in images))
    
    total_width = sum(widths)  # 总宽度为所有图片宽度之和
    max_height = max(heights)  # 高度取所有图片中的最大值

    # 创建一个新的空白图片，大小为总宽度和最大高度
    new_img = Image.new('RGB', (total_width, max_height))

    # 将每张图片按顺序粘贴到新图片上
    x_offset = 0
    for img in images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width

    # 保存合并后的图片
    new_img.save(output_path, quality=100)  # quality=100保持最高质量

# 使用示例
image_files = ['./FLOPs_prefill_depth16.png', './FLOPs_prefill_depth16_sum.png', './Total_FLOPs_by_Stage.png']  # 替换为你本地图片的路径
output_image = './merged_image.png'
merge_images(image_files, output_image)
