# coding=gbk


import os
import torch
import shap
import numpy as np
from torchvision import transforms
from PIL import Image

# ============================
# 1. 灰度图像预处理函数
# ============================
def preprocess_image(npy_path, device):
    """
    预处理灰度图像，将其转为模型的输入格式 [1, 1, H, W]。
    """
    preprocess = transforms.Compose([
        transforms.ToTensor(),  # 转为 Tensor 格式
        transforms.Normalize(mean=[0.456], std=[0.224])  # 标准化
    ])

    # 加载 NumPy 文件并预处理
    image = np.load(npy_path)  # 假设灰度图为单通道
    input_tensor = preprocess(image).unsqueeze(0).float().to(device)  # 转为 [1, 1, H, W]
    return input_tensor, image
    
def preprocess_image_1(npy, device):
    """
    预处理灰度图像，将其转为模型的输入格式 [1, 1, H, W]。
    """
    preprocess = transforms.Compose([
        transforms.ToTensor(),  # 转为 Tensor 格式
        transforms.Normalize(mean=[0.456], std=[0.224])  # 标准化
    ])

    # 加载 NumPy 文件并预处理
    image = npy  # 假设灰度图为单通道
    input_tensor = preprocess(image).unsqueeze(0).float().to(device)  # 转为 [1, 1, H, W]
    return input_tensor, image


# ============================
# 2. 计算单个模型的 SHAP 值
# ============================
def compute_shap_values(model, input_tensor, background_tensor):
    """
    计算单个模型的 SHAP 值。
    """
    explainer_1 = shap.PermutationExplainer(model, background_tensor, max_evals=513)
    shap_values_1 = explainer_1.shap_values(input_tensor)
    return shap_values_1[0]  # 假设只处理单通道灰度图


# ============================
# 3. 融合多个模型的 SHAP 结果
# ============================
def fuse_shap_values(shap_values_list):
    """
    融合多个模型的 SHAP 结果（按平均值融合）。
    """
    fused_shap = np.mean(shap_values_list, axis=0)  # 融合所有 SHAP 值
    return fused_shap


# ============================
# 4. 保存 SHAP 热图
# ============================
def save_heatmap(shap_values, original_image, save_path):
    """
    保存 SHAP 热图到指定路径。
    """
    heatmap = shap_values[0, 0]  # 提取 SHAP 值 (H, W)
    plt.figure(figsize=(8, 8))
    plt.imshow(original_image, cmap='gray', alpha=0.6)  # 显示原图
    plt.imshow(heatmap, cmap='jet', alpha=0.4)          # 叠加热图
    plt.axis('off')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"热图已保存到: {save_path}")
    
# ============================
# 5. 保存 SHAP 瀑布图
# ============================
def save_waterfall_plot(shap_values, save_path):
    """
    保存 SHAP 瀑布图到指定路径。
    """
    shap.summary_plot(shap_values, feature_names=["Pixel Contribution"], plot_type="waterfall", show=False)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"瀑布图已保存到: {save_path}")


# ============================
# 6. 主逻辑：处理多个模型并保存结果
# ============================
def process_multiple_models(models, input_image_path, background_image, heatmap_save_path, waterfall_save_path, device):
    """
    对多个模型计算 SHAP 值，融合结果并保存热图和瀑布图。
    """
    # 加载并预处理输入图像
    input_tensor, original_image = preprocess_image(input_image_path, device)

    # 加载并预处理背景图像
    background_tensor, _ = preprocess_image_1(background_image, device)

    # 计算每个模型的 SHAP 值
    shap_values_list = []
    for i, model in enumerate(models):
        model.to(device)
        model.eval()  # 确保模型处于评估模式
        print(f"计算模型 {i} 的 SHAP 值...")
        shap_values = compute_shap_values(model, input_tensor, background_tensor)
        shap_values_list.append(shap_values)

    # 融合 SHAP 结果
    print("融合多个模型的 SHAP 值...")
    fused_shap_values = fuse_shap_values(shap_values_list)

    # 保存融合后的热图
    save_heatmap(fused_shap_values, original_image, heatmap_save_path)

    # 保存融合后的瀑布图
    save_waterfall_plot(fused_shap_values, waterfall_save_path)

# ============================
# 7. 示例调用
# ============================
if __name__ == "__main__":
    # 模型列表（假设已加载多个模型）
    model_paths = [
        '/home/wangzeyu/ENAS_new_code/modeltest/0.h5',
        '/home/wangzeyu/ENAS_new_code/modeltest/1.h5',
        '/home/wangzeyu/ENAS_new_code/modeltest/2.h5',
        '/home/wangzeyu/ENAS_new_code/modeltest/3.h5',
        '/home/wangzeyu/ENAS_new_code/modeltest/4.h5',
        '/home/wangzeyu/ENAS_new_code/modeltest/5.h5',
        '/home/wangzeyu/ENAS_new_code/modeltest/6.h5'
    ]
    models = [torch.load(path) for path in model_paths]

    # 输入图像和背景图像路径
    input_image_path = "/home/wangzeyu/ENAS_new_code/local_2D_mix_5_1/val/1/RADCURE-0169-111.npy"        # 输入图像（.npy 格式）
    background_image = np.zeros((1,256,256))

    # 保存路径
    heatmap_save_path = "/home/wangzeyu/ENAS_new_code/pictrues_1/shap_results/heatmap.png"
    waterfall_save_path = "/home/wangzeyu/ENAS_new_code/pictrues_1/shap_results/waterfall.png"

    # 指定计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 运行主逻辑
    process_multiple_models(models, input_image_path, background_image, heatmap_save_path, waterfall_save_path, device)