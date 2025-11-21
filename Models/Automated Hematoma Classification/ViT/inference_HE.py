import torch
import torch.nn as nn
from torchvision import transforms
import os
from PIL import Image
import classic_models  # 替换为定义模型的正确模块
import logging

# 设置日志记录
logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

def log_and_print(message):
    # 打印到终端
    print(message)
    # 记录到日志文件
    logging.info(message)

# 加载模型及其权重的函数
def load_model(model_name, num_classes, weights_path, device):
    model = classic_models.find_model_using_name(model_name, num_classes=num_classes).to(device)  # 根据模型名称和类别数获取模型，并将其移到指定的设备上（如 GPU）
    model.load_state_dict(torch.load(weights_path, map_location=device))  # 加载指定路径下的模型权重
    model.eval()  # 将模型设置为评估模式
    return model  # 返回加载好的模型

# 预测函数
def predict(model, data, device):
    model.eval()  # 确保模型处于评估模式
    with torch.no_grad():  # 在不计算梯度的上下文中，减少内存消耗
        data = data.to(device)  # 将数据移到指定设备（如 GPU）
        outputs = model(data)  # 模型前向传播，获取输出
        soft_max_predict = torch.softmax(outputs, 1)  # 计算 softmax 预测
        _, predicted = torch.max(soft_max_predict, 1)  # 获取预测概率最高的类别
        predicted = 1 - predicted  # 对预测结果进行标签互换
    return predicted, soft_max_predict  # 返回调整后的预测类别和 softmax 预测值

def main(model, num_classes, weights, data_path):
    device = "cuda"  # 指定设备为 GPU
    # 加载模型
    model = load_model(model, num_classes, weights, device)

    # 定义数据预处理的转换（例如调整大小和归一化）
    transform = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 加载图像数据
    image = Image.open(data_path).convert('RGB')  # 打开图像并确保是 RGB 格式
    data = transform(image).unsqueeze(0)  # 应用预处理并添加批次维度

    # 进行预测
    predicted_class, soft_max_predict = predict(model, data, device)

    # 获取样本名称
    img_name = os.path.basename(data_path)

    # 真实标签 (需要根据具体情况设置)
    true_label = 1  # 示例值，根据实际情况修改

    # 打印和记录预测信息
    message = (
        f"样本名称: {img_name}\n"
        f"真实标签: {true_label}\n"
        f"预测概率值: {soft_max_predict.squeeze().tolist()}\n"
        f"预测标签: {predicted_class.item()}\n"
    )
    log_and_print(message)

    return predicted_class.item()

# 定义路径
paths = "data"
weights = "vit.pth"
count_0 = 0
count_1 = 0
total_files = 0

# 遍历指定路径下的文件
for i in os.listdir(paths):
    # 对每个文件调用 main 函数进行预测
    predicted_class = main(model="vision_transformer",
         data_path=os.path.join(paths, i),
         weights=weights,
         num_classes=2)
    
    # 统计预测为 0 和 1 的数量
    if predicted_class == 0:
        count_0 += 1
    else:
        count_1 += 1
    
    total_files += 1

# 打印结果
print(f"NHE的预测数量: {count_0}")
print(f"HE的预测数量: {count_1}")
print(f"总共预测的文件数: {total_files}")
acc0 = count_0 / total_files
acc1 = count_1 / total_files
# print(f"NHE推理acc为{acc0}")
print(f"HE推理acc为{acc1}")

# 记录到日志文件
log_and_print(f"NHE的预测数量: {count_0}")
log_and_print(f"HE的预测数量: {count_1}")
log_and_print(f"总共预测的文件数: {total_files}")
log_and_print(f"HE推理acc为{acc1}")
