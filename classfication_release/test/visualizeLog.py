# %%
import json
import os
import matplotlib.pyplot as plt

def load_log(file_path):
    """从日志文件中读取所有JSON行，返回包含每个epoch数据的列表"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"⚠️ skip line  {line.strip()}")
    return data


def extract_name(file_path):
    """根据路径提取模型名"""
    # 示例路径：/root/code/RMT/classfication_release/work_dirs/TokenNet/tokenGalerkin_t_30q/log.txt
    return os.path.basename(os.path.dirname(file_path))


def plot_metrics(log_files, metrics):
    plt.figure(figsize=(10, 6))

    for file_path in log_files:
        name = extract_name(file_path)
        data = load_log(file_path)
        epochs = [d['epoch'] for d in data]

        for metric in metrics:
            if metric not in data[0]:
                print(f"⚠️ file {name} does not contain {metric}")
                continue
            values = [d[metric] for d in data]
            plt.plot(epochs, values, label=f"{name} - {metric}")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Log Visualization")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("请输入日志文件路径（可输入多个，用空格分隔）：")
    log_files = input().strip().split()
    print("可选指标示例：train_loss, test_loss, test_acc1, test_acc5, train_lr")
    metrics = input("请输入要可视化的指标（可输入多个，用空格分隔）：").strip().split()
    plot_metrics(log_files, metrics)
