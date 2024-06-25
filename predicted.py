import os
import subprocess
import configparser

# 读取配置文件
config = configparser.ConfigParser()
config_file = "mixConfig.ini"
config.read(config_file)

# 获取配置文件中的参数
PredictPath = config["predicted"]["PredictPath"]
ModelPath = config["predicted"]["ModelPath"]
InputDir = config["predicted"]["InputDir"]
OutputDir = config["predicted"]["OutputDir"]


def all_predict(predict_path, model_path, input_dir, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取输入目录中的所有 .jpg 文件
    input_files = [f for f in os.listdir(input_dir) if f.endswith(".jpg")]
    for file in input_files:
        print(f"Processing file: {file}")  # 显示目前正在处理的文件
        input_path = os.path.join(input_dir, file)  # 完整的输入文件路径
        output_path = os.path.join(output_dir, file)  # 完整的输出文件路径

        # 检查输出目录中是否已经存在相同名称的文件，若有，则跳过这个文件
        if os.path.exists(output_path):
            print(f"File {file} already exists in output directory. Skipping...")
            continue

        # 执行 "python predict.py -m model_path -i input_path -o output_path" 指令
        subprocess.run(
            [
                "python",
                predict_path,
                "-m",
                model_path,
                "-i",
                input_path,
                "-o",
                output_path,
            ],
            check=True,
        )


def predicted():
    print("PredictPath:", PredictPath)
    print("ModelPath:", ModelPath)
    print("InputDir:", InputDir)
    print("OutputDir:", OutputDir)

    all_predict(PredictPath, ModelPath, InputDir, OutputDir)
