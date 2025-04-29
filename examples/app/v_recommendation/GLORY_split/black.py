import os
import subprocess


def format_py_files_with_black(directory):
    try:
        # 检查 black 是否已经安装
        result = subprocess.run(
            ['black', '--version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            print("black 未安装，请先安装 black: pip install black")
            return

        # 调用 black 格式化指定目录下的所有 .py 文件
        command = ['black', directory]
        process = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if process.returncode == 0:
            print("所有 .py 文件已使用 black 格式化完成。")
            print(process.stdout)
        else:
            print("格式化过程中出现错误：")
            print(process.stderr)
    except Exception as e:
        print(f"执行过程中出现错误: {e}")


if __name__ == "__main__":
    # 指定要处理的目录
    target_directory = '.'  # 当前目录，你可以根据需要修改
    format_py_files_with_black(target_directory)
