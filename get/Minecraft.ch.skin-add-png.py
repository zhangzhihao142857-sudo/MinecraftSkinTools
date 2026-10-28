import os
import shutil

# C:\MCLDownload\Game\.minecraft\assets\skins


def process_skin_files(root_dir):
    """
    1. 遍历指定目录下所有子文件夹，为无.png后缀的文件添加.png后缀
    2. 在根目录下创建.player_skin文件夹
    3. 将所有.png文件复制到.player_skin文件夹（同名文件跳过不复制）

    Args:
        root_dir (str): 要处理的根目录路径
    """
    # 检查根目录是否存在
    if not os.path.exists(root_dir):
        print(f"错误：目录 {root_dir} 不存在！")
        return

    # 定义.player_skin文件夹路径（在原根目录下创建）
    skin_target_dir = os.path.join(root_dir, ".player_skin")  # D:/root_dir/.player_skin

    # 创建.player_skin文件夹（不存在则创建）
    if not os.path.exists(skin_target_dir):
        os.makedirs(skin_target_dir)
        print(f"已创建文件夹：{skin_target_dir}")
    else:
        print(f"文件夹 {skin_target_dir} 已存在，无需重复创建")

    # 递归遍历根目录下所有子目录和文件
    for current_dir, subdirs, files in os.walk(root_dir):
        # 跳过.player_skin文件夹本身，避免重复处理
        if current_dir == skin_target_dir:
            continue

        for file_name in files:
            # 拼接文件原始完整路径
            old_file_path = os.path.join(current_dir, file_name)

            # 跳过文件夹（仅处理文件）
            if os.path.isdir(old_file_path):
                continue

            # 步骤1：为无.png后缀的文件添加后缀
            new_file_name = file_name
            if not file_name.lower().endswith(".png"):
                new_file_name = file_name + ".png"
                new_file_path = os.path.join(current_dir, new_file_name)

                # 防止重命名时目标文件已存在
                if os.path.exists(new_file_path):
                    print(f"警告：{new_file_path} 已存在，跳过添加后缀")
                else:
                    try:
                        os.rename(old_file_path, new_file_path)
                        print(f"已添加后缀：{old_file_path} -> {new_file_path}")
                        # 更新文件路径为添加后缀后的路径，用于后续复制
                        old_file_path = new_file_path
                    except Exception as e:
                        print(f"添加后缀失败 {old_file_path}：{str(e)}")
                        continue  # 后缀添加失败则跳过复制

            # 步骤2：将.png文件复制到.player_skin文件夹（同名文件跳过）
            if new_file_name.lower().endswith(".png"):
                # 拼接目标复制路径
                dest_file_path = os.path.join(skin_target_dir, new_file_name)

                # 核心判断：目标文件已存在则跳过复制
                if os.path.exists(dest_file_path):
                    print(f"跳过：{dest_file_path} 已存在，不重复复制")
                    continue

                try:
                    # 使用copy2保留文件元数据（创建/修改时间等）
                    shutil.copy2(old_file_path, dest_file_path)
                    print(f"已复制文件：{old_file_path} -> {dest_file_path}")
                except Exception as e:
                    print(f"复制文件失败 {old_file_path}：{str(e)}")


# 手动输入目标文件夹路径（自动去除首尾可能误输的单/双引号）
target_root = input("请输入目标文件夹路径：").strip("\"'")

# 执行主函数
if __name__ == "__main__":
    process_skin_files(target_root)
