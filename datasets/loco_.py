import os
import shutil

# 기존 폴더 및 파일 경로
base_dir = "/home/desk/patch_core_/datasets/mvtec_loco"
sub_dirs = ["breakfast_box/ground_truth", "juice_bottle/ground_truth", "pushpins/ground_truth", "screw_bag/ground_truth", "splicing_connectors/ground_truth"] 
subsub_dirs = ['logical_anomalies' , 'structural_anomalies']
# 새로운 파일 경로로 이동
for sub_dir in sub_dirs:
    full_sub_dir_path_1 = os.path.join(base_dir, sub_dir)
    for subsub_dir in subsub_dirs:
        full_sub_dir_path_2 = os.path.join(full_sub_dir_path_1, subsub_dir)
        
        folder_list = os.listdir(full_sub_dir_path_2)
        for folder_name in folder_list:
            old_path = os.path.join(full_sub_dir_path_2, folder_name)
            
            if os.path.exists(old_path) and os.path.isdir(old_path):
                for file_name in os.listdir(old_path):
                    file_old_path = os.path.join(old_path, file_name)
                    folder_names = f"{folder_name}.png"
                    file_new_path = os.path.join(full_sub_dir_path_2, f"{folder_names}")
                    
                    if os.path.isfile(file_old_path):
                        shutil.move(file_old_path, file_new_path)
                        print(f"Moved {file_old_path} to {file_new_path}")
                
                # 빈 디렉토리 삭제
                os.rmdir(old_path)
                print(f"Deleted directory {old_path}")
        # When Mask_gt has 2 or more image, It has error and can not make pixelwise auroc
                