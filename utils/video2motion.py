import os
import mediapipe
import pandas as pd
import cv2
import numpy as np


def get_resized_motion_per_video(video_path):
    
    keypoinys = 33
    clip_len = 16
    motion_x = np.empty((clip_len, keypoinys, 4), np.dtype('float32'))
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // clip_len
    
    current_frame = 0
    clip_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_frame % interval == 0:
            frame = frame[:, :, ::-1]    # BGR -> RGB
            results = mediapipe_pose.process(frame)
            if results.pose_landmarks:  # 如果有输出
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    motion_x[clip_idx, id, :] = [lm.x, lm.y, lm.z, lm.visibility]
            else:
                motion_x[clip_idx, :, :] = 0.0
                
            clip_idx += 1

        current_frame += 1
        
        if clip_idx == clip_len:
            break

    cap.release()
    
    return motion_x

def class_process(dir_path, dst_dir_path, class_name):
    class_path = os.path.join(dir_path, class_name)
    if not os.path.isdir(class_path):
        return

    dst_class_path = os.path.join(dst_dir_path, class_name)
    if not os.path.exists(dst_class_path):
        # os.mkdir(dst_class_path)                  # use in linxu
        os.makedirs(dst_class_path, exist_ok=True)  # use on windows

    for file_name in os.listdir(class_path):
        if '.avi' not in file_name:
            continue
        name, ext = os.path.splitext(file_name)
        video_file_path = os.path.join(class_path, file_name)
        motion = get_resized_motion_per_video(video_file_path)
        np.save(os.path.join(dst_class_path, name), motion)
        print(f'processing {video_file_path}')
        print('\n')

if __name__=="__main__":

    # 把avi视频文件解码，逐帧提取成图像文件

    # write your path
    dir_path = '../data/ucf101/UCF-101'
    dst_dir_path = '../data/ucf101/UCF101_npy'
    mediapipe_pose = mediapipe.solutions.pose.Pose()
    
    for class_name in os.listdir(dir_path):
        print(class_name)
        class_process(dir_path, dst_dir_path, class_name)