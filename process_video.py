import cv2
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# 裁剪去指定边框
def crop_black_border(frame, up, bottom, left, right):
    # 裁剪掉指定边框
    height, width = frame.shape[:2]
    cropped_frame = frame[up:height-bottom, left:width-right]
    
    return cropped_frame

# 提取视频帧
def extract_frames(video_path, output_dir, interval=30, up=0, bottom=0, left=0, right=0):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频的帧数和fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    # 确保目标目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 设置初始帧数
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 每隔interval帧，抽取一帧
        if frame_num % interval == 0:
            # 裁剪指定边框
            cropped_frame = crop_black_border(frame, up, bottom, left, right)
            
            # 保存帧图像
            output_filename = os.path.join(output_dir, f"frame{frame_num:05d}.png")
            cv2.imwrite(output_filename, cropped_frame)
        
        frame_num += 1

    cap.release()

# 示例使用
# video_paths = [
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P27_刘树元.mp4',
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P28_史建.mp4',
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P29_周新梅.mp4',
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P30_张亚林.mp4',
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P31_张立群.mp4',
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P32_杜以伟.mp4',
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P33_王兴军1.mp4',
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P33_王兴军2.mp4',
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P34_王培满.mp4',
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P35_袁银锁.mp4',
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P36_贾开朗.mp4',
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P37_赵文兰.mp4',
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P38_赵霞.mp4',
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P42_沈烈宝1.mp4',
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P42_沈烈宝2.mp4',
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P43_王亦亮.mp4',
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P44_董云生1.mp4',
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P44_董云生2.mp4'
# ]

# video_paths = [
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P40_卢希宏.mp4'
# ]
# video_paths = [
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P41_李瑞锋.mp4'
# ]

# video_paths = [
    # '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P23_张连清.mp4',
    # '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P17_陈长君.mp4',
    # '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P18_陈维文.mp4',
    # '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P19_刁振先.mp4',
    # '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P20_贾开朗.mp4',
    # '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P22_牛培生.mp4',
    # '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P26_张保东.mp4',
    
# ]

# video_paths = [
    # '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P15_杨红美1.mp4',
    # '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P15_杨红美2.mp4',
# ]

# video_paths = [
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P14_徐发远1.mp4',
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P14_徐发远2.mp4',
    # '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P14_徐发远3.mp4',
    # '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P14_徐发远4.mp4'
# ]
# video_paths = [
    # '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P8_刘德政1.mp4',
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P8_刘德政3.mp4',
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P8_刘德政4.mp4',
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P8_刘德政5.mp4'
# ]

# video_paths = [
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P6_李淑玲1.mp4',
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P6_李淑玲2.mp4',
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P6_李淑玲3.mp4'
# ]

# video_paths = [
    # '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P5_李干兵4.mp4',
    # '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P39_崔京忠.mp4',
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P21_蒋建华.mp4',
# ]

# video_paths = [
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P24_王传林.mp4',
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P25_李翠兰.mp4',
# ]

# video_paths = [
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P1_陈保华.mp4',
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P4_韩云波.mp4',
# ]
# video_paths = [
#     '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P45_程先杰.mp4'
# ]

video_paths = [
    '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P46_李敏.mp4',
    '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P47_于新华1.mp4',
    '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P47_于新华2.mp4',
    '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P47_于新华3.mp4',
    '/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/patient_raw_video/P47_于新华4.mp4'
]

def process_video(path):
    filename = os.path.basename(path).replace('.mp4', '')
    output_dir = f'/home/ren2/data2/mengya/mengya_dataset/ESD_Qilu_Bleeding/ESD_Bleeding_Dataset/images_extract_1fps/{filename}'  # 输出图片目录
    # extract_frames(path, output_dir, interval=30, up=26, bottom=26, left=466, right=40)
    # extract_frames(path, output_dir, interval=30, up=38, bottom=38, left=698, right=60) # P39 P5 P21
    # extract_frames(path, output_dir, interval=30, up=24, bottom=98, left=348, right=324) # P40
    # extract_frames(path, output_dir, interval=30f, up=8, bottom=8, left=386, right=8) # P41
    # extract_frames(path, output_dir, interval=30, up=26, bottom=26, left=466, right=40) # P23, P17, 18, 19, 20, 22, 26
    # extract_frames(path, output_dir, interval=30, up=0, bottom=0, left=658, right=20) # P15 P14 P6
    # extract_frames(path, output_dir, interval=30, up=58, bottom=56, left=42, right=652) # P8
    # extract_frames(path, output_dir, interval=30, up=46, bottom=46, left=458, right=34) # P24 P25
    # extract_frames(path, output_dir, interval=30, up=68, bottom=70, left=686, right=52) # P1 P4
    # extract_frames(path, output_dir, interval=30, up=26, bottom=26, left=468, right=42) # P45
    extract_frames(path, output_dir, interval=30, up=0, bottom=0, left=658, right=20) # P46 (sometimes NBI), P47-1, 2, 3, 4

# 使用线程池并行处理视频
num_threads = 20  # 你可以根据需要调整线程数
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    list(tqdm(executor.map(process_video, video_paths), total=len(video_paths), desc="Overall Progress"))