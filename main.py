#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liu shipeng
# datetime： 2023/6/12 11:52 
# ide： PyCharm
# description：主程序
import threading
import datetime
import time

import cv2
from loguru import logger
from yaml import safe_load

from yolox.data import CUS_CLASSES
from yolox.utils.visualize import vis
from model_utils import load_model, image_inference
from request_utils import get_image_url, request_api, scene_mapping, camera_info


# 总配置
args: dict


def submit_scene(infer_result, frame, request: bool = False):
    # 发生时间
    happened_time = str(datetime.datetime.now().replace(microsecond=0))

    if request:
        # 请求http，上传图像并获取url
        image_url = get_image_url(args.get('UserIp'), args.get('UserPort'), frame)

        # 遍历推理结果，有一种场景就请求一次
        norm_list = infer_result["norm"]
        for i in range(len(norm_list)):
            cur_object = norm_list[i]
            # 数据集中的类名
            cls_name = CUS_CLASSES[int(cur_object[-2])]
            if cls_name in scene_mapping:
                # 场景类型id
                scene_id = scene_mapping[cls_name]
                # 请求http，传输检测结果
                request_api(
                    "192.168.45.3" if args.get('VideoPath').endswith("mp4") else args.get('VideoPath'),
                    scene_id, happened_time, image_url, "liu", args.get('UserIp'), args.get('UserPort'))


# 主方法
def main():
    # 加载模型
    model = load_model(args.get('ModelConfig'), args.get('ModelWeights'))
    logger.info("检测模型加载完成.")

    # 加载视频
    if args.get('InputType') == "video":
        video_path = args.get('VideoPath')
        video = cv2.VideoCapture(video_path)
        logger.info("视频读取完成.")
    else:
        # video_path = f"rtsp://admin:Jx1234567.@192.168.70.201/Streaming/Channels/{camera_info.get(str(args.video_path))}/"
        video_path = f"rtsp://admin:Jx1234567.@192.168.70.201/Streaming/Channels/201/"
        video = cv2.VideoCapture(video_path)
        logger.info("连接摄像头成功,连接地址:{}".format(video_path))

    if args.get('Visualize'):
        # 保存的视频的文件名
        output_path = args.get("SavePath")
        # 获取视频的帧率、尺寸和编解码器
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编解码器，可以根据需要更改编解码器

        # 创建用于保存视频的VideoWriter对象
        video_writer = cv2.VideoWriter(output_path, codec, fps, (frame_width, frame_height))

    # 记录上次提交的时间
    last_submit_time = time.time()

    while video.isOpened():
        # 读取视频帧
        ret, frame = video.read()

        if not ret:
            logger.error("读取视频帧失败")
            break

        # 与上次提交的时间差如果小于0.5s，就不再进行检测，直接展示帧
        if time.time() - last_submit_time < 0.5:
            if args.get('Visualize'):
                cv2.imshow('Video', frame)
                video_writer.write(frame)
            else:
                pass
        else:
            infer_result = image_inference(model, frame)
            if infer_result is not None:
                # 获取帧图像
                ori = infer_result["ori"]
                frame = vis(frame, ori["bboxes"], ori["scores"], ori["cls"], ori["cls_conf"], CUS_CLASSES)

                # 提交
                thread = threading.Thread(target=submit_scene, args=(infer_result, frame, args.get('Request')))
                thread.daemon = True
                thread.start()

                # 更新上次提交时间
                last_submit_time = time.time()

            if args.get('Visualize'):
                cv2.imshow('Video', frame)
                video_writer.write(frame)

        # 按下q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    video.release()
    if args.get('Visualize'):
        video_writer.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    """
    远程启动与终止python进程
        启动
        ssh shipeng@192.168.1.53 "python /home/shipeng/thousand_eyes_vision/main.py"
        终止
        ssh shipeng@192.168.1.53 "pkill -f '/home/shipeng/thousand_eyes_vision/main.py'"

    本地命令行启动
        python main.py
    """
    config_path = './config.yaml'
    with open(config_path, mode='r', encoding='utf-8') as file:
        args = safe_load(file)
    main()
