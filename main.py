#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liu shipeng
# datetime： 2023/6/12 11:52 
# ide： PyCharm
# description：主程序
import threading
import time
import argparse
import datetime

import cv2
from loguru import logger

from yolox.data import CUS_CLASSES
from yolox.utils.visualize import vis
from model_utils import load_model, image_inference
from request_utils import get_image_url, request_api, scene_mapping


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_type", type=str, default="video", help="输入类型：video或camera")
    parser.add_argument("--video_path", type=str, default="E:/video/20230609_172203.mp4", help="摄像机ip地址如192.168.1.53，或视频文件路径")
    parser.add_argument("--user_ip", type=str, default="192.168.1.31", help="图片上传和接口请求ip")
    parser.add_argument("--user_port", type=str, default="8500", help="图片上传和接口请求port")
    parser.add_argument("--config", type=str, default="./exps/yolox_s.py", help="模型配置文件路径")
    parser.add_argument("--weights", type=str, default="./ckpts/1000eyes_3.pth", help="模型权重文件路径")
    parser.add_argument(
        "--vis",
        action="store_true",
        default=False,
        help="是否可视化并保存视频，启动命令不含该参数时为false，否则为true")
    parser.add_argument(
        "--request",
        action="store_true",
        default=False,
        help="检测到目标是否需要请求后端发送数据，启动命令不含该参数时为false，否则为true"
    )
    return parser


def submit_scene(infer_result, frame, request: bool = False):
    # 发生时间
    happened_time = str(datetime.datetime.now().replace(microsecond=0))
    # 获取帧图像
    ori = infer_result["ori"]
    frame = vis(frame, ori["bboxes"], ori["scores"], ori["cls"], ori["cls_conf"], CUS_CLASSES)

    if request:
        # 请求http，上传图像并获取url
        image_url = get_image_url(args.user_ip, args.user_port, frame)
        # print(image_url)
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
                    "192.168.45.3" if args.video_path.endswith("mp4") else args.video_path,
                    scene_id, happened_time, image_url, "liu", args.user_ip, args.user_port)


# 主方法
def main(args):
    # 解析args
    if args.input_type == "video":
        video_path = args.video_path
    else:
        video_path = f"rtsp://{args.video_path}/stream"

    # 加载模型、视频
    logger.info("模型加载中...")
    output_path = "example.mp4"

    model = load_model(args.config, args.weights)
    logger.info("检测模型加载完成.")

    video = cv2.VideoCapture(video_path)
    logger.info("视频读取完成.")

    if args.vis:
        # 获取视频的帧率、尺寸和编解码器
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编解码器，可以根据需要更改编解码器

        # 创建用于保存视频的VideoWriter对象
        video_writer = cv2.VideoWriter(output_path, codec, fps, (frame_width, frame_height))

        # 初始化帧率计算相关变量
        frame_count = 0
        start_time = time.time()

    while video.isOpened():
        # 读取视频帧
        ret, frame = video.read()

        if not ret:
            break

        if args.vis:
            # 计算帧率
            frame_count += 1
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = frame_count / elapsed_time

            # 在图像上显示帧率信息
            height, width, _ = frame.shape
            position = (int(0.02 * width), int(0.04 * height))  # 设置相对位置
            cv2.putText(frame, f"FPS: {fps:.2f}", position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        infer_result = image_inference(model, frame)
        if infer_result is not None:
            thread = threading.Thread(target=submit_scene, args=(infer_result, frame, args.request))
            thread.daemon = True
            thread.start()

        if args.vis:
            # 显示帧
            cv2.imshow('Video', frame)

            # 将帧写入输出视频文件
            video_writer.write(frame)

        # 按下q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if args.vis:
            # 更新开始时间
            start_time = end_time
            frame_count = 0

    # 释放资源
    video.release()
    if args.vis:
        video_writer.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    """
    远程启动与终止python进程
        启动
        ssh shipeng@192.168.1.53 "python /home/shipeng/thousand_eyes_vision/main.py --type camera -p 192.168.1.53"
        终止
        ssh shipeng@192.168.1.53 "pkill -f '/home/shipeng/thousand_eyes_vision/main.py'"

    本地命令行启动
        python main.py --type video -p E:/video/20230609_172203.mp4 --vis
    """
    args = make_parser().parse_args()
    main(args)
