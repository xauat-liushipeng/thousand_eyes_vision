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
# args: dict


def submit_scene(infer_result, frame, request, user_ip, user_port, video_path):
    # 发生时间
    happened_time = str(datetime.datetime.now().replace(microsecond=0))

    if request:
        # 请求http，上传图像并获取url
        image_url = get_image_url(user_ip, user_port, frame)

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
                    "192.168.45.3" if video_path.endswith("mp4") else video_path,
                    scene_id, happened_time, image_url, "liu", user_ip, user_port)


# 主方法
def main(args):
    # 加载模型
    model = load_model(args.get('ModelConfig'), args.get('ModelWeights'), cuda=args.get('UseCuda'))
    logger.info("检测模型加载完成.")

    # 加载视频
    if args.get('InputType') == "video":
        video_path = args.get('VideoPath')
        video = cv2.VideoCapture(video_path)
        logger.info("视频读取完成.")
    else:
        video_path = f"rtsp://admin:Jx1234567.@192.168.70.201/Streaming/Channels/{camera_info.get(str(args.get('VideoPath')))}/"
        video = cv2.VideoCapture(video_path)
        logger.info("连接摄像头成功,连接地址:{}".format(video_path))

    # 记录上次提交的时间
    last_submit_time = time.time()

    if args.get("Visualize"):
        window_name = args.get("VideoPath")
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 450)  # 设置宽度为 800，高度为 600

    while video.isOpened():
        # 读取视频帧
        ret, frame = video.read()

        if not ret:
            logger.error("读取视频帧失败")
            break

        # 与上次提交的时间差如果小于0.5s，就不再进行检测，直接展示帧
        if time.time() - last_submit_time < 0.5:
            if args.get('Visualize'):
                cv2.imshow(window_name, frame)
            else:
                pass
        else:
            infer_result = image_inference(model, frame)
            if infer_result is not None:
                # 获取帧图像
                ori = infer_result["ori"]
                frame = vis(frame, ori["bboxes"], ori["scores"], ori["cls"], ori["cls_conf"], CUS_CLASSES)

                # 提交
                thread = threading.Thread(
                    target=submit_scene,
                    args=(infer_result, frame, args.get('Request'), args.get('UserIp'), args.get('UserPort'), args.get('VideoPath')))
                thread.daemon = True
                thread.start()

                # 更新上次提交时间
                last_submit_time = time.time()

        if args.get('Visualize'):
            cv2.imshow(window_name, frame)

        # 按下q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    video.release()
    if args.get('Visualize'):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    """
    远程启动与终止python进程
        启动
        ssh hwpq5:971215@192.168.1.53 "python D:/jiuzhoucode/thousand_eyes_vision/main.py"
        终止
        ssh shipeng@192.168.1.53 "pkill -f '/home/shipeng/thousand_eyes_vision/main.py'"

    本地命令行启动
        python main.py
    """
    config_path = 'D:/jiuzhoucode/thousand_eyes_vision/config.yaml'
    with open(config_path, mode='r', encoding='utf-8') as file:
        cfg = safe_load(file)
    main(cfg)
