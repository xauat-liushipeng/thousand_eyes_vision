import time

import cv2
import requests

from loguru import logger

from yolox.data import CUS_CLASSES
from yolox.utils.visualize import vis
from model_utils import load_model, image_inference


# 检测到调用这个方法给java返回消息
def request_http(data):
    # 发送HTTP POST请求到后端
    url = 'http://localhost:8080/api/coal_block'
    response = requests.post(url, json=data)
    return response


# 主方法
def main():
    logger.info("System launch.")
    video_path = r"E:\video\20230609_172203.mp4"
    output_path = "example.mp4"

    model = load_model("./exps/yolox_s.py", "./ckpts/1000eyes_3.pth")
    logger.info("加载检测模型完成.")

    video = cv2.VideoCapture(video_path)
    logger.info("视频读取完成")

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
        if infer_result is None:
            cv2.imshow('Video', frame)
        else:
        # 绘制检测框和标签
            ori = infer_result["ori"]
            vis(frame, ori["bboxes"], ori["scores"], ori["cls"], ori["cls_conf"], CUS_CLASSES)

            # 显示帧
            cv2.imshow('Video', frame)

        # 将帧写入输出视频文件
        video_writer.write(frame)

        # 按下q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 更新开始时间
        start_time = end_time
        frame_count = 0

    video.release()
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
