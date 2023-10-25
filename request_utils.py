#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liu shipeng
# datetime： 2023/6/12 14:22 
# ide： PyCharm
# description：http请求工具
import io

import requests
from PIL import Image
from loguru import logger
from requests_toolbelt import MultipartEncoder


# ndarray 转字节
def get_image_handler(img_arr):
    an_image = Image.fromarray(img_arr)
    image_file = io.BytesIO()
    an_image.save(image_file, format='png')
    imagedata = image_file.getvalue()
    return imagedata


# 上传图片，获取url
def get_image_url(ip: str, port: str, image):
    image_handler = get_image_handler(image)

    url = 'http://' + ip + ':' + port + '/common/common/uploads'

    header = {}  # 请求头
    files = {
        'filename': "0.jpg",
        'Content-Disposition': 'form-data;',
        'Content-Type': 'multipart/form-data',
        'uploadMark': 'user',
        'file': ("0.jpg", image_handler, 'image/jpeg')
    }
    form_data = MultipartEncoder(files)  # 格式转换
    header['content-type'] = form_data.content_type
    r = requests.post(url, data=form_data, headers=header)  # 请求
    logger.success("图片上传地址".format(r.text))
    result = r.json()
    return result['data']['saveAddr']


# 检测到调用这个方法给java返回消息
def request_api(camera_ip, scene_id, happen_time, violation_img_url, violation_worker, user_ip, user_port):
    request_content = dict()
    # 相机ip
    request_content["cameraIp"] = camera_ip
    # 检测到的目标类型——对应项目中的场景类型id
    request_content["sceneId"] = scene_id

    request_content["happenedTime"] = happen_time
    request_content["violationWorker"] = violation_worker
    request_content["violationImg"] = violation_img_url

    # 发送HTTP POST请求到后端
    url = 'http://' + user_ip + ':' + user_port + '/system/scene/monitorrecord/add'
    response = requests.post(url, json=request_content)
    logger.success("请求结果：{}".format(response.text))
    return response


# 检测类型与项目场景序号映射
scene_mapping = {
    "unloaded": 2,
    "coal_lump": 4
}


camera_info = {
    "192.168.45.3": "401",
    "192.168.30.28": "301",
    "192.168.45.4": "901",
    "192.168.45.5": "1001",
    "192.168.45.6": "1101",
    "192.168.45.7": "1201",
    "192.168.45.8": "1301",
    "192.168.45.9": "2701",
    "192.168.45.10": "2801",
    "192.168.45.11": "2601",
    "192.168.45.12": "1701",
    "192.168.45.13": "1801",
    "192.168.45.14": "3601",
    "192.168.45.15": "2101",
    "192.168.45.16": "3501",
    # "192.168.45.17": "",
    "192.168.45.18": "3801",

    # "192.168.1.214": "2901",
    "192.168.70.31": "201"
}
