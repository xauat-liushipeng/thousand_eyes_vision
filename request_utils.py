#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liu shipeng
# datetime： 2023/6/12 14:22 
# ide： PyCharm
# description：http请求工具
import io

import requests
from PIL import Image
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
    result = r.json()
    return result['data']['saveAddr']


# 检测到调用这个方法给java返回消息
def request_api(camera_ip, scene_id, happen_time, violation_img_url, violation_worker, user_ip, user_port):
    request_content = dict()
    request_content["cameraIp"] = camera_ip
    request_content["sceneId"] = scene_id
    request_content["happenedTime"] = happen_time
    request_content["violationWorker"] = violation_worker
    request_content["violationImg"] = violation_img_url

    # 发送HTTP POST请求到后端
    url = 'http://' + user_ip + ':' + user_port + '/system/scene/monitorrecord/add'
    response = requests.post(url, json=request_content)
    return response


# 场景序号映射
scene_mapping = {
    "unloaded": 2,
    "coal_lump": 4
}
