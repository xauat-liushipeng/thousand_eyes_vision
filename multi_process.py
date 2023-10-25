#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liu shipeng
# datetime： 2023/6/16 15:55 
# ide： PyCharm
# description：同时启动多个摄像头检测
import sys
from threading import Thread

from yaml import safe_load

from main import main

def child_thread(config: tuple):
    main(dict(config))


ips = [
    "192.168.45.16",  # 101皮带
    "192.168.45.4",  # 二部皮带机头
    # "192.168.45.18",  # 二部皮带机尾
    "192.168.45.5",  # 三部皮带机头
    "192.168.45.6",  # 四部皮带机头
    "192.168.45.7",  # 4209转载机头
    "192.168.45.3",  # 副斜井
    "192.168.30.28",  # 副斜井

    # "192.168.1.214",  # 4304回采工作面-转载机头
    "192.168.70.31",  # 主斜井
]


for ip in ips:
    with open("D:/jiuzhoucode/thousand_eyes_vision/config.yaml", mode='r', encoding='utf-8') as f:
        cfg = safe_load(f)
    cfg["VideoPath"] = ip
    cfg["SavePath"] = cfg["SavePath"].replace("example", ip.replace(".", "-"))
    cfg = tuple(cfg.items())

    thread = Thread(target=child_thread, args=(cfg, ))
    thread.start()

if 0xFF == ord('q'):
    sys.exit()
