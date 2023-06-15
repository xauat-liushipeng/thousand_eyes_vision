#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import torch

from yolox.exp import get_exp
from yolox.data import CUS_CLASSES
from yolox.utils import postprocess
from yolox.data.data_augment import ValTransform


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = None
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        if output is None:
            return None
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        # 保留原始结果
        ori = dict()
        ori["bboxes"] = bboxes
        ori["scores"] = scores
        ori["cls"] = cls
        ori["cls_conf"] = cls_conf

        # 转换成list
        bboxes = bboxes.tolist()
        cls = cls.tolist()
        scores = scores.tolist()

        norm = []
        for i in range(len(bboxes)):
            if scores[i] >= cls_conf:
                bboxes[i].append(cls[i])
                bboxes[i].append(scores[i])
                norm.append(bboxes[i])


        # 最终结果json
        result = dict()
        result["norm"] = norm
        result["ori"] = ori

        return result


def image_inference(predictor, image):
    outputs, img_info = predictor.inference(image)

    # 输出格式为：[[x0, y0, x1, y1, cls_id, score]]
    result = predictor.visual(outputs[0], img_info, predictor.confthre)
    return result


def load_model(exp_path, ckpt_path, fp16=False, conf=0.3, nms=0.45, inp_size=416):
    # 模型配置文件
    exp = get_exp(exp_path)
    # 模型权重文件
    ckpt_file = ckpt_path

    # 推理参数
    exp.test_conf = conf
    exp.nmsthre = nms
    exp.test_size = (inp_size, inp_size)

    # 设备类型
    if torch.cuda.is_available():
        device = "gpu"
    else:
        device = "cpu"

    # 从模型配置读取模型架构
    model = exp.get_model()

    if device == "gpu":
        model.cuda()
        if fp16:
            model.half()  # to FP16
    model.eval()  # 模型选择推理模式

    ckpt = torch.load(ckpt_file, "cuda" if device == "gpu" else "cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])

    return Predictor(model, exp, CUS_CLASSES, None, device, fp16, False)
