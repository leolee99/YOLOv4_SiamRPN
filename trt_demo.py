from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import sys
from PIL import Image, ImageDraw,ImageFont
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
sys.path.append('./pytorch-YOLOv4/')
from tool.utils import *
import datetime

import os
import argparse

#import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')

class Yolov4_tiny(object):

    def __init__(self,model_path):

        # load model
        trt_logger = trt.Logger()  # This logger is required to build an engine
        runtime = trt.Runtime(trt_logger)
        f = open(model_path, "rb")
        self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.context.set_binding_shape(0, (1, 3, 416,416))

    def Inference(self,img,height,width):

        # Simple helper data class that's a little nicer to use than a 2-tuple.
        class HostDeviceMem(object):
            def __init__(self, host_mem, device_mem):
                self.host = host_mem
                self.device = device_mem

            def __str__(self):
                return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

            def __repr__(self):
                return self.__str__()

        # Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
        def allocate_buffers(engine, batch_size):
            inputs = []
            outputs = []
            bindings = []
            stream = cuda.Stream()
            for binding in engine:

                size = trt.volume(engine.get_binding_shape(binding)) * batch_size
                dims = engine.get_binding_shape(binding)
                
                # in case batch dimension is -1 (dynamic)
                if dims[0] < 0:
                    size *= -1
                
                dtype = trt.nptype(engine.get_binding_dtype(binding))
                # Allocate host and device buffers
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                # Append the device buffer to device bindings.
                bindings.append(int(device_mem))
                # Append to the appropriate list.
                if engine.binding_is_input(binding):
                    inputs.append(HostDeviceMem(host_mem, device_mem))
                else:
                    outputs.append(HostDeviceMem(host_mem, device_mem))
            return inputs, outputs, bindings, stream


        # This function is generalized for multiple inputs/outputs.
        # inputs and outputs are expected to be lists of HostDeviceMem objects.
        def do_inference(context, bindings, inputs, outputs, stream):
            # Transfer input data to the GPU.
            [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
            # Run inference.
            context.execute_async(bindings=bindings, stream_handle=stream.handle)
            # Transfer predictions back from the GPU.
            [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
            # Synchronize the stream
            stream.synchronize()
            # Return only the host outputs.
            return [out.host for out in outputs]

        image_cv = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        blob = np.expand_dims(image_cv[:, :, (2, 1, 0)].transpose(2, 0, 1), axis=0).astype("float32") 
        blob /= 255.0
        blob = np.ascontiguousarray(blob)

        self.buffers = allocate_buffers(self.engine,1)
        inputs, outputs, bindings, stream = self.buffers
        inputs[0].host = blob

        begin = datetime.datetime.now()
        trt_outputs = do_inference(self.context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        end = datetime.datetime.now()
        print('yolov4-tiny inference time:' ,end - begin)

        trt_outputs[0] = trt_outputs[0].reshape(1, -1, 1, 4)
        trt_outputs[1] = trt_outputs[1].reshape(1, -1, 80)

        boxes = post_processing(blob, 0.4, 0.6, trt_outputs)

        return boxes


class ReID(object):

    def __init__(self,model_path):

        # load model
        trt_logger = trt.Logger()  # This logger is required to build an engine
        runtime = trt.Runtime(trt_logger)
        f = open(model_path, "rb")
        self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.context.set_binding_shape(0, (1, 3, 256,128))

    def Inference(self,img,height,width):

        # Simple helper data class that's a little nicer to use than a 2-tuple.
        class HostDeviceMem(object):
            def __init__(self, host_mem, device_mem):
                self.host = host_mem
                self.device = device_mem

            def __str__(self):
                return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

            def __repr__(self):
                return self.__str__()

        # Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
        def allocate_buffers(engine, batch_size):
            inputs = []
            outputs = []
            bindings = []
            stream = cuda.Stream()
            for binding in engine:

                size = trt.volume(engine.get_binding_shape(binding)) * batch_size
                dims = engine.get_binding_shape(binding)
                
                # in case batch dimension is -1 (dynamic)
                if dims[0] < 0:
                    size *= -1
                
                dtype = trt.nptype(engine.get_binding_dtype(binding))
                # Allocate host and device buffers
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                # Append the device buffer to device bindings.
                bindings.append(int(device_mem))
                # Append to the appropriate list.
                if engine.binding_is_input(binding):
                    inputs.append(HostDeviceMem(host_mem, device_mem))
                else:
                    outputs.append(HostDeviceMem(host_mem, device_mem))
            return inputs, outputs, bindings, stream


        # This function is generalized for multiple inputs/outputs.
        # inputs and outputs are expected to be lists of HostDeviceMem objects.
        def do_inference(context, bindings, inputs, outputs, stream):
            # Transfer input data to the GPU.
            [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
            # Run inference.
            context.execute_async(bindings=bindings, stream_handle=stream.handle)
            # Transfer predictions back from the GPU.
            [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
            # Synchronize the stream
            stream.synchronize()
            # Return only the host outputs.
            return [out.host for out in outputs]

        image_cv = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        blob = np.expand_dims(image_cv[:, :, (2, 1, 0)].transpose(2, 0, 1), axis=0).astype("float32") 
        # blob /= 255.0
        # blob = np.ascontiguousarray(blob)

        self.buffers = allocate_buffers(self.engine,1)
        inputs, outputs, bindings, stream = self.buffers
        inputs[0].host = blob

        begin = datetime.datetime.now()
        trt_outputs = do_inference(self.context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        end = datetime.datetime.now()
        print('reid inference time:' ,end - begin)

        embedding = trt_outputs[0]

        return embedding

class SiamTracker(object):

	def __init__(self,args):
            # load config
    	    cfg.merge_from_file(args.config)
    	    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
            #device = torch.device('cuda')
    	    device = torch.device('cuda' if cfg.CUDA else 'cpu')
            #device = torch.device('cuda')

    	    # create model
    	    model = ModelBuilder()

    	    # load model
    	    model.load_state_dict(torch.load(args.snapshot,
                map_location=lambda storage, loc: storage.cpu()))
    	    model.eval().to(device)

 	    # build tracker
    	    self.tracker = build_tracker(model)

        
def LoadGallery(image_path):

    reid_model_path = 'trt_checkpoints/baseline_R50.trt'
    reid = ReID(reid_model_path)

    image = cv2.imread(image_path)

    gallery_embedding = reid.Inference(image,256,128).reshape(1,2048)
    gallery_norm = np.linalg.norm(gallery_embedding, axis=1).reshape(-1,1)
    gallery_norm += 1e-8

    gallery_normlize = np.divide(gallery_embedding,gallery_norm)

    return gallery_normlize

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "NotoSansCJK-Bold.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":

    tracker_args = parser.parse_args()
    siamtracker = SiamTracker(tracker_args)

    detection_model_path = 'trt_checkpoints/yolov4_tiny.trt'
    yolov4_tiny = Yolov4_tiny(detection_model_path)

    reid_model_path = 'trt_checkpoints/baseline_R50.trt'
    reid = ReID(reid_model_path)
    

    tracker_flag = False

    gallery_embedding = LoadGallery('test_data/find_person.jpg')

    video_path = 'video/2.flv'
    cap = cv2.VideoCapture(video_path)


    while(cap.isOpened()):

        success,frame = cap.read()
        #cv2.imshow('src',frame)
        if success == True:
            # frame = cv2.resize(frame,(1280,720))
            draw = frame.copy()
            if tracker_flag == False:
                boxes = yolov4_tiny.Inference(frame,416,416)
                namesfile = 'pytorch-YOLOv4/data/coco.names'
                class_names = load_class_names(namesfile)
                draw = plot_boxes_cv2(frame, boxes[0], class_names=class_names)

                for i in range(len(boxes[0])):
                    box = boxes[0][i]
                    if box[6] == 0:

                        width = frame.shape[1]
                        height = frame.shape[0]

                        x1 = max(0,int(box[0] * width))
                        y1 = max(0,int(box[1] * height))
                        x2 = min(int(box[2] * width),width-1)
                        y2 = min(int(box[3] * height),height-1)

                        person = frame[y1:y2,x1:x2]
                        # cv2.imwrite('find_person.jpg',person)
                        embedding = reid.Inference(person,256,128).reshape(1,2048)

                        face_norm = np.linalg.norm(embedding, axis=1).reshape(-1,1)
                        face_norm += 1e-8
                        face_normalize = np.divide(embedding,face_norm)
                        cos_dis = np.dot(gallery_embedding,face_normalize.T)

                        if (cos_dis > 0.95):
                            draw = cv2.rectangle(draw, (x1, y1), (x2, y2), (0,0,255), 5)
                            draw = cv2ImgAddText(draw,'find', x1, y1, (255, 255, 0), 20)
                            person_box = (x1,y1,x2-x1,y2-y1)
                            siamtracker.tracker.init(frame, person_box)
                            #tracker.init(frame, person_box)
                            tracker_flag = True
            else:
                begin = datetime.datetime.now()
                #ok, bbox = tracker.update(frame)
                outputs = siamtracker.tracker.track(frame)
                end = datetime.datetime.now()
                print('track time:',end - begin)

                if 'polygon' in outputs:
                    polygon = np.array(outputs['polygon']).astype(np.int32)
                    cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                                  True, (0, 255, 0), 3)
                    mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                    mask = mask.astype(np.uint8)
                    mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                    frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
                else:
                    bbox = list(map(int, outputs['bbox']))
                    cv2.rectangle(draw, (bbox[0], bbox[1]),
                                  (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                                  (0, 255, 0), 3)
                                
            cv2.imshow('test',draw)
            cv2.waitKey(1)
	    #out.write(draw)

        else:
            raise RuntimeError('can\'t no get the frame')
