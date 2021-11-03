from posix import listdir
import unittest
import os
import numpy as np
import torch
import time
import img_helper
from model_convertor import ModelConvertor
import mask_postprocess as mp
import cv2
import postprocess as op

dir_path = os.path.split(os.path.realpath(__file__))[0]
verbosity = "info"
nloop = 1 # 10000

class TestModelConvertor(unittest.TestCase):
    def test_fp32_mmdet_SSD(self):
        bsize = 1
        convertor = ModelConvertor()
        # convert onnx model to tensorRT model
        onnx_model = "/home/jliu/data/models/ssd512_coco_shape512x512_trtnms_topk1000.onnx"
        prefix_model= os.path.split(onnx_model)[-1].split('.')[0]
        trt_model_path = os.path.join(dir_path, "data/models/{}_fp32_bsize{}.trt".format(prefix_model, bsize))
        engine_path = convertor.load_model(
            onnx_model,
            None,
            onnx_model_path=None,
            engine_path=trt_model_path,
            explicit_batch=True,
            precision='fp32',
            verbosity=verbosity
        )

    def test_fp16_mmdet_SSD(self):
        bsize = 1
        convertor = ModelConvertor()
        # convert onnx model to tensorRT model
        onnx_model = "/home/jliu/data/models/ssd512_coco_shape512x512_trtnms_topk1000.onnx"
        prefix_model= os.path.split(onnx_model)[-1].split('.')[0]
        trt_model_path = os.path.join(dir_path, "data/models/{}_fp16_bsize{}.trt".format(prefix_model, bsize))
        engine_path = convertor.load_model(
            onnx_model,
            None,
            onnx_model_path=None,
            engine_path=trt_model_path,
            explicit_batch=True,
            precision='fp16',
            verbosity=verbosity
        )

    def test_int8_mmdet_SSD(self):
        bsize = 1
        convertor = ModelConvertor()
        # convert onnx model to tensorRT model
        onnx_model = "/home/yhuang/tensorRT_work/ssd_resnet18_tf2trt/ssd_resnet18_224.onnx"
        prefix_model= os.path.split(onnx_model)[-1].split('.')[0]
        trt_model_path = os.path.join(dir_path, "data/models/{}_int8cali_cocoval_calisize512_n500_precoco_bsize{}.trt".format(prefix_model, bsize))
        engine_path = convertor.load_model(
            onnx_model,
            None,
            onnx_model_path=None,
            engine_path=trt_model_path,
            explicit_batch=True,
            precision='int8',
            verbosity=verbosity,
            max_calibration_size=500,
            calibration_batch_size=32,
            calibration_data='/home/jliu/data/coco/images/val2017/',
            preprocess_func='preprocess_coco_mmdet_ssd',
            cali_input_shape=(512, 512),
            save_cache_if_exists=True
        )
    def test_fp16_mmdet_SSD_Resnet18(self):
        bsize = 1
        convertor = ModelConvertor()
        # convert onnx model to tensorRT model
        onnx_model = "/home/yhuang/tensorRT_work/ssd_resnet18_tf2trt/ssd_resnet18_224.onnx"
        prefix_model= os.path.split(onnx_model)[-1].split('.')[0]
        trt_model_path = os.path.join(dir_path, "data/models/{}_fp16_bsize{}.trt".format(prefix_model, bsize))
        engine_path = convertor.load_model(
            onnx_model,
            None,
            onnx_model_path=None,
            engine_path=trt_model_path,
            explicit_batch=True,
            precision='fp16',
            verbosity=verbosity
        )
    def test_int8_mmdet_SSD_Resnet18(self):
        bsize = 1
        convertor = ModelConvertor()
        # convert onnx model to tensorRT model
        onnx_model = "/home/yhuang/tensorRT_work/ssd_resnet18_tf2trt/ssd_resnet18_224.onnx"
        prefix_model= os.path.split(onnx_model)[-1].split('.')[0]
        trt_model_path = os.path.join(dir_path, "data/models/{}_int8cali_vocval_calisize224_n500_prevoc_bsize{}.trt".format(prefix_model, bsize))
        engine_path = convertor.load_model(
            onnx_model,
            None,
            onnx_model_path=None,
            engine_path=trt_model_path,
            explicit_batch=True,
            precision='int8',
            verbosity=verbosity,
            max_calibration_size=3000,
            calibration_batch_size=32,
            calibration_data='/home/yhuang/datasets/voc/VOCdevkit/VOC2007/JPEGImages/',
            preprocess_func='preprocess_voc_mmdet_ssd',
            cali_input_shape=(224, 224),
            save_cache_if_exists=True
        )
    
 
    
    def test_mmdet_SSD_loadtrt(self):
        bsize = 1
        convertor = ModelConvertor()
        img_readpath = '/home/jliu/data/coco/images/val2017/'
        img_savepath = os.path.join(dir_path, 'data/images/test_mmdet_ssd')
        # laod tensorRT model
        for trt_model_path in [ \
            "/home/jliu/data/models/ssd512_coco_shape512x512_trtnms_topk1000_fp32_bsize1.trt", \
            "/home/jliu/data/models/ssd512_coco_shape512x512_trtnms_topk1000_fp16_bsize1.trt", \
            "/home/jliu/data/models/ssd512_coco_shape512x512_trtnms_topk1000_int8nocalibytrtexec_bsize1.trt", \
            "/home/jliu/data/models/ssd512_coco_shape512x512_trtnms_topk1000_int8cali_cocoval_calisize512_n500_precoco_bsize1.trt",
        ]:
            convertor.load_model(trt_model_path, dummy_input=None, onnx_model_path=None, engine_path=None)

            # for relpath in os.listdir('/home/jliu/data/coco/val2017'):
            for relimgpath in [
                '000000000139.jpg',
                '000000255917.jpg',
                '000000037777.jpg',
                '000000397133.jpg',
                '000000000632.jpg',
                ]:
                # single_image_path = '/home/jliu/data/images/mmdetection/demo.jpg'
                single_image_path = os.path.join(img_readpath, relimgpath)
                input_config = {
                    'input_shape': (1, 3, 512, 512),
                    'input_path': single_image_path,
                    'normalize_cfg': {
                        'mean': (123.675, 116.28, 103.53),
                        'std': (1, 1, 1)
                        }
                    }
                one_img, one_meta = img_helper.preprocess_example_input(input_config)
                batch_in = one_img.contiguous().detach().numpy()
                stime = time.time()
                for iloop in range(0, nloop):
                    labels, boxes_and_scores = convertor.predict(batch_in)
                etime = time.time()
                print("boxes_and_scores: \n{}".format(boxes_and_scores))
                print("labels: \n{}".format(labels))

                # save image
                boxes_and_scores = np.squeeze(boxes_and_scores, axis=0)
                labels = np.rint(np.squeeze(labels, axis=0)).astype(np.int32)
                for score_thr in [0.02, 0.05, 0.1, 0.2, 0.3]: # modify here
                    prefix_image= os.path.split(single_image_path)[-1].split('.')[0]
                    prefix_model= os.path.split(trt_model_path)[-1].split('.')[0]
                    out_image_path = os.path.join(img_savepath, prefix_model+'_input_'+prefix_image+"_score"+str(score_thr)+'.jpg')
                    img_helper.imshow_det_bboxes(
                        one_meta['show_img'],
                        boxes_and_scores,
                        labels,
                        class_names=img_helper.coco_classes(),
                        score_thr=score_thr,
                        bbox_color='red',
                        text_color='red',
                        thickness=1,
                        font_size=4,
                        win_name="tensorrt",
                        out_file=out_image_path)
                    print('output image to {}'.format(out_image_path))  
                    print('time to infer for {} times={:.2f}s'.format(nloop, etime-stime))       
    def test_mmdet_maskrcnn_loadtrt(self):
        bsize = 1
        convertor = ModelConvertor()
        img_readpath = '/workspace/public/dataset/cv/coco/val2017/'
        img_savepath = os.path.join(dir_path, 'data/images/test_mmdet_maskrcnn/')
        # laod tensorRT model
        for trt_model_path in [ 
            # "/home/yhuang/trt_projects/MaskRCNN/build/Mask_RCNN_fp32.plan"
            "/home/yhuang/LT_BENCH_WORK/lt_bench/baseModel/build/cv/segmentation/mask_rcnn/mask_rcnn_r50_torch/mask_rcnn_r50_3x384x384_dynamicbatch_fp32.trt"
        ]:
            convertor.load_model(trt_model_path, dummy_input=None, onnx_model_path=None, engine_path=None)
            for relimgpath in [
                # '000000000139.jpg',
                '000000397133.jpg',
                # '000000000632.jpg',
                # '000000255917.jpg',
                ]:
                single_image_path = os.path.join(img_readpath, relimgpath)
                input_config = {
                    'input_shape': (1, 3, 384, 384),
                    'input_path': single_image_path,
                    'normalize_cfg': {
                        'mean': (123.675, 116.28, 103.53),
                        'std': (58.395, 57.12, 57.375)
                        }
                    }
                
                one_img, one_meta = img_helper.preprocess_example_input(input_config)
                batch_in = one_img.contiguous().detach().numpy()
                stime = time.time()
                for iloop in range(0, nloop):
                    _, bboxes, scores, labels, masks = convertor.predict(batch_in)  # TensorRT python model
                    # _, _, _, bboxes, scores, labels, _, masks = convertor.predict(batch_in) # TensorRT c++ model
                
                etime = time.time()
                print("bboxes type = ",type(bboxes))
                print("bboxes shape = ",bboxes.shape)
                print("scores type = ",type(scores))
                print("labels type = ",type(labels))
                print("masks type = ",type(masks))
                # save image
                bboxes = np.squeeze(bboxes, axis = 0)
                scores = np.squeeze(scores, axis = 0)
                labels = np.squeeze(labels, axis = 0)
                masks = np.squeeze(masks, axis = 0)
                print("bboxes shape = ",bboxes.shape)
                print("scores shape = ",scores.shape)
                print("labels shape = ",labels.shape)
                print("masks shape = ",masks.shape)
                for score_thr in [0.3, 0.4]:
                    prefix_image= os.path.split(single_image_path)[-1].split('.')[0]
                    prefix_model= os.path.split(trt_model_path)[-1].split('.')[0]
                    out_image_path = os.path.join(img_savepath, prefix_model+'_input_'+prefix_image+"_score"+str(score_thr)+'.jpg')
                    print("prefix_image = ", img_readpath+prefix_image+'.jpg')
                    src_img = cv2.imread(img_readpath+prefix_image+'.jpg')
                    image_array = cv2.resize(src_img, (384, 384))
                    cv2.imwrite(img_savepath+prefix_image+'.jpg', src_img)
                    results = mp.postprocess(labels, scores, bboxes, masks, conf_threshold=score_thr)
                    mp.vis_res(image_array, src_img, results, mask_threshold=0.3)

                    print('output image to {}'.format(out_image_path))  
                    print('time to infer for {} times={:.2f}s'.format(nloop, etime-stime))       

    def test_onlynms_convert(self):
        bsize = 1
        convertor = ModelConvertor()
        # convert onnx model to tensorRT model
        onnx_model =  "/home/jliu/data/models/cumstom_op_nms.onnx"
        prefix_model= os.path.split(onnx_model)[-1].split('.')[0]
        trt_model_path = os.path.join(dir_path, "data/models/{}_fp32_bsize{}.trt".format(prefix_model, bsize))
        engine_path = convertor.load_model(
            onnx_model,
            None,
            onnx_model_path=None,
            engine_path=trt_model_path,
            explicit_batch=True,
            precision='fp32',
            verbosity=verbosity
        )

    def test_onlynms_predict(self):
        bsize = 1
        convertor = ModelConvertor()
        # laod tensorRT model
        trt_model_path = "/home/jliu/data/models/cumstom_op_nms_fp32_bsize1.trt"
        convertor.load_model(trt_model_path, dummy_input=None, onnx_model_path=None, engine_path=None)
        # dummy input
        bsize = 1
        num_class = 80
        num_detections = 3652
        after_top_k = 200
        boxes = torch.rand(bsize, num_detections, 1, 4)
        scores = torch.rand(bsize, num_detections, num_class)
        max_output_boxes_per_class = torch.tensor([after_top_k], dtype=torch.int32)
        iou_threshold = torch.tensor([0.5], dtype=torch.float32)
        score_threshold = torch.tensor([0.02], dtype=torch.float32)      
        dummy_input = [boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold]
        # predict
        out = convertor.predict([boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold])
        print(out)


    def test_fp32_mmdet_yolov3(self):
        bsize = 8
        convertor = ModelConvertor()
        # convert onnx model to tensorRT model
        onnx_model = "/workspace/jupdate.onnx"
        prefix_model= os.path.split(onnx_model)[-1].split('.')[0]
        trt_model_path = os.path.join(dir_path, "data/models/{}_fp32_bsize{}.trt".format(prefix_model, bsize))
        engine_path = convertor.load_model(
            onnx_model,
            None,
            onnx_model_path=None,
            engine_path=trt_model_path,
            explicit_batch=True,
            precision='fp32',
            verbosity=verbosity
        )

    def test_int8_mmdet_yolov3(self):
        bsize = 8
        convertor = ModelConvertor()
        # convert onnx model to tensorRT model
        onnx_model = "/workspace/jupdate.onnx"
        prefix_model= os.path.split(onnx_model)[-1].split('.')[0]
        trt_model_path = os.path.join(dir_path, "data/models/{}_int8cali_cocoval_calisize320_n500_preyolov3_bsize{}.trt".format(prefix_model, bsize))
        engine_path = convertor.load_model(
            onnx_model,
            None,
            onnx_model_path=None,
            engine_path=trt_model_path,
            explicit_batch=True,
            precision='int8',
            verbosity=verbosity,
            max_calibration_size=500,
            calibration_batch_size=32,
            calibration_data='/home/jliu/data/coco/images/val2017/',
            preprocess_func='preprocess_coco_mmdet_yolov3',
            cali_input_shape=(320, 320),
            save_cache_if_exists=True
        )        

 
    def test_mmdet_retinanet_loadtrt(self):
        bsize = 1
        convertor = ModelConvertor()
        img_readpath = '/home/jliu/data/coco/images/val2017/'
        img_savepath = os.path.join(dir_path, 'data/images/test_mmdet_retina')
        # laod tensorRT model
        for trt_model_path in [ \
            "/home/jliu/data/models/RetinaNet_int8.plan", \
            #  "/home/jliu/data/models/RetinaNet_int8.plan", \
        ]:
            convertor.load_model(trt_model_path, dummy_input=None, onnx_model_path=None, engine_path=None)

            # for relpath in os.listdir('/home/jliu/data/coco/val2017'):
            for relimgpath in [
                '000000000139.jpg',
                # '000000255917.jpg',
                # '000000037777.jpg',
                # '000000397133.jpg',
                # '000000000632.jpg',
                ]:
                # single_image_path = '/home/jliu/data/images/mmdetection/demo.jpg'
                single_image_path = os.path.join(img_readpath, relimgpath)
                input_config = {
                    'input_shape': (1, 3, 384, 384),
                    'input_path': single_image_path,
                    'normalize_cfg': {
                        'mean': (123.675, 116.28, 103.53),
                        'std': (58.395, 57.12, 57.375)
                        }}
                one_img, one_meta = img_helper.preprocess_example_input(input_config)
                batch_in = one_img.contiguous().detach().numpy()
                stime = time.time()
                for iloop in range(0, nloop):
                    labels, boxes_and_scores = convertor.predict(batch_in)
                etime = time.time()
                print("boxes_and_scores: \n{}".format(boxes_and_scores))
                print("labels: \n{}".format(labels))

                # save image
                boxes_and_scores = np.squeeze(boxes_and_scores, axis=0)
                labels = np.rint(np.squeeze(labels, axis=0)).astype(np.int32)
                for score_thr in [0.02, 0.05, 0.1, 0.2, 0.3]: # modify here
                    prefix_image= os.path.split(single_image_path)[-1].split('.')[0]
                    prefix_model= os.path.split(trt_model_path)[-1].split('.')[0]
                    out_image_path = os.path.join(img_savepath, prefix_model+'_input_'+prefix_image+"_score"+str(score_thr)+'.jpg')
                    img_helper.imshow_det_bboxes(
                        one_meta['show_img'],
                        boxes_and_scores,
                        labels,
                        class_names=img_helper.coco_classes(),
                        score_thr=score_thr,
                        bbox_color='red',
                        text_color='red',
                        thickness=1,
                        font_size=4,
                        win_name="tensorrt",
                        out_file=out_image_path)
                    print('output image to {}'.format(out_image_path))  
                    print('time to infer for {} times={:.2f}s'.format(nloop, etime-stime))   

    def test_mmdet_faster_rcnn_loadtrt(self):
        bsize = 1
        convertor = ModelConvertor()
        img_readpath = '/home/yhuang/datasets/coco/val2017/'
        img_savepath = os.path.join(dir_path, 'data/images/test_mmdet_faster_rcnn')
        input_shape = (1, 3, 384, 384)
        # laod tensorRT model
        for trt_model_path in [ \
            "/workspace/public/basemodel/cv/object_detection/faster_rcnn/faster_rcnn_r50_torch/trt/faster_rcnn_r50_3x384x384_dynamicbatch_fp32.trt", \
            #  "/home/jliu/data/models/RetinaNet_int8.plan", \
        ]:
            convertor.load_model(trt_model_path, dummy_input=None, onnx_model_path=None, engine_path=None)

            # for relpath in os.listdir('/home/jliu/data/coco/val2017'):
            for relimgpath in [
                # '000000000139.jpg',
                # '000000255917.jpg',
                # '000000037777.jpg',
                '000000397133.jpg',
                # '000000000632.jpg',
                ]:
                single_image_path = os.path.join(img_readpath, relimgpath)
                input_config = {
                    'input_shape': input_shape,
                    'input_path': single_image_path,
                    'normalize_cfg': {
                        'mean': (123.675, 116.28, 103.53),
                        'std': (58.395, 57.12, 57.375)
                        }}
                one_img, one_meta = img_helper.preprocess_example_input(input_config)
                batch_in = one_img.contiguous().detach().numpy()
                stime = time.time()
                for iloop in range(0, nloop):
                    _, _, number_boxes, boxes, scores, labels = convertor.predict(batch_in)
                etime = time.time()
                
                # print("boxes : \n{}".format(boxes))
                # print("scores: \n{}".format(scores))
                # print("labels: \n{}".format(labels))
                # threshold = 0.4
                # op.vis_image(single_image_path, boxes, scores, labels, threshold, input_shape)
                classes = labels 
                bboxes = boxes
                threshold = 0.4
                op.plt_bboxes(single_image_path, classes, scores, bboxes, threshold)

                # boxes_and_scores = op.get_boxes_and_scores(boxes, scores)
                # thresh = 0.3
                # keep_dets = op.py_nms(boxes_and_scores, thresh)
                # print("kepp_dets = ", keep_dets)
                # print(boxes_and_scores[keep_dets])
                # save image
                # boxes_and_scores = np.squeeze(boxes_and_scores, axis=0)
                # labels = np.rint(np.squeeze(labels, axis=0)).astype(np.int32)
                # for score_thr in [0.3, 0.4]: # modify here
                #     prefix_image= os.path.split(single_image_path)[-1].split('.')[0]
                #     prefix_model= os.path.split(trt_model_path)[-1].split('.')[0]
                #     out_image_path = os.path.join(img_savepath, prefix_model+'_input_'+prefix_image+"_score"+str(score_thr)+'.jpg')
                #     img_helper.imshow_det_bboxes(
                #         one_meta['show_img'],
                #         boxes_and_scores,
                #         labels,
                #         class_names=img_helper.coco_classes(),
                #         score_thr=score_thr,
                #         bbox_color='red',
                #         text_color='red',
                #         thickness=1,
                #         font_size=4,
                #         win_name="tensorrt",
                #         out_file=out_image_path)
                #     print('output image to {}'.format(out_image_path))  
                #     print('time to infer for {} times={:.2f}s'.format(nloop, etime-stime))   

if __name__ == '__main__':
    # TestModelConvertor().test_fp32_mmdet_SSD()
    # TestModelConvertor().test_fp16_mmdet_SSD()
    # TestModelConvertor().test_int8_mmdet_SSD()
    # TestModelConvertor().test_mmdet_SSD_loadtrt() 
    # TestModelConvertor().test_onlynms_convert() # not OK yet
    # TestModelConvertor().test_onlynms_predict() # not OK yet
    # TestModelConvertor().test_fp32_mmdet_yolov3()
    # TestModelConvertor().test_int8_mmdet_yolov3()
    # TestModelConvertor().test_mmdet_retinanet_loadtrt()
    
    # TestModelConvertor().test_int8_mmdet_MaskRCNN()
    # TestModelConvertor().test_mmdet_maskrcnn_loadtrt()
    # TestModelConvertor().test_int8_mmdet_SSD_Resnet18()
    # TestModelConvertor().test_fp16_mmdet_SSD_Resnet18()

    TestModelConvertor().test_mmdet_faster_rcnn_loadtrt()
