## Standard library imports
import argparse
import json
import os
import time

## Third party Imports
import cv2
import numpy as np
import onnxruntime


class DetectedObject(object):
    def __init__(
        self,
        bbox,
        class_id,
        confidence,
        class_name=""
    ):

        self.bbox = bbox
        self.class_id = class_id
        self.confidence = confidence
        self.class_name = class_name

class CustomYolox_ONNX(object):

    def __init__(
        self,
        model_path="model_final.onnx",
        cls_info_file_path="model_classes.json",
        nms_thresh=0.10, 
        infer_size=640,
        cls_confidence=0.25, 
        is_visualize_mode=False
    ):

        self.is_visualize_mode = is_visualize_mode

        ## set detection thresholds
        self.infer_size = infer_size
        self.nms_thresh = nms_thresh
        self.cls_confidence = cls_confidence

        ## set model inference session
        try:
            ## model path - fix pattern
            self.infer_session = onnxruntime.InferenceSession(model_path)
            with open(cls_info_file_path, "r") as f:
                self.model_info = json.load(f)

        except Exception as e:
            print(e)
            self.infer_session = None
        else:
            self.input_name = self.infer_session.get_inputs()[0].name 

        ## set model classes and color for visulaization
        self._set_model_info_and_visulization_params()

    def _set_model_info_and_visulization_params(self):

        self.input_shape = (self.infer_size, self.infer_size)
        self.max_size_val = self.infer_size - 1 

        ## grids and expanded stride for model infer arch
        self._set_grids_and_expanded_strides()

        ## visualization
        self.classes = [
            d[0] for d in self.model_info["classes_and_colors"]
        ]
        self.colors = [
            d[1] for d in self.model_info["classes_and_colors"]
        ]
   
    def _set_grids_and_expanded_strides(self):
        
        grids, expanded_strides = [], []
        strides = [8, 16, 32]
        
        hsizes = [self.infer_size // stride for stride in strides]
        wsizes = [self.infer_size // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        self.grids = np.concatenate(grids, 1)
        self.expanded_strides = np.concatenate(expanded_strides, 1)

    def _nms(self, boxes, scores):
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

    def _multiclass_nms(self, boxes, scores):
    
        final_dets = []
        num_classes = scores.shape[1]

        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid_score_mask = cls_scores > self.cls_confidence
            if valid_score_mask.sum() == 0:
                continue
            else:
                valid_scores = cls_scores[valid_score_mask]
                valid_boxes = boxes[valid_score_mask]
                keep = self._nms(valid_boxes, valid_scores)
                
                if len(keep) > 0:
                    cls_inds = np.ones((len(keep), 1)) * cls_ind
                    dets = np.concatenate(
                        [
                            valid_boxes[keep], 
                            valid_scores[keep, None], 
                            cls_inds
                        ], 
                        1
                    )
                    final_dets.append(dets)
        
        if len(final_dets) == 0:
            return None
        
        return np.concatenate(final_dets, 0)
    
    def _preprocess_image(self, img, swap=(2, 0, 1)):
        
        og_height, og_width = img.shape[:2]
        scale_w, scale_h = og_width/self.infer_size, og_height/self.infer_size

        resized_img = cv2.resize(
            img,
            self.input_shape,
            interpolation=cv2.INTER_AREA
        )
        resized_img = resized_img.transpose(swap)
        resized_img = np.ascontiguousarray(resized_img, dtype=np.float32)

        return resized_img, scale_w, scale_h

    def _raw_output_postprocess(self, outputs, ratio=1):

        outputs[..., :2] = (outputs[..., :2] + self.grids) * self.expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * self.expanded_strides

        predictions = outputs[0]
        
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        # boxes_xyxy /= ratio # as ratio is 1

        dets = self._multiclass_nms(boxes_xyxy, scores)

        return dets

    # EMPTY DETECTED OBJECT
    def _empty_response(self):
        return []

    def update_bbox_and_remove_unnecessary_data(
        self, boxes, scores, cls_inds, scale_w, scale_h
    ):

        detected_objects = []

        for idx in range(len(boxes)):

            cls_id =  int(cls_inds[idx])

            bbox = boxes[idx]
            x1 = int(max(1, int(bbox[0]*scale_w)-1))
            y1 = int(max(1, int(bbox[1]*scale_h)-1))
            x2 = int(min(self.max_size_val*scale_w, int(bbox[2]*scale_w)+1))
            y2 = int(min(self.max_size_val*scale_h, int(bbox[3]*scale_h+1)))

            w, h = x2 - x1, y2 - y1
            if w < 5 or h < 5:
                continue
            
            detected_objects.append(
                DetectedObject(
                    [x1, y1, x2, y2, (y2-y1)*(x2-x1)],
                    cls_id,
                    round(float(scores[idx]), 2),
                    class_name=self.classes[cls_id]
                )
            )
          
        return detected_objects

    def visualize(self, img, boxes, scores, cls_ids):
       
        for i in range(len(boxes)):

            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            if score < self.cls_confidence:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            color = self.colors[cls_id]
            text = '{}:{:.1f}%'.format(self.classes[cls_id], score * 100)
            
            text = self.classes[cls_id]
            txt_color = (
                (0, 0, 0) 
                if np.mean(self.colors[cls_id]) > 0.5 
                else (255, 255, 255)
            )
            font = cv2.FONT_HERSHEY_SIMPLEX

            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

            # txt_bk_color = [int(bgr*0.7) for bgr in self.colors[cls_id]]
            # cv2.rectangle(
            #     img,
            #     (x0, y0 - int(1.5*txt_size[1])),
            #     (x0 + txt_size[0] + 1, y0),
            #     txt_bk_color,
            #     -1
            # )
            # cv2.putText(
            #     img, 
            #     text, 
            #     (x0, y0 - int(0.75*txt_size[1])), 
            #     font, 0.4, 
            #     txt_color, 
            #     thickness=1
            # )

        return img

    def extract(self, image):

        if self.infer_session is None:
            return self._empty_response()

        # st = time.time()
        img, scale_w, scale_h = self._preprocess_image(image)

        output = self.infer_session.run(None, {self.input_name: img[None, :, :, :]})
        dets = self._raw_output_postprocess(output[0])

        if dets is None:
            return self._empty_response()

        boxes, scores, cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        # print("time", time.time() -st)
        
        # Testing purpose only
        if self.is_visualize_mode:

            new_boxes = []
            for idx in range(len(boxes)):
                bbox = boxes[idx]

                x1 = float(max(1, int(bbox[0]*scale_w)))
                y1 = float(max(1, int(bbox[1]*scale_h)))

                x2 = float(min(self.max_size_val*scale_w, int(bbox[2]*scale_w)))
                y2 = float(min(self.max_size_val*scale_h, int(bbox[3]*scale_h)))
                
                new_boxes.append([x1, y1, x2, y2])

            vis_img = self.visualize(image.copy(), new_boxes, scores, cls_inds)
            cv2.imshow("disp", vis_img)
            # k = cv2.waitKey(0)
            # if  k == ord("q"):
            #     exit()

        return self.update_bbox_and_remove_unnecessary_data(
            boxes, scores, cls_inds, scale_w, scale_h
        )

# UTILS FUNCTIONS

def resize_to_desired_max_size_for_processing(img, max_size=1200):

    og_height, og_width = img.shape[:2]

    inter_ploation = (
        cv2.INTER_CUBIC
        if og_height < max_size and og_width < max_size
        else cv2.INTER_AREA
    )
    img_s_resized = (
        (int(og_width * max_size / og_height), max_size)
        if og_width < og_height
        else (max_size, int(og_height * max_size / og_width))
    )

    img = cv2.resize(img, img_s_resized, interpolation=inter_ploation)
    return img

def resize_to_desired_sqaure_size_image_with_padding(img, max_size=1280):

    img = resize_to_desired_max_size_for_processing(img, max_size=max_size)
    h, w, c = img.shape

    assert c == 3

    if h != max_size:
        pad_img = np.zeros(
            [max_size-h, w, c], dtype=np.uint8
        )
        img = cv2.vconcat([img, pad_img])

    elif w != max_size:

        pad_img = np.zeros(
            [h, max_size-w, c], dtype=np.uint8
        )
        img = cv2.hconcat([img, pad_img])

    return img

def cms_video_demo(args):
    
    if args.v_dir != "":
        v_dir = args.v_dir
        try:
            all_video_file_paths = [
                f"{v_dir}/{f}"
                for f in os.listdir(v_dir)
                if f.endswith((".mp4", ".avi"))
            ]
        except:
            print("Wrong dir given, please check")
            exit()
    elif args.v_path != "" and os.path.exists(args.v_path):
        all_video_file_paths = [args.v_path]
    else:
        print("please provide proper video info/path")
        exit()

    print("Model loading....", end="")
    # Load model
    cms_model = CustomYolox_ONNX(
        model_path="all_models/yolox_s_cms.onnx",
        cls_info_file_path="model_classes/cms.json",
        nms_thresh=0.10, 
        infer_size=640,
        cls_confidence=0.40, 
        is_visualize_mode=bool(args.vis)
    )
    print("Finished")

    # Display
    cv2.namedWindow("disp", cv2.WINDOW_NORMAL)

    ## run now
    for video_path in all_video_file_paths:
        print(f"\nStarting... session for {video_path}")

        stream = cv2.VideoCapture(video_path)
        if not stream.isOpened():
            print(f"Error in video reading -> {video_path}")
            continue
        
        shutters_info = {}

        while stream.isOpened():
            ret, frame = stream.read()
            if ret:
                frame = resize_to_desired_sqaure_size_image_with_padding(
                    frame, max_size=640
                )

                detected_objcets = cms_model.extract(frame)

                all_shutter_objects = [
                    d for d in detected_objcets if d.class_id == 3
                ]

            else:
                break
        
        stream.release()
        print(f"Session for {video_path} ended")
    
    cv2.destroyAllWindows()
    print("DEMO ENDED")

def make_parser():

    parser = argparse.ArgumentParser("onnxruntime cms demo")
    parser.add_argument(
        "--v_path", 
        default="test_files/cms/Cms-1 Shutter Open.mp4", 
        type=str, 
        help="run on this particular video"
    )

    parser.add_argument(
        "--v_dir", 
        default="", 
        type=str, 
        help="run on all video files in this particular directory"
    )

    parser.add_argument(
        "--vis", 
        default=1, 
        type=int, 
        help="display info"
    )
   
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    cms_video_demo(args)
