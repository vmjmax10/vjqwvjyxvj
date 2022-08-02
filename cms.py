## Standard library imports
import argparse
from contextlib import closing
import json
import os
import time
import math

## Third party Imports
import cv2
import numpy as np
import onnxruntime

NEXT_S_ID = 0
MISS_THRESH = 10

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
        cls_confidence=-1, 
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
        self.num_classes = self.model_info["num_classes"]

        ## grids and expanded stride for model infer arch
        self._set_grids_and_expanded_strides()

        if self.cls_confidence == -1:
            self.min_class_confidence = {
                int(k):v
                for k, v in self.model_info[
                    "min_class_confidences_for_detection"
                ].items()
            } 
        else:
            self.min_class_confidence = {
                i:self.cls_confidence for i in range(self.num_classes)
            }

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
        for cls_ind in range(self.num_classes):
            cls_scores = scores[:, cls_ind]
            valid_score_mask = cls_scores > self.min_class_confidence[cls_ind]
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
            x2 = int(min(self.max_size_val_w*scale_w, int(bbox[2]*scale_w)+1))
            y2 = int(min(self.max_size_val_h*scale_h, int(bbox[3]*scale_h+1)))

            w, h = x2 - x1, y2 - y1
            if w < 5 or h < 5:
                continue
            
            detected_objects.append(
                DetectedObject(
                    [x1, y1, x2, y2, w*h],
                    cls_id,
                    round(float(scores[idx]), 2),
                    class_name=self.classes[cls_id]
                )
            )
          
        return detected_objects

    def visualize(self, img, all_detected_objects): 
        for obj in all_detected_objects:
            bbox = obj.bbox
            color = self.colors[obj.class_id]
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        return img

    def put_text_on_image(
        self, 
        image, 
        text, 
        x1=1,
        y1=1,
        txt_color=(0, 0, 255),
        txt_background_color=(0, 0, 0),
        font = cv2.FONT_HERSHEY_DUPLEX,
        font_scale=1.0
    ):

        img_h, img_w = image.shape[:2]
        txt_size = cv2.getTextSize(text, font, font_scale, 1)[0]

        pos_x2 = min(img_w-1, x1+txt_size[0]+1)
        pos_y2 = int(min(img_h-1, y1+txt_size[1]*1.15+1))

        cv2.rectangle(
            image,
            (x1, y1),
            (pos_x2, pos_y2),
            txt_background_color,
            -1
        )
        cv2.putText(
            image, 
            text, 
            (x1, y1+int(txt_size[1]*1.075)), 
            font, 
            font_scale, 
            txt_color, 
            thickness=1
        )

    def extract(self, image):

        if self.infer_session is None:
            return self._empty_response()
        
        img_h, img_w = image.shape[:2]
        if img_h > img_w:
            self.max_size_val_w = int(img_w*(self.infer_size/img_h))-1
            self.max_size_val_h = self.infer_size - 1
        else:
            self.max_size_val_h = int(img_h*(self.infer_size/img_w))-1
            self.max_size_val_w = self.infer_size - 1

        image = resize_to_desired_sqaure_size_image_with_padding(
            image, max_size=self.infer_size, pad_color=255
        )
        self.image = image

        # st = time.time()
        img, scale_w, scale_h = self._preprocess_image(image)

        output = self.infer_session.run(None, {self.input_name: img[None, :, :, :]})
        dets = self._raw_output_postprocess(output[0])

        if dets is None:
            return self._empty_response()

        boxes, scores, cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        # print("time", time.time() -st)
        
        all_detected_objects = self.update_bbox_and_remove_unnecessary_data(
            boxes, scores, cls_inds, scale_w, scale_h
        )
        # Testing purpose only
        if self.is_visualize_mode:
            ord_quit = ord("q")
            vis_img = self.visualize(image.copy(), all_detected_objects)
            cv2.imshow("disp", vis_img)
            if cv2.waitKey(1) == ord_quit:
                exit()

        return all_detected_objects

# ##UTILS FUNCTIONS

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

def resize_to_desired_sqaure_size_image_with_padding(
    img, max_size=1280, pad_color=0
):

    img = resize_to_desired_max_size_for_processing(img, max_size=max_size)
    h, w, c = img.shape

    assert c == 3

    if h != max_size:
        pad_img = np.ones(
            [max_size-h, w, c], dtype=np.uint8
        )*pad_color
        img = cv2.vconcat([img, pad_img])

    elif w != max_size:

        pad_img = np.ones(
            [h, max_size-w, c], dtype=np.uint8
        )*pad_color
        img = cv2.hconcat([img, pad_img])

    return img

def get_iou(query_rect, available_rects):
    
    max_iou, max_a_intersection = 0, 0

    for req_rect in available_rects:

        x_right = min(query_rect[2], req_rect[2])
        x_left = max(query_rect[0], req_rect[0])
        y_bottom = min(query_rect[3], req_rect[3])
        y_top = max(query_rect[1], req_rect[1])

        if x_left < x_right and y_top < y_bottom:
            max_intersection = (x_right-x_left)*(y_bottom-y_top)
            min_area = min(query_rect[4], req_rect[4])
            iou = max_intersection/min_area
            if iou > max_iou and max_intersection > max_a_intersection:
                max_iou = iou
                max_a_intersection = max_intersection
            
    return max_iou

# ## 

def check_if_already_exists(obj, shutters_info, found_s_ids, min_iou_thresh=0.15):
    max_idx, max_iou = -1, min_iou_thresh
    for s_id, s_info in shutters_info.items():
        if s_id in found_s_ids:
            continue
        iou = get_iou(obj.bbox, s_info["last_n_locations"])
        if iou > max_iou:
            max_iou = iou
            max_idx = s_id
    return max_idx

def add_in_shutters_info(obj, shutters_info):
    global NEXT_S_ID
    if len(shutters_info) == 0:
        NEXT_S_ID = 0
    
    NEXT_S_ID += 1

    s_id = NEXT_S_ID
    shutters_info[s_id] = {
        "miss_thresh":MISS_THRESH,
        "last_n_locations":[obj.bbox for _ in range(MISS_THRESH)],
        "last_n_y2":[obj.bbox[3]],
        "min_y1":obj.bbox[1], 
        "max_y2":obj.bbox[3],
        "ideal_frames":0,
        "num_active":0,
        "_opening":False
    }
    return s_id

def update_in_shutters_info(obj, shutters_info, s_id):

    s_info = shutters_info[s_id]

    s_info["miss_thresh"] = MISS_THRESH
    bbox = obj.bbox

    s_info["num_active"] += 1

    ## IGNORE UPDATES UNTILL SOME MOVEMENT
    if s_info["last_n_y2"] and abs(bbox[3]-s_info["last_n_y2"][-1]) > 5:
        s_info["last_n_y2"] = s_info["last_n_y2"][-5:] + [bbox[3]]
        s_info["ideal_frames"] = 0
    else:
        s_info["ideal_frames"] += 1

    ## update _y1, _y2 --> y1 is top of shutter which is fixed entity
    s_info["min_y1"] = min(s_info["min_y1"], bbox[1])
    s_info["max_y2"] = max(s_info["max_y2"], bbox[3])
 
    s_info["last_n_locations"] = s_info["last_n_locations"][1:] + [bbox]
  
def remove_from_shutters_info(found_s_ids, shutters_info):

    for s_id in list(shutters_info.keys()):
        if s_id not in found_s_ids:
            shutters_info[s_id]["miss_thresh"] -= 1
            if shutters_info[s_id]["miss_thresh"] < 0:
                del shutters_info[s_id]
            else:
                shutters_info[s_id]["num_active"] = 0

def get_current_message_based_on_shutters_info(shutters_info, max_size_w, max_size_h):
    
    len_s_ids = len(shutters_info)

    if len_s_ids == 0:
        msg = "SHUTTERS OPEN"
    else:
        msgs = []
        for s_id in list(shutters_info.keys()):

            s_info = shutters_info[s_id]

            len_heights = len(s_info["last_n_y2"])

            if len_heights < 5:

                # print("came here", len_heights)
                if s_info["num_active"] > 10 and len_heights:
                    txt = "CLOSED" if s_info["last_n_y2"][-1] > max_size_h/2 else "OPEN"
                    msg = f"Shutter({s_id}) => {txt}" 
                    msgs.append(msg)
                else:
                    msg = f"Shutter({s_id}) => Analyzing" 
                    msgs.append(msg)

                continue

            msg = ""

            s_last_n_y2 = s_info["last_n_y2"]
            min_last_y2 = min(s_last_n_y2)
            max_last_y2 = max(s_last_n_y2)

            min_y2_index = s_last_n_y2.index(min_last_y2)
            max_y2_index = s_last_n_y2.index(max_last_y2)

            ## IDLE STAGE
            if s_info["ideal_frames"] > 4:

                if max_y2_index >= min_y2_index:
                    msg = f"Shutter({s_id}) => CLOSED"
                    del shutters_info[s_id]
                    # print(f"Shutter({s_id}) => CLOSED")
                else:
                    max_s_height = s_info["max_y2"] - s_info["min_y1"]
                    cur_s_height = s_last_n_y2[-1] - s_info["min_y1"]

                    percentage_open = 100 - round((cur_s_height/max_s_height)*100, 2)
                    if percentage_open > 90:
                        msg = f"Shutter({s_id}) => OPEN"
                        del shutters_info[s_id]
                        # print(f"Shutter({s_id}) => OPEN")
                    elif percentage_open > 25:
                        msg = f"Shutter({s_id}) => OPEN ({str(percentage_open)[:4]}%)"
                    elif percentage_open > 15:
                        msg = f"Shutter({s_id}) => OPENING ({str(percentage_open)[:4]}%)"
                    else:
                        msg = f"Shutter({s_id}) => Analyzing"

            else:
                if min_y2_index <= max_y2_index:
                    s_info["_opening"] = False
                    msg = f"Shutter({s_id}) => CLOSING"
                else:
                    msg = f"Shutter({s_id}) => OPENING"
                    s_info["_opening"] = True

            if msg:
                msgs.append(msg)

        msg = ", ".join(msgs)

    return msg


# ### CMS VIDEO DEMO FUNCTION
def cms_video_demo(args):
    
    if args.v_path != "" and os.path.exists(args.v_path):
        all_video_file_paths = [args.v_path]
    elif args.v_dir != "":
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
    else:
        print("please provide proper video info/path")
        exit()

    total_videos = len(all_video_file_paths)
    print(f"Total Video Files => {total_videos}")

    print("Model loading....", end="")
    # Load model
    cms_model = CustomYolox_ONNX(
        model_path="all_models/yolox_s_cms.onnx",
        cls_info_file_path="model_classes/cms.json",
        nms_thresh=0.10, 
        infer_size=640,
        cls_confidence=-1, ## special -1 value tells to use the per class confidnece
        is_visualize_mode=False
    )
    print("Finished")

    # Display
    cv2.namedWindow("disp", cv2.WINDOW_NORMAL)
    ord_quit, ord_next, ord_skip = ord("q"), ord("n"), ord("s")
    num_skip_frames = 75

    infer_size = cms_model.infer_size

    ## run now
    for v_idx, video_path in enumerate(all_video_file_paths, start=1):

        print(f"\nStarting... session for ({v_idx}/{total_videos}) -> {video_path}")

        stream = cv2.VideoCapture(video_path)
        if not stream.isOpened():
            print(f"Error in video reading -> {video_path}")
            continue
        
        width  = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width > height:
            max_size_w = infer_size
            max_size_h = int(height * (infer_size/width))
        else:
            max_size_h = infer_size
            max_size_w = int(width * (infer_size/height))

        ## allowed shutter size (both at the same time)
        max_size_hs = max_size_h*0.9
        max_size_ws = max_size_w*0.9

        # fps = stream.get(cv2.CAP_PROP_FPS)
        # frame_count = stream.get(cv2.CAP_PROP_FRAME_COUNT)
        # print(f"\nVideo Properties AW {max_size_ws}, AH {max_size_hs}")
        
        frame_idx, do_skip_idx = 0, False
        shutters_info = {}

        while stream.isOpened():
            ret, frame = stream.read()
            if ret:
                frame_idx += 1
                
                if frame_idx < do_skip_idx:
                    continue

                all_detected_objects = cms_model.extract(frame)
                
                found_s_ids = []
                ## add shutters info in session
                for obj in all_detected_objects:

                    x1, y1, x2, y2 , a = obj.bbox

                    if obj.class_id == 3:
                        if (y2-y1) > max_size_hs and (x2-x1) > max_size_ws:
                            continue
                        s_id = check_if_already_exists(obj, shutters_info, found_s_ids)
                        if s_id != -1:
                            found_s_ids.append(s_id)
                            update_in_shutters_info(obj, shutters_info, s_id)
                        else:
                            s_id = add_in_shutters_info(obj, shutters_info)
                            found_s_ids.append(s_id)
                
                remove_from_shutters_info(found_s_ids, shutters_info)

                ## display text
                text = get_current_message_based_on_shutters_info(
                    shutters_info, max_size_w, max_size_h
                )

                vis_img = cms_model.visualize(cms_model.image.copy(), all_detected_objects)
                cms_model.put_text_on_image(vis_img, text, 10, 10)

                cv2.imshow("disp", vis_img)
                k = cv2.waitKey(1)

                if k == ord_quit:
                    exit()
                elif k == ord_next:
                    break
                elif k == ord_skip:
                    do_skip_idx = frame_idx + num_skip_frames
                    print(f"skipping {num_skip_frames} frames")
                else:
                    do_skip_idx = -1
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
        default="", 
        type=str, 
        help="run on this particular video"
    )

    parser.add_argument(
        "--v_dir", 
        default="test_files/cms", 
        type=str, 
        help="run on all video files in this particular directory"
    )
    
    return parser

if __name__ == "__main__":
    args = make_parser().parse_args()
    cms_video_demo(args)
