if 1:
    import onnx, onnxruntime, numpy as np, cv2, time
    infer_session = onnxruntime.InferenceSession(
        "all_models/yolox_s_vjs.onnx",
        providers=[
            # "TensorrtExecutionProvider"
            "CUDAExecutionProvider", 
            # "CPUExecutionProvider"
        ]
    )
    input_tensor = infer_session.get_inputs()[0]
    input_tensor_name = input_tensor.name

    print(input_tensor_name, input_tensor.shape)
    
    infer_size = 640

    swap=(2, 0, 1)

    # print(img_s.shape)
    num_samples = 15

    img = cv2.imread("test.jpg")

    sample_images = [
        img
        # np.ones((infer_size,infer_size, 3), dtype=np.uint8)*255
        # np.random.randint(0, 256, size=(infer_size,infer_size, 3), dtype=np.uint8)
        for _ in range(num_samples)
    ]

    img_batch = []
    for idx in range(num_samples):
        resized_img = cv2.resize(
            sample_images[idx],
            (infer_size, infer_size),
            interpolation=cv2.INTER_AREA
        )
        resized_img = resized_img.transpose(swap)
        resized_img = np.ascontiguousarray(resized_img, dtype=np.float32)
        img_s = resized_img[None, :, :, :]
        img_batch.append(img_s)
    
    output_a = {}
    st = time.time()
    for idx in range(num_samples):
        st1 = time.time()
        output_a[idx] = infer_session.run(None, {input_tensor_name: img_batch[idx]})
        print("TIME -> ", idx, time.time()-st1, "\n")
    tt = time.time()-st

    # for _ in range(4):
    #     st = time.time()
    #     output_b = infer_session.run(None, {input_tensor_name: np.concatenate(img_batch)})
    #     dt = time.time()-st
    #     print("TIME BATCH -> ", dt)

    # print(output_b[0][num_samples-1].shape, output_a[num_samples-1][0][0].shape)

    # print(np.array_equal(output_b[0][num_samples-1], output_a[num_samples-1][0][0]))

    exit()

# from yolox.core import Trainer, launch
# from yolox.exp import get_exp
# from yolox.utils import configure_nccl, configure_omp, get_num_devices


# import cv2, numpy as np


# def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
#     r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
#     hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
#     dtype = img.dtype  # uint8

#     x = np.arange(0, 256, dtype=np.int16)
#     lut_hue = ((x * r[0]) % 180).astype(dtype)
#     lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
#     lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

#     img_hsv = cv2.merge(
#         (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
#     ).astype(dtype)
#     cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


# img = cv2.imread("1.jpg")

# # img1 = img.copy()
# # augment_hsv(img1)

# # img2 = img.copy()
# # augment_hsv(img2)

# # img3 = img.copy()
# # augment_hsv(img3)

# img1 = img[:, ::-1]

# cv2.namedWindow("disp", cv2.WINDOW_NORMAL)
# cv2.imshow("disp", cv2.hconcat([img, img1]))
# cv2.waitKey(0)

import re, time, math
import cv2
import numpy as np
import onnxruntime

class BaseRecLabelDecode(object):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False):
        self.beg_str = "sos"
        self.end_str = "eos"
        self.reverse = False
        self.character_str = []

        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
            if 'arabic' in character_dict_path:
                self.reverse = True

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def pred_reverse(self, pred):
        pred_re = []
        c_current = ''
        for c in pred:
            if not bool(re.search('[a-zA-Z0-9 :*./%+-]', c)):
                if c_current != '':
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ''
            else:
                c_current += c
        if c_current != '':
            pred_re.append(c_current)

        return ''.join(pred_re[::-1])

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[
                    batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id]
                for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = ''.join(char_list)

            if self.reverse:  # for arabic rec
                text = self.pred_reverse(text)

            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(CTCLabelDecode, self).__init__(character_dict_path,
                                             use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[-1]
        # if isinstance(preds, paddle.Tensor):
        #     preds = preds.numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character


class RecOnncInfer(object):

    def __init__(
        self, 
        onnx_model_path,
        character_dict_path, 
        use_space_char=True, 
        rec_image_shape=[3, 32, 320]
    ):  

        self.postprocess_op = CTCLabelDecode(
            character_dict_path=character_dict_path,
            use_space_char=use_space_char
        )
        self.rec_image_shape = rec_image_shape

        self.predictor = onnxruntime.InferenceSession(onnx_model_path)
        self.input_tensor = self.predictor.get_inputs()[0]
        self.input_tensor_name = self.input_tensor.name
        self.output_tensors = None

    def final_preprocess_for_model(self, img):
        img = img.astype("float32")
        img = img.transpose((2, 0, 1)) / 255
        img -= 0.5
        img /= 0.5

        return img

    def distortion_free_resize_word_image(
        self, img, req_img_width=None, req_img_height=None
    ):
        
        req_img_height = req_img_height or self.rec_image_shape[1]
        req_img_width = req_img_width or self.rec_image_shape[2]
        
        og_height, og_width = img.shape[:2]
        num_channels = len(img.shape)

        ## as text are horizontal, do it from height
        new_width = int(og_width * (req_img_height/og_height))
        new_height = req_img_height

        inter_ploation = (
            cv2.INTER_CUBIC if new_width < req_img_width else cv2.INTER_AREA
        )
        new_image = cv2.resize(
            img, (new_width, new_height), interpolation=inter_ploation
        )

        ## check if image width exceeds the req_img_width
        half_height = req_img_height * 0.75

        if new_width > req_img_width:

            og_height, og_width = new_image.shape[:2]

            new_width = req_img_width
            new_height = int(og_height*(req_img_width/og_width))

            ## THESE CASES JUST DO THE RESIZE WITHOUT ANY PADDING
            if new_height < half_height:
                new_image = cv2.resize(
                    new_image, 
                    (req_img_width, req_img_height), 
                    interpolation=cv2.INTER_AREA
                )  

            else:

                new_image = cv2.resize(
                    img, (new_width, new_height), interpolation=cv2.INTER_AREA
                )

                pad_h_up = (req_img_height - new_height) // 2
                pad_h_down = req_img_height - new_height - pad_h_up

                v_concat_img_list = []

                if pad_h_up:
                    v_concat_img_list.append(
                        np.zeros(
                            (pad_h_up, new_width, num_channels), dtype=np.uint8
                        ) 
                    )
            
                v_concat_img_list.append(new_image)
                
                if pad_h_down:
                    v_concat_img_list.append(
                        np.zeros(
                            (pad_h_down, new_width, num_channels), dtype=np.uint8
                        )
                    )

                new_image = cv2.vconcat(v_concat_img_list)

        else:
            ## pad remaining width
            pad_w_image = np.zeros(
                (req_img_height, req_img_width-new_width, num_channels), dtype=np.uint8
            )

            new_image = cv2.hconcat([new_image, pad_w_image])

        assert new_image.shape[:2] == (req_img_height, req_img_width)

        return new_image

    def get_smart_batch_norm_img(self, img, max_width):
        resized_image = self.distortion_free_resize_word_image(
            img, req_img_width=max_width
        )
        norm_img = self.final_preprocess_for_model(resized_image)
        norm_img = norm_img[np.newaxis, :]
        return norm_img

    def rec_using_smart_batch(self, img_list):

        img_num = len(img_list)
        rec_res = [["", 0.0]] * img_num
        prb_chars = [
            math.ceil(img.shape[1] / float(img.shape[0])) for img in img_list
        ]

        ## defined by trials
        # prb_chars_2_width = {
        #     0:32, 1:32, 2:32, 3:64, 4:64, 5:80, 6:80, 7:108, 8:108, 9:128, 10:160
        # }
        prb_chars_2_width = {
            0:24,
            **{i:28*i for i in range(1, 6)},
            **{i:24*i for i in range(6, 10)},
            **{10:320}
        }

        # prb_chars_2_width = {i:320 for i in range(11)}

        by_batch_img_idxs = {}
        for idx, _ in enumerate(img_list):
            pc = min(10, prb_chars[idx])
            try:
                by_batch_img_idxs[pc].append(idx)
            except:
                by_batch_img_idxs[pc] = [idx]

        for pc, img_idxes in by_batch_img_idxs.items():

            norm_img_batch = [
                self.get_smart_batch_norm_img(
                    img_list[idx], max_width=prb_chars_2_width[pc]
                )
                for idx in img_idxes
            ]

            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
            
            ## run onnx model
            input_dict = {}
            input_dict[self.input_tensor_name] = norm_img_batch
            outputs = self.predictor.run(self.output_tensors, input_dict)

            for idx, rec_val in enumerate(self.postprocess_op(outputs[0])):
                rec_res[img_idxes[idx]] = rec_val

        return rec_res



if 1:

    r = RecOnncInfer(
        "all_models/rec_v3_server.onnx",
        "all_models/rec_v3_server.txt",
        use_space_char=True,
        rec_image_shape=[3, 48 if 1 else 32, 320]
    )

    # print("Model input shape => ", r.input_tensor.shape)
    # exit()

    test_img = cv2.imread("test_word.jpg")
    img_list = [test_img]*500

    st = time.time()
    resp = r.rec_using_smart_batch(img_list)
    print(resp[0])

    print(f"Time taken for {len(img_list)} => ", time.time()-st)


