import os
import time
import argparse
from tqdm import tqdm

import tensorrt as trt
import onnx
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from tqdm import tqdm

# import sys
# sys.path.append("../../sam2")
from sam2.utils.kalman_filter import KalmanFilter

import cv2


colors = [
    (0, 0, 255),     # red 0
    (0, 255, 0),     # green 1
    (255, 0, 0),     # blue 2
    (255, 255, 0),   # cyan 3
    (255, 0, 255),   # magenta 4
    (0, 255, 255),   # yellow 5
    (255, 255, 255), # white 6
    (128, 128, 128), # gray 7
    (140, 140, 0),   # mars green 8
    (167, 47, 0),    # klein blue 9
    (39, 88, 232),   # hermes orange 10
    (32, 0, 128),    # burgundy 11
    (208, 216, 129), # tiffany blue 12
    (9, 0, 76),      # bordeaux 13
    (36, 220, 249),  # sennelier yellow 14
]

class SAM2TrackerTRT:
    def __init__(self, args):
        self.args = args
        self.onnx_file_prefix = args.onnx_model_path
        self.trt_engine_prefix = args.trt_engine_path
        self.fp16_mode = args.use_fp16

        # Create a TensorRT logger
        self.trt_logger = trt.Logger(trt.Logger.WARNING)

        # load and deserialize TRT engine
        self.engines = self.init_engine()
        
        self.buffers = []
        self.contexts = []
        for engine in self.engines:
            profile_idx = range(engine.num_optimization_profiles)[0]
            inputs, outputs, bindings, stream = self.allocate_buffers(engine, profile_idx=profile_idx)
            context = engine.create_execution_context()
            self.buffers.append((inputs, outputs, bindings, stream))
            self.contexts.append(context)

        self.image_size = 512
        self.video_W, self.video_H = 1280, 720

        self.maskmem_tpos_enc = None

        self.memory_bank = {}
        self.kf = KalmanFilter()
        self.kf_mean = None
        self.kf_covariance = None
        self.stable_frames = 0

        self.stable_frames_threshold = 15
        self.stable_ious_threshold = 0.3
        self.kf_score_weight = 0.25
        self.memory_bank_iou_threshold = 0.5
        self.memory_bank_obj_score_threshold = 0.0
        self.memory_bank_kf_score_threshold = 0.0
        self.max_obj_ptrs_in_encoder = 16
        self.num_maskmem = 7

    def init_engine(self):
        '''
        Initialize TensorRT engines from ONNX files.
        '''
        # Check if the engine file exists
        onnx_models = ["image_encoder.onnx", "memory_attention.onnx", "memory_encoder.onnx", "mask_decoder.onnx"]
        trt_engines = [model.replace(".onnx", "_fp16.engine") if self.fp16_mode else model.replace(".onnx", ".engine") for model in onnx_models]
        engines = []

        if self.trt_engine_prefix is not None:
            for i, engine in enumerate(trt_engines):
                engine_path = os.path.join(self.trt_engine_prefix, engine)
                if not os.path.exists(engine_path):
                    raise FileNotFoundError("The {} engine file does not exist!".format(engine))
                else:
                    print("loading the {} ...".format(engine_path))
                engines.append(self.load_engine(engine_path))
        elif self.onnx_file_prefix is not None:
            for i, model in enumerate(onnx_models):
                onnx_model_path = os.path.join(self.onnx_file_prefix, model)
                engine_path = os.path.join(self.onnx_file_prefix, trt_engines[i])
                if os.path.exists(engine_path):
                    print("The {} fonud, skip building!".format(engine_path))
                    engines.append(self.load_engine(engine_path))
                    continue

                # check onnx model
                onnx_model = onnx.load(onnx_model_path)
                onnx.checker.check_model(onnx_model)
                print("The {} model is valid! building the engine...".format(onnx_model_path))

                engine = self.build_engine(onnx_model_path, self.fp16_mode)
                self.save_engine(engine, engine_path)
                print("The {} is saved!".format(engine_path))

                engines.append(self.load_engine(engine_path))
        else:
            print("Please specify the path to the TRT engine or ONNX model!")

        return engines

    def build_engine(self, onnx_file_path, fp16_mode=False):
        '''
        Build a TensorRT engine from an ONNX file.
        '''
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(self.trt_logger) as builder, \
             builder.create_network(EXPLICIT_BATCH) as network, \
             trt.OnnxParser(network, self.trt_logger) as parser:

            # Parse the ONNX model
            with open(onnx_file_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None

            # Configure the builder (optional settings)
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # Set workspace size to 1GB

            # Optionally, set other configurations like precision
            # For example, to use FP16 precision:
            if builder.platform_has_fast_fp16 and fp16_mode:
                # config.set_flag(trt.BuilderFlag.FP16)
                config.set_flag(trt.BuilderFlag.BF16)
                config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

            if 'memory_attention' in onnx_file_path:
                # 设置动态输入尺寸
                profile = builder.create_optimization_profile()
                input_info = network.get_input
                
                # for i in range(network.num_inputs):
                #     print("Input {} {}: {}".format(i, input_info(i).name, input_info(i).shape))

                input_shape = input_info(2).shape
                profile.set_shape("maskmem_feats", 
                    (input_shape[0], 1, input_shape[2], input_shape[3]),  # 最小尺寸
                    (input_shape[0], 7, input_shape[2], input_shape[3]),  # 优化尺寸
                    (input_shape[0], 7, input_shape[2], input_shape[3]))  # 最大尺寸
                
                input_shape = input_info(3).shape
                profile.set_shape("memory_pos_embed", 
                    (input_shape[0], 1, input_shape[2], input_shape[3]),  # 最小尺寸
                    (input_shape[0], 7, input_shape[2], input_shape[3]),  # 优化尺寸
                    (input_shape[0], 7, input_shape[2], input_shape[3]))  # 最大尺寸
                
                input_shape = input_info(4).shape
                profile.set_shape("obj_ptrs", 
                    (1,  input_shape[1], input_shape[2]),  # 最小尺寸
                    (16, input_shape[1], input_shape[2]),  # 优化尺寸
                    (16, input_shape[1], input_shape[2]))  # 最大尺寸
                
                input_shape = input_info(5).shape
                profile.set_shape("obj_pos", 
                    (1,),  # 最小尺寸
                    (16,),  # 优化尺寸
                    (16,))  # 最大尺寸
                config.add_optimization_profile(profile)

            # Build the engine
            engine = builder.build_serialized_network(network, config)
            
            return engine

    def save_engine(self, engine, engine_file_path):
        ''' 
        Serialize the engine and save it to a file.
        '''
        with open(engine_file_path, "wb") as f:
            f.write(engine)

    def load_engine(self, engine_file_path):
        with open(engine_file_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def allocate_buffers(self, engine, context=None, profile_idx=None):
        '''
        Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
        '''
        inputs_buffer = []
        outputs_buffer = []
        bindings = []
        stream = cuda.Stream()

        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)

            if context is not None:
                shape = context.get_tensor_shape(tensor_name)
            elif profile_idx is not None:
                shape = engine.get_tensor_profile_shape(tensor_name, profile_idx)[-1]
            else:
                shape = engine.get_tensor_shape(tensor_name)

            shape_valid = np.all([s >= 0 for s in shape])
            if not shape_valid and profile_idx is None:
                raise ValueError(f'Binding "{tensor_name}" has dynamic shape, but no profile was specified.')

            size = trt.volume(shape)
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype) # page-locked memory buffer (won't swapped to disk)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer address to device bindings. When cast to int, it's a linear index into the context's memory (like memory address). See https://documen.tician.de/pycuda/driver.html#pycuda.driver.DeviceAllocation
            bindings.append(int(device_mem))

            # Append to the appropriate input/output list.
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs_buffer.append({"name": tensor_name, "host_mem": host_mem, "device_mem": device_mem, "shape": shape})
            else:
                outputs_buffer.append({"name": tensor_name, "host_mem": host_mem, "device_mem": device_mem, "shape": shape})

        return inputs_buffer, outputs_buffer, bindings, stream

    def inference(self, input_datas, engine, context, buffers):
        '''
        Perform inference on the TensorRT engine.
        '''
        execute_begin = time.time()

        inputs_buffer, outputs_buffer, bindings, stream = buffers

        for i, input in enumerate(inputs_buffer):
            # print("Input {}: {}, {}, {}".format(i, inputs_buffer[i]['name'], input_datas[i].shape, inputs_buffer[i]['shape']))
            np.copyto(inputs_buffer[i]['host_mem'], input_datas[i].ravel())
        
        # 数据传输与执行
        [cuda.memcpy_htod_async(inputs_buffer[i]['device_mem'], inputs_buffer[i]['host_mem'], stream) for i in range(len(inputs_buffer))]

        for i in range(engine.num_io_tensors):
            context.set_tensor_address(engine.get_tensor_name(i), bindings[i])

        context.execute_async_v3(stream_handle=stream.handle)

        [cuda.memcpy_dtoh_async(outputs_buffer[i]['host_mem'], outputs_buffer[i]['device_mem'], stream) for i in range(len(outputs_buffer))]

        stream.synchronize()
        execute_end = time.time()

        # print("execute_async_v3 time: {:.3} ms".format((execute_end - execute_begin) * 1000))

        retrieve_begin = time.time()
        outputs_data = [output['host_mem'].reshape(output['shape']).copy() for output in outputs_buffer]
        retrieve_end = time.time()
        # print("Retrieve time: {:.3} ms".format((retrieve_end - retrieve_begin) * 1000))

        # [cuda.memcpy_htod(inputs_buffer[i]['device_mem'], inputs_buffer[i]['host_mem']) for i in range(len(inputs_buffer))]
        # context.execute_v2(bindings=bindings)
        # [cuda.memcpy_dtoh(output['host_mem'], output['device_mem']) for output in outputs_buffer]
        # outputs_data = [output['host_mem'].reshape(output['shape']) for output in outputs_buffer]

        return outputs_data

    def image_encoder_inference(self, input_image):
        '''
        Image encoder inference.
        '''
        return self.inference([input_image,], self.engines[0], self.contexts[0], self.buffers[0])

    def memory_attention_inference_test(self, frame_idx, vision_feats, vision_pos, i):
        '''
        Memory attention inference for testing.
        '''
        m = min(i, 16)
        n = min(i, 7)
        memory = np.ones((1024, n, 1, 64)).astype(np.float32)
        memory_pos_embed = np.ones((1024, n, 1, 64)).astype(np.float32)
        object_ptrs = np.ones((m, 1, 256)).astype(np.float32)
        obj_pos_enc = np.ones((m,)).astype(np.int32)

        inputs_buffer, outputs_buffer, bindings, stream = self.buffers[1]
        # self.contexts[1].set_optimization_profile_async(0, stream=stream.handle)
        self.contexts[1].set_input_shape("maskmem_feats", (1024, n, 1, 64))
        self.contexts[1].set_input_shape("memory_pos_embed", (1024, n, 1, 64))
        self.contexts[1].set_input_shape("obj_ptrs", (m, 1, 256))
        self.contexts[1].set_input_shape("obj_pos", (m,))

        self.buffers[1] = self.allocate_buffers(self.engines[1], context=self.contexts[1])
        memory_attention_outputs = self.inference([vision_feats, vision_pos, memory, memory_pos_embed, object_ptrs, obj_pos_enc],
                                                  self.engines[1], self.contexts[1], self.buffers[1])
        return memory_attention_outputs

    def memory_attention_inference(self, frame_idx, vision_feats, vision_pos):
        '''
        Memory attention inference.
        '''
        memmask_features = [self.memory_bank[0]['maskmem_features'].copy()]
        memmask_pos_enc = [self.memory_bank[0]['maskmem_pos_enc'] + self.maskmem_tpos_enc[6]]
        object_ptrs = [self.memory_bank[0]['obj_ptr'].copy()]
        ## samurai----------------------------------------------- ##
        valid_indices = []
        if frame_idx > 1:
            for i in range(frame_idx - 1, 0, -1):  # Iterate backwards through previous frames
                # Check the number of valid indices
                if len(valid_indices) >= self.max_obj_ptrs_in_encoder - 1:
                    break
                iou_score = self.memory_bank[i]["best_iou_score"]  # Get mask affinity score
                obj_score = self.memory_bank[i]["obj_score_logits"]  # Get object score
                kf_score = self.memory_bank[i]["kf_score"]  # Get motion score if available
                # Check if the scores meet the criteria for being a valid index
                if iou_score > self.memory_bank_iou_threshold and \
                    obj_score > self.memory_bank_obj_score_threshold and \
                    (kf_score is None or kf_score > self.memory_bank_kf_score_threshold):
                    valid_indices.insert(0, i)
                # valid_indices.insert(0, i)

        # print("valid_indices: ", valid_indices, '\nprev_frame_idx : ', end='')
        # 最近6帧的memmask_features
        for prev_frame_idx in valid_indices[::-1]:
            # print(prev_frame_idx, end=', ')
            mem = self.memory_bank.get(prev_frame_idx, None)
            if mem is not None:
                memmask_features.insert(1, mem['maskmem_features'].copy())
                memmask_pos_enc.insert(1, mem['maskmem_pos_enc'].copy())
            if len(memmask_features) >= self.num_maskmem:
                break
        # print()
        ## samurai----------------------------------------------- ##

        obj_pos_enc = np.arange(1, frame_idx)[:15]
        obj_pos_enc = np.insert(obj_pos_enc, 0, frame_idx).astype(np.int32)
        obj_pos_enc = obj_pos_enc[:self.max_obj_ptrs_in_encoder]
        # print('obj_pos_enc : ', obj_pos_enc)
        # 最近15帧的object_ptrs
        # print("object_ptrs: ", end='')
        for i in range(frame_idx - 15, frame_idx):
            if i < 1:
                continue
            if len(object_ptrs) >= self.max_obj_ptrs_in_encoder:
                break
            mem = self.memory_bank.get(i, None)
            if mem is not None:
                # print(i, end=', ')
                object_ptrs.append(mem['obj_ptr'].copy())
        # print()

        for i, pos_enc in enumerate(reversed(memmask_pos_enc[1:])):
            pos_enc[:] = pos_enc[:] + self.maskmem_tpos_enc[i]
        
        memory = np.concatenate(memmask_features, axis=0)
        memory_pos_embed = np.concatenate(memmask_pos_enc, axis=0)
        memory = memory.reshape(-1, len(memmask_features), memory.shape[-2], memory.shape[-1])
        memory_pos_embed = memory_pos_embed.reshape(-1, len(memmask_pos_enc), memory_pos_embed.shape[-2], memory_pos_embed.shape[-1])

        object_ptrs = object_ptrs[0:1] + object_ptrs[1:][::-1]
        object_ptrs = np.stack(object_ptrs, axis=0)

        # dynamic_shapes = {
        #     "maskmem_feats": memory.shape,
        #     "memory_pos_embed": memory_pos_embed.shape,
        #     "obj_ptrs": object_ptrs.shape,
        #     "obj_pos": obj_pos_enc.shape,
        # }
        # print("dynamic_shapes: ", dynamic_shapes)
        inputs_buffer, outputs_buffer, bindings, stream = self.buffers[1]
        # self.contexts[1].set_optimization_profile_async(profile_idx=0, stream=stream)
        self.contexts[1].set_input_shape("maskmem_feats", memory.shape)
        self.contexts[1].set_input_shape("memory_pos_embed", memory_pos_embed.shape)
        self.contexts[1].set_input_shape("obj_ptrs", object_ptrs.shape)
        self.contexts[1].set_input_shape("obj_pos", obj_pos_enc.shape)
        self.buffers[1] = self.allocate_buffers(self.engines[1], context=self.contexts[1])

        memory_attention_outputs = self.inference([vision_feats, vision_pos, memory, memory_pos_embed, object_ptrs, obj_pos_enc],
                                                    self.engines[1], self.contexts[1], self.buffers[1])

        return memory_attention_outputs

    def memory_encoder_inference(self, vision_feats, high_res_feats, obj_score_logits, isMaskFromPts):
        '''
        Memory encoder inference.
        '''
        return self.inference([vision_feats, high_res_feats, obj_score_logits, isMaskFromPts],
                               self.engines[2], self.contexts[2], self.buffers[2])

    def mask_decoder_inference(self, input_points, input_labels, pixel_feat_with_memory, high_res_feats0, high_res_feats1):
        '''
        Mask decoder inference.
        '''
        return self.inference([input_points, input_labels, pixel_feat_with_memory, high_res_feats0, high_res_feats1],
                               self.engines[3], self.contexts[3], self.buffers[3])

    def add_first_frame_bbox(self, frame_idx, image, first_frame_bbox):
        '''
        Add the first bbox when the frame_idx is 0.
        '''
        input_image = image.astype(np.float32)[np.newaxis, ...]

        image_encoder_outputs = self.image_encoder_inference(input_image)
        high_res_features0, high_res_features1, low_res_features, _, pix_feat_with_mem = image_encoder_outputs
        # for o in image_encoder_outputs:
        #     print(f'image_encoder_outputs : {o.shape}, {o.sum()}, {o.min()}, {o.max()}, {o.mean()}, {o.std()}')
        # print()

        box_coords = np.array(first_frame_bbox).reshape((1, 2, 2))
        box_labels = np.array([2, 3]).reshape((1, 2))

        # video_H, video_W = frame.shape[:2]
        points = box_coords / np.array([self.video_W, self.video_H])

        input_points = (points * self.image_size).astype(np.float32)
        input_labels = box_labels.astype(np.int32)

        mask_decoder_outputs = self.mask_decoder_inference(input_points, input_labels, pix_feat_with_mem, high_res_features0, high_res_features1)
        low_res_multimasks, ious, obj_ptrs, object_score_logits, self.maskmem_tpos_enc = mask_decoder_outputs
        # for o in mask_decoder_outputs:
        #     print(f'mask_decoder_outputs : {o.shape}, {o.sum()}, {o.min()}, {o.max()}, {o.mean()}, {o.std()}')
        # print()

        pred_mask, high_res_masks_for_mem, best_iou_inds, kf_score = self._forward_sam_head(mask_decoder_outputs)

        # memory_encoder predict
        is_mask_from_pts = np.array([frame_idx==0]).astype(bool)
        memory_encoder_outputs = self.memory_encoder_inference(low_res_features, high_res_masks_for_mem, object_score_logits, is_mask_from_pts)
        maskmem_features, maskmem_pos_enc = memory_encoder_outputs
        # for o in memory_encoder_outputs:
        #     print(f'memory_encoder_outputs : {o.shape}, {o.sum()}, {o.min()}, {o.max()}, {o.mean()}, {o.std()}')
        # print()

        # save to memory bank
        self.memory_bank[0] = {
                                'maskmem_features': maskmem_features,
                                'maskmem_pos_enc': maskmem_pos_enc,
                                'obj_ptr': obj_ptrs[0, best_iou_inds],
                                'best_iou_score': ious[0, best_iou_inds],
                                'obj_score_logits': object_score_logits,
                                'kf_score': kf_score,
                                }

        return pred_mask.squeeze()

    def track_step(self, frame_idx, image):
        # print(f"\033[93mframe_idx: {frame_idx}\033[0m")

        # step 1:image_encoder predict, get image feature
        start = time.time()
        # input_image = self._normalize_image(image)
        # input_image = input_image[np.newaxis, ...]
        input_image = image.astype(np.float32)[np.newaxis, ...]

        image_encoder_outputs = self.image_encoder_inference(input_image)
        high_res_features0, high_res_features1, low_res_features, vision_pos_embeds, _ = image_encoder_outputs
        # for o in image_encoder_outputs:
        #     print(f'image_encoder_outputs : {o.shape}, {o.sum()}, {o.min()}, {o.max()}, {o.mean()}, {o.std()}')
        # print()

        # step 2: memory_attention predict
        memory_attention_outputs = self.memory_attention_inference(frame_idx, low_res_features, vision_pos_embeds)
        pix_feat_with_mem = memory_attention_outputs[0]
        # print(f"pix_feat_with_mem :  {pix_feat_with_mem.shape}, {pix_feat_with_mem.sum()}, {pix_feat_with_mem.min()}, {pix_feat_with_mem.max()}, {pix_feat_with_mem.mean()}, {pix_feat_with_mem.std()}")

        # step 3 : mask decoder predict
        input_points = np.zeros((1, 2, 2), dtype=np.float32)
        input_labels = -np.ones((1, 2), dtype=np.int32)
        mask_decoder_outputs = self.mask_decoder_inference(input_points, input_labels, pix_feat_with_mem, high_res_features0, high_res_features1)
        _, ious, obj_ptrs, object_score_logits, _ = mask_decoder_outputs
        # for o in mask_decoder_outputs:
        #     print(f'mask_decoder_outputs : {o.shape}, {o.sum()}, {o.min()}, {o.max()}, {o.mean()}, {o.std()}')
        # print()
        # print('ious : ', ious)
        # print('object_score_logits : ', object_score_logits)

        pred_mask, high_res_masks_for_mem, best_iou_inds, kf_score = self._forward_sam_head(mask_decoder_outputs)

        # step 4 : memory_encoder predict, save maskmem to memory bank
        is_mask_from_pts = np.array([frame_idx==0]).astype(bool)
        memory_encoder_outputs = self.memory_encoder_inference(low_res_features, high_res_masks_for_mem, object_score_logits, is_mask_from_pts)
        maskmem_features, maskmem_pos_enc = memory_encoder_outputs
        # for o in memory_encoder_outputs:
        #     print(f'memory_encoder_outputs : {o.shape}, {o.sum()}, {o.min()}, {o.max()}, {o.mean()}, {o.std()}')
        # print()

        self.memory_bank[frame_idx] = {
                                'maskmem_features': maskmem_features,
                                'maskmem_pos_enc': maskmem_pos_enc,
                                'obj_ptr': obj_ptrs[0, best_iou_inds],
                                'best_iou_score': ious[0, best_iou_inds],
                                'obj_score_logits': object_score_logits,
                                'kf_score': kf_score,
                                }
        
        return pred_mask.squeeze()

    def _forward_sam_head(self, mask_decoder_outputs):
        low_res_multimasks, ious, _, _, _ = mask_decoder_outputs
        # high_res_multimasks = F.interpolate(low_res_multimasks, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        high_res_multimasks = cv2.resize(low_res_multimasks[0].transpose(1, 2, 0), (self.image_size, self.image_size))
        high_res_multimasks = high_res_multimasks.transpose(2, 0, 1)[None, ...]

        ## samurai ---------------------------------------------------------------------##
        B = 1
        kf_ious = None
        if self.kf_mean is None and self.kf_covariance is None or self.stable_frames == 0:
            best_iou_inds = np.argmax(ious, axis=-1)
            batch_inds = np.arange(B)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds][:, None]
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds][:, None]
            non_zero_indices = np.argwhere(high_res_masks[0][0] > 0.0)
            if len(non_zero_indices) == 0:
                high_res_bbox = [0, 0, 0, 0]
            else:
                y_min, x_min = non_zero_indices.min(axis=0).tolist()
                y_max, x_max = non_zero_indices.max(axis=0).tolist()
                high_res_bbox = [x_min, y_min, x_max, y_max]
            self.kf_mean, self.kf_covariance = self.kf.initiate(self.kf.xyxy_to_xyah(high_res_bbox))

            self.stable_frames += 1
        elif self.stable_frames < self.stable_frames_threshold:
            self.kf_mean, self.kf_covariance = self.kf.predict(self.kf_mean, self.kf_covariance)
            best_iou_inds = np.argmax(ious, axis=-1)
            batch_inds = np.arange(B)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds][:, None]
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds][:, None]
            non_zero_indices = np.argwhere(high_res_masks[0][0] > 0.0)
            if len(non_zero_indices) == 0:
                high_res_bbox = [0, 0, 0, 0]
            else:
                y_min, x_min = non_zero_indices.min(axis=0).tolist()
                y_max, x_max = non_zero_indices.max(axis=0).tolist()
                high_res_bbox = [x_min, y_min, x_max, y_max]
            if ious[0][best_iou_inds] > self.stable_ious_threshold:
                self.kf_mean, self.kf_covariance = self.kf.update(self.kf_mean, self.kf_covariance, self.kf.xyxy_to_xyah(high_res_bbox))
                self.stable_frames += 1
            else:
                self.stable_frames = 0
        else:
            self.kf_mean, self.kf_covariance = self.kf.predict(self.kf_mean, self.kf_covariance)
            high_res_multibboxes = []
            batch_inds = np.arange(B)
            for i in range(ious.shape[1]):
                non_zero_indices = np.argwhere(high_res_multimasks[batch_inds, i][0] > 0.0)
                if len(non_zero_indices) == 0:
                    high_res_multibboxes.append([0, 0, 0, 0])
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    high_res_multibboxes.append([x_min, y_min, x_max, y_max])
            # compute the IoU between the predicted bbox and the high_res_multibboxes
            kf_ious = np.array(self.kf.compute_iou(self.kf_mean[:4], high_res_multibboxes))
            # weighted iou
            weighted_ious = self.kf_score_weight * kf_ious + (1 - self.kf_score_weight) * ious
            best_iou_inds = np.argmax(weighted_ious, axis=-1)
            batch_inds = np.arange(B)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds][:, None]
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds][:, None]

            if ious[0][best_iou_inds] < self.stable_ious_threshold:
                self.stable_frames = 0
            else:
                self.kf_mean, self.kf_covariance = self.kf.update(self.kf_mean, self.kf_covariance, self.kf.xyxy_to_xyah(high_res_multibboxes[best_iou_inds.item()]))
        
        ## sam2 ---------------------------------------------------------------------##
        # best_iou_inds = np.argmax(ious, axis=-1)
        # batch_inds = np.arange(1)
        # low_res_masks = low_res_multimasks[batch_inds, best_iou_inds][:, None]
        # high_res_masks = high_res_multimasks[batch_inds, best_iou_inds][:, None]

        best_iou_score = ious[0][best_iou_inds]
        kf_score = kf_ious[best_iou_inds] if kf_ious is not None else None
        pred_mask = low_res_masks
        high_res_masks_for_mem = high_res_masks

        return pred_mask, high_res_masks_for_mem, best_iou_inds, kf_score

def load_txt(gt_path):
    with open(gt_path, 'r') as f:
        gt = f.readlines()
    prompts = {}
    for fid, line in enumerate(gt):
        x, y, w, h = map(float, line.split(','))
        x, y, w, h = int(x), int(y), int(w), int(h)
        prompts[fid] = ((x, y, x + w, y + h), 0)
    return prompts

def main(args):
    tracker = SAM2TrackerTRT(args)

    image_size = tracker.image_size
    cap = cv2.VideoCapture(args.video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('./trt_demo.mp4', fourcc, 30, (frame_width, frame_height))

    name_window = os.path.basename(args.video_path)
    cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)

    start = time.time()
    for frame_idx in tqdm(range(num_frames), desc="Processing video frames"):
        # print(f"\033[93mframe_idx: {frame_idx}\033[0m")
        # start = time.time()
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Failed to read frame from video.")

        input_image = cv2.resize(frame, (image_size, image_size))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        if frame_idx == 0:
            # bbox = cv2.selectROI(name_window, frame) # (x, y, w, h)
            # x, y, w, h = bbox
            # first_frame_bbox = [x, y, x + w, y + h]

            first_frame_bbox = load_txt(args.txt_path)[0][0]

            mask = tracker.add_first_frame_bbox(0, input_image, first_frame_bbox)
        else:
            mask = tracker.track_step(frame_idx, input_image)

        mask = cv2.resize(mask, (frame_width, frame_height))
        mask = mask > 0.0
        non_zero_indices = np.argwhere(mask)
        if len(non_zero_indices) == 0:
            bbox = [0, 0, 0, 0]
        else:
            y_min, x_min = non_zero_indices.min(axis=0).tolist()
            y_max, x_max = non_zero_indices.max(axis=0).tolist()
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

        mask_img = np.zeros((frame_height, frame_width, 3), np.uint8)
        mask_img[mask] = colors[1]
        frame = cv2.addWeighted(frame, 1, mask_img, 0.4, 0)

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), colors[1], 2)

        cv2.imshow(name_window, frame)
        cv2.waitKey(1)

    elapsed = (time.time() - start) * 1000
    print(f"Elapsed time: {elapsed:.3f} ms")
    print(f"every frame spend time: {elapsed / (frame_idx + 1):.2f}ms")
    print(f"fps: {1000 / (elapsed / (frame_idx + 1)):.2f}")

    cap.release()
    # out.release()
    cv2.destroyAllWindows()

def test(args):
    tracker = SAM2TrackerTRT(args)

    # # test image_encoder inference
    # input_data = np.ones((1, 512, 512, 3)).astype(np.float32)
    # outputs = tracker.image_encoder_inference(input_data)
    # for i, output in enumerate(outputs):
    #     print("Output {}: {}, {}".format(i, output.shape, output.sum()))

    # test memory_attention inference
    for i in range(20):
        vision_feats = np.ones((1024, 1, 256)).astype(np.float32)
        vision_pos = np.ones((1024, 1, 256)).astype(np.float32)
        outputs = tracker.memory_attention_inference_test(0, vision_feats, vision_pos, i+1)
        for i, output in enumerate(outputs):
            print("Output {}: {}, {}".format(i, output.shape, output.sum()), output.min(), output.max(), output.mean(), output.std())

    # # test memory_encoder inference
    # vision_feats = np.ones((1024, 1, 256)).astype(np.float32)
    # high_res_feats = np.ones((1, 1, 512, 512)).astype(np.float32)
    # obj_score_logits = np.ones((1, 1)).astype(np.float32)
    # isMaskFromPts = np.array(0).astype(np.bool)
    # outputs = tracker.memory_encoder_inference(vision_feats, high_res_feats, obj_score_logits, isMaskFromPts)
    # for i, output in enumerate(outputs):
    #     print("Output {}: {}, {}".format(i, output.shape, output.sum()))

    # # test mask_decoder inference
    # # input_points = np.array([[[307.2000, 432.3556],
    # #                         [580.8000, 881.7778]]], dtype=np.float32)
    # # input_labels = np.array([2, 3], dtype=np.int32)
    # input_points = np.ones((1, 2, 2)).astype(np.float32)
    # input_labels = np.ones((1, 2)).astype(np.int32)

    # pixel_feat_with_memory = np.ones((1, 256, 32, 32)).astype(np.float32)

    # high_res_feats0 = np.ones((1, 32, 128, 128)).astype(np.float32)
    # high_res_feats1 = np.ones((1, 64, 64, 64)).astype(np.float32)

    # outputs = tracker.mask_decoder_inference(input_points, input_labels, pixel_feat_with_memory, high_res_feats0, high_res_feats1)
    # for i, output in enumerate(outputs):
    #     print(f"Output {i} ", output.shape, output.sum())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="./data/1917-1.mp4", help="Input video path or directory of frames.")
    parser.add_argument("--txt_path", default="./first_frame_bbox.txt", help="Path to ground truth text file.")
    parser.add_argument("--onnx_model_path", default="./onnx_model", help="Path to the onnx model.")
    parser.add_argument("--trt_engine_path", help="Path to the tensorRT model.")
    parser.add_argument("--video_output_path", default="demo.mp4", help="Path to save the output video.")
    parser.add_argument("--save_to_video", default=False, help="Save results to a video.")
    parser.add_argument("--use_fp16", default=True, help="Use FP16 precision for inference.")
    args = parser.parse_args()

    main(args)
    # test(args)
