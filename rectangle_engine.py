"""Full RoMa with a fixed reference and optional reference-feature caching."""
import gc
import threading
import time
from types import MethodType

from runtime_config import ROOT
from match_settings import SETTINGS
import numpy as np
import torch
from PIL import Image
from romatch import roma_indoor, roma_outdoor
from romatch.utils import get_tuple_transform_ops

torch.set_float32_matmul_precision('highest')


class RectangleMatcher:
    def __init__(self, reference):
        self.reference = reference
        self.lock = threading.RLock()
        self.model = None
        self.variant = None
        self.features = {}
        self.cache_enabled = True
        self.state = '模型尚未预加载'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_seconds = 0

    def load(self, variant=SETTINGS['variant']):
        with self.lock:
            if self.variant == variant:
                return
            if variant not in ('indoor','outdoor'):
                raise ValueError('未知权重')
            self.state = '正在加载 '+variant
            self.model = None
            self.original_extract = None
            self.reference_tensors = None
            self.features.clear()
            self.variant = None
            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                torch.backends.cudnn.benchmark = True
            started = time.perf_counter()
            factory = roma_indoor if variant=='indoor' else roma_outdoor
            self.model = factory(device=self.device, use_custom_corr=False).eval()
            self.original_extract = self.model.extract_backbone_features
            engine = self

            def cached_extract(model, batch, batched=True, upsample=False):
                if not engine.cache_enabled or not batched or 'unique_images' in batch:
                    return engine.original_extract(batch, batched, upsample)
                if upsample not in engine.features:
                    engine.features[upsample] = model.encoder(batch['im_A'], upsample=upsample)
                query = model.encoder(batch['im_B'], upsample=upsample)
                return {scale:torch.cat((value,query[scale]),dim=0)
                        for scale,value in engine.features[upsample].items()}

            self.model.extract_backbone_features = MethodType(cached_extract,self.model)
            self.transforms = [get_tuple_transform_ops(resize=(560,560),normalize=True),
                               get_tuple_transform_ops(resize=(864,864),normalize=True)]
            self.reference_tensors = [transform((self.reference,))[0][None].to(self.device)
                                      for transform in self.transforms]
            self.variant = variant
            self.load_seconds = time.perf_counter()-started
            self.state = variant+' 已加载'

    def match(self, current, variant=SETTINGS['variant'], cache=SETTINGS['cache']):
        with self.lock,torch.inference_mode():
            self.load(variant)
            self.cache_enabled = cache
            if self.device=='cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            tensors=[transform((current,))[0][None].to(self.device) for transform in self.transforms]
            warp,certainty=self.model.match(self.reference_tensors[0],tensors[0],
                                            im_A_high_res=self.reference_tensors[1],
                                            im_B_high_res=tensors[1],device=self.device)
            if self.device=='cuda':
                torch.cuda.synchronize()
            return warp,certainty,{'inference_seconds':time.perf_counter()-start,
                                   'peak_gpu_MB':round(torch.cuda.max_memory_allocated()/1024**2,1) if self.device=='cuda' else None,
                                   'reference_feature_cache':cache,'load_seconds':self.load_seconds}

    def preload(self):
        try:
            self.state='正在预加载和预热 '+SETTINGS['variant']
            self.match(self.reference)
            self.state=SETTINGS['variant']+' 已预热，可开始匹配'
        except Exception as error:
            self.state='预加载失败：'+str(error)
