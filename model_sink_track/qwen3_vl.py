# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3VL model compatible with HuggingFace weights."""
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature
from transformers.models.qwen2_vl import Qwen2VLImageProcessorFast
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    smart_resize as image_smart_resize)
from transformers.models.qwen3_vl import (Qwen3VLProcessor,
                                          Qwen3VLVideoProcessor)
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig, Qwen3VLVisionConfig)
from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    smart_resize as video_smart_resize)
from transformers.video_utils import VideoMetadata

from vllm.attention.layer import check_upstream_fa_availability
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargsItem,
                                    MultiModalKwargsItems, VideoItem)
from vllm.multimodal.parse import (ImageSize, MultiModalDataItems,
                                   MultiModalDataParser)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        PromptReplacement, PromptUpdate,
                                        PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.platforms import _Backend
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import uses_mrope
from vllm.utils import is_list_of

from .interfaces import (MultiModalEmbeddings, SupportsLoRA,
                         SupportsMultiModal, SupportsPP)
from .qwen2_5_vl import (Qwen2_5_VisionAttention,
                         Qwen2_5_VisionRotaryEmbedding,
                         Qwen2_5_VLImageEmbeddingInputs, Qwen2_5_VLImageInputs,
                         Qwen2_5_VLImagePixelInputs,
                         Qwen2_5_VLVideoEmbeddingInputs, Qwen2_5_VLVideoInputs,
                         Qwen2_5_VLVideoPixelInputs)
from .qwen2_vl import Qwen2VLProcessingInfo
from .qwen3 import Qwen3ForCausalLM, Qwen3Model
from .utils import (AutoWeightsLoader, PPMissingLayer, WeightsMapper,
                    maybe_prefix, merge_multimodal_embeddings)
from .vision import get_vit_attn_backend, run_dp_sharded_mrope_vision_model

from vllm.model_executor.layers.layernorm import RMSNorm

logger = init_logger(__name__)

# Official recommended max pixels is 24576 * 32 * 32
_MAX_FRAMES_PER_VIDEO = 24576

import logging
import os
import sys

logger = logging.getLogger(__name__)

if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        '%(processName)s(pid=%(process)d)[%(levelname)s] %(name)s:%(lineno)d - %(message)s',
        datefmt='%H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    logger.propagate = False
# =====================================

import torch._dynamo as dynamo

def _sink_debug_log(msg: str, *args, level: str = "debug") -> None:
    if dynamo.is_compiling():
        return
    if level == "warning":
        logger.warning(msg, *args)
    elif level == "error":
        logger.error(msg, *args)
    else:
        logger.debug(msg, *args)


class Qwen3_VisionPatchEmbed(nn.Module):

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(in_channels,
                              hidden_size,
                              kernel_size=kernel_size,
                              stride=kernel_size,
                              bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L, C = x.shape
        x = x.view(L, -1, self.temporal_patch_size, self.patch_size,
                   self.patch_size)
        x = self.proj(x).view(L, self.hidden_size)
        return x


class Qwen3_VisionMLP(nn.Module):

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 bias: bool = False,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "",
                 use_data_parallel: bool = False):
        super().__init__()
        self.linear_fc1 = ColumnParallelLinear(in_features,
                                               hidden_features,
                                               bias=bias,
                                               quant_config=quant_config,
                                               return_bias=False,
                                               prefix=f"{prefix}.linear_fc1",
                                               disable_tp=use_data_parallel)
        self.linear_fc2 = RowParallelLinear(hidden_features,
                                            in_features,
                                            bias=bias,
                                            quant_config=quant_config,
                                            return_bias=False,
                                            prefix=f"{prefix}.linear_fc2",
                                            disable_tp=use_data_parallel)
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor):
        mlp_output = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return mlp_output


class Qwen3_VisionBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
        attn_backend: _Backend = _Backend.TORCH_SDPA,
        use_upstream_fa: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Qwen2_5_VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            use_data_parallel=use_data_parallel,
            attn_backend=attn_backend,
            use_upstream_fa=use_upstream_fa)
        self.mlp = Qwen3_VisionMLP(dim,
                                   mlp_hidden_dim,
                                   act_fn=act_fn,
                                   bias=True,
                                   quant_config=quant_config,
                                   prefix=f"{prefix}.mlp",
                                   use_data_parallel=use_data_parallel)

    def forward(
            self,
            x: torch.Tensor,
            cu_seqlens: torch.Tensor,
            rotary_pos_emb: torch.Tensor,
            max_seqlen: Optional[int] = None,  # Only used for Flash Attention
            seqlens: Optional[list[int]] = None,  # Only used for xFormers
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x),
                          cu_seqlens=cu_seqlens,
                          rotary_pos_emb=rotary_pos_emb,
                          max_seqlen=max_seqlen,
                          seqlens=seqlens)

        x = x + self.mlp(self.norm2(x))
        return x


class Qwen3_VisionPatchMerger(nn.Module):

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)

        self.use_postshuffle_norm = use_postshuffle_norm
        if self.use_postshuffle_norm:
            context_dim = self.hidden_size

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(context_dim)
        self.linear_fc1 = ColumnParallelLinear(self.hidden_size,
                                               self.hidden_size,
                                               bias=True,
                                               quant_config=quant_config,
                                               prefix=f"{prefix}.linear_fc1",
                                               disable_tp=use_data_parallel)
        self.act_fn = nn.GELU()
        self.linear_fc2 = RowParallelLinear(self.hidden_size,
                                            d_model,
                                            bias=True,
                                            quant_config=quant_config,
                                            prefix=f"{prefix}.linear_fc2",
                                            disable_tp=use_data_parallel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)

        x_parallel, _ = self.linear_fc1(x)
        x_parallel = self.act_fn(x_parallel)
        out, _ = self.linear_fc2(x_parallel)
        return out


class Qwen3_VisionTransformer(nn.Module):

    def __init__(
        self,
        vision_config: Qwen3VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.num_position_embeddings = vision_config.num_position_embeddings
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.deepstack_visual_indexes = vision_config.deepstack_visual_indexes
        self.use_data_parallel = use_data_parallel
        self.num_grid_per_side = int(self.num_position_embeddings**0.5)

        # NOTE: This is used for creating empty tensor for all_gather for
        # DP ViT. Here out_hidden_size is enlarged due to deepstack
        self.out_hidden_size = (vision_config.out_hidden_size *
                                (1 + len(self.deepstack_visual_indexes)))

        self.patch_embed = Qwen3_VisionPatchEmbed(
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            in_channels=vision_config.in_channels,
            hidden_size=self.hidden_size,
        )

        self.pos_embed = nn.Embedding(self.num_position_embeddings,
                                      self.hidden_size)

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.merger = Qwen3_VisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=self.hidden_size,
            norm_layer=norm_layer,
            spatial_merge_size=self.spatial_merge_size,
            quant_config=quant_config,
            prefix=f"{prefix}.merger",
            use_data_parallel=use_data_parallel,
        )

        self.deepstack_merger_list = nn.ModuleList([
            Qwen3_VisionPatchMerger(
                d_model=vision_config.out_hidden_size,
                context_dim=self.hidden_size,
                spatial_merge_size=self.spatial_merge_size,
                use_postshuffle_norm=True,
                norm_layer=norm_layer,
                quant_config=quant_config,
                prefix=f"{prefix}.deepstack_merger_list.{layer_idx}",
                use_data_parallel=use_data_parallel)
            for layer_idx in range(len(self.deepstack_visual_indexes))
        ])

        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim, dtype=torch.get_default_dtype())
        use_upstream_fa = False
        if self.attn_backend != _Backend.FLASH_ATTN and \
            check_upstream_fa_availability(
                torch.get_default_dtype()):
            self.attn_backend = _Backend.FLASH_ATTN
            use_upstream_fa = True

        if self.attn_backend not in {
                _Backend.FLASH_ATTN, _Backend.TORCH_SDPA, _Backend.XFORMERS,
                _Backend.ROCM_AITER_FA
        }:
            raise RuntimeError(
                f"Qwen3-VL does not support {self.attn_backend} backend now.")

        self.blocks = nn.ModuleList([
            Qwen3_VisionBlock(
                dim=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_dim=vision_config.intermediate_size,
                act_fn=_ACTIVATION_REGISTRY[vision_config.hidden_act],
                norm_layer=norm_layer,
                quant_config=quant_config,
                prefix=f"{prefix}.blocks.{layer_idx}",
                use_data_parallel=use_data_parallel,
                attn_backend=self.attn_backend,
                use_upstream_fa=use_upstream_fa)
            for layer_idx in range(vision_config.depth)
        ])

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        # Support both Tensor and list inputs for DP path
        if isinstance(grid_thw, list):
            grid_list = grid_thw
            max_grid_size = max(max(h, w) for _, h, w in grid_list)
        else:
            grid_list = grid_thw.tolist()
            max_grid_size = int(grid_thw[:, 1:].max().item())
        for t, h, w in grid_list:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(
                torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def fast_pos_embed_interpolate(self,
                                   grid_thw: list[list[int]]) -> torch.Tensor:

        num_grid_per_side = self.num_grid_per_side
        m_size = self.spatial_merge_size
        hidden_dim = self.pos_embed.embedding_dim

        outputs = []
        for t, h, w in grid_thw:
            h_idxs = torch.linspace(0,
                                    num_grid_per_side - 1,
                                    h,
                                    dtype=torch.float32,
                                    device=self.device)
            w_idxs = torch.linspace(0,
                                    num_grid_per_side - 1,
                                    w,
                                    dtype=torch.float32,
                                    device=self.device)

            h_floor = h_idxs.to(torch.long)
            w_floor = w_idxs.to(torch.long)
            h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
            w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            # Create meshgrid view for all h, w vars
            dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing='ij')
            h_floor_grid, w_floor_grid = torch.meshgrid(h_floor,
                                                        w_floor,
                                                        indexing='ij')
            h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil,
                                                      w_ceil,
                                                      indexing='ij')
            h_floor_grid_idx = h_floor_grid * num_grid_per_side
            h_ceil_grid_idx = h_ceil_grid * num_grid_per_side

            # original computation of weights
            # w00 = (1 - dh_grid) * (1 - dw_grid)
            # w01 = (1 - dh_grid) * dw_grid
            # w10 = dh_grid * (1 - dw_grid)
            # w11 = dh_grid * dw_grid
            # we reuse w11 here to avoid duplicate
            # dh_grid * dw_grid computation
            w11 = dh_grid * dw_grid
            w10 = dh_grid - w11
            w01 = dw_grid - w11
            w00 = 1 - dh_grid - dw_grid + w11

            idx00 = h_floor_grid_idx + w_floor_grid
            idx01 = h_floor_grid_idx + w_ceil_grid
            idx10 = h_ceil_grid_idx + w_floor_grid
            idx11 = h_ceil_grid_idx + w_ceil_grid

            indices = torch.stack([idx00, idx01, idx10, idx11],
                                  dim=0).reshape(4, -1)
            weights = torch.stack([w00, w01, w10, w11],
                                  dim=0).reshape(4, -1, 1)
            weights = weights.to(dtype=self.dtype, device=self.device)

            embeds = self.pos_embed(indices)
            weighted_embeds = embeds * weights
            p0, p1, p2, p3 = weighted_embeds.unbind(dim=0)
            combined = p0 + p1 + p2 + p3

            combined = combined.view(h * w, hidden_dim)
            repeated = combined.unsqueeze(0).expand(t, -1, -1).contiguous()
            repeated = repeated.view(t, h // m_size, m_size, w // m_size,
                                     m_size, hidden_dim)
            repeated = repeated.permute(0, 1, 3, 2, 4,
                                        5).reshape(-1, hidden_dim)
            outputs.append(repeated)

        return torch.cat(outputs, dim=0)

    def compute_attn_mask_seqlen(
        self,
        cu_seqlens: torch.Tensor,
    ) -> tuple[Optional[int], Optional[list[int]]]:
        max_seqlen, seqlens = None, None
        if self.attn_backend == _Backend.FLASH_ATTN:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        elif self.attn_backend == _Backend.XFORMERS:
            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        return max_seqlen, seqlens

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: list[list[int]],
    ) -> torch.Tensor:
        hidden_states = x.to(device=self.device, dtype=self.dtype)
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        grid_thw_tensor = torch.tensor(grid_thw,
                                       device=self.device,
                                       dtype=torch.int32)

        cu_seqlens = torch.repeat_interleave(
            grid_thw_tensor[:, 1] * grid_thw_tensor[:, 2],
            grid_thw_tensor[:, 0]).cumsum(
                dim=0,
                dtype=grid_thw_tensor.dtype
                if torch.jit.is_tracing() else torch.int32,
            )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        hidden_states = hidden_states.unsqueeze(1)
        rotary_pos_emb = rotary_pos_emb.to(hidden_states.device)
        max_seqlen, seqlens = self.compute_attn_mask_seqlen(cu_seqlens)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states,
                                cu_seqlens=cu_seqlens,
                                rotary_pos_emb=rotary_pos_emb,
                                max_seqlen=max_seqlen,
                                seqlens=seqlens)
            if layer_num in self.deepstack_visual_indexes:
                deepstack_merger_idx = self.deepstack_visual_indexes.index(
                    layer_num)
                deepstack_feature = self.deepstack_merger_list[
                    deepstack_merger_idx](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)
        hidden_states = self.merger(hidden_states)
        hidden_states = torch.cat(
            [hidden_states] + deepstack_feature_lists,
            dim=1)  # [seq_len, hidden_size * (1 + depth_of_deepstack)]
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("attn.qkv.", "attn.q.", "q"),
            ("attn.qkv.", "attn.k.", "k"),
            ("attn.qkv.", "attn.v.", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen3VLProcessingInfo(Qwen2VLProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3VLConfig)

    def get_hf_processor(self, **kwargs: object) -> Qwen3VLProcessor:
        return self.ctx.get_hf_processor(
            Qwen3VLProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )

    def get_tokenizer(self):
        return self.ctx.tokenizer

    def get_image_processor(self,
                            **kwargs: object) -> Qwen2VLImageProcessorFast:
        return self.get_hf_processor(**kwargs).image_processor

    def get_video_processor(self, **kwargs: object) -> Qwen3VLVideoProcessor:
        return self.get_hf_processor(**kwargs).video_processor

    def _get_vision_info(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int = 2,
        do_resize: bool = True,
        image_processor: Optional[Union[Qwen2VLImageProcessorFast,
                                        Qwen3VLVideoProcessor]],
    ) -> tuple[ImageSize, int]:
        if image_processor is None and num_frames > 1:
            image_processor = self.get_video_processor()
        elif image_processor is None:
            image_processor = self.get_image_processor()

        is_video = isinstance(image_processor, Qwen3VLVideoProcessor)

        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        patch_size = vision_config.patch_size
        merge_size = vision_config.spatial_merge_size
        temporal_patch_size = vision_config.temporal_patch_size

        if do_resize:
            if is_video:
                smart_resize = video_smart_resize
                extra_kwargs = {
                    "num_frames": num_frames,
                    "temporal_factor": temporal_patch_size
                }
            else:
                smart_resize = image_smart_resize
                extra_kwargs = {}
            resized_height, resized_width = smart_resize(
                height=image_height,
                width=image_width,
                factor=patch_size * merge_size,
                min_pixels=image_processor.size["shortest_edge"],
                max_pixels=image_processor.size["longest_edge"],
                **extra_kwargs,
            )
            preprocessed_size = ImageSize(width=resized_width,
                                          height=resized_height)
        else:
            preprocessed_size = ImageSize(width=image_width,
                                          height=image_height)

        padded_num_frames = num_frames + num_frames % temporal_patch_size

        grid_t = max(padded_num_frames // temporal_patch_size, 1)
        grid_h = preprocessed_size.height // patch_size
        grid_w = preprocessed_size.width // patch_size

        num_patches = grid_t * grid_h * grid_w
        num_vision_tokens = num_patches // (merge_size**2)

        return preprocessed_size, num_vision_tokens

    def _get_max_video_frames(self,
                              max_tokens: int,
                              start_num_frames: int = 2) -> int:
        return super()._get_max_video_frames(max_tokens,
                                             start_num_frames=start_num_frames)

    def get_num_frames_with_most_features(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        return super().get_num_frames_with_most_features(
            seq_len, mm_counts, max_frames_per_video=_MAX_FRAMES_PER_VIDEO)

    def get_max_video_tokens(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        target_width, target_height = self.get_image_size_with_most_features()
        video_soft_tokens = self.get_num_video_tokens(
            image_width=target_width,
            image_height=target_height,
            num_frames=self.get_num_frames_with_most_features(
                seq_len, mm_counts),
            image_processor=None,
        )

        # NOTE: By default in Qwen3-VL, one video token is converted to
        # "<{timestamp} seconds>" (on average 9.5 tokens) + vision_start_token + video_token + vision_end_token # noqa: E501
        formatted_video_soft_tokens = video_soft_tokens * 12.5
        return int(formatted_video_soft_tokens)

    def _calculate_timestamps(self, indices: list[int] | torch.Tensor,
                              video_fps: float, merge_size: int):
        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            # don't update metadata's frames_indices directly
            indices = indices + [indices[-1]
                                 ] * (merge_size - len(indices) % merge_size)
        timestamps = [idx / video_fps for idx in indices]
        timestamps = [(timestamps[i] + timestamps[i + merge_size - 1]) / 2
                      for i in range(0, len(timestamps), merge_size)]
        return timestamps

    def _get_video_second_idx(
            self,
            metadata: dict[str, Any],
            out_item: MultiModalKwargsItem,
            do_sample_frames: Optional[bool] = None,
            sampled_fps: Optional[float] = None) -> list[int]:
        video_processor = self.get_video_processor()
        merge_size = video_processor.merge_size
        indices = metadata["frames_indices"]

        # metadata["fps"] refers to the true fps of the input video.
        video_fps = metadata["fps"]
        if do_sample_frames is None:
            do_sample_frames = metadata.get("do_sample_frames", False)

        # If video frames are sampled in HF processor (instead of vLLM
        # video loader), we need to re-calculate the indices from original
        # metadata.
        if do_sample_frames:
            # here video_fps is the fps of the sampled video, and
            # metadata["fps"] refers to the fps of the original video.
            video_fps = sampled_fps if sampled_fps else video_processor.fps
            total_num_frames = metadata["total_num_frames"]
            num_frames = int(total_num_frames / metadata["fps"] * video_fps)
            num_frames = min(
                min(max(num_frames, video_processor.min_frames),
                    video_processor.max_frames), total_num_frames)
            indices = np.linspace(0, total_num_frames - 1,
                                  num_frames).round().astype(int).tolist()
        timestamps = self._calculate_timestamps(indices, video_fps, merge_size)
        return timestamps


class Qwen3VLDummyInputsBuilder(BaseDummyInputsBuilder[Qwen3VLProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        image_token = "<|vision_start|><|image_pad|><|vision_end|>"
        video_token = "<|vision_start|><|video_pad|><|vision_end|>"

        return image_token * num_images + video_token * num_videos

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        target_width, target_height = (
            self.info.get_image_size_with_most_features())
        target_num_frames = self.info.get_num_frames_with_most_features(
            seq_len, mm_counts)
        target_video_size, _ = self.info._get_vision_info(
            image_width=target_width,
            image_height=target_height,
            num_frames=target_num_frames,
            image_processor=self.info.get_video_processor(),
        )
        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images),
            "video":
            self._get_dummy_videos(
                width=target_video_size.width,
                height=target_video_size.height,
                num_frames=target_num_frames,
                num_videos=num_videos,
            ),
        }

    def _get_dummy_videos(
        self,
        *,
        width: int,
        height: int,
        num_frames: int,
        num_videos: int,
    ) -> list[VideoItem]:
        num_frames = max(num_frames, 2)
        video = np.full((num_frames, width, height, 3), 255, dtype=np.uint8)
        video_items = []
        for i in range(num_videos):
            video_metadata = {
                "fps": 2.0,
                "duration": num_frames / 2.0,
                "total_num_frames": num_frames,
                "frames_indices": [i for i in range(num_frames)],
                "video_backend": "opencv",
                "do_sample_frames": False,
            }
            video_item = (video.copy(), video_metadata)
            video_items.append(video_item)
        return video_items

    def get_dummy_processor_inputs(self, seq_len, mm_counts):
        processor_inputs = super().get_dummy_processor_inputs(
            seq_len, mm_counts)
        # HACK(Isotr0py): We set do_resize to False here to reuse Qwen2-VL's
        # profiling logic, which will be problematic for configurable mm
        # profiling.
        # TODO(Isotr0py): Switch to the implementation in
        # https://github.com/vllm-project/vllm/pull/25557
        # after supporting configurable mm profiling.
        processor_inputs.hf_processor_mm_kwargs = {"do_resize": False}
        return processor_inputs


class Qwen3VLMultiModalProcessor(BaseMultiModalProcessor[Qwen3VLProcessingInfo]
                                 ):

    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(video_needs_metadata=True)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        processor = self.info.get_hf_processor(**mm_kwargs)

        # Separate video processing from image processing. Because the videos
        # are processed into serval image patches
        if ("videos" in mm_data and isinstance(mm_data["videos"], list)
                and len(mm_data["videos"]) > 0):
            video_grid_thw_lst = []
            pixel_values_videos_lst = []

            for item_idx, item in enumerate(mm_data.pop("videos", [])):
                video_array, metadata = item

                # NOTE: @JJJYmmm new attr metadata.frames_indices indicates
                # the sampled frames indices of pre-sampled videos, which is
                # used to calculate the timestamps. Make sure that
                # do_sample_frames in mm_kwargs is false for presampled videos.

                # NOTE: a copy of is created to update do_sample_frames,
                # otherwise mm_hash for the object will be incorrect.
                video_mm_kwargs = dict(**mm_kwargs)
                if "do_sample_frames" not in video_mm_kwargs:
                    # qwen_vl_utils already has "do_sample_frames" in
                    # mm_kwargs, don't overwrite it.
                    video_mm_kwargs["do_sample_frames"] = metadata.get(
                        "do_sample_frames", False)

                metadata = VideoMetadata(**{
                    k: metadata[k]
                    for k in metadata if k != "do_sample_frames"
                })

                video_mm_data = dict()
                video_mm_data["videos"] = [[video_array]]
                video_mm_data["video_metadata"] = [[metadata]]

                video_outputs = super()._call_hf_processor(
                    prompt="<|vision_start|><|video_pad|><|vision_end|>",
                    mm_data=video_mm_data,
                    mm_kwargs=video_mm_kwargs,
                    tok_kwargs=tok_kwargs,
                )
                input_ids = video_outputs.pop("input_ids")
                video_placeholder = processor.tokenizer.batch_decode(
                    input_ids)[0]
                prompt = prompt.replace(
                    "<|vision_start|><|video_pad|><|vision_end|>",
                    video_placeholder,
                    1,
                )

                video_grid_thw_lst.append(video_outputs["video_grid_thw"])
                pixel_values_videos_lst.append(
                    video_outputs["pixel_values_videos"])
            video_outputs = dict(
                pixel_values_videos=torch.cat(pixel_values_videos_lst),
                video_grid_thw=torch.cat(video_grid_thw_lst),
            )
        else:
            video_outputs = dict()

        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        combined_outputs = dict(
            processed_outputs,
            **video_outputs,
        )
        return BatchFeature(combined_outputs)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
        image_grid_sizes = image_grid_thw.prod(-1)

        video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
        video_grid_sizes = video_grid_thw.prod(-1)

        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", image_grid_sizes),
            image_embeds=MultiModalFieldConfig.flat_from_sizes(
                "image", image_grid_sizes),
            image_grid_thw=MultiModalFieldConfig.batched("image"),
            pixel_values_videos=MultiModalFieldConfig.flat_from_sizes(
                "video", video_grid_sizes),
            video_embeds=MultiModalFieldConfig.flat_from_sizes(
                "video", video_grid_sizes),
            video_grid_thw=MultiModalFieldConfig.batched("video"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(
            **hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        hf_config = self.info.get_hf_config()

        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        vision_end_token_id = hf_config.vision_end_token_id

        merge_length = image_processor.merge_size**2

        def get_image_replacement_qwen3vl(item_idx: int):
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item["image_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            num_tokens = int(grid_thw.prod()) // merge_length
            return [hf_processor.image_token_id] * num_tokens

        def get_video_replacement_qwen3vl(item_idx: int):
            out_item = out_mm_kwargs["video"][item_idx]
            grid_thw = out_item["video_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            video, metadata = mm_items["video"][item_idx]
            do_sample_frames = hf_processor_mm_kwargs.get("do_sample_frames")
            sampled_fps = hf_processor_mm_kwargs.get("fps")
            if is_list_of(sampled_fps, float):
                sampled_fps = sampled_fps[item_idx]
            timestamps = self.info._get_video_second_idx(
                metadata, out_item, do_sample_frames, sampled_fps)

            assert len(timestamps) == grid_thw[0], (
                f"The timestamps length({len(timestamps)}) should be equal "
                f"video length ({grid_thw[0]}).")

            frames_idx_token = [
                tokenizer.encode(f"<{curr_time:.1f} seconds>",
                                 add_special_tokens=False)
                for curr_time in timestamps
            ]
            num_tokens_per_frame = int(grid_thw[1:].prod()) // merge_length
            placeholder = []
            for frame_idx in frames_idx_token:
                placeholder.extend(frame_idx)
                placeholder.extend([vision_start_token_id] +
                                   [video_token_id] * num_tokens_per_frame +
                                   [vision_end_token_id])
            return PromptUpdateDetails.select_token_id(placeholder,
                                                       video_token_id)

        return [
            PromptReplacement(
                modality="image",
                target=hf_processor.image_token,
                replacement=get_image_replacement_qwen3vl,
            ),

            # NOTE: We match string on purpose since searching sequence of
            # token ids takes more time.
            PromptReplacement(
                modality="video",
                target="<|vision_start|><|video_pad|><|vision_end|>",
                replacement=get_video_replacement_qwen3vl,
            ),
        ]


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        # the same shape as input_embeds
        "deepstack_input_embeds": 0
    })
class Qwen3LLMModel(Qwen3Model):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if not get_pp_group().is_first_rank:
            assert self.start_layer >= len(
                vllm_config.model_config.hf_config.vision_config.
                deepstack_visual_indexes), (
                    "start_layer should be greater than or equal to "
                    "len(deepstack_visual_indexes)")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        # args for deepstack
        deepstack_input_embeds: Optional[IntermediateTensors] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer_idx, layer in enumerate(
                self.layers[self.start_layer:self.end_layer]):
            layer_idx = layer_idx + self.start_layer

            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

            if deepstack_input_embeds is not None and \
                    layer_idx in range(0, len(deepstack_input_embeds)):
                hidden_states = hidden_states + deepstack_input_embeds[
                    f"deepstack_input_embeds_{layer_idx}"]

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3LLMForCausalLM(Qwen3ForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3ForCausalLM, self).__init__()
        config = vllm_config.model_config.hf_config.text_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = Qwen3LLMModel(vllm_config=vllm_config, prefix=prefix)

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(config.vocab_size,
                                              config.hidden_size,
                                              quant_config=quant_config,
                                              prefix="lm_head")
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)


@MULTIMODAL_REGISTRY.register_processor(Qwen3VLMultiModalProcessor,
                                        info=Qwen3VLProcessingInfo,
                                        dummy_inputs=Qwen3VLDummyInputsBuilder)
class Qwen3VLForConditionalGeneration(nn.Module, SupportsMultiModal,
                                      SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    supports_encoder_tp_data = True

    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
        })

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"

        raise ValueError("Only image or video modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__()
        config: Qwen3VLConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        if not multimodal_config.get_limit_per_prompt("image") and \
            not multimodal_config.get_limit_per_prompt("video"):
            self.visual = None
        else:
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
                use_data_parallel=self.use_data_parallel,
            )

        self.language_model = Qwen3LLMForCausalLM(vllm_config=vllm_config,
                                                  prefix=maybe_prefix(
                                                      prefix,
                                                      "language_model"))

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

        self.use_deepstack = hasattr(config.vision_config,
                                     'deepstack_visual_indexes')
        self.deepstack_num_level = len(
            config.vision_config.deepstack_visual_indexes
        ) if self.use_deepstack else 0
        # register buffer for deepstack
        if self.use_deepstack and self.visual is not None:
            self.deepstack_input_embeds = [
                torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    config.text_config.hidden_size)
                for _ in range(self.deepstack_num_level)
            ]
        else:
            self.deepstack_input_embeds = None
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

    def _get_deepstack_input_embeds(self,
                                    num_tokens: int) -> IntermediateTensors:
        # get deepstack_input_embeds from buffer, and clear the buffer
        return IntermediateTensors({
            f"deepstack_input_embeds_{idx}":
            self.deepstack_input_embeds[idx][:num_tokens]
            for idx in range(self.deepstack_num_level)
        })

    def _set_deepstack_input_embeds(
            self, deepstack_input_embeds: torch.Tensor) -> None:
        # set deepstack_input_embeds to buffer
        num_tokens = deepstack_input_embeds.size(1)
        if num_tokens > self.deepstack_input_embeds[0].size(0):
            self.deepstack_input_embeds = [
                torch.zeros(num_tokens,
                            self.config.text_config.hidden_size,
                            device=self.deepstack_input_embeds[0].device,
                            dtype=self.deepstack_input_embeds[0].dtype)
                for _ in range(self.deepstack_num_level)
            ]
        for idx in range(self.deepstack_num_level):
            self.deepstack_input_embeds[idx][:num_tokens].copy_(
                deepstack_input_embeds[idx])

    def _clear_deepstack_input_embeds(self, num_tokens: int) -> None:
        # clear deepstack_input_embeds in buffer
        if num_tokens > 0:
            for idx in range(self.deepstack_num_level):
                self.deepstack_input_embeds[idx][:num_tokens].zero_()

    def _validate_and_reshape_mm_tensor(self, mm_input: object,
                                        name: str) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. "
                             f"Got type: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            if mm_input.ndim == 2:
                return mm_input
            if mm_input.ndim != 3:
                raise ValueError(f"{name} should be 2D or batched 3D tensor. "
                                 f"Got ndim: {mm_input.ndim} "
                                 f"(shape={mm_input.shape})")
            return torch.concat(list(mm_input))
        else:
            return torch.concat(mm_input)

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[Qwen2_5_VLImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = self._validate_and_reshape_mm_tensor(
                pixel_values, "image pixel values")
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw")

            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return Qwen2_5_VLImagePixelInputs(type="pixel_values",
                                              pixel_values=pixel_values,
                                              image_grid_thw=image_grid_thw)

        if image_embeds is not None:
            image_embeds = self._validate_and_reshape_mm_tensor(
                image_embeds, "image embeds")
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw")

            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")
            return Qwen2_5_VLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw)

    def _parse_and_validate_video_input(
            self, **kwargs: object) -> Optional[Qwen2_5_VLVideoInputs]:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            pixel_values_videos = self._validate_and_reshape_mm_tensor(
                pixel_values_videos, "video pixel values")
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw")

            return Qwen2_5_VLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
            )

        if video_embeds is not None:
            video_embeds = self._validate_and_reshape_mm_tensor(
                video_embeds, "video embeds")
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw")

            if not isinstance(video_embeds, torch.Tensor):
                raise ValueError("Incorrect type of video embeddings. "
                                 f"Got type: {type(video_embeds)}")
            return Qwen2_5_VLVideoEmbeddingInputs(
                type="video_embeds",
                video_embeds=video_embeds,
                video_grid_thw=video_grid_thw)

    def _process_image_input(
            self,
            image_input: Qwen2_5_VLImageInputs) -> tuple[torch.Tensor, ...]:

        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(self.visual.dtype)
            if self.use_data_parallel:
                return run_dp_sharded_mrope_vision_model(self.visual,
                                                         pixel_values,
                                                         grid_thw_list,
                                                         rope_type="rope_3d")
            else:
                image_embeds = self.visual(pixel_values,
                                           grid_thw=grid_thw_list)

        # Split concatenated embeddings for each image item.
        # Using prod on grid_thw_list instead of grid_thw.prod avoids CUDA sync
        merge_size = self.visual.spatial_merge_size
        sizes = (torch.tensor(grid_thw_list, dtype=torch.long).prod(-1) //
                 (merge_size * merge_size)).tolist()
        return image_embeds.split(sizes)

    def _process_video_input(
            self,
            video_input: Qwen2_5_VLVideoInputs) -> tuple[torch.Tensor, ...]:

        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"].type(
                self.visual.dtype)
            if self.use_data_parallel:
                return run_dp_sharded_mrope_vision_model(self.visual,
                                                         pixel_values_videos,
                                                         grid_thw_list,
                                                         rope_type="rope_3d")
            else:
                video_embeds = self.visual(pixel_values_videos,
                                           grid_thw=grid_thw_list)

        # Split concatenated embeddings for each video item.
        # Using prod on grid_thw_list instead of grid_thw.prod avoids CUDA sync
        merge_size = self.visual.spatial_merge_size
        sizes = (torch.tensor(grid_thw_list, dtype=torch.long).prod(-1) //
                 (merge_size * merge_size)).tolist()
        return video_embeds.split(sizes)

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}
        for input_key in kwargs:
            if input_key in ("pixel_values", "image_embeds"
                             ) and "image" not in mm_input_by_modality:
                mm_input_by_modality[
                    "image"] = self._parse_and_validate_image_input(**kwargs)
            if input_key in ("pixel_values_videos", "video_embeds"
                             ) and "video" not in mm_input_by_modality:
                mm_input_by_modality[
                    "video"] = self._parse_and_validate_video_input(**kwargs)
        return mm_input_by_modality

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:

        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(
            **kwargs)
        if not mm_input_by_modality:
            return None

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                vision_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings += vision_embeddings
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                multimodal_embeddings += video_embeddings
        return multimodal_embeddings

    def _compute_deepstack_embeds(
            self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor,
            multimodal_embeddings: MultiModalEmbeddings) -> torch.Tensor:
        visual_lens = [
            x.shape[0] if isinstance(x, torch.Tensor) else len(x)
            for x in multimodal_embeddings
        ]
        multimodal_embeddings_cat = torch.cat(multimodal_embeddings, dim=0)

        multimodal_embeddings_main, multimodal_embeddings_multiscale = torch.split(  # noqa:E501
            multimodal_embeddings_cat, [self.visual_dim, self.multiscale_dim],
            dim=-1)

        multimodal_embeddings = torch.split(multimodal_embeddings_main,
                                            visual_lens,
                                            dim=0)
        multimodal_embeddings_multiscale = torch.split(
            multimodal_embeddings_multiscale, visual_lens, dim=0)

        deepstack_input_embeds = inputs_embeds.new_zeros(
            inputs_embeds.size(0),
            self.deepstack_num_level * inputs_embeds.size(1))

        deepstack_input_embeds = merge_multimodal_embeddings(
            input_ids,
            deepstack_input_embeds,
            multimodal_embeddings_multiscale,
            placeholder_token_id=[
                self.config.image_token_id, self.config.video_token_id
            ],
        )
        deepstack_input_embeds = deepstack_input_embeds.view(
            inputs_embeds.shape[0], self.deepstack_num_level, self.visual_dim)
        deepstack_input_embeds = deepstack_input_embeds.permute(1, 0, 2)
        return deepstack_input_embeds, multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        deepstack_input_embeds = None
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            if self.use_deepstack:
                deepstack_input_embeds, multimodal_embeddings = self._compute_deepstack_embeds(  # noqa:E501
                    input_ids, inputs_embeds, multimodal_embeddings)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                [self.config.image_token_id, self.config.video_token_id])

        if self.use_deepstack:
            if deepstack_input_embeds is None:
                deepstack_input_embeds = torch.zeros_like(
                    inputs_embeds).unsqueeze(0).repeat(
                        self.deepstack_num_level, 1, 1).contiguous()
            self._set_deepstack_input_embeds(deepstack_input_embeds)

        return inputs_embeds

    def get_input_embeddings_v0(
        self,
        input_ids: torch.Tensor,
        image_input: Optional[Qwen2_5_VLImageInputs] = None,
        video_input: Optional[Qwen2_5_VLVideoInputs] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.get_input_embeddings(input_ids)

        if self.use_deepstack:
            visual_dim = inputs_embeds.shape[-1]
            deepstack_input_embeds = None
            if image_input is not None or video_input is not None:
                deepstack_input_embeds = torch.zeros_like(
                    inputs_embeds).unsqueeze(1).repeat(
                        1, self.deepstack_num_level, 1).flatten(1)

        if image_input is not None:
            image_embeds = self._process_image_input(image_input)
            if self.use_deepstack:
                image_embeds = torch.cat(image_embeds)

                image_embeds, image_embeds_multiscale = image_embeds.split(
                    [visual_dim, visual_dim * self.deepstack_num_level],
                    dim=-1)

                deepstack_input_embeds = merge_multimodal_embeddings(
                    input_ids,
                    deepstack_input_embeds,
                    image_embeds_multiscale,
                    placeholder_token_id=self.config.image_token_id,
                )

            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                image_embeds,
                placeholder_token_id=self.config.image_token_id,
            )

        if video_input is not None:
            video_embeds = self._process_video_input(video_input)
            if self.use_deepstack:
                video_embeds = torch.cat(video_embeds)

                video_embeds, video_embeds_multiscale = video_embeds.split(
                    [visual_dim, visual_dim * self.deepstack_num_level],
                    dim=-1)

                deepstack_input_embeds = merge_multimodal_embeddings(
                    input_ids,
                    deepstack_input_embeds,
                    video_embeds_multiscale,
                    placeholder_token_id=self.config.video_token_id,
                )

            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                video_embeds,
                placeholder_token_id=self.config.video_token_id,
            )

        if self.use_deepstack and deepstack_input_embeds is not None:
            deepstack_input_embeds = deepstack_input_embeds.view(
                inputs_embeds.shape[0], self.deepstack_num_level,
                visual_dim).permute(1, 0, 2).contiguous()
            self._set_deepstack_input_embeds(deepstack_input_embeds)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Run forward pass for Qwen3VL.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen3VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
            intermediate_tensors: Intermediate tensors from previous pipeline
                stages.
            inputs_embeds: Pre-computed input embeddings.
            **kwargs: Additional keyword arguments including:
                - pixel_values: Pixel values to be fed to a model.
                    `None` if no images are passed.
                - image_grid_thw: Tensor `(n_images, 3)` of image 3D grid in
                    LLM. `None` if no images are passed.
                - pixel_values_videos: Pixel values of videos to be fed to a
                    model. `None` if no videos are passed.
                - video_grid_thw: Tensor `(n_videos, 3)` of video 3D grid in
                    LLM. `None` if no videos are passed.
        """

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner from
        # `get_multimodal_embeddings` and `get_input_embeddings`, this
        # condition is only for v0 compatibility.
        elif inputs_embeds is None:
            image_input = self._parse_and_validate_image_input(**kwargs)
            video_input = self._parse_and_validate_video_input(**kwargs)

            if image_input is None and video_input is None:
                inputs_embeds = None
            else:
                if uses_mrope(self.config):
                    assert positions.ndim == 2 and positions.size(0) == 3, (
                        "multimodal section rotary embedding requires "
                        f"(3, seq_len) positions, but got {positions.size()}")
                inputs_embeds = self.get_input_embeddings_v0(
                    input_ids,
                    image_input=image_input,
                    video_input=video_input)
                input_ids = None

        if self.use_deepstack and inputs_embeds is not None and get_pp_group(
        ).is_first_rank:
            deepstack_input_embeds = self._get_deepstack_input_embeds(
                inputs_embeds.size(0))
        else:
            deepstack_input_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            # args for deepstack
            deepstack_input_embeds=deepstack_input_embeds,
        )

        if inputs_embeds is not None and get_pp_group().is_first_rank:
            self._clear_deepstack_input_embeds(inputs_embeds.size(0))

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:

        skip_prefixes = []
        if self.visual is None:
            skip_prefixes.extend(["visual."])
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="model.visual.merger",
            tower_model="model.visual.",
        )

class SinkCrossAttention(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        sink_hidden: torch.Tensor,             # [num_seqs, 1, hidden]
        visual_hidden: torch.Tensor,            # [num_seqs, max_vis_len, hidden]
        visual_valid_mask: torch.BoolTensor,    # [num_seqs, max_vis_len]
    ) -> torch.Tensor:
        num_seqs = sink_hidden.shape[0]
        dtype = sink_hidden.dtype

        q = self.q_proj(sink_hidden)
        k = self.k_proj(visual_hidden)
        v = self.v_proj(visual_hidden)

        q = q.view(num_seqs, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(num_seqs, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(num_seqs, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        mask = ~visual_valid_mask
        mask = mask.unsqueeze(1).unsqueeze(2)
        attn_weights = attn_weights.masked_fill(mask, torch.finfo(dtype).min)

        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(num_seqs, 1, self.hidden_size)

        return self.o_proj(attn_output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):

        params_dict = dict(self.named_parameters())
        loaded = []
        for name, loaded_weight in weights:
            if name in params_dict:
                param = params_dict[name]
                param.data.copy_(loaded_weight)
                loaded.append(name)
        return loaded

@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        # the same shape as input_embeds
        "deepstack_input_embeds": 0
    })
class StudentQwen3LLMModel(Qwen3Model):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # sink track config
        self.sink_token_id = 151644 # im_start_token_id
        self.image_token_id = 151655
        self.video_token_id = 151656
        self.enable_sink_track = True
        self.sink_only_first_step = True
        self.sink_alpha = 0.3
        self.sink_interval = 5

        self._cached_input_ids: Optional[torch.Tensor] = None

        hf_config = vllm_config.model_config.hf_config
        logger.info(f"hf_config: {hf_config}")
        if hasattr(hf_config, "text_config"):
            text_config = hf_config.text_config
        else:
            text_config = hf_config

        hidden_size = text_config.hidden_size
        num_heads = text_config.num_attention_heads
        num_layers = text_config.num_hidden_layers
        rms_norm_eps = getattr(text_config, "rms_norm_eps", 1e-6)

        sink_layer_indices = [
            idx for idx in range(num_layers)
            if idx % self.sink_interval == 0
        ]
        self.sink_layer_indices_set = set(sink_layer_indices)

        self.sink_cross_attns = nn.ModuleDict({
            str(idx): SinkCrossAttention(hidden_size, num_heads)
            for idx in sink_layer_indices
        })
        self.sink_q_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.sink_kv_norm = RMSNorm(hidden_size, eps=rms_norm_eps)


        if not get_pp_group().is_first_rank:
            assert self.start_layer >= len(
                vllm_config.model_config.hf_config.vision_config.
                deepstack_visual_indexes), (
                    "start_layer should be greater than or equal to "
                    "len(deepstack_visual_indexes)")
        
        logger.warning("🔥 StudentQwen3LLMModel initialized! PID=%d", os.getpid())

    def cache_input_ids(self, input_ids: Optional[torch.Tensor]):
        if input_ids is not None:
            self._cached_input_ids = input_ids.detach().clone()
            # _sink_debug_log(f"🔥 [cache] input_ids cached, shape={input_ids.shape}")
        else:
            self._cached_input_ids = None

    def _resolve_input_ids(
        self,
        input_ids: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if input_ids is not None:
            return input_ids
        if self._cached_input_ids is not None:
            resolved = self._cached_input_ids
            # _sink_debug_log(f"🔥 [resolve] Using cached input_ids, shape={resolved.shape}")
            return resolved
        # _sink_debug_log("🔥 [resolve] No input_ids available")
        return None

    def _expand_ids_to_match_hidden(
        self,
        input_ids: torch.Tensor,
        target_len: int,
    ) -> Optional[torch.Tensor]:

        orig_len = input_ids.shape[0]
        if orig_len == target_len:
            return input_ids
        if orig_len > target_len:
            # _sink_debug_log(f"⚠️ orig_len({orig_len}) > target_len({target_len})")
            return None

        diff = target_len - orig_len
        image_token_id = self.image_token_id
        video_token_id = self.video_token_id
        is_visual = (input_ids == image_token_id) | (input_ids == video_token_id)
        num_placeholders = is_visual.sum().item()

        if num_placeholders == 0:
            # _sink_debug_log(f"⚠️ No visual placeholders but length mismatch")
            return None


        extra_per = diff // num_placeholders
        remainder = diff % num_placeholders

        repeat_counts = torch.ones(orig_len, dtype=torch.long, device=input_ids.device)
        visual_indices = is_visual.nonzero(as_tuple=True)[0]
        repeat_counts[visual_indices] = 1 + extra_per
        if remainder > 0:
            repeat_counts[visual_indices[:remainder]] += 1

        expanded = input_ids.repeat_interleave(repeat_counts)

        if expanded.shape[0] != target_len:
            return None

        return expanded


    def _get_sink_positions_fast(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if input_ids is None or self.sink_token_id is None:
            return None

        total_tokens = input_ids.shape[0]
        device = input_ids.device
        seq_starts, seq_ids = self._get_sequence_info(input_ids, positions)
        num_sequences = seq_starts.shape[0]

        is_sink = (input_ids == self.sink_token_id)

        sink_positions = torch.full(
            (num_sequences,), -1, dtype=torch.long, device=device
        )

        if is_sink.any():
            sink_global_indices = is_sink.nonzero(as_tuple=True)[0]
            sink_seq_ids = seq_ids[sink_global_indices]

            temp = torch.full(
                (num_sequences,), total_tokens, dtype=torch.long, device=device
            )
            temp.scatter_reduce_(
                0, sink_seq_ids, sink_global_indices, reduce="amin"
            )

            found_mask = temp < total_tokens
            sink_positions[found_mask] = temp[found_mask]

        return sink_positions

    def _get_sequence_info(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ):
        total_tokens = input_ids.shape[0]
        device = input_ids.device

        is_seq_start = torch.zeros(total_tokens, dtype=torch.bool, device=device)
        is_seq_start[0] = True

        if positions is not None:
            if positions.ndim == 2 and positions.size(0) == 3:
                pos = positions[0]  # temporal
            elif positions.ndim == 2:
                pos = positions[0]
            else:
                pos = positions

            if total_tokens > 1:
                is_seq_start[1:] = pos[1:] < pos[:-1]

        seq_starts = is_seq_start.nonzero(as_tuple=True)[0]
        seq_ids = is_seq_start.long().cumsum(0) - 1
        return seq_starts, seq_ids

    def _sink_track_update(
        self,
        hidden_states: torch.Tensor,
        sink_positions: torch.Tensor,
        seq_ids: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if visual_mask is None:
            return hidden_states

        device = hidden_states.device
        total_tokens, hidden_dim = hidden_states.shape
        num_sequences = sink_positions.shape[0]
        mm_mask = visual_mask.to(device=device, dtype=torch.bool)

        if mm_mask.shape[0] != total_tokens:
            return hidden_states
        if not mm_mask.any():
            return hidden_states


        valid_seq_mask = sink_positions >= 0

        mm_hidden = hidden_states * mm_mask.unsqueeze(-1)
        mm_sum = torch.zeros(
            num_sequences, hidden_dim, device=device, dtype=hidden_states.dtype
        )
        mm_sum.index_add_(0, seq_ids, mm_hidden)

        mm_count = torch.zeros(
            num_sequences, device=device, dtype=hidden_states.dtype
        )
        mm_count.index_add_(0, seq_ids, mm_mask.to(hidden_states.dtype))

        has_mm = (mm_count > 0) & valid_seq_mask

        if not has_mm.any():
            return hidden_states

        mm_count = mm_count.clamp(min=1.0).unsqueeze(-1)
        mm_avg = mm_sum / mm_count

        safe_sink_positions = sink_positions.clamp(min=0)

        sink_hidden = hidden_states[safe_sink_positions]
        sink_hidden_new = sink_hidden.clone()
        sink_hidden_new[has_mm] = sink_hidden[has_mm] + self.sink_alpha * (
            mm_avg[has_mm] - sink_hidden[has_mm]
        )

        hidden_states = hidden_states.clone()

        valid_indices = has_mm.nonzero(as_tuple=True)[0]
        for idx in valid_indices:
            pos = sink_positions[idx]
            hidden_states[pos] = sink_hidden_new[idx]

        return hidden_states

    def _get_visual_mask(
        self,
        input_ids: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Get visual token mask from image/video placeholder tokens in input_ids.
        Returns:
            visual_mask: [total_tokens] bool
        """
        if input_ids is None:
            return None

        image_token_id = self.image_token_id
        video_token_id = self.video_token_id

        visual_mask = (input_ids == image_token_id) | (input_ids == video_token_id)
        return visual_mask

    def _is_prefill_with_multimodal(self, input_ids: Optional[torch.Tensor]) -> bool:

        if not self.enable_sink_track:
            return False
        if input_ids is None:
            return False
        has_sink = (input_ids == self.sink_token_id).any().item()
        has_visual = (
            (input_ids == self.image_token_id) |
            (input_ids == self.video_token_id)
        ).any().item()
        result = has_sink and has_visual
        return result



    def _extract_visual_hidden_per_seq(
        self,
        hidden_states: torch.Tensor,       # [total_tokens, hidden]
        visual_mask: torch.BoolTensor,      # [total_tokens]
        seq_ids: torch.Tensor,              # [total_tokens]
        num_sequences: int,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.BoolTensor]]:

        device = hidden_states.device
        dtype = hidden_states.dtype
        hidden_dim = hidden_states.shape[-1]

        if not visual_mask.any():
            return None, None

        vis_count_per_seq = torch.zeros(num_sequences, dtype=torch.long, device=device)
        vis_global_indices = visual_mask.nonzero(as_tuple=True)[0]
        vis_seq = seq_ids[vis_global_indices]
        vis_count_per_seq.scatter_add_(0, vis_seq, torch.ones_like(vis_seq))

        max_vis_len = vis_count_per_seq.max().item()
        if max_vis_len == 0:
            return None, None

        visual_hidden = torch.zeros(
            num_sequences, max_vis_len, hidden_dim, device=device, dtype=dtype
        )
        visual_valid_mask = torch.zeros(
            num_sequences, max_vis_len, device=device, dtype=torch.bool
        )

        local_indices = torch.zeros_like(vis_seq)
        
        sorted_order = torch.argsort(vis_seq, stable=True)
        sorted_vis_seq = vis_seq[sorted_order]
        
        ones = torch.ones(sorted_vis_seq.shape[0], dtype=torch.long, device=device)
        
        seq_change = torch.zeros_like(sorted_vis_seq, dtype=torch.bool)
        seq_change[0] = True
        if sorted_vis_seq.shape[0] > 1:
            seq_change[1:] = sorted_vis_seq[1:] != sorted_vis_seq[:-1]
        
        global_cumsum = ones.cumsum(0)  # [1, 2, 3, 4, 5, ...]
        
        group_start_cumsum = torch.zeros_like(global_cumsum)
        group_start_cumsum[seq_change] = global_cumsum[seq_change]
        group_start_cumsum = torch.cummax(group_start_cumsum, dim=0).values
        
        sorted_local_indices = global_cumsum - group_start_cumsum
        
        local_indices_result = torch.empty_like(sorted_local_indices)
        local_indices_result[sorted_order] = sorted_local_indices

        local_indices_result = local_indices_result.clamp(0, max_vis_len - 1)

        vis_hidden = hidden_states[vis_global_indices]
        visual_hidden[vis_seq, local_indices_result] = vis_hidden
        visual_valid_mask[vis_seq, local_indices_result] = True

        return visual_hidden, visual_valid_mask

    def _sink_track_cross_attention(
        self,
        hidden_states: torch.Tensor,
        sink_positions: torch.Tensor,
        seq_ids: torch.Tensor,
        visual_mask: Optional[torch.Tensor],
        layer_idx: int,
    ) -> torch.Tensor:

        if visual_mask is None:
            return hidden_states

        dtype = hidden_states.dtype
        num_sequences = sink_positions.shape[0]
        valid_seq_mask = sink_positions >= 0

        visual_hidden, visual_valid_mask = self._extract_visual_hidden_per_seq(
            hidden_states, visual_mask.bool(), seq_ids, num_sequences
        )

        if visual_hidden is None:
            return hidden_states

        has_visual = visual_valid_mask.any(dim=1)
        active_mask = valid_seq_mask & has_visual

        if not active_mask.any():
            return hidden_states

        # Gather sink hidden
        safe_sink_pos = sink_positions.clamp(min=0)
        sink_hidden = hidden_states[safe_sink_pos].unsqueeze(1)

        active_indices = active_mask.nonzero(as_tuple=True)[0]
        active_sink = sink_hidden[active_indices]
        active_vis = visual_hidden[active_indices]
        active_vis_mask = visual_valid_mask[active_indices]

        # Norm + Cross-Attention
        active_sink_normed = self.sink_q_norm(active_sink)
        active_vis_normed = self.sink_kv_norm(active_vis)

        cross_attn = self.sink_cross_attns[str(layer_idx)]
        sink_update = cross_attn(
            sink_hidden=active_sink_normed,
            visual_hidden=active_vis_normed,
            visual_valid_mask=active_vis_mask,
        )

        # Residual
        active_sink_new = active_sink + sink_update


        # if logger.isEnabledFor(logging.DEBUG):
        #     with torch.no_grad():
        #         sink_norm = active_sink.float().norm(dim=-1).mean().item()
        #         update_norm = sink_update.float().norm(dim=-1).mean().item()
        #         new_norm = active_sink_new.float().norm(dim=-1).mean().item()
        #         ratio = update_norm / max(sink_norm, 1e-8)

        #         sink_abs_mean = active_sink.float().abs().mean().item()
        #         update_abs_mean = sink_update.float().abs().mean().item()
        #         abs_ratio = update_abs_mean / max(sink_abs_mean, 1e-8)

        #         logger.debug(
        #             "[SinkTrack L%d] "
        #             "sink_norm=%.4f, update_norm=%.4f, new_norm=%.4f, "
        #             "norm_ratio=%.6f, abs_mean_ratio=%.6f, "
        #             "n_active=%d",
        #             layer_idx,
        #             sink_norm, update_norm, new_norm,
        #             ratio, abs_ratio,
        #             active_sink.shape[0],
        #         )


        hidden_states = hidden_states.clone()
        active_sink_positions = sink_positions[active_indices]
        hidden_states[active_sink_positions] = active_sink_new.squeeze(1).to(dtype)

        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        # args for deepstack
        deepstack_input_embeds: Optional[IntermediateTensors] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]


        resolved_ids = self._resolve_input_ids(input_ids)
        sink_positions = None
        seq_ids = None
        visual_mask = None
        do_sink = False

        if resolved_ids is not None:
            if resolved_ids.device != hidden_states.device:
                resolved_ids = resolved_ids.to(hidden_states.device)


            if self._is_prefill_with_multimodal(resolved_ids):

                expanded_ids = self._expand_ids_to_match_hidden(
                    resolved_ids, hidden_states.shape[0]
                )
                if expanded_ids is not None:
                    do_sink = True
                    _, seq_ids = self._get_sequence_info(expanded_ids, positions)
                    sink_positions = self._get_sink_positions_fast(
                        expanded_ids, hidden_states, positions
                    )
                    visual_mask = self._get_visual_mask(expanded_ids)


        self._cached_input_ids = None

        sink_layer_indices = set(
            idx for idx in range(self.start_layer, self.end_layer)
            if idx % self.sink_interval == 0
        )

        is_prefill_like = do_sink
        sp_list = (
            sink_positions.detach().cpu().tolist()
            if sink_positions is not None and sink_positions.numel() > 0
            else None
        )

        for layer_idx, layer in enumerate(
                self.layers[self.start_layer:self.end_layer]):
            layer_idx = layer_idx + self.start_layer

            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

            if deepstack_input_embeds is not None and \
                    layer_idx in range(0, len(deepstack_input_embeds)):
                hidden_states = hidden_states + deepstack_input_embeds[
                    f"deepstack_input_embeds_{layer_idx}"]


            # if do_sink and sink_positions is not None and sink_positions.numel() > 0 and layer_idx in sink_layer_indices:
            #     hidden_states = self._sink_track_update(
            #         hidden_states=hidden_states,
            #         sink_positions=sink_positions,
            #         seq_ids=seq_ids,
            #         visual_mask=visual_mask,
            #     )

            # ★ Cross-Attention SinkTrack
            if (do_sink
                and sink_positions is not None
                and sink_positions.numel() > 0
                and layer_idx in self.sink_layer_indices_set):
                hidden_states = self._sink_track_cross_attention(
                    hidden_states=hidden_states,
                    sink_positions=sink_positions,
                    seq_ids=seq_ids,
                    visual_mask=visual_mask,
                    layer_idx=layer_idx,
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):

        sink_weights_buffer = []
        other_weights_buffer = []

        for name, loaded_weight in weights:
            if "sink_cross_attn" in name or "sink_q_norm" in name or "sink_kv_norm" in name:
                sink_weights_buffer.append((name, loaded_weight))
            else:
                other_weights_buffer.append((name, loaded_weight))


        loaded = set()
        if other_weights_buffer:
            result = super().load_weights(iter(other_weights_buffer))
            if result is not None:
                loaded.update(result)

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in sink_weights_buffer:
            if name in params_dict:
                params_dict[name].data.copy_(loaded_weight)
                loaded.add(name)
            else:
                logger.warning(f"[SinkTrack] Unmatched weight: {name}")

        return loaded


class StudentQwen3LLMForCausalLM(Qwen3ForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3ForCausalLM, self).__init__()
        config = vllm_config.model_config.hf_config.text_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = StudentQwen3LLMModel(vllm_config=vllm_config, prefix=prefix)

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(config.vocab_size,
                                              config.hidden_size,
                                              quant_config=quant_config,
                                              prefix="lm_head")
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)


@MULTIMODAL_REGISTRY.register_processor(Qwen3VLMultiModalProcessor,
                                        info=Qwen3VLProcessingInfo,
                                        dummy_inputs=Qwen3VLDummyInputsBuilder)
class StudentQwen3VLForConditionalGeneration(nn.Module, SupportsMultiModal,
                                      SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    supports_encoder_tp_data = True

    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
        })

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"

        raise ValueError("Only image or video modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # super().__init__()
        nn.Module.__init__(self)
        config: Qwen3VLConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        if not multimodal_config.get_limit_per_prompt("image") and \
            not multimodal_config.get_limit_per_prompt("video"):
            self.visual = None
        else:
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
                use_data_parallel=self.use_data_parallel,
            )

        self.language_model = StudentQwen3LLMForCausalLM(vllm_config=vllm_config,
                                                  prefix=maybe_prefix(
                                                      prefix,
                                                      "language_model"))

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

        self.use_deepstack = hasattr(config.vision_config,
                                     'deepstack_visual_indexes')
        self.deepstack_num_level = len(
            config.vision_config.deepstack_visual_indexes
        ) if self.use_deepstack else 0
        # register buffer for deepstack
        if self.use_deepstack and self.visual is not None:
            self.deepstack_input_embeds = [
                torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    config.text_config.hidden_size)
                for _ in range(self.deepstack_num_level)
            ]
        else:
            self.deepstack_input_embeds = None
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

    def _get_deepstack_input_embeds(self,
                                    num_tokens: int) -> IntermediateTensors:
        # get deepstack_input_embeds from buffer, and clear the buffer
        return IntermediateTensors({
            f"deepstack_input_embeds_{idx}":
            self.deepstack_input_embeds[idx][:num_tokens]
            for idx in range(self.deepstack_num_level)
        })

    def _set_deepstack_input_embeds(
            self, deepstack_input_embeds: torch.Tensor) -> None:
        # set deepstack_input_embeds to buffer
        num_tokens = deepstack_input_embeds.size(1)
        if num_tokens > self.deepstack_input_embeds[0].size(0):
            self.deepstack_input_embeds = [
                torch.zeros(num_tokens,
                            self.config.text_config.hidden_size,
                            device=self.deepstack_input_embeds[0].device,
                            dtype=self.deepstack_input_embeds[0].dtype)
                for _ in range(self.deepstack_num_level)
            ]
        for idx in range(self.deepstack_num_level):
            self.deepstack_input_embeds[idx][:num_tokens].copy_(
                deepstack_input_embeds[idx])

    def _clear_deepstack_input_embeds(self, num_tokens: int) -> None:
        # clear deepstack_input_embeds in buffer
        if num_tokens > 0:
            for idx in range(self.deepstack_num_level):
                self.deepstack_input_embeds[idx][:num_tokens].zero_()

    def _validate_and_reshape_mm_tensor(self, mm_input: object,
                                        name: str) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. "
                             f"Got type: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            if mm_input.ndim == 2:
                return mm_input
            if mm_input.ndim != 3:
                raise ValueError(f"{name} should be 2D or batched 3D tensor. "
                                 f"Got ndim: {mm_input.ndim} "
                                 f"(shape={mm_input.shape})")
            return torch.concat(list(mm_input))
        else:
            return torch.concat(mm_input)

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[Qwen2_5_VLImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = self._validate_and_reshape_mm_tensor(
                pixel_values, "image pixel values")
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw")

            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return Qwen2_5_VLImagePixelInputs(type="pixel_values",
                                              pixel_values=pixel_values,
                                              image_grid_thw=image_grid_thw)

        if image_embeds is not None:
            image_embeds = self._validate_and_reshape_mm_tensor(
                image_embeds, "image embeds")
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw")

            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")
            return Qwen2_5_VLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw)

    def _parse_and_validate_video_input(
            self, **kwargs: object) -> Optional[Qwen2_5_VLVideoInputs]:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            pixel_values_videos = self._validate_and_reshape_mm_tensor(
                pixel_values_videos, "video pixel values")
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw")

            return Qwen2_5_VLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
            )

        if video_embeds is not None:
            video_embeds = self._validate_and_reshape_mm_tensor(
                video_embeds, "video embeds")
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw")

            if not isinstance(video_embeds, torch.Tensor):
                raise ValueError("Incorrect type of video embeddings. "
                                 f"Got type: {type(video_embeds)}")
            return Qwen2_5_VLVideoEmbeddingInputs(
                type="video_embeds",
                video_embeds=video_embeds,
                video_grid_thw=video_grid_thw)

    def _process_image_input(
            self,
            image_input: Qwen2_5_VLImageInputs) -> tuple[torch.Tensor, ...]:

        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(self.visual.dtype)
            if self.use_data_parallel:
                return run_dp_sharded_mrope_vision_model(self.visual,
                                                         pixel_values,
                                                         grid_thw_list,
                                                         rope_type="rope_3d")
            else:
                image_embeds = self.visual(pixel_values,
                                           grid_thw=grid_thw_list)

        # Split concatenated embeddings for each image item.
        # Using prod on grid_thw_list instead of grid_thw.prod avoids CUDA sync
        merge_size = self.visual.spatial_merge_size
        sizes = (torch.tensor(grid_thw_list, dtype=torch.long).prod(-1) //
                 (merge_size * merge_size)).tolist()
        return image_embeds.split(sizes)

    def _process_video_input(
            self,
            video_input: Qwen2_5_VLVideoInputs) -> tuple[torch.Tensor, ...]:

        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"].type(
                self.visual.dtype)
            if self.use_data_parallel:
                return run_dp_sharded_mrope_vision_model(self.visual,
                                                         pixel_values_videos,
                                                         grid_thw_list,
                                                         rope_type="rope_3d")
            else:
                video_embeds = self.visual(pixel_values_videos,
                                           grid_thw=grid_thw_list)

        # Split concatenated embeddings for each video item.
        # Using prod on grid_thw_list instead of grid_thw.prod avoids CUDA sync
        merge_size = self.visual.spatial_merge_size
        sizes = (torch.tensor(grid_thw_list, dtype=torch.long).prod(-1) //
                 (merge_size * merge_size)).tolist()
        return video_embeds.split(sizes)

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}
        for input_key in kwargs:
            if input_key in ("pixel_values", "image_embeds"
                             ) and "image" not in mm_input_by_modality:
                mm_input_by_modality[
                    "image"] = self._parse_and_validate_image_input(**kwargs)
            if input_key in ("pixel_values_videos", "video_embeds"
                             ) and "video" not in mm_input_by_modality:
                mm_input_by_modality[
                    "video"] = self._parse_and_validate_video_input(**kwargs)
        return mm_input_by_modality

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:

        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(
            **kwargs)
        if not mm_input_by_modality:
            return None

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                vision_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings += vision_embeddings
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                multimodal_embeddings += video_embeddings
        return multimodal_embeddings

    def _compute_deepstack_embeds(
            self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor,
            multimodal_embeddings: MultiModalEmbeddings) -> torch.Tensor:
        visual_lens = [
            x.shape[0] if isinstance(x, torch.Tensor) else len(x)
            for x in multimodal_embeddings
        ]
        multimodal_embeddings_cat = torch.cat(multimodal_embeddings, dim=0)

        multimodal_embeddings_main, multimodal_embeddings_multiscale = torch.split(  # noqa:E501
            multimodal_embeddings_cat, [self.visual_dim, self.multiscale_dim],
            dim=-1)

        multimodal_embeddings = torch.split(multimodal_embeddings_main,
                                            visual_lens,
                                            dim=0)
        multimodal_embeddings_multiscale = torch.split(
            multimodal_embeddings_multiscale, visual_lens, dim=0)

        deepstack_input_embeds = inputs_embeds.new_zeros(
            inputs_embeds.size(0),
            self.deepstack_num_level * inputs_embeds.size(1))

        deepstack_input_embeds = merge_multimodal_embeddings(
            input_ids,
            deepstack_input_embeds,
            multimodal_embeddings_multiscale,
            placeholder_token_id=[
                self.config.image_token_id, self.config.video_token_id
            ],
        )
        deepstack_input_embeds = deepstack_input_embeds.view(
            inputs_embeds.shape[0], self.deepstack_num_level, self.visual_dim)
        deepstack_input_embeds = deepstack_input_embeds.permute(1, 0, 2)
        return deepstack_input_embeds, multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:


        if input_ids is not None and multimodal_embeddings is not None:
            # _sink_debug_log(
            #     f"🔥 [get_input_embeddings V1] Caching input_ids "
            #     f"shape={input_ids.shape}, "
            #     f"has_img={int((input_ids == self.config.image_token_id).sum())}, "
            #     f"has_vid={int((input_ids == self.config.video_token_id).sum())}, "
            #     f"has_sink={int((input_ids == 151644).sum())}"
            # )
            self.language_model.model.cache_input_ids(input_ids.clone())

        deepstack_input_embeds = None
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            if self.use_deepstack:
                deepstack_input_embeds, multimodal_embeddings = self._compute_deepstack_embeds(  # noqa:E501
                    input_ids, inputs_embeds, multimodal_embeddings)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                [self.config.image_token_id, self.config.video_token_id])

        if self.use_deepstack:
            if deepstack_input_embeds is None:
                deepstack_input_embeds = torch.zeros_like(
                    inputs_embeds).unsqueeze(0).repeat(
                        self.deepstack_num_level, 1, 1).contiguous()
            self._set_deepstack_input_embeds(deepstack_input_embeds)

        return inputs_embeds

    def get_input_embeddings_v0(
        self,
        input_ids: torch.Tensor,
        image_input: Optional[Qwen2_5_VLImageInputs] = None,
        video_input: Optional[Qwen2_5_VLVideoInputs] = None,
    ) -> torch.Tensor:


        if input_ids is not None and (image_input is not None or video_input is not None):
            self.language_model.model.cache_input_ids(input_ids.clone())

        inputs_embeds = self.get_input_embeddings(input_ids)

        if self.use_deepstack:
            visual_dim = inputs_embeds.shape[-1]
            deepstack_input_embeds = None
            if image_input is not None or video_input is not None:
                deepstack_input_embeds = torch.zeros_like(
                    inputs_embeds).unsqueeze(1).repeat(
                        1, self.deepstack_num_level, 1).flatten(1)

        if image_input is not None:
            image_embeds = self._process_image_input(image_input)
            if self.use_deepstack:
                image_embeds = torch.cat(image_embeds)

                image_embeds, image_embeds_multiscale = image_embeds.split(
                    [visual_dim, visual_dim * self.deepstack_num_level],
                    dim=-1)

                deepstack_input_embeds = merge_multimodal_embeddings(
                    input_ids,
                    deepstack_input_embeds,
                    image_embeds_multiscale,
                    placeholder_token_id=self.config.image_token_id,
                )

            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                image_embeds,
                placeholder_token_id=self.config.image_token_id,
            )

        if video_input is not None:
            video_embeds = self._process_video_input(video_input)
            if self.use_deepstack:
                video_embeds = torch.cat(video_embeds)

                video_embeds, video_embeds_multiscale = video_embeds.split(
                    [visual_dim, visual_dim * self.deepstack_num_level],
                    dim=-1)

                deepstack_input_embeds = merge_multimodal_embeddings(
                    input_ids,
                    deepstack_input_embeds,
                    video_embeds_multiscale,
                    placeholder_token_id=self.config.video_token_id,
                )

            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                video_embeds,
                placeholder_token_id=self.config.video_token_id,
            )

        if self.use_deepstack and deepstack_input_embeds is not None:
            deepstack_input_embeds = deepstack_input_embeds.view(
                inputs_embeds.shape[0], self.deepstack_num_level,
                visual_dim).permute(1, 0, 2).contiguous()
            self._set_deepstack_input_embeds(deepstack_input_embeds)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Run forward pass for Qwen3VL.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen3VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
            intermediate_tensors: Intermediate tensors from previous pipeline
                stages.
            inputs_embeds: Pre-computed input embeddings.
            **kwargs: Additional keyword arguments including:
                - pixel_values: Pixel values to be fed to a model.
                    `None` if no images are passed.
                - image_grid_thw: Tensor `(n_images, 3)` of image 3D grid in
                    LLM. `None` if no images are passed.
                - pixel_values_videos: Pixel values of videos to be fed to a
                    model. `None` if no videos are passed.
                - video_grid_thw: Tensor `(n_videos, 3)` of video 3D grid in
                    LLM. `None` if no videos are passed.
        """
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner from
        # `get_multimodal_embeddings` and `get_input_embeddings`, this
        # condition is only for v0 compatibility.
        elif inputs_embeds is None:
            image_input = self._parse_and_validate_image_input(**kwargs)
            video_input = self._parse_and_validate_video_input(**kwargs)

            if image_input is None and video_input is None:
                inputs_embeds = None
            else:
                if uses_mrope(self.config):
                    assert positions.ndim == 2 and positions.size(0) == 3, (
                        "multimodal section rotary embedding requires "
                        f"(3, seq_len) positions, but got {positions.size()}")
                
                inputs_embeds = self.get_input_embeddings_v0(
                    input_ids,
                    image_input=image_input,
                    video_input=video_input)
                input_ids = None

        if self.use_deepstack and inputs_embeds is not None and get_pp_group(
        ).is_first_rank:
            deepstack_input_embeds = self._get_deepstack_input_embeds(
                inputs_embeds.size(0))
        else:
            deepstack_input_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            # args for deepstack
            deepstack_input_embeds=deepstack_input_embeds,
        )

        if inputs_embeds is not None and get_pp_group().is_first_rank:
            self._clear_deepstack_input_embeds(inputs_embeds.size(0))

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:

        skip_prefixes = []
        if self.visual is None:
            skip_prefixes.extend(["visual."])
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="model.visual.merger",
            tower_model="model.visual.",
        )