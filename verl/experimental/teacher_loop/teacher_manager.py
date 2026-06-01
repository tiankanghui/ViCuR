# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import asyncio
from typing import Any, Optional
from uuid import uuid4

import ray
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from torch.nn import functional as F

from verl.experimental.agent_loop import AsyncLLMServerManager
from verl.protocol import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.config import DistillationConfig, DistillationLossConfig


def _get_teacher_sampling_params(
    distillation_config: DistillationConfig,
    distillation_loss_config: DistillationLossConfig,
) -> dict[str, Any]:
    """Get sampling parameters for teacher model when computing log probabilities for distillation."""
    if distillation_config.teacher_model.inference.temperature != 1.0:
        raise NotImplementedError("vLLM does not support temperature for prompt_logprobs.")

    num_logprobs = distillation_loss_config.topk if distillation_loss_config.loss_settings.use_topk else 0
    return {
        "max_tokens": 1,
        "temperature": distillation_config.teacher_model.inference.temperature,
        "prompt_logprobs": num_logprobs,
    }


def _pad_teacher_outputs(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    prompt_width: int,
    response_width: int,
    prompt_length: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO(wuxibin): remove padding and use tensordict.
    left_pad_size = prompt_width - prompt_length
    right_pad_size = response_width - response_length
    padding = (0, 0, left_pad_size, right_pad_size)
    return (
        F.pad(teacher_ids, padding, value=pad_token_id).unsqueeze(0),
        F.pad(teacher_logprobs, padding, value=0.0).unsqueeze(0),
    )


def _unpad_teacher_inputs(data: DataProto) -> tuple[list[int], int, int]:
    """Unpad valid sequence ids and prompt/response lengths from a single sample.
    The sample is a left-padded prompt concatenated with a right-padded response.
    TODO(wuxibin): remove padding and use tensordict.
    """
    assert len(data) == 1, "Teacher logprob computation expects a single sample"

    input_ids = data.batch["input_ids"][0]
    attention_mask = data.batch["attention_mask"][0]
    prompt_width = data.batch["prompts"][0].shape[0]
    response_width = data.batch["responses"][0].shape[0]
    assert attention_mask.shape[0] == prompt_width + response_width, (
        "attention_mask sequence length must match prompt and response widths"
    )
    valid_prompt_length = int(attention_mask[:prompt_width].sum().item())
    valid_response_length = int(attention_mask[-response_width:].sum().item())
    prompt_num_padding = prompt_width - valid_prompt_length
    sequence_ids = input_ids[prompt_num_padding : prompt_width + valid_response_length]
    sequence_ids = normalize_token_ids(sequence_ids)
    return sequence_ids, valid_prompt_length, valid_response_length


def _align_teacher_outputs_to_student_layout(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    student_prompt_length: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project teacher outputs to student's padded layout.

    Teacher prompt can differ from student prompt (q+c vs q). We only need response-token supervision to align
    with the student's trajectory. This helper keeps response tokens intact and trims/pads teacher prompt segment
    to student_prompt_length so downstream tensors keep the expected shape.
    """
    if teacher_ids.dim() == 1:
        teacher_ids = teacher_ids.unsqueeze(-1)
    if teacher_logprobs.dim() == 1:
        teacher_logprobs = teacher_logprobs.unsqueeze(-1)

    total_len = int(teacher_ids.shape[0])
    if response_length > total_len:
        raise ValueError(
            f"response_length ({response_length}) cannot exceed teacher output length ({total_len})"
        )

    teacher_prompt_len = total_len - response_length
    teacher_prompt_ids = teacher_ids[:teacher_prompt_len]
    teacher_prompt_lps = teacher_logprobs[:teacher_prompt_len]
    teacher_response_ids = teacher_ids[teacher_prompt_len:]
    teacher_response_lps = teacher_logprobs[teacher_prompt_len:]

    if teacher_prompt_len >= student_prompt_length:
        teacher_prompt_ids = teacher_prompt_ids[-student_prompt_length:]
        teacher_prompt_lps = teacher_prompt_lps[-student_prompt_length:]
    else:
        left_pad = student_prompt_length - teacher_prompt_len
        teacher_prompt_ids = F.pad(teacher_prompt_ids, (0, 0, left_pad, 0), value=pad_token_id)
        teacher_prompt_lps = F.pad(teacher_prompt_lps, (0, 0, left_pad, 0), value=0.0)

    return (
        torch.cat([teacher_prompt_ids, teacher_response_ids], dim=0),
        torch.cat([teacher_prompt_lps, teacher_response_lps], dim=0),
    )


class AsyncTeacherLLMServerManager(AsyncLLMServerManager):
    """Teacher-specific async client used for distillation logprob computation."""

    def __init__(
        self,
        config: DictConfig,
        servers: list[tuple[str, ray.actor.ActorHandle]],
        load_balancer_handle: ray.actor.ActorHandle,
        distillation_config: DictConfig | DistillationConfig,
        pad_token_id: int,
    ):
        super().__init__(config=config, servers=servers, load_balancer_handle=load_balancer_handle)
        if isinstance(distillation_config, DistillationConfig):
            self.distillation_config = distillation_config
        else:
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(distillation_config)
        self.distillation_loss_config: DistillationLossConfig = self.distillation_config.distillation_loss
        self.pad_token_id = pad_token_id

    # async def compute_teacher_logprobs_single(
    #     self,
    #     sequence_ids: list[int],
    #     multi_modal_data: Optional[dict[str, Any]] = None,
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     """Compute teacher log probabilities for a single unpadded sequence."""
    #     multi_modal_data = multi_modal_data or {}
    #     teacher_output = await self.generate(
    #         request_id=uuid4().hex,
    #         prompt_ids=sequence_ids,
    #         sampling_params=_get_teacher_sampling_params(self.distillation_config, self.distillation_loss_config),
    #         image_data=multi_modal_data.get("images"),
    #         video_data=multi_modal_data.get("videos"),
    #     )
    #     # Shapes: # S, (1 or K), where S is the response length, K is either 1 or topk depending on
    #     # the distillation loss settings.
    #     teacher_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
    #     teacher_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])
    #     assert teacher_ids.shape[0] == teacher_logprobs.shape[0] == len(sequence_ids)
    #     return teacher_ids, teacher_logprobs
    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
        expected_len: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute teacher log probabilities for a single unpadded sequence."""
        multi_modal_data = multi_modal_data or {}
        teacher_output = await self.generate(
            request_id=uuid4().hex,
            prompt_ids=sequence_ids,
            sampling_params=_get_teacher_sampling_params(self.distillation_config, self.distillation_loss_config),
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
        )
        teacher_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
        teacher_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])

        # Validate with expected_len (length after processor expansion)
        # If not provided, fall back to sequence_ids length (compatible with non-video scenarios)
        check_len = expected_len if expected_len is not None else len(sequence_ids)
        assert teacher_ids.shape[0] == teacher_logprobs.shape[0] == check_len, (
            f"Length mismatch: teacher_ids={teacher_ids.shape[0]}, "
            f"teacher_logprobs={teacher_logprobs.shape[0]}, "
            f"expected={check_len}, sequence_ids={len(sequence_ids)}"
        )
        return teacher_ids, teacher_logprobs

    async def compute_teacher_logprobs_batch(self, data: DataProto) -> DataProto:
        """Compute teacher log probabilities for a batch of prompt-response pairs."""
        multi_modal_data_batch = data.non_tensor_batch.get("teacher_multi_modal_data")
        teacher_sequence_ids_batch = data.non_tensor_batch.get("teacher_sequence_ids")
        tasks = []
        lengths = []
        use_explicit_teacher_sequences = []
        prompt_width = data.batch["prompts"].shape[1]
        response_width = data.batch["responses"].shape[1]

        # Compute logprobs for each sample in the batch
        for i in range(len(data)):
            item = data[i : i + 1]
            default_sequence_ids, prompt_length, response_length = _unpad_teacher_inputs(item)
            sequence_ids = default_sequence_ids
            use_explicit_sequence = False
            if teacher_sequence_ids_batch is not None and teacher_sequence_ids_batch[i] is not None:
                sequence_ids = normalize_token_ids(teacher_sequence_ids_batch[i])
                use_explicit_sequence = True
            multi_modal_data = None if multi_modal_data_batch is None else multi_modal_data_batch[i]
            lengths.append((prompt_length, response_length))
            use_explicit_teacher_sequences.append(use_explicit_sequence)
            tasks.append(
                asyncio.create_task(
                    self.compute_teacher_logprobs_single(
                        sequence_ids=sequence_ids,
                        expected_len=len(sequence_ids),
                        multi_modal_data=multi_modal_data,
                    )
                )
            )
        outputs = await asyncio.gather(*tasks)

        # Pad the teacher logprobs and ids
        padded_teacher_ids = []
        padded_teacher_logprobs = []
        for (teacher_ids, teacher_logprobs), (prompt_length, response_length), use_explicit_sequence in zip(
            outputs, lengths, use_explicit_teacher_sequences, strict=True
        ):
            if use_explicit_sequence:
                teacher_ids, teacher_logprobs = _align_teacher_outputs_to_student_layout(
                    teacher_ids=teacher_ids,
                    teacher_logprobs=teacher_logprobs,
                    student_prompt_length=prompt_length,
                    response_length=response_length,
                    pad_token_id=self.pad_token_id,
                )
            padded_ids, padded_logprobs = _pad_teacher_outputs(
                teacher_ids,
                teacher_logprobs,
                prompt_width=prompt_width,
                response_width=response_width,
                prompt_length=prompt_length,
                response_length=response_length,
                pad_token_id=self.pad_token_id,
            )
            padded_teacher_ids.append(padded_ids)
            padded_teacher_logprobs.append(padded_logprobs)

        batch = TensorDict(
            {
                "teacher_ids": torch.cat(padded_teacher_ids),
                "teacher_logprobs": torch.cat(padded_teacher_logprobs),
            },
            batch_size=len(data),
        )
        return DataProto(batch=batch)
