# Copyright 2025 The MjLab Developers
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

import torch


def configure_torch_backends(allow_tf32: bool = True, deterministic: bool = False):
  """Configure PyTorch CUDA and cuDNN backends for performance/reproducibility."""
  # https://docs.pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere
  torch.backends.cuda.matmul.allow_tf32 = allow_tf32
  torch.backends.cudnn.allow_tf32 = allow_tf32

  torch.backends.cudnn.deterministic = deterministic
  torch.backends.cudnn.benchmark = not deterministic
