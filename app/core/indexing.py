from app.core.models import Chunk, IngestResult
import numpy as np
from numpy.typing import NDArray
import torch
from typing import Any

def add_to_index(
    all_chunks: list[Chunk],
    embeddings: list[torch.Tensor] | NDArray[Any] | torch.Tensor,
) -> None:
