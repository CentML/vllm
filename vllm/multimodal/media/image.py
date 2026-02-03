# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from io import BytesIO
from pathlib import Path
from typing import TypeAlias

import pybase64
import torch
from PIL import Image

from vllm.logger import init_logger

from ..image import convert_image_mode, rgba_to_rgb
from .base import MediaIO, MediaWithBytes

logger = init_logger(__file__)

# Image output can be either PIL Image or Tensor (from nvJPEG)
ImageOutput: TypeAlias = Image.Image | torch.Tensor


class ImageMediaIO(MediaIO[ImageOutput]):
    # Class-level counters for nvJPEG statistics
    _nvjpeg_success_count: int = 0
    _nvjpeg_fallback_count: int = 0
    _nvjpeg_available: bool | None = None  # Lazy initialization

    def __init__(self, image_mode: str = "RGB", **kwargs) -> None:
        super().__init__()

        self.image_mode = image_mode
        # `kwargs` contains custom arguments from
        # --media-io-kwargs for this modality.
        # They can be passed to the underlying
        # media loaders (e.g. custom implementations)
        # for flexible control.
        self.kwargs = kwargs

        # Extract RGBA background color from kwargs if provided
        # Default to white background for backward compatibility
        rgba_bg = kwargs.get("rgba_background_color", (255, 255, 255))
        # Convert list to tuple for consistency
        if isinstance(rgba_bg, list):
            rgba_bg = tuple(rgba_bg)

        # Validate rgba_background_color format
        if not (
            isinstance(rgba_bg, tuple)
            and len(rgba_bg) == 3
            and all(isinstance(c, int) and 0 <= c <= 255 for c in rgba_bg)
        ):
            raise ValueError(
                "rgba_background_color must be a list or tuple of 3 integers "
                "in the range [0, 255]."
            )
        self.rgba_background_color = rgba_bg

        # Check nvJPEG availability on first instantiation
        if ImageMediaIO._nvjpeg_available is None:
            ImageMediaIO._nvjpeg_available = self._check_nvjpeg_available()

    @staticmethod
    def _check_nvjpeg_available() -> bool:
        """Check if nvJPEG is available (CUDA + torchvision decode_jpeg)."""
        try:
            # torch.cuda.is_available() can raise RuntimeError if CUDA driver fails
            if not torch.cuda.is_available():
                logger.debug("nvJPEG not available: CUDA not available")
                return False
            # Check if torchvision decode_jpeg is available
            from torchvision.io import decode_jpeg  # noqa: F401
            logger.info("nvJPEG available: using GPU-accelerated JPEG decoding")
            return True
        except ImportError:
            logger.debug("nvJPEG not available: torchvision.io.decode_jpeg not found")
            return False
        except RuntimeError as e:
            # CUDA driver initialization can fail with RuntimeError
            logger.debug(f"nvJPEG not available: CUDA driver error - {e}")
            return False
        except Exception as e:
            logger.debug(f"nvJPEG not available: {e}")
            return False

    @staticmethod
    def _is_jpeg(data: bytes) -> bool:
        """Detect JPEG format from magic bytes."""
        return len(data) >= 3 and data[:3] == b'\xff\xd8\xff'

    def _decode_with_nvjpeg(self, data: bytes) -> torch.Tensor | None:
        """
        Try to decode JPEG using nvJPEG (GPU-accelerated).
        
        Returns:
            torch.Tensor in CHW format on CPU, or None on failure.
            Note: Decoding happens on GPU for speed, then moved to CPU
            for compatibility with vLLM's memory pinning.
        """
        try:
            from torchvision.io import decode_jpeg, ImageReadMode
            
            # Convert bytes to tensor
            data_tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8)
            
            # Select mode based on image_mode
            if self.image_mode == "RGB":
                mode = ImageReadMode.RGB
            elif self.image_mode == "L":
                mode = ImageReadMode.GRAY
            else:
                mode = ImageReadMode.UNCHANGED
            
            # Decode on GPU using nvJPEG
            tensor = decode_jpeg(data_tensor, mode=mode, device='cuda')
            
            # Move to CPU for compatibility with vLLM's memory pinning
            tensor = tensor.cpu()
            
            # Update success counter and log periodically
            ImageMediaIO._nvjpeg_success_count += 1
            self._log_stats_if_needed()
            
            return tensor  # CHW tensor on CPU
            
        except Exception as e:
            logger.debug(f"nvJPEG decode failed, falling back to PIL: {e}")
            ImageMediaIO._nvjpeg_fallback_count += 1
            return None

    def _log_stats_if_needed(self) -> None:
        """Log nvJPEG statistics periodically."""
        total = ImageMediaIO._nvjpeg_success_count + ImageMediaIO._nvjpeg_fallback_count
        if total > 0 and total % 100 == 0:
            logger.info(
                f"nvJPEG decode stats: {ImageMediaIO._nvjpeg_success_count} successful, "
                f"{ImageMediaIO._nvjpeg_fallback_count} fallback to PIL"
            )

    def _convert_image_mode(
        self, image: Image.Image | MediaWithBytes[Image.Image]
    ) -> Image.Image:
        """Convert image mode with custom background color."""
        if isinstance(image, MediaWithBytes):
            image = image.media
        if image.mode == self.image_mode:
            return image
        elif image.mode == "RGBA" and self.image_mode == "RGB":
            return rgba_to_rgb(image, self.rgba_background_color)
        else:
            return convert_image_mode(image, self.image_mode)

    def load_bytes(self, data: bytes) -> MediaWithBytes[ImageOutput]:
        # Try nvJPEG for JPEG images when available
        if ImageMediaIO._nvjpeg_available and self._is_jpeg(data):
            tensor = self._decode_with_nvjpeg(data)
            if tensor is not None:
                return MediaWithBytes(tensor, data)
        
        # Fallback to PIL for non-JPEG or when nvJPEG fails
        image = Image.open(BytesIO(data))
        return MediaWithBytes(self._convert_image_mode(image), data)

    def load_base64(self, media_type: str, data: str) -> MediaWithBytes[ImageOutput]:
        return self.load_bytes(pybase64.b64decode(data, validate=True))

    def load_file(self, filepath: Path) -> MediaWithBytes[ImageOutput]:
        with open(filepath, "rb") as f:
            data = f.read()
        
        # Try nvJPEG for JPEG images when available
        if ImageMediaIO._nvjpeg_available and self._is_jpeg(data):
            tensor = self._decode_with_nvjpeg(data)
            if tensor is not None:
                return MediaWithBytes(tensor, data)
        
        # Fallback to PIL for non-JPEG or when nvJPEG fails
        image = Image.open(BytesIO(data))
        return MediaWithBytes(self._convert_image_mode(image), data)

    def encode_base64(
        self,
        media: Image.Image,
        *,
        image_format: str | None = None,
    ) -> str:
        if image_format is None:
            logger.warning_once(
                "The default format of `ImageMediaIO.encode_base64` will be changed "
                'from "JPEG" to "PNG" in v0.15 to avoid lossy compression. '
                "To continue using the old default, "
                'pass `format="JPEG"` explicitly to silence this warning.'
            )
            image_format = "JPEG"

        image = media

        with BytesIO() as buffer:
            image = self._convert_image_mode(image)
            image.save(buffer, image_format)
            data = buffer.getvalue()

        return pybase64.b64encode(data).decode("utf-8")


class ImageEmbeddingMediaIO(MediaIO[torch.Tensor]):
    def __init__(self) -> None:
        super().__init__()

    def load_bytes(self, data: bytes) -> torch.Tensor:
        buffer = BytesIO(data)
        # Enable sparse tensor integrity checks to prevent out-of-bounds
        # writes from maliciously crafted tensors
        with torch.sparse.check_sparse_tensor_invariants():
            tensor = torch.load(buffer, weights_only=True)
            return tensor.to_dense()

    def load_base64(self, media_type: str, data: str) -> torch.Tensor:
        return self.load_bytes(pybase64.b64decode(data, validate=True))

    def load_file(self, filepath: Path) -> torch.Tensor:
        # Enable sparse tensor integrity checks to prevent out-of-bounds
        # writes from maliciously crafted tensors
        with torch.sparse.check_sparse_tensor_invariants():
            tensor = torch.load(filepath, weights_only=True)
            return tensor.to_dense()

    def encode_base64(self, media: torch.Tensor) -> str:
        return pybase64.b64encode(media.numpy()).decode("utf-8")
