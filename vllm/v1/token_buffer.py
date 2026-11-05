# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token ids held as int32 that read like a list (``VLLM_PROMPT_TOKEN_BUFFER``).

A long prompt held as a ``list[int]`` costs one Python int per token, built
(``tolist`` + ``copy``) on the EngineCore input thread and freed again on the
main thread when the request finishes. ``TokenBuffer`` keeps the ids in one
int32 ``array``: building it from the packed int32 prompt is a memcpy.

Reads stay list-like, so every consumer of ``Request.prompt_token_ids`` and
``Request._all_token_ids`` sees the values it saw before: indexing returns
``int``, slices and ``copy()`` return ``list``, it compares equal to a list or
tuple with the same tokens, ``+`` gives a list. Writes (``append``, ``extend``,
``pop``, ``del`` of a slice, slice assignment) keep the int32 storage. Token ids
fit int32 (the packed transport, ``packed_prompt_ids``, already requires it).

Design after vLLM PR #62702 ("Keep prompt token ids as int32 buffers from
vllm-rs to the workers", open; ``vllm/v1/token_buffer.py`` there), re-written
here with explicit pickling and ``clear``.
"""

from array import array


def _from_bytes(data: bytes) -> "TokenBuffer":
    buf = TokenBuffer()
    buf.frombytes(data)
    return buf


class TokenBuffer(array):
    __slots__ = ()

    def __new__(cls, tokens=()):
        # From an int32 array (or another TokenBuffer) this is a memcpy.
        return super().__new__(cls, "i", tokens)

    def __getitem__(self, index):
        item = array.__getitem__(self, index)
        return item.tolist() if isinstance(index, slice) else item

    def __setitem__(self, index, value):
        if isinstance(index, slice) and not isinstance(value, array):
            value = array("i", value)
        array.__setitem__(self, index, value)

    def __eq__(self, other):
        if isinstance(other, (list, tuple)):
            return len(self) == len(other) and self.tolist() == list(other)
        return array.__eq__(self, other)

    def __ne__(self, other):
        eq = self.__eq__(other)
        return eq if eq is NotImplemented else not eq

    __hash__ = None  # type: ignore[assignment]

    def __add__(self, other):
        return self.tolist() + list(other)

    def __radd__(self, other):
        return list(other) + self.tolist()

    def __iadd__(self, other):
        self.extend(other)
        return self

    def __reduce_ex__(self, protocol):
        return (_from_bytes, (self.tobytes(),))

    def copy(self) -> list[int]:
        return self.tolist()

    def clear(self) -> None:
        del self[:]
