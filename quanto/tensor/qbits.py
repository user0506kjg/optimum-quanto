# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import ast
from copy import copy

import torch
from torch.autograd import Function

from .core import ungroup
from .packed import PackedTensor
from .qtensor import QTensor, qfallback
from .qtype import qtypes


__all__ = ["QBitsTensor"]


class QBitsDequantizer(Function):
    @staticmethod
    def forward(ctx, t):
        unpacked = t._data.unpack()
        int8_data = unpacked.to(torch.int8) - t._zeropoint.to(torch.int8)
        if t.qtype.is_floating_point:
            # Upcast explicitly to the scale dtype
            dqt = t._scale * int8_data.to(t._scale.dtype)
        else:
            dqt = t._scale * int8_data
        if t.axis is None:
            return dqt
        # Restore the original shape (if needed)
        return ungroup(dqt, axis=t.axis, orig_shape=t.shape)

    @staticmethod
    def backward(ctx, gO):
        return gO


class QBitsTensor(QTensor):
    @staticmethod
    def __new__(cls, qtype, axis, size, stride, data, scale, zeropoint, requires_grad=False):
        assert isinstance(data, PackedTensor)
        assert data.device == scale.device
        assert data.device == zeropoint.device
        return torch.Tensor._make_wrapper_subclass(
            cls, size, strides=stride, dtype=scale.dtype, device=data.device, requires_grad=requires_grad
        )

    def __init__(self, qtype, axis, size, stride, data, scale, zeropoint, requires_grad=False):
        super().__init__(qtype, axis)
        self._data = data
        self._scale = scale
        self._zeropoint = zeropoint

    def __repr__(self):
        return f"QBitsTensor({self._data}, scale={self._scale}, zeropoint={self._zeropoint}, dtype={self.dtype})"

    def dequantize(self):
        return QBitsDequantizer.apply(self)

    def __tensor_flatten__(self):
        inner_tensors = ["_data", "_scale", "_zeropoint"]
        # Since meta can be used for serialization, use only strings
        meta = {
            "qtype": self._qtype.name,
            "axis": str(self._axis),
            "size": str(list(self.size())),
            "stride": str(list(self.stride())),
        }
        return inner_tensors, meta

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert len(inner_tensors) == 3
        assert len(meta) == 4
        data, scale, zeropoint = inner_tensors["_data"], inner_tensors["_scale"], inner_tensors["_zeropoint"]
        # Meta should only contain strings, AST compatible except qtype
        qtype = qtypes[meta["qtype"]]
        axis = ast.literal_eval(meta["axis"])
        size = ast.literal_eval(meta["size"])
        stride = ast.literal_eval(meta["stride"])
        return QBitsTensor(qtype, axis, size, stride, data, scale, zeropoint)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        from .func import get_qtensor_func

        kwargs = kwargs or {}

        # Look for a func accepting QTensor inputs
        qfunc = get_qtensor_func(func)
        if qfunc is not None:
            return qfunc(*args, **kwargs)
        # Defer to dispatcher to look instead for QTensor operations
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, op, types, args, kwargs=None):
        if op.overloadpacket is torch.ops.aten.detach:
            # Detach is required when copying and deserializing
            t = args[0]
            data = op(t._data)
            scale = op(t._scale)
            zeropoint = op(t._zeropoint)
            return QBitsTensor(t._qtype, t._axis, t.size(), t.stride(), data, scale, zeropoint)
        elif op.overloadpacket is torch.ops.aten._to_copy:
            t = args[0]
            # Copy scale
            scale = op(t._scale, **kwargs)
            # Move data and zeropoint, ignoring dtype (it only applies to scale)
            data_kwargs = copy(kwargs)
            data_kwargs["dtype"] = torch.uint8
            data = op(t._data, **data_kwargs)
            zeropoint_kwargs = copy(kwargs)
            zeropoint_kwargs["dtype"] = torch.int8
            zeropoint = op(t._zeropoint, **data_kwargs)
            return QBitsTensor(t._qtype, t._axis, t.size(), t.stride(), data, scale, zeropoint)
        # No dispatch available: qfallback
        return qfallback(op, *args, **kwargs)
