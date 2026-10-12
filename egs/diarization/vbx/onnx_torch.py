"""ONNX x-vector extractors as torch modules (``onnx2torch``), to run them
batched on the GPU.  onnxruntime-gpu needs CUDA 12, the project is on CUDA 13.

The VBx models need two fixes on top of ``onnx2torch``:

* converters for newer opset versions of ops whose semantics did not change
  for these graphs (``Cast`` 19, ``Constant`` 19, ``ReduceMean`` 18, ...) are
  aliased to the latest available version;
* ``auto_pad=SAME_*`` of stride-1 convolutions / pools is replaced by explicit
  ``pads`` (onnx2torch does not implement it);
* ``Gelu`` (opset 20) is added;
* omitted ``min`` / ``max`` inputs of ``Clip`` become constants;
* constant ``axes`` inputs of ``Reduce*`` (opset 18) become attributes, as in
  the opset 13 version their converter is aliased to.
"""

from __future__ import annotations

import re
import typing as tp
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper
from onnx2torch import convert
from onnx2torch.node_converters.registry import _CONVERTER_REGISTRY, OperationDescription, add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult, onnx_mapping_from_node

_POOLS = {"Conv", "MaxPool", "AveragePool", "LpPool"}


@add_converter(operation_type="Gelu", version=20)
def _gelu(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    approximate = node.attributes.get("approximate", "none")
    return OperationConverterResult(
        torch_module=torch.nn.GELU(approximate=approximate), onnx_mapping=onnx_mapping_from_node(node)
    )


def _explicit_pads(model: onnx.ModelProto) -> None:
    for node in model.graph.node:
        if node.op_type not in _POOLS:
            continue
        attrs = {a.name: a for a in node.attribute}
        mode = helper.get_attribute_value(attrs["auto_pad"]).decode() if "auto_pad" in attrs else "NOTSET"
        if mode in ("NOTSET", "VALID"):
            continue
        kernel = helper.get_attribute_value(attrs["kernel_shape"])
        strides = helper.get_attribute_value(attrs["strides"]) if "strides" in attrs else [1] * len(kernel)
        dilations = helper.get_attribute_value(attrs["dilations"]) if "dilations" in attrs else [1] * len(kernel)
        if any(s != 1 for s in strides):
            raise NotImplementedError(f"{node.name}: auto_pad {mode} with strides {strides}")
        total = [d * (k - 1) for k, d in zip(kernel, dilations)]
        small = [t // 2 for t in total]
        large = [t - s for t, s in zip(total, small)]
        begin, end = (small, large) if mode == "SAME_UPPER" else (large, small)
        node.attribute.remove(attrs["auto_pad"])
        node.attribute.append(helper.make_attribute("auto_pad", "NOTSET"))
        node.attribute.append(helper.make_attribute("pads", begin + end))


def _clip_bounds(model: onnx.ModelProto) -> None:
    graph = model.graph
    for node in graph.node:
        if node.op_type != "Clip" or len(node.input) < 2:
            continue
        for i, bound in ((1, -np.inf), (2, np.inf)):
            if i < len(node.input) and node.input[i] == "":
                name = f"{node.output[0]}__clip{i}"
                graph.initializer.append(numpy_helper.from_array(np.array(bound, dtype=np.float32), name))
                node.input[i] = name


_REDUCE_ATTR_AXES = {"ReduceMean", "ReduceProd", "ReduceMax", "ReduceMin", "ReduceL1", "ReduceL2",
                     "ReduceLogSum", "ReduceLogSumExp", "ReduceSumSquare"}


def _reduce_axes(model: onnx.ModelProto) -> None:
    graph = model.graph
    constants = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            constants[node.output[0]] = numpy_helper.to_array(helper.get_attribute_value(node.attribute[0]))
    for node in graph.node:
        if node.op_type not in _REDUCE_ATTR_AXES or len(node.input) < 2:
            continue
        if node.input[1] and node.input[1] not in constants:
            raise NotImplementedError(f"{node.name}: dynamic axes")
        if node.input[1]:
            node.attribute.append(helper.make_attribute("axes", [int(a) for a in constants[node.input[1]].ravel()]))
        del node.input[1:]


def _alias(op: str, version: int) -> bool:
    older = [k.version for k in _CONVERTER_REGISTRY if k.domain == "" and k.operation_type == op and k.version < version]
    if not older:
        return False
    key = OperationDescription(domain="", operation_type=op, version=version)
    _CONVERTER_REGISTRY[key] = _CONVERTER_REGISTRY[OperationDescription(domain="", operation_type=op, version=max(older))]
    return True


def input_shape(path: tp.Union[str, Path]) -> tp.List[tp.Union[int, str]]:
    """Shape of the first input, symbolic dimensions by name."""
    model = onnx.load(str(path), load_external_data=False)
    dims = model.graph.input[0].type.tensor_type.shape.dim
    return [d.dim_value if d.HasField("dim_value") else d.dim_param for d in dims]


def load_onnx(path: tp.Union[str, Path]) -> torch.nn.Module:
    model = onnx.load(str(path))
    _explicit_pads(model)
    _clip_bounds(model)
    _reduce_axes(model)
    while True:
        try:
            return convert(model).eval()
        except NotImplementedError as e:
            match = re.search(r"operation_type='(\w+)', version=(\d+)", str(e))
            if match is None or not _alias(match.group(1), int(match.group(2))):
                raise
