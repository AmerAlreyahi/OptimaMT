"""Shows where the output projection lives in a merged decoder, before and
after the rebuild, and what ONNX Runtime keeps of it after its graph
optimizations (written to results/graph.json, used by the report).

    python scripts/inspect_graph.py [pair]      # default en-ar
"""

import sys
import tempfile
from pathlib import Path

import onnx

from common import DEC, RESULTS, v1_dir, v2_decoder, write_json

BIG = 10 * 2**20  # initializers above 10 MiB


def size(t) -> int:
    n = 1
    for d in t.dims:
        n *= d
    return n * onnx.helper.tensor_dtype_to_np_dtype(t.data_type).itemsize


def large(g, where="main") -> list[dict]:
    out = [{"graph": where, "name": t.name, "shape": list(t.dims),
            "dtype": onnx.TensorProto.DataType.Name(t.data_type).lower(), "mib": round(size(t) / 2**20, 1)}
           for t in g.initializer if size(t) > BIG]
    for n in g.node:
        for a in n.attribute:
            if a.type == onnx.AttributeProto.GRAPH:
                out += large(a.g, a.g.name)
    return out


def projection(g, where="main") -> list[dict]:
    """The Transpose -> DequantizeLinear nodes that build the output projection."""
    out = []
    for n in g.node:
        if ((n.op_type == "Transpose" and "shared.weight" in n.input[0])
                or (n.op_type == "DequantizeLinear" and "weight_transposed" in n.input[0])):
            out.append({"graph": where, "op": n.op_type, "name": n.name, "inputs": list(n.input),
                        "outputs": list(n.output)})
        for a in n.attribute:
            if a.type == onnx.AttributeProto.GRAPH:
                out += projection(a.g, a.g.name)
    return out


def optimized(path: Path, qdq_off: bool) -> list[dict]:
    import onnxruntime as ort
    with tempfile.TemporaryDirectory() as tmp:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.optimized_model_filepath = str(Path(tmp) / "optimized.onnx")
        if qdq_off:
            so.add_session_config_entry("session.disable_quant_qdq", "1")
        ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        return large(onnx.load(so.optimized_model_filepath).graph)


def main() -> None:
    pair = sys.argv[1] if len(sys.argv) > 1 else "en-ar"
    out = {"pair": pair}
    for name, path in (("original", v1_dir(pair) / "onnx" / DEC), ("rebuilt", v2_decoder(pair))):
        m = onnx.load(str(path))
        iff = next(n for n in m.graph.node if n.op_type == "If")
        out[name] = {
            "if_node": iff.name, "if_input": iff.input[0],
            "branches": [a.g.name for a in iff.attribute if a.type == onnx.AttributeProto.GRAPH],
            "projection_nodes": projection(m.graph), "large_initializers": large(m.graph),
            "after_ort_default": optimized(path, False), "after_ort_qdq_off": optimized(path, True),
        }
    write_json(RESULTS / "graph.json", out)
    for name in ("original", "rebuilt"):
        print(name)
        for k in ("after_ort_default", "after_ort_qdq_off"):
            for t in out[name][k]:
                print(f"  {k:18} {t['graph']:10} {t['name']} {t['shape']} {t['dtype']} {t['mib']} MiB")


if __name__ == "__main__":
    main()
