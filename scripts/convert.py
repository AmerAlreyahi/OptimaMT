"""Rebuilds each pair's merged decoder with the output projection's weight
dequantized once, before the decoder's `If`, instead of once in each branch.

The merged decoder (optimum's `decoder_model_merged`) is one `If` node: the
`no_past` branch runs the first step, `with_past` every later step. Each branch
holds its own copy of

    Transpose(model.shared.weight_merged_0_quantized)          int8 [vocab, 512]
      -> DequantizeLinear(.., model.shared.weight_merged_0_scale, .._zero_point)
      -> MatMul (logits)

Both copies read the same initializers and compute the same value. This script
moves the Transpose and the DequantizeLinear (and any initializer they read
that lives in a branch) into the main graph, right before the `If`; both
branches then read the one outer value. Nothing else changes: same inputs and
outputs, same int8 weights, same tokens.

    python scripts/convert.py            # all pairs
    python scripts/convert.py en-ar ja-en

Writes work/build/opus-mt-<pair>-decoder_model_merged_quantized.onnx and
work/build/SHA256SUMS.txt, and checks every file against the published sha256
in data/sources.json (the output is byte-for-byte reproducible).
"""

import sys

import onnx

from common import BUILD, DEC, ROOT, sha256, source, sources, v1_dir, v2_decoder, write_json


def hoist(src, dst) -> None:
    m = onnx.load(str(src))
    g = m.graph
    iff = [n for n in g.node if n.op_type == "If"]
    assert len(iff) == 1
    iff = iff[0]
    branches = [a.g for a in iff.attribute if a.type == onnx.AttributeProto.GRAPH]
    assert len(branches) == 2
    outer = {i.name for i in g.initializer}
    moved = None
    for b in branches:
        # the output projection: the one DequantizeLinear fed by a Transpose
        dq = [n for n in b.node if n.op_type == "DequantizeLinear"
              and any(t.op_type == "Transpose" and t.output[0] == n.input[0] for t in b.node)]
        assert len(dq) == 1, [n.output[0] for n in dq]
        dq = dq[0]
        tr = next(t for t in b.node if t.output[0] == dq.input[0])
        local = {i.name: i for i in b.initializer}
        consts = list(tr.input) + list(dq.input[1:])
        # every input is a constant: an initializer of the branch or of the main graph
        assert all(n in local or n in outer for n in consts), consts
        if moved is None:
            moved = (onnx.helper.printable_node(tr), onnx.helper.printable_node(dq))
            for n in consts:
                if n in local:
                    assert n not in outer
                    g.initializer.append(local[n])
                    outer.add(n)
            at = list(g.node).index(iff)
            g.node.insert(at, dq)
            g.node.insert(at, tr)
        else:
            # the other branch built exactly the same value
            assert moved == (onnx.helper.printable_node(tr), onnx.helper.printable_node(dq))
            for n in consts:
                if n in local:
                    assert local[n].SerializeToString() == next(
                        i for i in g.initializer if i.name == n).SerializeToString()
        keep = [n for n in b.node if n is not dq and n is not tr]
        del b.node[:]
        b.node.extend(keep)
        for n in consts:
            if n in local:
                b.initializer.remove(local[n])
    onnx.checker.check_model(m)
    tmp = dst.with_name(dst.name + ".part")
    onnx.save(m, str(tmp))
    tmp.replace(dst)


def main() -> None:
    pairs = sys.argv[1:] or [e["pair"] for e in sources()]
    BUILD.mkdir(parents=True, exist_ok=True)
    bad = []
    for pair in pairs:
        dst = v2_decoder(pair)
        pin = source(pair)["v2_decoder"]
        if pin is None:  # a new pair: build it and pin the result in data/sources.json
            hoist(v1_dir(pair) / "onnx" / DEC, dst)
            pin = {"size": dst.stat().st_size, "sha256": sha256(dst), "url": None}
            new = sources()
            next(e for e in new if e["pair"] == pair)["v2_decoder"] = pin
            write_json(ROOT / "data" / "sources.json", new)
        if not (dst.exists() and dst.stat().st_size == pin["size"] and sha256(dst) == pin["sha256"]):
            hoist(v1_dir(pair) / "onnx" / DEC, dst)
        same = sha256(dst) == pin["sha256"]
        bad += [] if same else [pair]
        print(f"{pair}  {dst.stat().st_size:>11,} bytes  {'matches the published sha256' if same else 'DIFFERS'}",
              flush=True)
    sums = sorted(f"{sha256(f)}  {f.name}" for f in BUILD.glob(f"opus-mt-*-{DEC}"))
    (BUILD / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    if bad:
        raise SystemExit(f"not reproduced: {' '.join(bad)}")


if __name__ == "__main__":
    main()
