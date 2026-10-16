"""Print the structure of an Assetto Corsa .kn5 file (textures, materials, nodes).

Reads the file as a stream and skips texture and vertex payloads, so it works
on 300 MB track files. Used to check our own exports against Kunos tracks.

    python kn5_inspect.py track.kn5 [--nodes N] [--grep ROAD]
"""
import argparse
import struct
import sys


class Reader:
    def __init__(self, f):
        self.f = f

    def raw(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("truncated kn5")
        return b

    def skip(self, n):
        self.f.seek(n, 1)

    def i32(self):
        return struct.unpack("<i", self.raw(4))[0]

    def u32(self):
        return struct.unpack("<I", self.raw(4))[0]

    def u8(self):
        return self.raw(1)[0]

    def f32(self, n=1):
        v = struct.unpack("<%df" % n, self.raw(4 * n))
        return v[0] if n == 1 else v

    def s(self):
        return self.raw(self.i32()).decode("utf-8", "replace")


def read_header(r):
    if r.raw(6) != b"sc6969":
        raise ValueError("not a kn5 file")
    version = r.i32()
    extra = r.i32() if version > 5 else 0
    return version, extra


def read_textures(r, keep_data=False):
    out = []
    for _ in range(r.i32()):
        active = r.i32()
        name = r.s()
        size = r.u32()
        if keep_data:
            data = r.raw(size)
        else:
            head = r.raw(min(size, 16))
            r.skip(size - len(head))
            data = head
        out.append({"active": active, "name": name, "size": size, "data": data})
    return out


def read_materials(r):
    out = []
    for _ in range(r.i32()):
        m = {"name": r.s(), "shader": r.s(), "blend": r.u8(), "alpha_tested": r.u8(),
             "depth": r.i32(), "props": {}, "textures": []}
        for _ in range(r.i32()):
            name = r.s()
            a = r.f32()
            b = r.f32(2)
            c = r.f32(3)
            d = r.f32(4)
            m["props"][name] = (a, b, c, d)
        for _ in range(r.i32()):
            m["textures"].append((r.s(), r.i32(), r.s()))
        out.append(m)
    return out


def read_node(r, nodes, depth=0, keep_mesh=False):
    n = {"cls": r.i32(), "name": r.s(), "depth": depth}
    nchild = r.i32()
    n["active"] = r.u8()
    if n["cls"] == 1:
        n["matrix"] = r.f32(16)
    elif n["cls"] == 2:
        n["cast_shadows"], n["visible"], n["transparent"] = r.u8(), r.u8(), r.u8()
        nv = r.u32()
        if keep_mesh:
            n["vbytes"] = r.raw(44 * nv)
        else:
            first = r.f32(11) if nv else None
            r.skip(44 * nv - (44 if nv else 0))
            n["first_vertex"] = first
        n["nv"] = nv
        ni = r.u32()
        if keep_mesh:
            n["ibytes"] = r.raw(2 * ni)
        else:
            r.skip(2 * ni)
        n["ni"] = ni
        n["material"] = r.u32()
        n["layer"] = r.u32()
        n["lod_in"], n["lod_out"] = r.f32(), r.f32()
        n["bs_center"] = r.f32(3)
        n["bs_radius"] = r.f32()
        n["renderable"] = r.u8()
    else:
        raise ValueError("unsupported node class %d (%s)" % (n["cls"], n["name"]))
    n["children"] = nchild
    nodes.append(n)
    for _ in range(nchild):
        read_node(r, nodes, depth + 1, keep_mesh)


def load(path, keep_data=False):
    with open(path, "rb") as f:
        r = Reader(f)
        version, extra = read_header(r)
        tex = read_textures(r, keep_data)
        mats = read_materials(r)
        nodes = []
        read_node(r, nodes, keep_mesh=keep_data)
        rest = f.read()
    return {"version": version, "extra": extra, "textures": tex, "materials": mats,
            "nodes": nodes, "trailing_bytes": len(rest)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kn5")
    ap.add_argument("--nodes", type=int, default=40, help="max nodes to print")
    ap.add_argument("--grep", default="", help="only print nodes whose name contains this")
    ap.add_argument("--materials", type=int, default=20, help="max materials to print")
    a = ap.parse_args()
    k = load(a.kn5)
    print("version %d extra %d, %d textures, %d materials, %d nodes, %d trailing bytes" % (
        k["version"], k["extra"], len(k["textures"]), len(k["materials"]), len(k["nodes"]),
        k["trailing_bytes"]))
    for t in k["textures"][:15]:
        print("  tex %-40s %9d B active=%d head=%r" % (t["name"], t["size"], t["active"], t["data"][:4]))
    for m in k["materials"][:a.materials]:
        props = ", ".join("%s=%s" % (p, round(v[0], 3) if p != "ksEmissive" else tuple(round(x, 3) for x in v[2]))
                          for p, v in m["props"].items())
        print("  mat %-28s %-22s blend=%d at=%d depth=%d | %s | %s" % (
            m["name"], m["shader"], m["blend"], m["alpha_tested"], m["depth"], props,
            ["%s:%d:%s" % t for t in m["textures"]]))
    shown = 0
    for n in k["nodes"]:
        if a.grep and a.grep not in n["name"]:
            continue
        if shown >= a.nodes:
            break
        shown += 1
        ind = "  " * n["depth"]
        if n["cls"] == 1:
            print("  %sDUMMY %s children=%d m=%s" % (ind, n["name"], n["children"],
                                                   [round(x, 3) for x in n["matrix"]]))
        else:
            print("  %sMESH %s nv=%d ni=%d mat=%d layer=%d lod=%g/%g bs=%s r=%.1f flags=%d%d%d%d v0=%s" % (
                ind, n["name"], n["nv"], n["ni"], n["material"], n["layer"], n["lod_in"], n["lod_out"],
                [round(x, 1) for x in n["bs_center"]], n["bs_radius"], n["cast_shadows"], n["visible"],
                n["transparent"], n["renderable"],
                [round(x, 3) for x in n["first_vertex"]] if n["first_vertex"] else None))


if __name__ == "__main__":
    sys.exit(main())
