"""Build the 5-junction SUMO network and random traffic files.

Layout (3 lanes/dir: right / straight / left). Mid-block zebra at CW on N2→J2:

                 N5        N1        N2        N3
                  |         |         |         |
                  |         |        CW
                  |         |         |         |
   W --- J5 ----- J1 ----- J2 ----- J3 ----- J4 --- E
         (T)     (4-way)  (4-way)  (4-way)   (T)
                  |         |         |         |
                            S1        S2        S3        S4

    python make_scenario.py
    python make_scenario.py --demand-only --n-routes 100 --seed 7
"""
import argparse, json, math, os, random, shutil, subprocess, sys, time

S = 250.0
CROSSWALK_NODE = "CW"

NODES = {
    "W": (-2 * S, 0, "end"), "J5": (-S, 0, "junction"), "J1": (0, 0, "junction"),
    "J2": (S, 0, "junction"), "J3": (2 * S, 0, "junction"), "J4": (3 * S, 0, "junction"),
    "E": (4 * S, 0, "end"),
    "N5": (-S, S, "end"), "N1": (0, S, "end"), "S1": (0, -S, "end"),
    "N2": (S, S, "end"), "S2": (S, -S, "end"), "N3": (2 * S, S, "end"), "S3": (2 * S, -S, "end"),
    "S4": (3 * S, -S, "end"),
    "CW": (S, 0.55 * S, "crosswalk"),
}

LINKS = [("W", "J5"), ("J5", "J1"), ("J1", "J2"), ("J2", "J3"), ("J3", "J4"), ("J4", "E"),
         ("J5", "N5"), ("J1", "N1"), ("J1", "S1"),
         ("J2", "CW"), ("CW", "N2"), ("J2", "S2"),
         ("J3", "N3"), ("J3", "S3"), ("J4", "S4")]
JUNCTIONS = ["J1", "J2", "J3", "J4", "J5", "CW"]
ENDS = [n for n, v in NODES.items() if v[2] == "end"]
ARTERIAL_ENDS = ["W", "E"]

CROSSWALK_EDGES = ("N2_CW", "CW_N2")
SIDEWALK_EDGES = {"N2_CW", "CW_N2", "CW_J2", "J2_CW"}

VEHICLE_TYPES = {
    "car":        ({"length": "4.5", "width": "1.8", "accel": "2.6", "decel": "4.5",
                    "maxSpeed": "13.9", "sigma": "0.5", "guiShape": "passenger",
                    "color": "0,100,200"}, 0.70),
    "van":        ({"length": "5.5", "width": "2.0", "accel": "2.2", "decel": "4.0",
                    "maxSpeed": "13.9", "sigma": "0.5", "guiShape": "passenger/van",
                    "color": "0,160,120"}, 0.10),
    "truck":      ({"length": "8.0", "width": "2.4", "accel": "1.5", "decel": "3.5",
                    "maxSpeed": "11.0", "sigma": "0.5", "guiShape": "truck",
                    "color": "180,80,20"}, 0.08),
    "bus":        ({"length": "12.0", "width": "2.5", "accel": "1.2", "decel": "3.5",
                    "maxSpeed": "12.0", "sigma": "0.3", "guiShape": "bus",
                    "color": "220,180,0"}, 0.05),
    "motorcycle": ({"length": "2.2", "width": "0.9", "accel": "3.0", "decel": "6.0",
                    "maxSpeed": "13.9", "sigma": "0.7", "guiShape": "motorcycle",
                    "color": "200,40,40"}, 0.07),
}


def sumo_tools_dir():
    home = os.environ.get("SUMO_HOME")
    if not home:
        try:
            import sumo
            home = sumo.SUMO_HOME
        except ImportError:
            sys.exit("Cannot find SUMO. Set SUMO_HOME or run: pip install eclipse-sumo")
    return os.path.join(home, "tools")


def _exe(name):
    return shutil.which(name) or name


def _movement(a, b, c):
    v1 = (b[0] - a[0], b[1] - a[1]); v2 = (c[0] - b[0], c[1] - b[1])
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n = math.hypot(*v1) * math.hypot(*v2)
    if dot < -0.9 * n:
        return "uturn"
    if abs(cross) < 0.3 * n:
        return "straight"
    return "left" if cross > 0 else "right"


def lane_roles(moves, lanes):
    has = set(moves)
    roles = {l: set() for l in range(lanes)}
    if lanes >= 3 and "straight" in has and "right" in has and "left" in has:
        roles[0] = {"right"}
        for l in range(1, lanes - 1):
            roles[l] = {"straight"}
        roles[lanes - 1] = {"left"}
        return roles
    if lanes == 2:
        if "straight" in has:
            roles[0].add("straight"); roles[1].add("straight")
        if "right" in has:
            roles[0].add("right")
        if "left" in has:
            roles[1].add("left")
        if "straight" not in has and "right" in has and "left" in has:
            roles[0] = {"right"}; roles[1] = {"left"}
        elif not any(roles.values()):
            m = next(iter(has)); roles[0].add(m); roles[1].add(m)
        return roles
    if "straight" in has:
        for l in range(lanes):
            roles[l].add("straight")
        if "right" in has:
            roles[0] = {"right"}
        if "left" in has:
            roles[lanes - 1] = {"left"}
    else:
        half = max(1, lanes // 2)
        for l in range(lanes):
            if "right" in has and "left" in has:
                roles[l] = {"right" if l < half else "left"}
            else:
                roles[l] = {next(iter(has))}
    return roles


def build_network(out, lanes):
    if lanes < 2:
        sys.exit("--lanes must be at least 2")
    os.makedirs(out, exist_ok=True)

    nod = ['<nodes>']
    for n, (x, y, kind) in NODES.items():
        t = "traffic_light" if kind in ("junction", "crosswalk") else "dead_end"
        nod.append(f'  <node id="{n}" x="{x}" y="{y}" type="{t}"/>')
    nod.append('</nodes>')

    edges = []
    for a, b in LINKS:
        edges += [(f"{a}_{b}", a, b), (f"{b}_{a}", b, a)]

    edg = ['<edges>']
    for i, a, b in edges:
        if i in SIDEWALK_EDGES:
            edg.append(f'  <edge id="{i}" from="{a}" to="{b}" numLanes="{lanes}" '
                       f'speed="13.9" sidewalkWidth="3.0"/>')
        else:
            edg.append(f'  <edge id="{i}" from="{a}" to="{b}" numLanes="{lanes}" speed="13.9"/>')
    edg.append('</edges>')

    con = ['<connections>']
    for j in JUNCTIONS:
        for ei, a, b in [e for e in edges if e[2] == j]:
            options = []
            for eo, c, d in [e for e in edges if e[1] == j]:
                m = _movement(NODES[a][:2], NODES[j][:2], NODES[d][:2])
                if m != "uturn":
                    options.append((eo, m))
            roles = lane_roles([m for _, m in options], lanes)
            for eo, m in options:
                for lane, moves in roles.items():
                    if m in moves:
                        con.append(f'  <connection from="{ei}" to="{eo}" fromLane="{lane}" '
                                   f'toLane="{min(lane, lanes - 1)}"/>')

    e1, e2 = CROSSWALK_EDGES
    con.append(f'  <crossing node="{CROSSWALK_NODE}" edges="{e1} {e2}" priority="0" width="4.0"/>')
    con.append('</connections>')

    for name, lines in (("nodes.nod.xml", nod), ("edges.edg.xml", edg), ("cons.con.xml", con)):
        with open(os.path.join(out, name), "w") as f:
            f.write("\n".join(lines))

    subprocess.run([_exe("netconvert"),
                    "--node-files", os.path.join(out, "nodes.nod.xml"),
                    "--edge-files", os.path.join(out, "edges.edg.xml"),
                    "--connection-files", os.path.join(out, "cons.con.xml"),
                    "--no-turnarounds", "true",
                    "--no-warnings", "true",
                    "--tls.guess-signals", "true",
                    "-o", os.path.join(out, "net.net.xml")], check=True)
    print(f"Network: {lanes} lanes/dir. Mid-block zebra at {CROSSWALK_NODE} on {e1}/{e2}.")


def compute_phases(out):
    import sumolib
    net = sumolib.net.readNet(os.path.join(out, "net.net.xml"), withPrograms=True)
    phases = {}
    for j in JUNCTIONS:
        tls = net.getTLS(j)
        conns = tls.getConnections()
        progs = tls.getPrograms()
        if progs:
            n_links = len(next(iter(progs.values())).getPhases()[0].state)
        else:
            n_links = max(c[2] for c in conns) + 1
        veh_idxs = {c[2] for c in conns}
        ped_idxs = set(range(n_links)) - veh_idxs

        if j == CROSSWALK_NODE:
            cars = "".join("G" if i in veh_idxs else "r" for i in range(n_links))
            walk = "".join("G" if i in ped_idxs else "r" for i in range(n_links))
            phases[j] = [cars, walk] if ped_idxs else [cars]
            print(f"  {j}: {len(phases[j])} phases (cars/walk), "
                  f"state_len={n_links}, ped_links={len(ped_idxs)}")
            continue

        groups = {}
        approaches = {"EW": set(), "NS": set()}
        for in_l, out_l, idx in conns:
            ein, eout = in_l.getEdge(), out_l.getEdge()
            a = ein.getFromNode().getCoord()
            b = ein.getToNode().getCoord()
            c = eout.getToNode().getCoord()
            m = _movement(a, b, c)
            axis = "EW" if abs(b[0] - a[0]) > abs(b[1] - a[1]) else "NS"
            approaches[axis].add(ein.getID())
            kind = "left" if m == "left" else "through"
            groups.setdefault((axis, kind), set()).add(idx)

        plist = []
        for axis in ("NS", "EW"):
            if len(approaches[axis]) == 0:
                continue
            if len(approaches[axis]) == 1:
                plist.append(groups.get((axis, "through"), set()) | groups.get((axis, "left"), set()))
            else:
                for kind in ("through", "left"):
                    if groups.get((axis, kind)):
                        plist.append(groups[(axis, kind)])

        phase_strs = ["".join("G" if i in p else "r" for i in range(n_links)) for p in plist]
        phases[j] = phase_strs
        print(f"  {j}: {len(phase_strs)} phases, state_len={n_links}")
    with open(os.path.join(out, "phases.json"), "w") as f:
        json.dump(phases, f, indent=1)
    return phases


PROFILES = ["steady", "morning_rush", "evening_rush", "side_street_surge", "random_waves"]


def _multipliers(kind, n, rng):
    x = [i / max(n - 1, 1) for i in range(n)]
    if kind == "steady":
        return [1.0] * n
    if kind in ("morning_rush", "evening_rush"):
        return [0.4 + 1.1 * math.exp(-((v - 0.5) ** 2) / 0.05) for v in x]
    if kind == "side_street_surge":
        c = rng.uniform(0.2, 0.8)
        return [0.6 + 1.0 * math.exp(-((v - c) ** 2) / 0.02) for v in x]
    return [rng.uniform(0.3, 1.5) for _ in range(n)]


def _vtype_xml():
    lines = ['  <vTypeDistribution id="mixed">']
    for vid, (attrs, prob) in VEHICLE_TYPES.items():
        attr_s = " ".join(f'{k}="{v}"' for k, v in attrs.items())
        lines.append(f'    <vType id="{vid}" probability="{prob}" {attr_s}/>')
    lines.append('  </vTypeDistribution>')
    return lines


def _pedestrian_xml(horizon):
    return [
        '  <vType id="ped" vClass="pedestrian" speed="1.2" guiShape="pedestrian" color="255,220,0"/>',
        f'  <personFlow id="ped_across" type="ped" begin="0" end="{horizon}" personsPerHour="80">',
        '    <walk from="N2_CW" to="CW_J2"/>',
        '  </personFlow>',
        f'  <personFlow id="ped_back" type="ped" begin="0" end="{horizon}" personsPerHour="80">',
        '    <walk from="J2_CW" to="CW_N2"/>',
        '  </personFlow>',
    ]


def generate_demand(out, n_files, seed, horizon=1800, slice_s=300, base_vph=5000):
    rng = random.Random(seed)
    net_file = os.path.join(out, "net.net.xml")
    n_slices = horizon // slice_s
    for k in range(n_files):
        kind = PROFILES[k % len(PROFILES)]
        level = rng.uniform(0.6, 1.4)
        mult = _multipliers(kind, n_slices, rng)
        w_orig = {e: (3.0 if e in ARTERIAL_ENDS else 1.0) * rng.uniform(0.6, 1.4) for e in ENDS}
        w_dest = {e: (3.0 if e in ARTERIAL_ENDS else 1.0) * rng.uniform(0.6, 1.4) for e in ENDS}
        if kind == "morning_rush":
            w_orig["W"] *= 2.5; w_dest["E"] *= 2.5
        elif kind == "evening_rush":
            w_orig["E"] *= 2.5; w_dest["W"] *= 2.5
        elif kind == "side_street_surge":
            for e in ENDS:
                if e not in ARTERIAL_ENDS:
                    w_orig[e] *= 3.0
        so, sd = sum(w_orig.values()), sum(w_dest.values())
        rows = ['<routes>'] + _vtype_xml() + _pedestrian_xml(horizon)
        fid = 0
        for s in range(n_slices):
            for o in ENDS:
                for d in ENDS:
                    if o == d:
                        continue
                    vph = base_vph * level * mult[s] * (w_orig[o] / so) * (w_dest[d] / sd)
                    if vph < 1:
                        continue
                    rows.append(f'  <flow id="f{fid}" type="mixed" begin="{s * slice_s}" '
                                f'end="{(s + 1) * slice_s}" '
                                f'from="{o}_{_first(o)}" to="{_first(d)}_{d}" vehsPerHour="{vph:.1f}" '
                                f'departLane="best" departSpeed="max"/>')
                    fid += 1
        rows.append('</routes>')
        flow_file = os.path.join(out, f"flows_{k}.xml")
        with open(flow_file, "w") as f:
            f.write("\n".join(rows))
        subprocess.run([_exe("duarouter"), "-n", net_file, "-r", flow_file,
                        "-o", os.path.join(out, f"routes_{k}.rou.xml"),
                        "--ignore-errors", "true", "--no-warnings", "true"],
                       check=True, stdout=subprocess.DEVNULL)
        _ensure_pedestrians_in_routes(os.path.join(out, f"routes_{k}.rou.xml"), horizon)
        os.remove(flow_file)
        alt = os.path.join(out, f"routes_{k}.rou.alt.xml")
        if os.path.exists(alt):
            os.remove(alt)
    fleet = ", ".join(f"{k}={p:.0%}" for k, (_, p) in VEHICLE_TYPES.items())
    print(f"Created {n_files} traffic files in {out}")
    print(f"Vehicle mix: {fleet}")
    print(f"Pedestrians: mid-block zebra at {CROSSWALK_NODE}")


def _ensure_pedestrians_in_routes(path, horizon):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if 'personFlow id="ped_across"' in text or 'person id="ped_across' in text:
        return
    ped = "\n".join(_pedestrian_xml(horizon))
    text = text.replace("</routes>", ped + "\n</routes>")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _first(end_node):
    for a, b in LINKS:
        if a == end_node:
            return b
        if b == end_node:
            return a


def _remove_old_routes(out):
    for f in os.listdir(out) if os.path.isdir(out) else []:
        if f.startswith("routes_"):
            path = os.path.join(out, f)
            for attempt in range(5):
                try:
                    os.remove(path)
                    break
                except PermissionError:
                    if attempt == 4:
                        print(f"Warning: could not delete {path}. Close SUMO-GUI and retry.")
                        raise
                    time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scenario")
    ap.add_argument("--lanes", type=int, default=3)
    ap.add_argument("--n-routes", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=1800)
    ap.add_argument("--slice", type=int, default=300)
    ap.add_argument("--base-vph", type=float, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demand-only", action="store_true")
    a = ap.parse_args()
    if not a.demand_only:
        build_network(a.out, a.lanes)
        phases = compute_phases(a.out)
        print({j: len(p) for j, p in phases.items()}, "green phases per junction")
    _remove_old_routes(a.out)
    generate_demand(a.out, a.n_routes, a.seed, a.horizon, a.slice, a.base_vph)


if __name__ == "__main__":
    main()
