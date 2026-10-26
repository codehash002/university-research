"""Export SUMO net.xml geometry into a compact JSON payload for the WebGL client."""
from __future__ import annotations

import math
from collections import defaultdict

import sumolib


DEFAULT_LANE_WIDTH = 3.2


def _parse_shape(shape):
    """sumolib shape → list of [x, y]."""
    if not shape:
        return []
    return [[float(x), float(y)] for x, y in shape]


def _nav_angle(dx: float, dy: float) -> float:
    """SUMO navigational degrees: 0 = north, clockwise."""
    return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0


def _vehicle_lanes(edge):
    """Lanes that carry vehicles (not ped-only)."""
    out = []
    for ln in edge.getLanes():
        allow = ln.getPermissions() or set()
        is_ped_only = (
            "pedestrian" in allow
            and "passenger" not in allow
            and "bus" not in allow
            and "truck" not in allow
            and "taxi" not in allow
        )
        if not is_ped_only:
            out.append(ln)
    return out or list(edge.getLanes())


def _classify_turn(from_lane, to_lane) -> str:
    """Return L / S / R from approach→exit geometry (right-hand traffic)."""
    fs = from_lane.getShape()
    ts = to_lane.getShape()
    if not fs or len(fs) < 2 or not ts or len(ts) < 2:
        return "S"
    p0, p1 = fs[-2], fs[-1]
    q0, q1 = ts[0], ts[1] if len(ts) > 1 else ts[0]
    fdx, fdy = p1[0] - p0[0], p1[1] - p0[1]
    tdx, tdy = q1[0] - q0[0], q1[1] - q0[1]
    fl = math.hypot(fdx, fdy) or 1.0
    tl = math.hypot(tdx, tdy) or 1.0
    ux, uy = fdx / fl, fdy / fl
    vx, vy = tdx / tl, tdy / tl
    cross = ux * vy - uy * vx
    dot = ux * vx + uy * vy
    ang = math.degrees(math.atan2(cross, dot))
    if abs(ang) < 35.0:
        return "S"
    if ang > 0.0:
        return "L"
    return "R"


def _export_tls_heads(net) -> list:
    """Gantry-ready heads: one L/S/R signal set per approach edge (link subsets)."""
    tls = []
    seen = set()

    def add_tl(tid: str):
        if tid in seen or not net.hasNode(tid):
            return
        seen.add(tid)
        node = net.getNode(tid)
        nx, ny = node.getCoord()

        # from_lane_id → list of (link_index, turn)
        lane_turns: dict[str, list[tuple[int, str]]] = defaultdict(list)
        try:
            tl_obj = net.getTLS(tid) if hasattr(net, "getTLS") else None
            if tl_obj is None:
                for t in net.getTrafficLights():
                    if (t.getID() if hasattr(t, "getID") else str(t)) == tid:
                        tl_obj = t
                        break
            if tl_obj is not None:
                for conn in tl_obj.getConnections():
                    from_lane, to_lane, idx = conn[0], conn[1], conn[2]
                    turn = _classify_turn(from_lane, to_lane)
                    lane_turns[from_lane.getID()].append((int(idx), turn))
        except Exception:
            pass

        heads = []
        for edge in node.getIncoming():
            if edge.getFunction():
                continue
            lanes = _vehicle_lanes(edge)
            if not lanes:
                continue

            # Left-hand traffic: lane 0 = leftmost (outer curb / nearside).
            ref = lanes[0]
            shape = ref.getShape()
            if not shape or len(shape) < 2:
                continue
            p0, p1 = shape[-2], shape[-1]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            length = math.hypot(dx, dy) or 1.0
            ux, uy = dx / length, dy / length
            travel_nav = _nav_angle(dx, dy)
            rx, ry = uy, -ux  # right of travel
            lx, ly = -rx, -ry  # left of travel (nearside curb in LHT)
            half_w = 0.5 * sum(float(ln.getWidth() or DEFAULT_LANE_WIDTH) for ln in lanes)
            # Pole flush with LEFT roadside (on the curb, not out in the grass)
            pole_lat = half_w + 0.05
            base_x = p1[0] - ux * 5.0 + lx * pole_lat
            base_y = p1[1] - uy * 5.0 + ly * pole_lat

            # Aggregate link indices by turn across all lanes of this approach
            by_turn: dict[str, list[int]] = {"L": [], "S": [], "R": []}
            lane_links: dict[int, list[int]] = {}
            for lane in lanes:
                li = int(lane.getIndex())
                links_here = []
                for idx, turn in lane_turns.get(lane.getID(), []):
                    by_turn.setdefault(turn, []).append(idx)
                    links_here.append(idx)
                lane_links[li] = sorted(set(links_here))
            for t in by_turn:
                by_turn[t] = sorted(set(by_turn[t]))

            # LHT lane roles: 0=left, mid=straight, last=right — fill any missing turn
            idxs = sorted(lane_links.keys())
            if idxs:
                if not by_turn["L"]:
                    by_turn["L"] = list(lane_links.get(idxs[0], []))
                if not by_turn["R"]:
                    by_turn["R"] = list(lane_links.get(idxs[-1], []))
                if not by_turn["S"]:
                    mid_links = []
                    for li in idxs[1:-1] if len(idxs) > 2 else idxs:
                        mid_links.extend(lane_links.get(li, []))
                    by_turn["S"] = sorted(set(mid_links)) or list(
                        lane_links.get(idxs[len(idxs) // 2], [])
                    )

            # Always show L/S/R on 3-lane approaches (4-way arms); never drop a head
            # just because one turn bucket was empty after classification.
            if len(lanes) >= 3:
                active = [("L", 0), ("S", 1), ("R", 2)]
            elif len(lanes) == 2:
                active = [("L", 0), ("R", 1)]
                if by_turn.get("S") and not by_turn.get("L"):
                    active = [("S", 0), ("R", 1)]
            else:
                active = [(t, i) for i, t in enumerate(("L", "S", "R")) if by_turn.get(t)]
                if not active:
                    active = [("S", 0)]

            n_heads = len(active)
            # Span exactly this approach width (lane count × width), not into opposing
            span = max(sum(float(ln.getWidth() or DEFAULT_LANE_WIDTH) for ln in lanes), 6.0)
            for slot, (turn, _) in enumerate(active):
                # Center each head over its lane band
                t = (slot + 0.5) / max(n_heads, 1)
                along = span * t
                px = base_x + rx * along
                py = base_y + ry * along
                heads.append({
                    "edge": edge.getID(),
                    "lane": ref.getID(),
                    "lane_index": int(ref.getIndex()),
                    "turn": turn,
                    "x": float(px),
                    "y": float(py),
                    "facing": float((travel_nav + 180.0) % 360.0),
                    "travel": float(travel_nav),
                    "links": by_turn.get(turn, []),
                })

        tls.append({
            "id": tid,
            "x": float(nx),
            "y": float(ny),
            "heads": heads,
        })

    try:
        for tl in net.getTrafficLights():
            tid = tl.getID() if hasattr(tl, "getID") else str(tl)
            add_tl(tid)
    except Exception:
        pass

    for node in net.getNodes():
        if node.getType() == "traffic_light":
            add_tl(node.getID())

    return tls


def export_network(net_file: str) -> dict:
    net = sumolib.net.readNet(net_file, withInternal=True)

    lanes = []
    for edge in net.getEdges():
        function = edge.getFunction() or ""
        # Skip walking areas — they bleed into the carriageway in the 3D view.
        # Crossings kept as zebra; sidewalks are generated client-side beside roads.
        if function in ("walkingarea",):
            continue
        for lane in edge.getLanes():
            shape = _parse_shape(lane.getShape())
            if len(shape) < 2:
                continue
            width = float(lane.getWidth() or DEFAULT_LANE_WIDTH)
            allow = lane.getPermissions() or set()
            is_ped_only = (
                "pedestrian" in allow
                and "passenger" not in allow
                and "bus" not in allow
                and "truck" not in allow
                and "taxi" not in allow
            )
            # Ped-only edge lanes: skip mesh (client builds curb sidewalks).
            # Keep crossings for zebra.
            if is_ped_only and function != "crossing":
                continue
            lanes.append({
                "id": lane.getID(),
                "edge": edge.getID(),
                "index": lane.getIndex(),
                "width": width,
                "shape": shape,
                "kind": "crossing" if function == "crossing" else "road",
                "internal": edge.isSpecial(),
            })

    junctions = []
    for node in net.getNodes():
        shape = _parse_shape(node.getShape())
        junctions.append({
            "id": node.getID(),
            "x": float(node.getCoord()[0]),
            "y": float(node.getCoord()[1]),
            "type": node.getType(),
            "shape": shape,
        })

    boundary = net.getBoundary()
    return {
        "bounds": {
            "xmin": float(boundary[0]),
            "ymin": float(boundary[1]),
            "xmax": float(boundary[2]),
            "ymax": float(boundary[3]),
        },
        "lanes": lanes,
        "junctions": junctions,
        "tls": _export_tls_heads(net),
    }
