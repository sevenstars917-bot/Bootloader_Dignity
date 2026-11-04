#!/usr/bin/env python3
"""
AG Scientific Co-Processor (ag_calc)
Deterministic Math & Logic Co-Processor for AI Agents & LLMs.
Zero External Dependencies (Python Standard Library Only).
"""

import sys
import os
import ast
import math
import json
import re
import argparse
from datetime import datetime, timezone, timedelta

JST = timezone(timedelta(hours=9))

# ==========================================
# 0. State & Memory Management (Ans & Variables)
# ==========================================
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ag_calc_state.json")

def load_memory():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"ans": 0, "vars": {}}

def save_memory(mem):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(mem, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ==========================================
# 1. Math Module (Safe AST Evaluator)
# ==========================================
def nth_root(x, n):
    """n-th root: root(125, 3) -> 5.0"""
    return x ** (1.0 / n)

def parallel_sum(*resistors):
    """Parallel resistance / Reciprocal sum: 1 / (1/r1 + 1/r2 + ...)"""
    return 1.0 / sum(1.0 / r for r in resistors)

SAFE_MATH_NAMES = {
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'asin': math.asin,
    'acos': math.acos,
    'atan': math.atan,
    'atan2': math.atan2,
    'sqrt': math.sqrt,
    'cbrt': getattr(math, 'cbrt', lambda x: x ** (1.0 / 3.0)),
    'root': nth_root,
    'nroot': nth_root,
    'pow': math.pow,
    'exp': math.exp,
    'log': math.log,
    'log2': math.log2,
    'log10': math.log10,
    'deg': math.radians,  # deg(45) -> radians
    'rad': math.degrees,  # rad(pi/4) -> degrees
    'deg2rad': math.radians,
    'rad2deg': math.degrees,
    'pi': math.pi,
    'π': math.pi,
    'e': math.e,
    'abs': abs,
    'round': round,
    'floor': math.floor,
    'ceil': math.ceil,
    'parallel': parallel_sum,
    'para': parallel_sum,
}

def safe_eval_expr(node, env):
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = safe_eval_expr(node.left, env)
        right = safe_eval_expr(node.right, env)
        if isinstance(node.op, ast.Add): return left + right
        if isinstance(node.op, ast.Sub): return left - right
        if isinstance(node.op, ast.Mult): return left * right
        if isinstance(node.op, ast.Div): return left / right
        if isinstance(node.op, ast.FloorDiv): return left // right
        if isinstance(node.op, ast.Mod): return left % right
        if isinstance(node.op, ast.Pow): return left ** right
    elif isinstance(node, ast.UnaryOp):
        operand = safe_eval_expr(node.operand, env)
        if isinstance(node.op, ast.UAdd): return +operand
        if isinstance(node.op, ast.USub): return -operand
    elif isinstance(node, ast.Call):
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if func_name and func_name in env:
            args = [safe_eval_expr(arg, env) for arg in node.args]
            return env[func_name](*args)
        raise ValueError(f"Unsupported function call: {func_name}")
    elif isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise ValueError(f"Undefined variable: {node.id}")
    elif isinstance(node, ast.List):
        return [safe_eval_expr(elt, env) for elt in node.elts]
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")

def normalize_math_string(s):
    """Normalize input string to standard math format (SI prefixes, full-width symbols, etc.)"""
    # 0. Full-width to half-width symbols
    s = s.replace('×', '*').replace('÷', '/').replace('＋', '+').replace('－', '-')
    s = s.replace('（', '(').replace('）', ')')
    
    # 1. Caret exponent: 10^3 -> 10**3
    s = re.sub(r'\^', '**', s)
    
    # 2. Engineering SI Prefixes (e.g., 10k, 2.2u, 100M, 4.7p)
    # T: 1e12, G: 1e9, M: 1e6, k: 1e3, m: 1e-3, u/μ: 1e-6, n: 1e-9, p: 1e-12
    # Ensure it doesn't match function names or hex (e.g. mod, sin)
    prefix_map = {
        'T': 'e12',
        'G': 'e9',
        'M': 'e6',
        'k': 'e3',
        'm': 'e-3',
        'u': 'e-6',
        'μ': 'e-6',
        'n': 'e-9',
        'p': 'e-12',
    }
    # Pattern: number followed immediately by a prefix character, not followed by word character
    for p, exp in prefix_map.items():
        pattern = rf'(\b\d+(?:\.\d+)?){p}(?![a-zA-Z0-9_])'
        s = re.sub(pattern, rf'\1{exp}', s)

    return s

def eval_math(expr_str):
    expr_str = expr_str.strip()
    mem = load_memory()
    
    # Build environment: math builtins + ans + user vars
    env = dict(SAFE_MATH_NAMES)
    env['ans'] = mem.get("ans", 0)
    env.update(mem.get("vars", {}))

    # Check for assignment: var = expr
    assign_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$', expr_str)
    if assign_match:
        var_name = assign_match.group(1)
        sub_expr = normalize_math_string(assign_match.group(2))
        parsed = ast.parse(sub_expr, mode='eval')
        val = safe_eval_expr(parsed.body, env)
        
        mem["vars"][var_name] = val
        mem["ans"] = val
        save_memory(mem)
        return f"{var_name} = {val}"
    
    # Pure evaluation
    norm_str = normalize_math_string(expr_str)
    parsed = ast.parse(norm_str, mode='eval')
    val = safe_eval_expr(parsed.body, env)
    
    mem["ans"] = val
    save_memory(mem)
    return val

def calc_stats(numbers):
    n = len(numbers)
    if n == 0:
        return {}
    s = sum(numbers)
    mean = s / n
    variance = sum((x - mean) ** 2 for x in numbers) / (n - 1) if n > 1 else 0.0
    stdev = math.sqrt(variance)
    sorted_nums = sorted(numbers)
    median = (sorted_nums[n // 2] if n % 2 == 1 
              else (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2.0)
    return {
        "count": n,
        "sum": s,
        "mean": mean,
        "stdev": stdev,
        "median": median,
        "min": sorted_nums[0],
        "max": sorted_nums[-1]
    }

# ==========================================
# 2. Vision & PTZ Geometry Module
# ==========================================
def calc_fov_offset(fov_deg, resolution_px, target_pos_px):
    """
    Calculate camera pan/tilt angle offset from center based on linear FoV approximation.
    offset_angle = ((pos - center) / res) * fov
    """
    center_px = resolution_px / 2.0
    diff_px = target_pos_px - center_px
    angle_offset = (diff_px / resolution_px) * fov_deg
    return {
        "fov_deg": fov_deg,
        "resolution_px": resolution_px,
        "target_pos_px": target_pos_px,
        "diff_px": diff_px,
        "angle_offset_deg": round(angle_offset, 3)
    }

def convert_crop_coords(width, height, norm_coords=None, pixel_coords=None):
    """Convert between normalized (0.0-1.0) and absolute pixel bounding boxes."""
    if norm_coords:
        parts = [float(x.strip()) for x in norm_coords.split(',')]
        ymin, xmin, ymax, xmax = parts
        x1 = int(round(xmin * width))
        y1 = int(round(ymin * height))
        x2 = int(round(xmax * width))
        y2 = int(round(ymax * height))
        return {
            "norm": [ymin, xmin, ymax, xmax],
            "pixel_box": [x1, y1, x2, y2],
            "w": x2 - x1,
            "h": y2 - y1,
            "area_ratio": round(((x2 - x1) * (y2 - y1)) / (width * height), 4)
        }
    elif pixel_coords:
        parts = [int(x.strip()) for x in pixel_coords.split(',')]
        x1, y1, x2, y2 = parts
        ymin = round(y1 / height, 4)
        xmin = round(x1 / width, 4)
        ymax = round(y2 / height, 4)
        xmax = round(x2 / width, 4)
        return {
            "pixel_box": [x1, y1, x2, y2],
            "norm": [ymin, xmin, ymax, xmax],
            "w": x2 - x1,
            "h": y2 - y1,
            "area_ratio": round(((x2 - x1) * (y2 - y1)) / (width * height), 4)
        }
    raise ValueError("Either norm_coords or pixel_coords must be provided")

# ==========================================
# 3. Text & Social Metrics Module
# ==========================================
def count_x_points(text):
    """
    X (Twitter) Weighted Character Counter.
    CJK / Full-width: 2 points (Max 140 chars = 280 points)
    ASCII / Half-width: 1 point (Max 280 chars = 280 points)
    URLs: 23 points fixed
    """
    url_regex = re.compile(r'https?://[^\s]+')
    urls = url_regex.findall(text)
    text_no_url = url_regex.sub('', text)
    
    score = len(urls) * 23
    for char in text_no_url:
        cp = ord(char)
        # Halfwidth ASCII & Katakana
        if cp <= 0x007F or (0xFF61 <= cp <= 0xFF9F):
            score += 1
        else:
            score += 2

    return {
        "points": score,
        "max_points": 280,
        "remaining_points": 280 - score,
        "fullwidth_equivalent": score / 2.0,
        "remaining_fullwidth": (280 - score) / 2.0,
        "is_valid": score <= 280,
        "urls_detected": len(urls)
    }

# ==========================================
# 4. Time & Calendar Module
# ==========================================
def parse_date_flexible(s):
    s = s.strip()
    if s.lower() == 'now':
        return datetime.now(JST)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=JST)
        except ValueError:
            pass
    raise ValueError(f"Cannot parse date format: {s}")

def calc_time_diff(start_str, end_str="now"):
    d_start = parse_date_flexible(start_str)
    d_end = parse_date_flexible(end_str)
    diff = d_end - d_start
    total_seconds = int(diff.total_seconds())
    days = diff.days
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    return {
        "start": d_start.strftime("%Y-%m-%d %H:%M:%S JST"),
        "end": d_end.strftime("%Y-%m-%d %H:%M:%S JST"),
        "total_days": round(diff.total_seconds() / 86400, 2),
        "days": days,
        "hours": hours,
        "minutes": minutes,
        "total_seconds": total_seconds
    }

# ==========================================
# Cheatsheet & Documentation
# ==========================================
CHEAT_SHEET = """======================================================================
  AG Scientific Co-Processor (ag_calc) Cheat Sheet for AI Agents
======================================================================
[Math & Engineering Units]
  python ag_calc.py math "560k * 34.5u"             # SI Units (k, M, G, m, u/μ, n, p)
  python ag_calc.py math "1 / (2 * π * 10k * 0.1u)"   # Greek π, pi, e
  python ag_calc.py math "parallel(100, 200, 300)"    # Parallel resistance
  python ag_calc.py math "root(125, 3)"               # nth-root (or cbrt)
  python ag_calc.py math "10^(-1.5)"                  # Exponents (^ or **)
  python ag_calc.py math --stat "[10, 20, 30, 40]"    # Mean, stdev, median

[Memory & Chaining]
  python ag_calc.py math "ans / 2"                    # Re-use previous result
  python ag_calc.py math "r = 10k"                    # Store variable
  python ag_calc.py mem                               # View stored variables
  python ag_calc.py mem --clear                       # Reset memory

[Active Vision & PTZ Geometry]
  python ag_calc.py vision fov --fov 60 --res 1920 --pos 1420
  python ag_calc.py vision crop --res 1920x1080 --norm 0.2,0.3,0.8,0.7

[Text & Metrics]
  python ag_calc.py text x "投稿ドラフト..."           # 280pt strict check
  python ag_calc.py text hash "文字列"                # SHA256 & MD5

[Time & Calendar]
  python ag_calc.py time diff "2026-07-10" "now"      # Elapsed days & hours
  python ag_calc.py time now                          # JST ISO & Epoch
======================================================================"""

# ==========================================
# CLI Entry Point
# ==========================================
def main():
    argv = sys.argv[1:]
    # Poka-yoke: Prevent negative numbers (e.g. -12*3) from being parsed as flags
    if len(argv) >= 2 and argv[0] == "math" and argv[1].startswith("-") and argv[1] not in ("--stat", "-h", "--help"):
        if argv[1] != "--":
            argv.insert(1, "--")

    # Show cheat sheet when called without arguments or with help/cheat
    if not argv or argv[0] in ("-h", "--help", "help", "cheat"):
        print(CHEAT_SHEET)
        sys.exit(0)

    parser = argparse.ArgumentParser(description="AG Scientific Co-Processor (ag_calc)", epilog=CHEAT_SHEET, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="subcommand")

    # Math
    p_math = subparsers.add_parser("math", help="Mathematical evaluation and statistics")
    p_math.add_argument("expr", nargs="?", help="Math expression string e.g. 'sin(deg(45))*1920'")
    p_math.add_argument("--stat", help="JSON or comma-separated numbers for statistical analysis")

    # Memory
    p_mem = subparsers.add_parser("mem", help="Memory & Variable inspection")
    p_mem.add_argument("--clear", action="store_true", help="Clear all stored memory and variables")

    # Vision
    p_vision = subparsers.add_parser("vision", help="Active Vision & PTZ Geometry")
    p_vision_sub = p_vision.add_subparsers(dest="vision_cmd")
    
    p_fov = p_vision_sub.add_parser("fov", help="Calculate pan/tilt offset angle")
    p_fov.add_argument("--fov", type=float, required=True, help="Camera Field of View in degrees")
    p_fov.add_argument("--res", type=float, required=True, help="Resolution in pixels along the axis")
    p_fov.add_argument("--pos", type=float, required=True, help="Target pixel coordinate")

    p_crop = p_vision_sub.add_parser("crop", help="Convert normalized coordinates to pixel bbox")
    p_crop.add_argument("--res", required=True, help="Source image resolution e.g. 1920x1080")
    p_crop.add_argument("--norm", help="Normalized bbox ymin,xmin,ymax,xmax (0.0-1.0)")
    p_crop.add_argument("--pixel", help="Pixel bbox x1,y1,x2,y2")

    # Text
    p_text = subparsers.add_parser("text", help="Text metrics and X/Twitter points")
    p_text.add_argument("cmd", choices=["x", "len", "hash"], help="Command")
    p_text.add_argument("content", nargs="?", help="Text string (or pass via stdin)")

    # Time
    p_time = subparsers.add_parser("time", help="Date & Time calculations")
    p_time.add_argument("cmd", choices=["diff", "now", "epoch", "from-epoch"], help="Time command")
    p_time.add_argument("args", nargs="*", help="Arguments for time command")

    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        sys.exit(0)

    # 1. Math
    if args.subcommand == "math":
        if args.stat:
            try:
                nums = json.loads(args.stat) if args.stat.startswith('[') else [float(x) for x in args.stat.split(',')]
                print(json.dumps(calc_stats(nums), ensure_ascii=False, indent=2))
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
        elif args.expr:
            try:
                res = eval_math(args.expr)
                if isinstance(res, float):
                    print(int(res) if res.is_integer() else round(res, 6))
                else:
                    print(res)
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            p_math.print_help()

    # 1.5 Memory
    elif args.subcommand == "mem":
        if args.clear:
            save_memory({"ans": 0, "vars": {}})
            print("Memory cleared.")
        else:
            mem = load_memory()
            print(json.dumps(mem, ensure_ascii=False, indent=2))

    # 2. Vision
    elif args.subcommand == "vision":
        if args.vision_cmd == "fov":
            res = calc_fov_offset(args.fov, args.res, args.pos)
            print(json.dumps(res, ensure_ascii=False, indent=2))
        elif args.vision_cmd == "crop":
            w, h = [int(v) for v in args.res.lower().split('x')]
            res = convert_crop_coords(w, h, norm_coords=args.norm, pixel_coords=args.pixel)
            print(json.dumps(res, ensure_ascii=False, indent=2))
        else:
            p_vision.print_help()

    # 3. Text
    elif args.subcommand == "text":
        content = args.content
        if not content and not sys.stdin.isatty():
            content = sys.stdin.read()
        if content is None:
            print("Error: No text provided", file=sys.stderr)
            sys.exit(1)

        if args.cmd == "x":
            res = count_x_points(content)
            print(json.dumps(res, ensure_ascii=False, indent=2))
        elif args.cmd == "len":
            print(json.dumps({
                "chars": len(content),
                "bytes": len(content.encode('utf-8')),
                "lines": len(content.splitlines())
            }, indent=2))
        elif args.cmd == "hash":
            import hashlib
            b = content.encode('utf-8')
            print(json.dumps({
                "sha256": hashlib.sha256(b).hexdigest(),
                "md5": hashlib.md5(b).hexdigest()
            }, indent=2))

    # 4. Time
    elif args.subcommand == "time":
        if args.cmd == "now":
            dt = datetime.now(JST)
            print(json.dumps({
                "iso": dt.isoformat(),
                "readable": dt.strftime('%Y-%m-%d %H:%M:%S JST'),
                "epoch": int(dt.timestamp())
            }, indent=2))
        elif args.cmd == "diff":
            if not args.args:
                print("Error: Provide start date [and optional end date]", file=sys.stderr)
                sys.exit(1)
            start = args.args[0]
            end = args.args[1] if len(args.args) > 1 else 'now'
            res = calc_time_diff(start, end)
            print(json.dumps(res, ensure_ascii=False, indent=2))
        elif args.cmd == "epoch":
            print(int(datetime.now(JST).timestamp()))
        elif args.cmd == "from-epoch":
            if not args.args:
                print("Error: Provide epoch timestamp", file=sys.stderr)
                sys.exit(1)
            ts = float(args.args[0])
            dt = datetime.fromtimestamp(ts, JST)
            print(json.dumps({
                "iso": dt.isoformat(),
                "readable": dt.strftime('%Y-%m-%d %H:%M:%S JST')
            }, indent=2))

if __name__ == "__main__":
    main()

