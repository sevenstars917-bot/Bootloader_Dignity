#!/usr/bin/env python3
"""
AG Scientific Co-Processor (ag_calc)
LLM（大脳皮質）のための外部頭頂葉（決定論的数理コプロセッサ CLI）。
Zero External Dependencies (標準ライブラリのみ)。
"""

import sys
import os
import ast
import math
import json
import re
import unicodedata
import argparse
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone, timedelta

JST = timezone(timedelta(hours=9))

# ==========================================
# 0. State & Memory Management (Ans & Variables)
# ==========================================
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ag_calc_state.json")
ERROR_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ag_calc_errors.log")


def sanitize_cmd_args(args):
    """ログに投稿ドラフト本文等の長文テキストを生書きしないようサニタイズ（サブコマンド名は残す）"""
    clean = []
    for i, a in enumerate(args):
        if len(args) >= 2 and args[0] == "text" and i >= 2 and not a.startswith("-"):
            clean.append(f"[text: {len(a)} chars]")
        elif len(a) > 60 and not a.startswith("-"):
            clean.append(f"[arg: {len(a)} chars]")
        else:
            clean.append(a)
    return clean


def log_error(cmd_args, error_msg):
    try:
        clean_args = sanitize_cmd_args(cmd_args)
        with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now(JST).isoformat()}] CMD: {' '.join(clean_args)} | ERROR: {error_msg}\n")
    except Exception:
        pass


def load_memory():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return {"ans": data.get("ans", 0), "vars": data.get("vars", {})}
        except Exception as e:
            print(f"Warning: Failed to read state file ({e}). Resetting memory.", file=sys.stderr)
    return {"ans": 0, "vars": {}}


def save_memory(mem):
    tmp_file = f"{STATE_FILE}.tmp.{os.getpid()}"
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(mem, f, ensure_ascii=False, indent=2)
        os.replace(tmp_file, STATE_FILE)
    except Exception as e:
        print(f"Warning: Failed to save state file: {e}", file=sys.stderr)
        if os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except OSError:
                pass


# ==========================================
# 1. Math Module (Safe AST Evaluator)
# ==========================================
def nth_root(x, n):
    """n乗根: root(125, 3) -> 5.0, root(-8, 3) -> -2.0 (負数の奇数乗根は実数解を返し、偶数乗根はエラー)"""
    if n == 0:
        raise ValueError("Zero-th root is undefined")
    if x < 0:
        if isinstance(n, (int, float)) and float(n).is_integer() and int(n) % 2 != 0:
            return -((-x) ** (1.0 / n))
        raise ValueError(f"Real {n}-th root of negative number ({x}) is undefined")
    return x ** (1.0 / n)


def cbrt_real(x):
    """実数立方根: cbrt(-8) -> -2.0"""
    if hasattr(math, "cbrt"):
        return math.cbrt(x)
    return nth_root(x, 3)


def parallel_sum(*resistors):
    """並列合成抵抗 / 逆数和の逆数: 1 / (1/r1 + 1/r2 + ...)"""
    if not resistors:
        raise ValueError("parallel() requires at least one argument")
    for r in resistors:
        if r == 0:
            return 0.0
    return 1.0 / sum(1.0 / r for r in resistors)


def safe_factorial(n):
    if isinstance(n, float) and n.is_integer():
        n = int(n)
    if not isinstance(n, int) or n < 0 or n > 1000:
        raise ValueError(f"factorial(n) requires integer 0 <= n <= 1000 (got {n})")
    return math.factorial(n)


def safe_comb(n, k):
    if isinstance(n, float) and n.is_integer(): n = int(n)
    if isinstance(k, float) and k.is_integer(): k = int(k)
    if not (isinstance(n, int) and isinstance(k, int)):
        raise ValueError("comb(n, k) requires integer arguments")
    if n < 0 or n > 1000 or k < 0 or k > n:
        raise ValueError(f"comb(n, k) requires integers 0 <= k <= n <= 1000 (got n={n}, k={k})")
    if hasattr(math, 'comb'):
        return math.comb(n, k)
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


def safe_perm(n, k=None):
    if isinstance(n, float) and n.is_integer(): n = int(n)
    if not isinstance(n, int) or n < 0 or n > 1000:
        raise ValueError(f"perm(n, k) requires integer 0 <= n <= 1000 (got {n})")
    if k is not None:
        if isinstance(k, float) and k.is_integer(): k = int(k)
        if not isinstance(k, int) or k < 0 or k > n:
            raise ValueError(f"perm(n, k) requires integers 0 <= k <= n <= 1000 (got n={n}, k={k})")
        if hasattr(math, 'perm'):
            return math.perm(n, k)
        return math.factorial(n) // math.factorial(n - k)
    return math.factorial(n)


def round_half_up(val, ndigits=0):
    """
    算術四捨五入（ROUND_HALF_UP）：
    Python標準の銀行丸め（偶数丸め：round(2.5)=2, round(0.5)=0）を排し、
    人間やLLMが期待する決定論的な小学校の四捨五入（round(2.5)=3, round(0.5)=1）を保証。
    負の桁（round(1234, -2) -> 1200）にも完全対応。
    """
    if isinstance(ndigits, float) and ndigits.is_integer():
        ndigits = int(ndigits)
    if not isinstance(ndigits, int):
        raise ValueError("round(val, ndigits) requires integer ndigits")
    d = Decimal(str(val))
    q = Decimal(1).scaleb(-ndigits)
    res = d.quantize(q, rounding=ROUND_HALF_UP)
    return int(res) if ndigits <= 0 else float(res)


SAFE_MATH_NAMES = {
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'asin': math.asin,
    'acos': math.acos,
    'atan': math.atan,
    'atan2': math.atan2,
    'sqrt': math.sqrt,
    'cbrt': cbrt_real,
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
    'hypot': math.hypot,
    'gcd': math.gcd,
    'lcm': getattr(math, 'lcm', lambda a, b: abs(a * b) // math.gcd(a, b) if a and b else 0),
    'factorial': safe_factorial,
    'comb': safe_comb,
    'perm': safe_perm,
    'min': min,
    'max': max,
    'clamp': lambda x, low, high: max(low, min(x, high)),
    'pi': math.pi,
    'π': math.pi,
    'e': math.e,
    'abs': abs,
    'round': round_half_up,
    'floor': math.floor,
    'ceil': math.ceil,
    'parallel': parallel_sum,
    'para': parallel_sum,
}

RESERVED_NAMES = set(SAFE_MATH_NAMES.keys()) | {'ans', 'Ans'}


def safe_eval_expr(node, env):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
        return node.value
    elif isinstance(node, ast.BinOp):
        left = safe_eval_expr(node.left, env)
        right = safe_eval_expr(node.right, env)
        if isinstance(node.op, ast.Add):
            res = left + right
        elif isinstance(node.op, ast.Sub):
            res = left - right
        elif isinstance(node.op, ast.Mult):
            res = left * right
        elif isinstance(node.op, ast.Div):
            res = left / right
        elif isinstance(node.op, ast.FloorDiv):
            res = left // right
        elif isinstance(node.op, ast.Mod):
            res = left % right
        elif isinstance(node.op, ast.Pow):
            if abs(right) > 10000:
                raise ValueError(f"Exponent too large: {right} (max 10000)")
            if isinstance(left, (int, float)) and abs(left) > 1 and right > 0:
                if right * math.log10(abs(left)) > 300:
                    raise ValueError("Power result exceeds safe numeric range (max ~1e300)")
            res = left ** right
        else:
            raise ValueError(f"Unsupported binary operator: {ast.dump(node.op)}")
        if isinstance(res, complex):
            raise ValueError("Complex numbers are not supported (result is not real)")
        return res
    elif isinstance(node, ast.UnaryOp):
        operand = safe_eval_expr(node.operand, env)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
    elif isinstance(node, ast.Call):
        if node.keywords:
            raise ValueError("Keyword arguments are not supported in function calls (use positional arguments)")
        if isinstance(node.func, ast.Name) and node.func.id in env:
            func = env[node.func.id]
            if not callable(func):
                raise ValueError(f"'{node.func.id}' is not callable")
            args = [safe_eval_expr(arg, env) for arg in node.args]
            res = func(*args)
            if isinstance(res, complex):
                raise ValueError("Complex numbers are not supported (result is not real)")
            return res
    elif isinstance(node, ast.Name):
        if node.id in env:
            val = env[node.id]
            if callable(val):
                raise ValueError(f"'{node.id}' is a function, not a variable")
            return val
    raise ValueError(f"Unsupported math syntax or unknown variable: {ast.dump(node)}")


SI_PREFIXES = {
    'T': '*1e12',
    'G': '*1e9',
    'M': '*1e6',
    'k': '*1e3',
    'K': '*1e3',
    'm': '*1e-3',
    'u': '*1e-6',
    'μ': '*1e-6',  # U+03BC Greek Small Letter Mu
    'µ': '*1e-6',  # U+00B5 Micro Sign
    'n': '*1e-9',
    'p': '*1e-12',
}

# 上付き・下付き数字の検出（NFKC前の誤答防止）
SUPERSCRIPT_SUBSCRIPT_REGEX = re.compile(r'[⁰¹²³⁴-⁹₀-₉]')

# 不正なゼロ連続トークン（000）の検出（引数誤認・サイレント誤答防止）
ZERO_TOKEN_REGEX = re.compile(r'(?<![0-9.])0{2,}(?![0-9.])')

# SI接頭辞（直後に工学・時間単位サフィックス F, H, W, V, A, Hz, Ω, ohm, s が付いていても展開）
SI_PREFIX_UNIT_REGEX = re.compile(
    r'(?<![a-zA-Z0-9_.])(\d+(?:\.\d+)?|\.\d+)\s*([TGMkKmunpμµ])(?:(F|H|W|V|A|Hz|hz|[\u03A9\u2126]|ohm|s))?(?![a-zA-Z0-9_])'
)

# 接頭辞なしの工学・時間単位サフィックス（例: 3.3V, 100Ω, 500W, 10A, 50Hz, 2s）を数値化
BARE_UNIT_REGEX = re.compile(
    r'(?<![a-zA-Z0-9_.])(\d+(?:\.\d+)?|\.\d+)\s*(F|H|W|V|A|Hz|hz|[\u03A9\u2126]|ohm|s)(?![a-zA-Z0-9_])'
)


def normalize_math_expr(expr_str):
    """数式のNFKC正規化（全角英数・µ→μ）・記号置換・変数代入抽出・SI/単位展開を行い (target_var, normalized_expr, units_stripped) を返す"""
    # 1. NFKC前のサイレント誤答防止（5² が 52 になる退行を遮断）
    if SUPERSCRIPT_SUBSCRIPT_REGEX.search(expr_str):
        raise ValueError("Superscript/subscript digits (e.g. 5² or 10³) are not supported. Use '^' or '**' (e.g. 5^2, 10^3)")

    # 2. 不正なゼロ連続トークン（000）の誤爆遮断（引数区切り誤認で 0 になるバグを防止）
    if ZERO_TOKEN_REGEX.search(expr_str):
        raise ValueError("Bare zero tokens (e.g. '000') are not allowed. Check for accidental thousand separators.")

    expr_str = unicodedata.normalize("NFKC", expr_str).strip()
    expr_str = expr_str.replace('×', '*').replace('÷', '/').replace('^', '**').replace('−', '-')

    # 代入構文のチェック: 例 "r1 = 10k" (比較演算子 ==, !=, <=, >= を除外)
    assign_match = re.match(r'^([a-zA-Z_\u0370-\u03FF][a-zA-Z0-9_\u0370-\u03FF]*)\s*=(?!=)\s*(.+)$', expr_str)
    target_var = None
    if assign_match:
        target_var = assign_match.group(1)
        if target_var in RESERVED_NAMES:
            raise ValueError(f"Cannot overwrite reserved constant or function: '{target_var}'")
        expr_str = assign_match.group(2)

    units_stripped = []

    def replace_si(match):
        num = match.group(1)
        prefix = match.group(2)
        unit = match.group(3)
        if unit:
            units_stripped.append(unit)
        return f"({num}{SI_PREFIXES[prefix]})"

    def replace_bare(match):
        num = match.group(1)
        unit = match.group(2)
        units_stripped.append(unit)
        return f"({num})"

    # SI接頭辞（単位付き含む）の展開
    normalized = SI_PREFIX_UNIT_REGEX.sub(replace_si, expr_str)
    # 接頭辞なしの工学単位サフィックス（3.3V等）の数値化
    normalized = BARE_UNIT_REGEX.sub(replace_bare, normalized)

    return target_var, normalized, units_stripped


def format_math_result(res):
    """微小値(nF/pF等)を0.0に潰さず、かつ整数は整数として表示する有効桁フォーマット"""
    if isinstance(res, bool):
        return str(res)
    if isinstance(res, int):
        return str(res)
    if isinstance(res, float):
        if math.isnan(res) or math.isinf(res):
            return str(res)
        if res.is_integer() and abs(res) < 1e16:
            return str(int(res))
        return f"{res:.10g}"
    return str(res)


def sig_round(val, sig=10):
    """JSON出力用に有効桁10桁で丸め、微小値の消失と浮動小数点ゴミの両方を防ぐ"""
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        if math.isnan(val) or math.isinf(val):
            return val
        fval = float(f"{val:.{sig}g}")
        if fval.is_integer() and abs(fval) < 1e16:
            return int(fval)
        return fval
    return val


def eval_math(expr_str, explain=False):
    mem = load_memory()
    prev_ans = mem.get('ans', 0)
    env = dict(SAFE_MATH_NAMES)
    env['ans'] = prev_ans
    env['Ans'] = prev_ans
    # 保存済み変数が予約定数・関数を上書きしないようガード
    for k, v in mem.get('vars', {}).items():
        if k not in RESERVED_NAMES:
            env[k] = v

    target_var, normalized, units_stripped = normalize_math_expr(expr_str)

    tree = ast.parse(normalized, mode='eval')
    used_ans = any(isinstance(n, ast.Name) and n.id in ('ans', 'Ans') for n in ast.walk(tree))
    res = safe_eval_expr(tree.body, env)

    # メモリー更新
    mem['ans'] = res
    if target_var:
        if 'vars' not in mem or not isinstance(mem['vars'], dict):
            mem['vars'] = {}
        mem['vars'][target_var] = res
    save_memory(mem)

    if explain:
        return {
            "input": expr_str,
            "normalized": normalized,
            "target_var": target_var,
            "used_ans": used_ans,
            "ans_value": sig_round(prev_ans, 12) if used_ans else None,
            "units_stripped": units_stripped if units_stripped else None,
            "result": sig_round(res, 12),
            "formatted": format_math_result(res),
        }
    return res


def calc_stats(numbers):
    n = len(numbers)
    if n == 0:
        return {"count": 0}
    mean = sum(numbers) / n
    variance = sum((x - mean) ** 2 for x in numbers) / (n - 1 if n > 1 else 1)
    stdev = math.sqrt(variance)
    sorted_nums = sorted(numbers)
    median = (sorted_nums[n // 2] if n % 2 != 0 else (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2)
    return {
        "count": n,
        "sum": sig_round(sum(numbers), 10),
        "mean": sig_round(mean, 10),
        "stdev": sig_round(stdev, 10),
        "median": sig_round(median, 10),
        "min": sig_round(min(numbers), 10),
        "max": sig_round(max(numbers), 10),
    }


# ==========================================
# 2. Vision Module (FoV & Crop Geometry)
# ==========================================
def calc_fov_offset(fov_deg, resolution_px, target_px):
    """
    中心からのズレ(target_px - resolution_px/2)から、
    カメラを回転させるべき角度（offset_deg）をピンホールカメラ幾何で厳密計算。
    """
    if fov_deg <= 0 or fov_deg >= 180:
        raise ValueError(f"fov must be between 0 and 180 degrees (got {fov_deg})")
    if resolution_px <= 0:
        raise ValueError(f"res must be positive (got {resolution_px})")

    center_px = resolution_px / 2.0
    focal_px = center_px / math.tan(math.radians(fov_deg / 2.0))
    offset_px = target_px - center_px
    if offset_px == 0:
        offset_deg = 0.0
        direction = "center"
    else:
        offset_deg = round(math.degrees(math.atan2(offset_px, focal_px)), 3)
        direction = "right/down" if offset_px > 0 else "left/up"

    return {
        "target_px": target_px,
        "offset_px": offset_px,
        "offset_deg": offset_deg,
        "direction": direction,
    }


def convert_crop_coords(width, height, norm_coords=None, pixel_coords=None, scale=1.0):
    """
    正規化座標 [ymin, xmin, ymax, xmax] (デフォルト 0.0-1.0、--scale 1000 で Gemini 0-1000 対応) と
    ピクセル座標 [x1, y1, x2, y2] の双方向変換。
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid resolution: {width}x{height}")
    if scale <= 0:
        raise ValueError(f"scale must be positive (got {scale})")

    if norm_coords:
        parts = [float(v) for v in norm_coords.split(',')]
        if len(parts) != 4:
            raise ValueError("--norm requires 4 comma-separated values: ymin,xmin,ymax,xmax")
        ymin, xmin, ymax, xmax = parts
        if any(v < 0 or v > scale for v in (ymin, xmin, ymax, xmax)):
            hint = " (Hint: pass --scale 1000 for Gemini 0-1000 bbox)" if scale == 1.0 and any(v > 1.0 for v in parts) else ""
            raise ValueError(f"Normalized coordinates {parts} out of range [0, {scale}]{hint}")
        x1 = int(math.floor(((xmin / scale) * width) + 0.5))
        y1 = int(math.floor(((ymin / scale) * height) + 0.5))
        x2 = int(math.floor(((xmax / scale) * width) + 0.5))
        y2 = int(math.floor(((ymax / scale) * height) + 0.5))
        return {
            "scale": scale,
            "norm": [ymin, xmin, ymax, xmax],
            "pixel_bbox": [x1, y1, x2, y2],
            "crop_wh": [x2 - x1, y2 - y1],
            "pillow_crop": [x1, y1, x2, y2],
        }
    elif pixel_coords:
        parts = [int(v) for v in pixel_coords.split(',')]
        if len(parts) != 4:
            raise ValueError("--pixel requires 4 comma-separated integers: x1,y1,x2,y2")
        x1, y1, x2, y2 = parts
        ymin = sig_round((y1 / height) * scale, 6)
        xmin = sig_round((x1 / width) * scale, 6)
        ymax = sig_round((y2 / height) * scale, 6)
        xmax = sig_round((x2 / width) * scale, 6)
        return {
            "scale": scale,
            "pixel_bbox": [x1, y1, x2, y2],
            "norm": [ymin, xmin, ymax, xmax],
            "crop_wh": [x2 - x1, y2 - y1],
        }
    raise ValueError("Provide either --norm or --pixel")


# ==========================================
# 3. Text Module (X-count v3 Equivalent & Hash)
# ==========================================
# 主要gTLDリスト（長い選択肢を先に置いて最短一致誤認を防止）
_COMMON_GTLDS = (
    r"com\.au|co\.jp|ne\.jp|or\.jp|ac\.jp|go\.jp|co\.uk|org\.uk|"
    r"com|org|net|edu|gov|mil|int|info|biz|io|dev|ai|app|me|co|xyz|tech|online|site|link|work|top|club|vip|"
    r"space|cloud|store|blog|design|live|world|today|zone|media|center|group|team|company|network|agency|studio"
)
# IANA公式 / twitter-text validCCTLD.js 準拠の255実在ccTLD（Node.js等の.jsや.tsは非ccTLDのためURL判定から除外）
_IANA_CCTLDS = (
    r"ac|ad|ae|af|ag|ai|al|am|ao|aq|ar|as|at|au|aw|ax|az|"
    r"ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bw|by|bz|"
    r"ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cu|cv|cw|cx|cy|cz|"
    r"de|dj|dk|dm|do|dz|"
    r"ec|ee|eg|eh|er|es|et|eu|"
    r"fi|fj|fk|fm|fo|fr|"
    r"ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|"
    r"hk|hm|hn|hr|ht|hu|"
    r"id|ie|il|im|in|io|iq|ir|is|it|"
    r"je|jm|jo|jp|"
    r"ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|"
    r"la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|"
    r"ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|"
    r"na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|"
    r"om|"
    r"pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|"
    r"qa|"
    r"re|ro|rs|ru|rw|"
    r"sa|sb|sc|sd|se|sg|sh|si|sj|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|"
    r"tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tr|tt|tv|tw|tz|"
    r"ua|ug|uk|us|uy|uz|"
    r"va|vc|ve|vg|vi|vn|vu|"
    r"wf|ws|"
    r"ye|yt|"
    r"za|zm|zw"
)
_VALID_TLD = rf"(?:{_COMMON_GTLDS}|{_IANA_CCTLDS})"

# プロトコル付き URL (http:// or https://) で、ホスト部にドメインドットと有効TLDを含むもの (localhost等の単一ホストは除外)
_PROTOCOL_URL = (
    r"https?://"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+" + _VALID_TLD +
    r"(?::[0-9]{1,5})?"
    r"(?:/[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]*)?"
)

# プロトコルなしドメイン (example.com, t.co, it.so, SKILL.md, test.py 等)
_BARE_DOMAIN_URL = (
    r"(?<![A-Za-z0-9@_/.-])"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+" + _VALID_TLD +
    r"(?::[0-9]{1,5})?"
    r"(?:/[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]*)?"
    r"(?![A-Za-z0-9_])"
)

URL_PATTERN = re.compile(
    rf"(?:{_PROTOCOL_URL}|{_BARE_DOMAIN_URL})(?<![.,!?;:)])",
    re.IGNORECASE,
)

# 絵文字書記素クラスタ（❤️ 等の異体字セレクタ付き絵文字、ZWJ連結絵文字 👨‍👩‍👧‍👦、国旗 🇯🇵、キーキャップ 1️⃣）を1単位(2pt)として抽出する正規表現
_EMOJI_BASE = (
    r"[\u203C\u2049\u2122\u2139\u2194-\u2199\u21A9\u21AA\u231A\u231B\u2328\u23CF"
    r"\u23E9-\u23F3\u23F8-\u23FA\u24C2\u25AA\u25AB\u25B6\u25C0\u25FB-\u25FE"
    r"\u2600-\u27BF\u2934\u2935\u2B05-\u2B07\u2B1B\u2B1C\u2B50\u2B55\u3030\u303D"
    r"\u3297\u3299\U0001F000-\U0001FAFF]"
)
_EMOJI_MOD = r"[\uFE0E\uFE0F\U0001F3FB-\U0001F3FF\U000E0020-\U000E007F]*"
EMOJI_CLUSTER_PATTERN = re.compile(
    r"(?:"
    r"[\U0001F1E6-\U0001F1FF]{2}"
    r"|[0-9#*]\uFE0F?\u20E3"
    r"|" + _EMOJI_BASE + _EMOJI_MOD + r"(?:\u200D" + _EMOJI_BASE + _EMOJI_MOD + r")*"
    r")"
)

_X_LINKABLE_EXTENSIONS = ('.md', '.py', '.sh')


def is_x_weight_one(code: int) -> bool:
    """
    twitter-text v3.json 公式定義に基づく weight 1 (1pt) コードポイント範囲:
    - 0x0000 - 0x10FF (ラテン文字、アクセント付き文字、ギリシャ文字 π/μ、キリル文字等)
    - 0x2000 - 0x200D (一般句読点スペース、ZWJ等)
    - 0x2010 - 0x201F (ハイフン、ダッシュ、欧文引用符)
    - 0x2032 - 0x2037 (プライム記号)
    """
    return (
        (0x0000 <= code <= 0x10FF)
        or (0x2000 <= code <= 0x200D)
        or (0x2010 <= code <= 0x201F)
        or (0x2032 <= code <= 0x2037)
    )


def count_x_points(text):
    # 1. twitter-text 仕様に従い NFC 正規化を適用
    normalized_text = unicodedata.normalize("NFC", text)

    # 2. URL (https://... およびプロトコルなしccTLD/gTLDドメイン) を抽出して各 23pt 加算
    urls = URL_PATTERN.findall(normalized_text)
    text_no_url = URL_PATTERN.sub("", normalized_text)

    # 3. 絵文字クラスタ (❤️, 👨‍👩‍👧‍👦, 🇯🇵 等) を抽出して各 2pt 加算
    emojis = EMOJI_CLUSTER_PATTERN.findall(text_no_url)
    text_remaining = EMOJI_CLUSTER_PATTERN.sub("", text_no_url)

    score = len(urls) * 23 + len(emojis) * 2
    for char in text_remaining:
        code = ord(char)
        if is_x_weight_one(code):
            score += 1
        else:
            score += 2

    # ファイル名がXのccTLD規則で意図せずURL扱いされた場合のポカヨケ警告
    url_warnings = [
        f"'{u}' is automatically linkified as a URL (23pt) on X due to its ccTLD ({ext}). Avoid periods or separate with spaces if unintended."
        for u in urls
        for ext in _X_LINKABLE_EXTENSIONS
        if u.lower().endswith(ext) or f"{ext}/" in u.lower()
    ]

    is_ok = score <= 280
    res = {
        "score_pt": score,
        "max_pt": 280,
        "char_count": len(normalized_text),
        "url_count": len(urls),
        "emoji_count": len(emojis),
        "fullwidth_equiv": score / 2.0,
        "remaining_pt": 280 - score,
        "remaining_fullwidth": (280 - score) / 2.0,
        "status": "OK" if is_ok else "OVERFLOW",
        "detected_urls": urls,
    }
    if url_warnings:
        res["url_warnings"] = url_warnings
    return res


# ==========================================
# 4. Time Module (Diff & Epoch)
# ==========================================
def parse_date(date_str):
    s = date_str.strip()
    if s.lower() == 'now':
        return datetime.now(JST)
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d', '%Y/%m/%d', '%Y/%m/%d %H:%M:%S', '%Y/%m/%d %H:%M'):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=JST)
        except ValueError:
            pass
    # ISO 8601 (例: 2026-07-10T09:00, 2026-07-10T00:00:00Z, +09:00 等) のフォールバック
    try:
        iso_str = s[:-1] + '+00:00' if s.endswith(('Z', 'z')) else s
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=JST)
        return dt.astimezone(JST)
    except ValueError:
        pass
    raise ValueError(f"Cannot parse date: {date_str}")


def calc_time_diff(start_str, end_str='now'):
    t1 = parse_date(start_str)
    t2 = parse_date(end_str)
    diff = t2 - t1
    total_sec = int(diff.total_seconds())
    sign = "-" if total_sec < 0 else ""
    abs_sec = abs(total_sec)
    days = abs_sec // 86400
    hours = (abs_sec % 86400) // 3600
    minutes = (abs_sec % 3600) // 60
    return {
        "start": t1.strftime('%Y-%m-%d %H:%M:%S'),
        "end": t2.strftime('%Y-%m-%d %H:%M:%S'),
        "total_days": round(diff.total_seconds() / 86400, 2),
        "total_hours": round(diff.total_seconds() / 3600, 2),
        "readable": f"{sign}{days}日 {hours}時間 {minutes}分",
        "seconds": total_sec,
    }


CHEAT_SHEET = """======================================================================
  AG Scientific Co-Processor (ag_calc) Cheat Sheet for Antigravity
======================================================================
[Math & Engineering Units (NFKC & SI Supported)]
  python ag_calc.py math "560k * 34.5u"             # SI Units (T, G, M, k, m, u/μ/µ, n, p)
  python ag_calc.py math ".5k + 10n"                # Leading dot & nano/pico precision
  python ag_calc.py math --explain "1 / (2*π*10k*0.1u)" # Show normalized SI expansion & ans usage
  python ag_calc.py math "parallel(100, 200, 300)"  # Parallel resistance
  python ag_calc.py math "root(-8, 3)"              # nth-root (supports odd roots of negative nums)
  python ag_calc.py math --stat "[10, 20, 30, 40]"  # Mean, stdev, median

[Memory & Chaining]
  python ag_calc.py math "ans / 2"                  # Re-use previous result
  python ag_calc.py math "r1 = 10k"                 # Store variable (constants protected)
  python ag_calc.py mem                             # View stored variables
  python ag_calc.py mem --clear                     # Reset memory

[Active Vision & PTZ Geometry]
  python ag_calc.py vision fov --fov 60 --res 1920 --pos 1420
  python ag_calc.py vision crop --res 1920x1080 --norm 0.2,0.3,0.8,0.7
  python ag_calc.py vision crop --res 1920x1080 --norm 200,300,800,700 --scale 1000

[Text & Metrics (X 280pt / NFC / Emoji Cluster & Bare URL Support)]
  python ag_calc.py text x "投稿ドラフト..."         # 280pt check (NFC, ❤️/ZWJ=2pt, JP URL safe)
  python ag_calc.py text hash "文字列"              # SHA256 & MD5

[Time & Calendar (Default TZ: JST / ISO 8601 Supported)]
  python ag_calc.py time diff "2026-07-10" "now"    # Elapsed days & hours (JST default)
  python ag_calc.py time now                        # JST ISO & Epoch
======================================================================"""


class LoggingArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        log_error(sys.argv[1:], f"CLI argument error: {message}")
        super().error(message)


# ==========================================
# CLI Entry Point
# ==========================================
def main():
    argv = sys.argv[1:]

    # --stat の次が負数で始まる場合（例: --stat "-1,2,3"）のポカヨケ結合
    for i in range(len(argv) - 1):
        if argv[i] == "--stat" and argv[i + 1].startswith("-"):
            argv[i] = f"--stat={argv[i + 1]}"
            del argv[i + 1]
            break

    # 負の数で始まる数式（例: math "-12*3" や math --explain "-12*3"）がフラグと誤認されるのを防ぐポカヨケ
    if len(argv) >= 2 and argv[0] == "math" and "--" not in argv:
        for i in range(1, len(argv)):
            tok = argv[i]
            if tok in ("--explain", "-h", "--help"):
                continue
            if tok.startswith("--stat"):
                break
            if tok.startswith("--"):
                continue
            if tok.startswith("-"):
                argv.insert(i, "--")
                break

    # 引数なし、または help / cheat の場合は寝起き用チートシートを表示
    if not argv or argv[0] in ("-h", "--help", "help", "cheat"):
        print(CHEAT_SHEET)
        sys.exit(0)

    parser = LoggingArgumentParser(
        description="AG Scientific Co-Processor (ag_calc)",
        epilog=CHEAT_SHEET,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    # Math
    p_math = subparsers.add_parser("math", help="Mathematical evaluation and statistics")
    p_math.add_argument("expr", nargs="?", help="Math expression string e.g. 'sin(deg(45))*1920'")
    p_math.add_argument("--stat", help="JSON or comma-separated numbers for statistical analysis")
    p_math.add_argument("--explain", action="store_true", help="Show normalized expression after SI expansion")

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
    p_crop.add_argument("--norm", help="Normalized bbox ymin,xmin,ymax,xmax")
    p_crop.add_argument("--pixel", help="Pixel bbox x1,y1,x2,y2")
    p_crop.add_argument("--scale", type=float, default=1.0, help="Normalization scale (default 1.0; use 1000 for Gemini bbox)")

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

    try:
        # 1. Math
        if args.subcommand == "math":
            if args.stat:
                nums = json.loads(args.stat) if args.stat.startswith('[') else [float(x) for x in args.stat.split(',')]
                print(json.dumps(calc_stats(nums), ensure_ascii=False, indent=2))
            elif args.expr:
                res = eval_math(args.expr, explain=args.explain)
                if args.explain:
                    print(json.dumps(res, ensure_ascii=False, indent=2))
                else:
                    _, _, units_stripped = normalize_math_expr(args.expr)
                    if units_stripped:
                        print(f"Note: units stripped: {', '.join(units_stripped)}", file=sys.stderr)
                    print(format_math_result(res))
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
                if 'x' not in args.res.lower():
                    raise ValueError(f"Invalid --res format '{args.res}' (expected WxH e.g. 1920x1080)")
                w, h = [int(v) for v in args.res.lower().split('x', 1)]
                res = convert_crop_coords(w, h, norm_coords=args.norm, pixel_coords=args.pixel, scale=args.scale)
                print(json.dumps(res, ensure_ascii=False, indent=2))
            else:
                p_vision.print_help()

        # 3. Text
        elif args.subcommand == "text":
            content = args.content
            if content is None and not sys.stdin.isatty():
                content = sys.stdin.read()
            if content is None:
                raise ValueError("No text provided")

            if args.cmd == "x":
                res = count_x_points(content)
                print(json.dumps(res, ensure_ascii=False, indent=2))
            elif args.cmd == "len":
                print(json.dumps({
                    "chars": len(content),
                    "bytes": len(content.encode('utf-8')),
                    "lines": len(content.splitlines()),
                }, indent=2))
            elif args.cmd == "hash":
                import hashlib
                b = content.encode('utf-8')
                print(json.dumps({
                    "sha256": hashlib.sha256(b).hexdigest(),
                    "md5": hashlib.md5(b).hexdigest(),
                }, indent=2))

        # 4. Time
        elif args.subcommand == "time":
            if args.cmd == "now":
                dt = datetime.now(JST)
                print(json.dumps({
                    "iso": dt.isoformat(),
                    "readable": dt.strftime('%Y-%m-%d %H:%M:%S JST'),
                    "epoch": int(dt.timestamp()),
                }, indent=2))
            elif args.cmd == "diff":
                if not args.args:
                    raise ValueError("Provide start date [and optional end date]")
                start = args.args[0]
                end = args.args[1] if len(args.args) > 1 else 'now'
                res = calc_time_diff(start, end)
                print(json.dumps(res, ensure_ascii=False, indent=2))
            elif args.cmd == "epoch":
                print(int(datetime.now(JST).timestamp()))
            elif args.cmd == "from-epoch":
                if not args.args:
                    raise ValueError("Provide epoch timestamp")
                ts = float(args.args[0])
                dt = datetime.fromtimestamp(ts, JST)
                print(json.dumps({
                    "iso": dt.isoformat(),
                    "readable": dt.strftime('%Y-%m-%d %H:%M:%S JST'),
                }, indent=2))

    except Exception as e:
        err_msg = str(e)
        log_error(sys.argv[1:], err_msg)
        print(f"Error: {err_msg}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
