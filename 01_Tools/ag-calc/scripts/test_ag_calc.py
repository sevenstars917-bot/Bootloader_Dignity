#!/usr/bin/env python3
"""
Golden Test Suite for AG Scientific Co-Processor (ag_calc.py)
レビュー指摘事項および境界条件（SI単位・AST安全性・X文字数v3・状態汚染防止・幾何/時間）の完全検証。
"""

import unittest
import subprocess
import sys
import os
import json

SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ag_calc.py")


def run_cli(*args):
    """CLIとしてag_calc.pyを実行し、(returncode, stdout, stderr)を返す"""
    cmd = [sys.executable, SCRIPT_PATH] + list(args)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
    return res.returncode, res.stdout.strip(), res.stderr.strip()


class TestAgCalcGolden(unittest.TestCase):

    def setUp(self):
        run_cli("mem", "--clear")

    def tearDown(self):
        run_cli("mem", "--clear")

    # ------------------------------------------
    # 1. 微小値・SI接頭辞・有効桁表示
    # ------------------------------------------
    def test_small_si_values_not_zero(self):
        """10n や 100p が round(..., 6) で 0.0 に潰れないこと"""
        code, out, err = run_cli("math", "10n")
        self.assertEqual(code, 0, msg=err)
        self.assertNotEqual(out, "0.0")
        self.assertAlmostEqual(float(out), 1e-8, places=15)

        code, out, err = run_cli("math", "100p")
        self.assertEqual(code, 0, msg=err)
        self.assertAlmostEqual(float(out), 1e-10, places=17)

    def test_leading_dot_si_prefix(self):
        """.5k が構文エラーにならず 500 と評価されること"""
        code, out, err = run_cli("math", ".5k")
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(out, "500")

        code, out, err = run_cli("math", ".25M + .5k")
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(out, "250500")

    def test_stats_small_values(self):
        """calc_stats でも微小値が 0.0 に丸められないこと"""
        code, out, err = run_cli("math", "--stat", "[1e-8, 2e-8, 3e-8]")
        self.assertEqual(code, 0, msg=err)
        data = json.loads(out)
        self.assertAlmostEqual(data["mean"], 2e-8, places=15)
        self.assertAlmostEqual(data["median"], 2e-8, places=15)
        self.assertGreater(data["stdev"], 0)

    def test_math_explain(self):
        """math --explain で正規化・SI展開後の式が確認できること"""
        code, out, err = run_cli("math", "--explain", ".5k + 10n")
        self.assertEqual(code, 0, msg=err)
        data = json.loads(out)
        self.assertIn("(.5*1e3)", data["normalized"])
        self.assertIn("(10*1e-9)", data["normalized"])
        self.assertAlmostEqual(data["result"], 500.00000001, places=10)

    # ------------------------------------------
    # 2. AST安全性（巨大指数・非数値定数の拒否）
    # ------------------------------------------
    def test_ast_giant_power_rejected(self):
        """9**9**9 がハングせず即座にエラー終了すること"""
        code, out, err = run_cli("math", "9**9**9")
        self.assertNotEqual(code, 0)
        self.assertIn("Error:", err)

    def test_ast_string_constant_rejected(self):
        """'a'*100 などの文字列リテラル演算が拒否されること"""
        code, out, err = run_cli("math", "'a' * 100")
        self.assertNotEqual(code, 0)
        self.assertIn("Error:", err)

    # ------------------------------------------
    # 3. 状態ファイルと予約語（定数・関数）保護
    # ------------------------------------------
    def test_reserved_constant_overwrite_rejected(self):
        """e = 5 や pi = 3 など予約定数・関数の上書きを拒否すること"""
        for reserved in ("e", "pi", "π", "sin", "ans", "Ans"):
            code, out, err = run_cli("math", f"{reserved} = 5")
            self.assertNotEqual(code, 0, msg=f"{reserved} should be rejected")
            self.assertIn("Error:", err)

        # 自然対数の底 e が無傷であること
        code, out, err = run_cli("math", "e")
        self.assertEqual(code, 0)
        self.assertAlmostEqual(float(out), 2.718281828, places=6)

    def test_variable_store_and_chain(self):
        """通常の変数代入と ans チェーンは正常に機能すること"""
        code, out, err = run_cli("math", "r1 = 10k")
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(out, "10000")

        code, out, err = run_cli("math", "r1 + ans")
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(out, "20000")

    # ------------------------------------------
    # 4. X文字数カウント（URL日本語隣接・NFC・twitter-text v3範囲）
    # ------------------------------------------
    def test_x_url_followed_by_japanese(self):
        """スペースなしでURLに続く日本語（例: https://example.com/aです）がURLに吸われないこと"""
        code, out, err = run_cli("text", "x", "https://example.com/aです")
        self.assertEqual(code, 0, msg=err)
        data = json.loads(out)
        # URL(23pt) + 「です」(2文字 * 2pt = 4pt) = 27pt
        self.assertEqual(data["score_pt"], 27)

    def test_x_weight_ranges_and_nfc(self):
        """ラテン文字・ギリシャ文字(π, μ)・NFC結合文字が1ptとしてカウントされること"""
        # NFDの 'e' + 結合アキュートアクセント (\u0301) -> NFCで1文字(é: 0x00E9) = 1pt
        nfd_e_acute = "e\u0301"
        code, out, err = run_cli("text", "x", nfd_e_acute)
        self.assertEqual(code, 0, msg=err)
        data = json.loads(out)
        self.assertEqual(data["score_pt"], 1)

        # ギリシャ文字 π (0x03C0), μ (0x03BC) -> 0x0000-0x10FF の範囲内なので各1pt
        code, out, err = run_cli("text", "x", "πμ")
        self.assertEqual(code, 0, msg=err)
        data = json.loads(out)
        self.assertEqual(data["score_pt"], 2)

        # 日本語「あ」(2pt) + 絵文字「🍵」(2pt) = 4pt
        code, out, err = run_cli("text", "x", "あ🍵")
        self.assertEqual(code, 0, msg=err)
        data = json.loads(out)
        self.assertEqual(data["score_pt"], 4)

    # ------------------------------------------
    # 5. Vision / Time / Error フォーマット統一
    # ------------------------------------------
    def test_fov_zero_offset_center(self):
        """fov で pos == res/2 のとき offset_deg == 0 かつ direction == 'center' になること"""
        code, out, err = run_cli("vision", "fov", "--fov", "60", "--res", "1920", "--pos", "960")
        self.assertEqual(code, 0, msg=err)
        data = json.loads(out)
        self.assertEqual(data["offset_px"], 0.0)
        self.assertEqual(data["offset_deg"], 0.0)
        self.assertEqual(data["direction"], "center")

    def test_crop_scale_1000_and_bounds(self):
        """crop で --scale 1000（Gemini bbox仕様）が使えること、また未指定で >1.0 はエラーになること"""
        code, out, err = run_cli("vision", "crop", "--res", "1920x1080", "--norm", "200,300,800,700", "--scale", "1000")
        self.assertEqual(code, 0, msg=err)
        data = json.loads(out)
        self.assertEqual(data["pixel_bbox"], [576, 216, 1344, 864])

        # --scale なしで 200,300,800,700 を渡した場合はポカヨケエラー
        code, out, err = run_cli("vision", "crop", "--res", "1920x1080", "--norm", "200,300,800,700")
        self.assertNotEqual(code, 0)
        self.assertIn("Error:", err)

    def test_negative_time_diff_readable(self):
        """負の時間差が '-2日 23時間' のような剰余バグにならず '-1日 1時間 0分' と直感的に出ること"""
        code, out, err = run_cli("time", "diff", "2026-11-04 12:00:00", "2026-11-03 11:00:00")
        self.assertEqual(code, 0, msg=err)
        data = json.loads(out)
        self.assertEqual(data["readable"], "-1日 1時間 0分")

    def test_unified_error_output(self):
        """vision や time の不正入力でも生のTracebackではなく Error: 形式で出力されること"""
        code, out, err = run_cli("time", "diff", "invalid-date", "now")
        self.assertNotEqual(code, 0)
        self.assertTrue(err.startswith("Error:"), msg=f"Unexpected stderr: {err}")
        self.assertNotIn("Traceback", err)

        code, out, err = run_cli("vision", "crop", "--res", "invalid_res", "--norm", "0.1,0.1,0.5,0.5")
        self.assertNotEqual(code, 0)
        self.assertTrue(err.startswith("Error:"), msg=f"Unexpected stderr: {err}")
        self.assertNotIn("Traceback", err)

    # ------------------------------------------
    # 6. Round 2 査読指摘（キーワード引数・複素数・NFKC・絵文字・空文字・ISO日時）
    # ------------------------------------------
    def test_keyword_args_rejected(self):
        """round(2.567, ndigits=1) のようにキーワード引数が黙って無視されずエラーになること"""
        code, out, err = run_cli("math", "round(2.567, ndigits=1)")
        self.assertNotEqual(code, 0)
        self.assertIn("Error:", err)

        # 位置引数 round(2.567, 1) は正常に 2.6 を返すこと
        code, out, err = run_cli("math", "round(2.567, 1)")
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(out, "2.6")

    def test_complex_numbers_rejected_and_odd_negative_root(self):
        """(-1)**0.5 はエラーになり、root(-8, 3) は実数 -2 を返すこと"""
        code, out, err = run_cli("math", "(-1)**0.5")
        self.assertNotEqual(code, 0)
        self.assertIn("Error:", err)

        code, out, err = run_cli("math", "root(-8, 2)")
        self.assertNotEqual(code, 0)
        self.assertIn("Error:", err)

        code, out, err = run_cli("math", "root(-8, 3)")
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(out, "-2")

    def test_explain_with_leading_minus_and_ans_tracking(self):
        """math --explain "-12*3" がエラーにならず、また ans 参照有無が出力されること"""
        code, out, err = run_cli("math", "--explain", "-12*3")
        self.assertEqual(code, 0, msg=err)
        data = json.loads(out)
        self.assertEqual(data["result"], -36)
        self.assertFalse(data["used_ans"])

        code, out, err = run_cli("math", "--explain", "ans + 6")
        self.assertEqual(code, 0, msg=err)
        data2 = json.loads(out)
        self.assertEqual(data2["result"], -30)
        self.assertTrue(data2["used_ans"])
        self.assertEqual(data2["ans_value"], -36)

    def test_nfkc_fullwidth_and_micro_sign(self):
        """マイクロ記号 µ (U+00B5) や全角数字・全角単位（４.７μ*１ｋ）が正しくSI展開されること"""
        code, out, err = run_cli("math", "4.7\u00b5*1k")
        self.assertEqual(code, 0, msg=err)
        self.assertAlmostEqual(float(out), 0.0047, places=8)

        code, out, err = run_cli("math", "４.７μ*１ｋ")
        self.assertEqual(code, 0, msg=err)
        self.assertAlmostEqual(float(out), 0.0047, places=8)

    def test_x_emoji_clusters_and_bare_domain_and_empty_string(self):
        """❤️ (U+2764+U+FE0F) が 2pt、プロトコルなし example.com が 23pt、空文字 "" が固まらず 0pt になること"""
        code, out, err = run_cli("text", "x", "❤️")
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(json.loads(out)["score_pt"], 2)

        code, out, err = run_cli("text", "x", "👨‍👩‍👧‍👦")
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(json.loads(out)["score_pt"], 2)

        code, out, err = run_cli("text", "x", "example.com")
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(json.loads(out)["score_pt"], 23)

        code, out, err = run_cli("text", "x", "example.com/aです")
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(json.loads(out)["score_pt"], 27)

        # 空文字 "" でも stdin 待ちでハングしないこと
        code, out, err = run_cli("text", "x", "")
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(json.loads(out)["score_pt"], 0)

    def test_time_diff_iso8601(self):
        """ISO 8601 (T区切りやZ付き) が time diff でパースできること"""
        code, out, err = run_cli("time", "diff", "2026-07-10T09:00:00+09:00", "2026-07-11T09:00:00+09:00")
        self.assertEqual(code, 0, msg=err)
        data = json.loads(out)
        self.assertEqual(data["seconds"], 86400)

        # UTC (Z) の 00:00 は JST の 09:00 と同時刻なので差分0秒
        code, out, err = run_cli("time", "diff", "2026-07-10T00:00:00Z", "2026-07-10 09:00:00")
        self.assertEqual(code, 0, msg=err)
        data2 = json.loads(out)
        self.assertEqual(data2["seconds"], 0)

    # ------------------------------------------
    # 7. Round 3 査読指摘（上付きサイレント誤答防止・単位サフィックス・負数stat・テレメトリ）
    # ------------------------------------------
    def test_superscript_subscript_rejected(self):
        """5² や 10³ が 52 や 103 にサイレント変質せず即座にエラー拒否されること"""
        for expr in ("5²", "10³", "2⁴", "x₁ = 10"):
            code, out, err = run_cli("math", expr)
            self.assertNotEqual(code, 0, msg=f"Should reject {expr}")
            self.assertIn("Superscript/subscript", err)

    def test_engineering_unit_suffixes(self):
        """モデルが書きがちな工学単位サフィックス（µF, nF, kΩ, V, A, MHz, mW等）が自動展開・計算されること"""
        code, out, err = run_cli("math", "4.7µF * 1kΩ")
        self.assertEqual(code, 0, msg=err)
        self.assertAlmostEqual(float(out), 0.0047, places=8)

        code, out, err = run_cli("math", "10V / 2A")
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(out, "5")

        code, out, err = run_cli("math", "100nF")
        self.assertEqual(code, 0, msg=err)
        self.assertAlmostEqual(float(out), 1e-7, places=15)

        code, out, err = run_cli("math", "500mW + 0.5W")
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(out, "1")

        code, out, err = run_cli("math", "100MHz")
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(out, "100000000")

    def test_stat_with_negative_first_item(self):
        """math --stat "-1,2,3" が argparse エラーにならず正常に計算できること"""
        code, out, err = run_cli("math", "--stat", "-1,2,3")
        self.assertEqual(code, 0, msg=err)
        data = json.loads(out)
        self.assertEqual(data["mean"], 1.333333333)
        self.assertEqual(data["min"], -1)
        self.assertEqual(data["max"], 3)

    def test_error_logging_telemetry(self):
        """エラー発生時に .ag_calc_errors.log に記録されること"""
        log_file = os.path.join(os.path.dirname(SCRIPT_PATH), ".ag_calc_errors.log")
        code, out, err = run_cli("math", "invalid_var_xyz")
        self.assertNotEqual(code, 0)
        self.assertTrue(os.path.exists(log_file))
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("invalid_var_xyz", content)

    # ------------------------------------------
    # 8. Round 4 永続装甲化（カンマ千の位防止・追加数学関数・時間サフィックス・URL精密化）
    # ------------------------------------------
    def test_comma_thousands_rejected(self):
        """000 トークンや裸のカンマ数値が誤爆せず、スペースなしの関数引数は正常に通ること"""
        # スペースなしの関数引数は正常に通ること（前回の退行修正）
        code, out, err = run_cli("math", "parallel(100,200,300)")
        self.assertEqual(code, 0, msg=err)
        self.assertAlmostEqual(float(out), 54.5454545, places=5)

        code, out, err = run_cli("math", "gcd(120,180)")
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(out, "60")

        code, out, err = run_cli("math", "max(100,200)")
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(out, "200")

        # 000 トークン（parallel(1,000, 2,000) 等）やタプルになる裸のカンマ数値は拒否されること
        for expr in ("parallel(1,000, 2,000)", "round(1,000)", "000", "1,000 * 3"):
            code, out, err = run_cli("math", expr)
            self.assertNotEqual(code, 0, msg=f"Should reject: {expr}")

    def test_additional_math_functions(self):
        """hypot, gcd, lcm, min, max, clamp, factorial, comb, perm が正しく動作し、上限(1000)が効くこと"""
        code, out, err = run_cli("math", "hypot(3, 4)")
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(out, "5")

        code, out, err = run_cli("math", "gcd(24, 36)")
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(out, "12")

        code, out, err = run_cli("math", "lcm(4, 6)")
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(out, "12")

        code, out, err = run_cli("math", "min(10, 5, 20) + max(10, 5, 20)")
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(out, "25")

        code, out, err = run_cli("math", "clamp(15, 0, 10)")
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(out, "10")

        code, out, err = run_cli("math", "factorial(5)")
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(out, "120")

        code, out, err = run_cli("math", "comb(5, 2)")
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(out, "10")

        code, out, err = run_cli("math", "perm(5, 2)")
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(out, "20")

        # 巨大入力（n > 1000）の安全遮断
        for expr in ("factorial(1001)", "comb(1001, 2)", "perm(1001, 2)"):
            code, out, err = run_cli("math", expr)
            self.assertNotEqual(code, 0, msg=f"Should reject: {expr}")
            self.assertIn("1000", err)

    def test_time_unit_suffixes(self):
        """時間サフィックス 10ms, 50us, 200ns, 2s が自動展開・計算されること"""
        code, out, err = run_cli("math", "10ms + 50us")
        self.assertEqual(code, 0, msg=err)
        self.assertAlmostEqual(float(out), 0.01005, places=7)

        code, out, err = run_cli("math", "1s / 50ms")
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(out, "20")

        code, out, err = run_cli("math", "200ns")
        self.assertEqual(code, 0, msg=err)
        self.assertAlmostEqual(float(out), 2e-7, places=15)

    def test_round_half_up(self):
        """Python標準の銀行丸め（round(2.5)=2）を排し、算術四捨五入（round(2.5)=3, round(0.5)=1）になること"""
        code, out, err = run_cli("math", "round(2.5)")
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(out, "3")

        code, out, err = run_cli("math", "round(0.5)")
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(out, "1")

        code, out, err = run_cli("math", "round(2.675, 2)")
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(out, "2.68")

        code, out, err = run_cli("math", "round(-2.5)")
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(out, "-3")

        # 負の桁数による百の位・千の位丸め（round(1234, -2) -> 1200）
        code, out, err = run_cli("math", "round(1234, -2)")
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(out, "1200")

    def test_x_domain_and_code_file_warning(self):
        """localhost・Node.js・config.json は通常文字扱い、SKILL.md や it.so は 23pt かつ url_warnings が出ること"""
        # localhost:8000 はドメインではないので URL(23pt) にならず通常テキスト(28文字=28pt)
        code, out, err = run_cli("text", "x", "http://localhost:8000/a/b/c")
        self.assertEqual(code, 0, msg=err)
        data = json.loads(out)
        self.assertEqual(data["score_pt"], len("http://localhost:8000/a/b/c"))
        self.assertEqual(data["url_count"], 0)

        # Node.js や Next.js は .js が実在ccTLDにないためURL扱いされない（通常文字カウント：7pt）
        code, out, err = run_cli("text", "x", "Node.js")
        self.assertEqual(code, 0, msg=err)
        data_node = json.loads(out)
        self.assertEqual(data_node["score_pt"], 7)
        self.assertEqual(data_node["url_count"], 0)

        # config.json や works.Then は非TLDなのでURL扱いされない
        code, out, err = run_cli("text", "x", "config.json と works.Then")
        self.assertEqual(code, 0, msg=err)
        data_dev = json.loads(out)
        self.assertEqual(data_dev["url_count"], 0)

        # it.so (ソマリアccTLD) はプロトコルなしでも 23pt
        code, out, err = run_cli("text", "x", "it.so")
        self.assertEqual(code, 0, msg=err)
        data = json.loads(out)
        self.assertEqual(data["score_pt"], 23)

        # SKILL.md は 23pt かつ url_warnings が付与されること
        code, out, err = run_cli("text", "x", "SKILL.md を読んだ")
        self.assertEqual(code, 0, msg=err)
        data = json.loads(out)
        # SKILL.md(23) + 半角スペース(1) + "を読んだ"(4文字*2=8) = 32pt
        self.assertEqual(data["score_pt"], 32)
        self.assertIn("url_warnings", data)
        self.assertTrue(any("SKILL.md" in w for w in data["url_warnings"]))

    def test_explain_units_stripped(self):
        """math --explain で units_stripped が返され、通常実行でも stderr に Note が出ること"""
        code, out, err = run_cli("math", "--explain", "3.3V + 5A")
        self.assertEqual(code, 0, msg=err)
        data = json.loads(out)
        self.assertEqual(data["result"], 8.3)
        self.assertEqual(data["units_stripped"], ["V", "A"])

        # 通常実行時でも stderr に通知が出ること
        code, out, err = run_cli("math", "3.3V + 5A")
        self.assertEqual(code, 0)
        self.assertEqual(out, "8.3")
        self.assertIn("Note: units stripped: V, A", err)

    def test_argparse_error_logging_and_sanitization(self):
        """CLI引数エラーがログに記録され、text サブコマンド名が保持され本文がマスクされること"""
        log_file = os.path.join(os.path.dirname(SCRIPT_PATH), ".ag_calc_errors.log")
        code, out, err = run_cli("math", "--unknown-flag-xyz")
        self.assertNotEqual(code, 0)
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("CLI argument error", content)

        # text サブコマンドでエラーを起こした際、サブコマンド名 x が保持され本文がマスクされること
        run_cli("text", "x", "a" * 300, "--invalid-flag")
        with open(log_file, "r", encoding="utf-8") as f:
            log_lines = f.readlines()
        last_log = log_lines[-1]
        self.assertIn("text x [text: 300 chars]", last_log)


if __name__ == "__main__":
    unittest.main()

