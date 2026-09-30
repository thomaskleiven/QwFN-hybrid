#!/usr/bin/env python3
"""Check this repository's C++ sources against the coding rules in docs/CODING_RULES.md.

    python3 scripts/check_rules.py [--quiet]      exit 1 if any rule is violated

Needs `pip install lizard` (function boundaries). Rule 10 (zero warnings) is checked by
scripts/check_rules.sh, which builds with -Werror before running this script.

A deviation is allowed only where it is marked in the source with a comment naming the rule
and pointing at docs/CODING_RULES.md, e.g. `// rule 2 deviation: daemon loop, see CODING_RULES.md`.
"""
import glob, os, re, sys

try:
    import lizard
except ImportError:
    sys.exit("check_rules: needs `pip install lizard`")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = sorted(glob.glob(os.path.join(ROOT, "src", "*.cpp")) + glob.glob(os.path.join(ROOT, "src", "*.h"))
               + glob.glob(os.path.join(ROOT, "tools", "*.cpp")))
MAX_LINES = 60
MIN_ASSERTS = 2
ASSERT_RE = re.compile(r"\bassert_that\s*\(")
FUNC_MACROS_ALLOWED = set()


def deviation(lines, i, rule):
    """True when line i (0-based) or the line above carries a documented deviation for `rule`."""
    tag = f"rule {rule} deviation"
    return any(tag in lines[j] for j in (i, i - 1) if 0 <= j < len(lines))


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def main():
    quiet = "--quiet" in sys.argv
    bad = []   # (rule, where, message)
    for path in FILES:
        rel = os.path.relpath(path, ROOT)
        raw = open(path, errors="replace").read()
        lines = raw.split("\n")
        code_lines = strip_comments(raw).split("\n")
        info = lizard.analyze_file(path)
        for fn in info.function_list:
            where = f"{rel}:{fn.start_line} {fn.name}"
            body = lines[fn.start_line - 1:fn.end_line]
            code = code_lines[fn.start_line - 1:fn.end_line]
            n = fn.end_line - fn.start_line + 1
            # rule 4: a function fits on one page
            if n > MAX_LINES and not deviation(lines, fn.start_line - 1, 4):
                bad.append((4, where, f"{n} lines > {MAX_LINES}"))
            # rule 5: assertion density. Documented exemption: trivial accessors (<= 5 lines, no
            # branch or loop) have nothing to assert.
            n_assert = sum(len(ASSERT_RE.findall(l)) for l in code)
            trivial = n <= 5 and not re.search(r"\b(if|for|while|switch|case)\b|\?", "\n".join(code))
            if n_assert < MIN_ASSERTS and not trivial and not deviation(lines, fn.start_line - 1, 5):
                bad.append((5, where, f"{n_assert} asserts < {MIN_ASSERTS}"))
            # rule 1: no direct recursion (the function's own unqualified name called in its body)
            short = fn.name.split("::")[-1]
            if short and not short.startswith("operator") and len(code) > 1:
                if re.search(r"(?<![\w.>:])" + re.escape(short) + r"\s*\(", "\n".join(code[1:])):
                    if not deviation(lines, fn.start_line - 1, 1):
                        bad.append((1, where, "calls itself"))
        for i, l in enumerate(code_lines):
            loc = f"{rel}:{i + 1}"
            # rule 1: no goto / setjmp / longjmp
            if re.search(r"\b(goto|setjmp|longjmp)\b", l):
                bad.append((1, loc, "goto/setjmp/longjmp"))
            # rule 2: loops without a static bound must say what bounds them
            if re.search(r"for\s*\(\s*;\s*;\s*\)|while\s*\(\s*(true|1)\s*\)", l) and not deviation(lines, i, 2):
                bad.append((2, loc, "unbounded loop without a documented bound"))
            # rule 8: no function-like macros beyond the allowed ones
            m = re.match(r"\s*#\s*define\s+(\w+)\(", l)
            if m and m.group(1) not in FUNC_MACROS_ALLOWED:
                bad.append((8, loc, f"function-like macro {m.group(1)}"))
            # rule 9: no double indirection or function pointers in our own declarations
            if re.search(r"\w\s*\*\s*\*\s*\w", l) and "argv" not in l and not deviation(lines, i, 9):
                bad.append((9, loc, "double pointer"))
            if re.search(r"std::function\s*<|\(\s*\*\s*\w+\s*\)\s*\(", l) and not deviation(lines, i, 9):
                bad.append((9, loc, "function pointer / std::function"))
            # rule 7: no unchecked atoi/atof-style parsing
            if re.search(r"\b(atoi|atol|atoll|atof)\s*\(", l):
                bad.append((7, loc, "unchecked numeric parse (use qwfn::parse_int / parse_float)"))
    by_rule = {}
    for r, w, m in bad:
        by_rule.setdefault(r, []).append((w, m))
    for r in sorted(by_rule):
        print(f"rule {r}: {len(by_rule[r])} violation(s)")
        if not quiet:
            for w, m in by_rule[r]:
                print(f"  {w}: {m}")
    print("check_rules:", "PASS" if not bad else f"FAIL ({len(bad)} violations)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
