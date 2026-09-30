#pragma once
// Runtime checks for docs/CODING_RULES.md.
//
// assert_that (rule 5): always on, in release builds too. A failed assertion is a bug, not an
// input error, so it reports where and why, then aborts: continuing would compute on corrupt state.
// It is a function, not a macro (rule 8); std::source_location supplies the call site.
//
// parse_int / parse_float (rule 7): checked replacements for atoi/atof, which silently return 0
// on garbage and accept out-of-range values.

#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <source_location>
#include <system_error>

namespace qwfn {

// rule 5 deviation: the assertion primitive itself, see docs/CODING_RULES.md
inline void assert_that(bool ok, const char * what,
                        const std::source_location loc = std::source_location::current()) {
    if (ok) return;
    std::fprintf(stderr, "[qwfn] assertion failed: %s (%s:%u, %s)\n", what, loc.file_name(),
                 (unsigned) loc.line(), loc.function_name());
    std::abort();
}

// The whole string must be a number within [lo, hi]; `out` is untouched otherwise.
inline bool parse_int(const char * s, long long lo, long long hi, long long & out) {
    assert_that(lo <= hi, "parse_int range is not empty");
    if (!s || !*s) return false;
    long long v = 0;
    const char * end = s + std::strlen(s);
    const auto r = std::from_chars(s, end, v);
    if (r.ec != std::errc() || r.ptr != end || v < lo || v > hi) return false;
    assert_that(v >= lo && v <= hi, "accepted value lies in range");
    out = v;
    return true;
}

inline bool parse_float(const char * s, double lo, double hi, double & out) {
    assert_that(lo <= hi, "parse_float range is not empty");
    if (!s || !*s) return false;
    char * end = nullptr;
    const double v = std::strtod(s, &end);
    if (end == s || *end != '\0' || !(v >= lo && v <= hi)) return false;
    assert_that(v >= lo && v <= hi, "accepted value lies in range");
    out = v;
    return true;
}

// Command-line values: a bad value is a usage error, reported with the flag's name.
inline long long arg_int(const char * flag, const char * s, long long lo, long long hi) {
    assert_that(flag != nullptr, "arg_int names its flag");
    assert_that(lo <= hi, "arg_int range is not empty");
    long long v = 0;
    if (!parse_int(s, lo, hi, v)) {
        std::fprintf(stderr, "error: %s expects an integer in [%lld, %lld], got '%s'\n", flag, lo, hi, s ? s : "");
        std::exit(2);
    }
    return v;
}

inline double arg_float(const char * flag, const char * s, double lo, double hi) {
    assert_that(flag != nullptr, "arg_float names its flag");
    assert_that(lo <= hi, "arg_float range is not empty");
    double v = 0.0;
    if (!parse_float(s, lo, hi, v)) {
        std::fprintf(stderr, "error: %s expects a number in [%g, %g], got '%s'\n", flag, lo, hi, s ? s : "");
        std::exit(2);
    }
    return v;
}

// Environment knobs: unset or invalid falls back to the default (and says so when invalid).
inline long long env_int(const char * name, long long def, long long lo, long long hi) {
    assert_that(name != nullptr, "env_int names its name");
    assert_that(lo <= hi, "env_int range is not empty");
    const char * s = std::getenv(name);
    long long v = def;
    if (s && !parse_int(s, lo, hi, v)) {
        std::fprintf(stderr, "[qwfn] ignoring %s='%s': expected an integer in [%lld, %lld]\n", name, s, lo, hi);
        v = def;
    }
    return v;
}

inline double env_float(const char * name, double def, double lo, double hi) {
    assert_that(name != nullptr, "env_float names its name");
    assert_that(lo <= hi, "env_float range is not empty");
    const char * s = std::getenv(name);
    double v = def;
    if (s && !parse_float(s, lo, hi, v)) {
        std::fprintf(stderr, "[qwfn] ignoring %s='%s': expected a number in [%g, %g]\n", name, s, lo, hi);
        v = def;
    }
    return v;
}

}  // namespace qwfn
