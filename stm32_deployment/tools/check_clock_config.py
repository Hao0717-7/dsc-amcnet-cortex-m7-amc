"""
check_clock_config.py
=====================
Sanity-check that the CubeMX project file (``.ioc``) and the hand-maintained
``main`` source agree on the clock tree, and warn about the two settings that
CubeMX does **not** store in the .ioc for this project.

Background (2026-10-30): the ``.ioc`` only contains the RCC/PLL parameters
(HSI 64 MHz -> PLL1 M=8 / N=68 / P=1 => 544 MHz, HPRE /2, APB /2).  The
voltage scaling (VOS0), the LDO supply selection and the Flash wait states
(FLASH_LATENCY_3) live only in the generated ``main.c``; a naive CubeMX
"Generate Code" would fall back to defaults unless they are re-checked by
hand.  This script makes that checklist mechanical.

Usage:
    python stm32_deployment/tools/check_clock_config.py
    python stm32_deployment/tools/check_clock_config.py --ioc <path> --main <path>
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IOC = ROOT / "stm32_deployment/custom_engine/DSC_AMCNet_Deployment.ioc"
DEFAULT_MAIN = ROOT / "stm32_deployment/custom_engine/app_main.c"

def div_of(value: str, prefix: str) -> int:
    """``RCC_APB3_DIV2`` / ``RCC_HCLK_DIV2`` -> 2; returns 0 if it does not match."""
    value = (value or "").strip()
    if not value.startswith(prefix):
        return 0
    m = re.search(r"(\d+)$", value)
    return int(m.group(1)) if m else 0


def parse_ioc(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ioc", default=str(DEFAULT_IOC))
    ap.add_argument("--main", default=str(DEFAULT_MAIN))
    args = ap.parse_args()

    ioc_path, main_path = Path(args.ioc), Path(args.main)
    ok = True
    fails: list[str] = []

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        mark = "OK  " if cond else "FAIL"
        print(f"[{mark}] {label}" + (f"  ({detail})" if detail else ""))
        if not cond:
            ok = False
            fails.append(label)

    if not ioc_path.exists():
        print(f"ioc not found: {ioc_path}")
        return 2
    ioc = parse_ioc(ioc_path)
    print(f"ioc : {ioc_path.relative_to(ROOT)}")
    print(f"main: {main_path.relative_to(ROOT) if main_path.exists() else '(missing)'}")
    print()

    # --- PLL ------------------------------------------------------------
    try:
        m = int(ioc.get("RCC.DIVM1", "0"))
        n = int(ioc.get("RCC.DIVN1", "0"))
        p = int(ioc.get("RCC.DIVP1", "0"))
        fracn = int(ioc.get("RCC.PLLFRACN", "0"))
    except ValueError:
        m = n = p = fracn = 0
    src = ioc.get("RCC.PLLSourceVirtual") or ioc.get("RCC.PLL1Source") or "HSI (default)"
    print(f"PLL1: source={src} M={m} N={n} FRACN={fracn}/8192 P={p}")
    check("PLL1 M/N/P present", m > 0 and n > 0 and p > 0)
    if m and n and p:
        vco_in = 64_000_000 // m          # HSI 64 MHz / M
        sysclk = (vco_in * n + vco_in * fracn // 8192) // p
        check("PLL1 VCO input in 1-16 MHz", 1_000_000 <= vco_in <= 16_000_000, f"{vco_in / 1e6:.1f} MHz")
        check("SYSCLK == 550 MHz", sysclk == 550_000_000, f"{sysclk / 1e6:.1f} MHz")

    # --- bus dividers ---------------------------------------------------
    hpre = div_of(ioc.get("RCC.HPRE", ""), "RCC_HCLK_DIV")
    check("HPRE = /2 (HCLK 275 MHz)", hpre == 2, ioc.get("RCC.HPRE", "missing"))
    for key, label, prefix in (("RCC.D1PPRE", "APB3", "RCC_APB3_DIV"),
                               ("RCC.D2PPRE1", "APB1", "RCC_APB1_DIV"),
                               ("RCC.D2PPRE2", "APB2", "RCC_APB2_DIV"),
                               ("RCC.D3PPRE", "APB4", "RCC_APB4_DIV")):
        div = div_of(ioc.get(key, ""), prefix)
        check(f"{label} prescaler = /2 (137.5 MHz)", div == 2, ioc.get(key, "missing"))

    # --- values recorded by CubeMX --------------------------------------
    for key, expected in (("RCC.SYSCLKFreq_VALUE", 550_000_000),
                          ("RCC.HCLKFreq_Value", 275_000_000),
                          ("RCC.APB1Freq_Value", 137_500_000)):
        try:
            got = int(ioc.get(key, "0"))
        except ValueError:
            got = 0
        check(f"{key} == {expected / 1e6:g} MHz", got == expected, f"{got / 1e6:g} MHz")

    # --- settings NOT stored in .ioc -> must be present in main.c -------
    print()
    print("settings not stored in the .ioc (CubeMX regenerates defaults) -- verified in main source:")
    main_txt = main_path.read_text(encoding="utf-8", errors="replace") if main_path.exists() else ""
    check("main.c: PWR_LDO_SUPPLY", "PWR_LDO_SUPPLY" in main_txt)
    check("main.c: PWR_REGULATOR_VOLTAGE_SCALE0 (VOS0)", "PWR_REGULATOR_VOLTAGE_SCALE0" in main_txt)
    check("main.c: VOSRDY wait loop", "PWR_FLAG_VOSRDY" in main_txt)
    check("main.c: FLASH_LATENCY_4 (550 MHz)", "FLASH_LATENCY_4" in main_txt)
    check("main.c: USART3 (PD8/PD9 VCP)", "USART3" in main_txt)

    print()
    if ok:
        print("RESULT: clock config consistent (550 MHz / HCLK 275 / APB 137.5, VOS0 + LDO + FLASH_LATENCY_4 present).")
        return 0
    print("RESULT: FAILED -> " + "; ".join(fails))
    print("See docs/BUILD.md section 1.1 (CubeMX regeneration checklist).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
