#!/usr/bin/env python3
"""Silence transformers' auto_docstring lint, which prints [ERROR] lines through a raw print().

The newer transformers in the 0.28-era image validates that every kwarg in a processor's
InitKwargs class is documented, and reports misses (min_frames/max_frames on
Qwen3VLVideoProcessor) as "[ERROR] ... but not documented" AT BOOT. It is a lint for
transformers developers, not an error: the emit is a bare print(), so TRANSFORMERS_VERBOSITY
cannot suppress it, and it scares log readers for a checkpoint we serve text-only anyway.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
AD = SP / "transformers" / "utils" / "auto_docstring.py"

ANCHOR = '''    if len(undocumented_parameters) > 0:
        print("\\n".join(undocumented_parameters))
'''
NEW = '''    if len(undocumented_parameters) > 0:
        # radiance 028 port: docstring lint for transformers developers, emitted via a raw
        # print() that no verbosity setting reaches. Silenced for serving.
        pass
'''


def main():
    apply(AD, ANCHOR, NEW, "docstring lint for transformers developers",
          "transformers: silence auto_docstring undocumented-parameter lint")


if __name__ == "__main__":
    main()
