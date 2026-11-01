#!/usr/bin/env python3
"""Build the two native libraries in native/ from their C++ sources (in place).

  python scripts/build_native.py

Windows: uses cl.exe from the PATH, or else finds Visual Studio's x64 C++ tools
with vswhere. Linux / macOS: g++ or clang++ with C++17. The package ships Windows
x64 DLLs built from the same sources; any other platform must build its own .so files.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

NATIVE = Path(__file__).resolve().parents[1] / "native"
SOURCES = ("dp_native", "prefix_search")


def vcvars64():
    base = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    finder = base / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not finder.is_file():
        sys.exit("cl.exe not found: install the Visual Studio C++ build tools or use an x64 developer prompt")
    found = subprocess.run([str(finder), "-latest", "-products", "*", "-requires",
                            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                            "-property", "installationPath"], capture_output=True, text=True, check=True)
    path = Path(found.stdout.strip()) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if not found.stdout.strip() or not path.is_file():
        sys.exit("no Visual Studio x64 C++ tools found")
    return path


def main():
    setup = None
    if sys.platform == "win32" and not shutil.which("cl"):
        setup = vcvars64()
    for name in SOURCES:
        if sys.platform == "win32":
            cmd = subprocess.list2cmdline(["cl.exe", "/nologo", "/O2", "/EHsc", "/std:c++17", "/MT", "/LD",
                                           f"{name}.cpp", f"/Fe:{name}.dll"])
            if setup:
                cmd = "call " + subprocess.list2cmdline([str(setup)]) + " >nul && " + cmd
            print(cmd)
            # Pass the command line directly: cmd.exe does not undo list2cmdline quoting of /c.
            subprocess.run('cmd.exe /d /s /c "' + cmd + '"', cwd=NATIVE, check=True)
        else:
            cxx = shutil.which("g++") or shutil.which("clang++")
            if not cxx:
                sys.exit("no C++ compiler (g++ or clang++) found")
            cmd = [cxx, "-O3", "-std=c++17", "-shared", "-fPIC", f"{name}.cpp", "-o", f"{name}.so"]
            print(" ".join(cmd))
            subprocess.run(cmd, cwd=NATIVE, check=True)
    for junk in list(NATIVE.glob("*.obj")) + list(NATIVE.glob("*.exp")) + list(NATIVE.glob("*.lib")):
        junk.unlink()


if __name__ == "__main__":
    main()
