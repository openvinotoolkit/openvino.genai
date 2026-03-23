"""
Workaround for PyAV (av) DLL import failure on Windows Server Core.

Problem: Windows Server Core containers lack AVICAP32.dll (Video for Windows),
which is a transitive dependency of av._core → avdevice → AVICAP32.dll.
Additionally, os.add_dll_directory() may be ineffective in AKS containers.

Root cause (dependency chain):
  _core.cp311-win_amd64.pyd
    → avdevice-62-*.dll (bundled in av.libs/)
      → AVICAP32.dll     ← MISSING on Server Core
        → MSVFW32.dll    ← MISSING on Server Core
        → WINMM.dll      ← may be missing

Solution:
1. Create stub DLLs for missing system DLLs (AVICAP32.dll, MSVFW32.dll, WINMM.dll)
   using a pure-Python PE builder (no compiler required)
2. Patch av/__init__.py to pre-load bundled FFmpeg DLLs via ctypes.CDLL

Usage (in CI, after `pip install av`):
    python .github/scripts/patch_pyav_for_servercore.py
"""

import os
import pathlib
import platform
import struct
import sys
import sysconfig


def get_av_libs_dir():
    """Find the av.libs directory where bundled FFmpeg DLLs live."""
    purelib = sysconfig.get_paths().get("purelib")
    libs_dir = os.path.join(purelib, "av.libs")
    if os.path.isdir(libs_dir):
        return libs_dir
    raise FileNotFoundError(f"av.libs not found at {libs_dir}. Is PyAV installed?")


def get_av_init_path():
    """Find av/__init__.py."""
    purelib = sysconfig.get_paths().get("purelib")
    init_path = os.path.join(purelib, "av", "__init__.py")
    if os.path.isfile(init_path):
        return init_path
    raise FileNotFoundError(f"av/__init__.py not found at {init_path}")


def _align(value, alignment):
    """Align value up to the given alignment."""
    return (value + alignment - 1) & ~(alignment - 1)


def build_pe_dll(dll_name, export_names):
    """
    Build a minimal valid PE32+ (64-bit) DLL entirely in Python.

    The DLL exports the specified function names as stubs that return 0.
    No compiler or linker is required. The exported functions are never
    actually called - they just need to resolve at DLL load time to satisfy
    the Windows loader's import resolution for avdevice.

    Returns the DLL content as bytes.
    """
    # PE layout:
    # - DOS header + stub
    # - PE signature
    # - COFF header
    # - Optional header (PE32+)
    # - Section headers (.text, .edata)
    # - .text section (ret 0 stub)
    # - .edata section (export directory)

    FILE_ALIGNMENT = 0x200
    SECTION_ALIGNMENT = 0x1000
    IMAGE_BASE = 0x10000000

    # --- Build export data (.edata) ---
    num_exports = len(export_names)
    dll_name_bytes = dll_name.encode("ascii") + b"\x00"

    # Export directory table (40 bytes)
    # Then: address table, name pointer table, ordinal table, name strings, dll name
    edt_size = 40
    addr_table_offset = edt_size
    addr_table_size = num_exports * 4
    name_ptr_table_offset = addr_table_offset + addr_table_size
    name_ptr_table_size = num_exports * 4
    ordinal_table_offset = name_ptr_table_offset + name_ptr_table_size
    ordinal_table_size = num_exports * 2
    dll_name_offset = ordinal_table_offset + ordinal_table_size
    names_offset = dll_name_offset + len(dll_name_bytes)

    # Build name strings
    name_bytes_list = [n.encode("ascii") + b"\x00" for n in sorted(export_names)]

    # Calculate total edata size
    total_names_size = sum(len(b) for b in name_bytes_list)
    edata_raw_size = names_offset + total_names_size

    # --- Build .text section (stub code: xor eax,eax; ret) ---
    # Single stub shared by all exports: "xor eax, eax; ret"
    text_code = b"\x31\xc0\xc3"  # xor eax, eax; ret

    # Virtual addresses (after section alignment)
    text_rva = SECTION_ALIGNMENT  # .text at 0x1000
    edata_rva = text_rva + SECTION_ALIGNMENT  # .edata at 0x2000
    text_stub_rva = text_rva  # All exports point to same stub

    # Now build the export directory with correct RVAs
    edata = bytearray()

    # Export Directory Table
    edata += struct.pack("<I", 0)  # Characteristics
    edata += struct.pack("<I", 0)  # TimeDateStamp
    edata += struct.pack("<H", 0)  # MajorVersion
    edata += struct.pack("<H", 0)  # MinorVersion
    edata += struct.pack("<I", edata_rva + dll_name_offset)  # Name RVA
    edata += struct.pack("<I", 1)  # OrdinalBase
    edata += struct.pack("<I", num_exports)  # NumberOfFunctions
    edata += struct.pack("<I", num_exports)  # NumberOfNames
    edata += struct.pack("<I", edata_rva + addr_table_offset)  # AddressOfFunctions
    edata += struct.pack("<I", edata_rva + name_ptr_table_offset)  # AddressOfNames
    edata += struct.pack("<I", edata_rva + ordinal_table_offset)  # AddressOfNameOrdinals

    # Export Address Table (all point to the same stub in .text)
    for i in range(num_exports):
        edata += struct.pack("<I", text_stub_rva)

    # Name Pointer Table (RVAs to name strings, sorted alphabetically)
    name_offset_cur = edata_rva + names_offset
    for nb in name_bytes_list:
        edata += struct.pack("<I", name_offset_cur)
        name_offset_cur += len(nb)

    # Ordinal Table
    for i in range(num_exports):
        edata += struct.pack("<H", i)

    # DLL name string
    edata += dll_name_bytes

    # Export name strings (sorted)
    for nb in name_bytes_list:
        edata += nb

    edata_raw_size = _align(len(edata), FILE_ALIGNMENT)

    # --- Build PE headers ---
    dos_header_size = 0x80  # Minimal DOS header
    pe_sig_offset = 0x80
    coff_header_size = 20
    opt_header_size = 240  # PE32+ optional header size (standard)
    num_sections = 2  # .text, .edata
    section_header_size = 40 * num_sections
    headers_raw_size = _align(
        dos_header_size + 4 + coff_header_size + opt_header_size + section_header_size,
        FILE_ALIGNMENT,
    )

    # Section file offsets
    text_file_offset = headers_raw_size
    edata_file_offset = text_file_offset + _align(len(text_code), FILE_ALIGNMENT)

    # Total image size (virtual)
    image_size = _align(edata_rva + SECTION_ALIGNMENT, SECTION_ALIGNMENT)

    pe = bytearray()

    # DOS Header
    pe += b"MZ"  # e_magic
    pe += b"\x00" * (0x3C - 2)  # padding
    pe += struct.pack("<I", pe_sig_offset)  # e_lfanew
    pe += b"\x00" * (pe_sig_offset + 4 - len(pe))  # pad to PE sig (overwrite sig area)

    # Rewrite from pe_sig_offset
    pe = pe[:pe_sig_offset]
    pe += b"PE\x00\x00"  # PE signature

    # COFF Header
    pe += struct.pack("<H", 0x8664)  # Machine: AMD64
    pe += struct.pack("<H", num_sections)  # NumberOfSections
    pe += struct.pack("<I", 0)  # TimeDateStamp
    pe += struct.pack("<I", 0)  # PointerToSymbolTable
    pe += struct.pack("<I", 0)  # NumberOfSymbols
    pe += struct.pack("<H", opt_header_size)  # SizeOfOptionalHeader
    pe += struct.pack("<H", 0x2022)  # Characteristics: DLL | EXECUTABLE | LARGE_ADDRESS_AWARE

    # Optional Header (PE32+)
    pe += struct.pack("<H", 0x20B)  # Magic: PE32+
    pe += struct.pack("<B", 14)  # MajorLinkerVersion
    pe += struct.pack("<B", 0)  # MinorLinkerVersion
    pe += struct.pack("<I", _align(len(text_code), FILE_ALIGNMENT))  # SizeOfCode
    pe += struct.pack("<I", edata_raw_size)  # SizeOfInitializedData
    pe += struct.pack("<I", 0)  # SizeOfUninitializedData
    pe += struct.pack("<I", 0)  # AddressOfEntryPoint (no entry point)
    pe += struct.pack("<I", text_rva)  # BaseOfCode

    # PE32+ fields
    pe += struct.pack("<Q", IMAGE_BASE)  # ImageBase
    pe += struct.pack("<I", SECTION_ALIGNMENT)  # SectionAlignment
    pe += struct.pack("<I", FILE_ALIGNMENT)  # FileAlignment
    pe += struct.pack("<H", 6)  # MajorOperatingSystemVersion
    pe += struct.pack("<H", 0)  # MinorOperatingSystemVersion
    pe += struct.pack("<H", 0)  # MajorImageVersion
    pe += struct.pack("<H", 0)  # MinorImageVersion
    pe += struct.pack("<H", 6)  # MajorSubsystemVersion
    pe += struct.pack("<H", 0)  # MinorSubsystemVersion
    pe += struct.pack("<I", 0)  # Win32VersionValue
    pe += struct.pack("<I", image_size)  # SizeOfImage
    pe += struct.pack("<I", headers_raw_size)  # SizeOfHeaders
    pe += struct.pack("<I", 0)  # CheckSum
    pe += struct.pack("<H", 3)  # Subsystem: WINDOWS_CUI
    pe += struct.pack("<H", 0x160)  # DllCharacteristics: DYNAMIC_BASE|NX_COMPAT|NO_SEH
    pe += struct.pack("<Q", 0x100000)  # SizeOfStackReserve
    pe += struct.pack("<Q", 0x1000)  # SizeOfStackCommit
    pe += struct.pack("<Q", 0x100000)  # SizeOfHeapReserve
    pe += struct.pack("<Q", 0x1000)  # SizeOfHeapCommit
    pe += struct.pack("<I", 0)  # LoaderFlags
    pe += struct.pack("<I", 16)  # NumberOfRvaAndSizes

    # Data Directories (16 entries)
    for i in range(16):
        if i == 0:  # Export Table
            pe += struct.pack("<I", edata_rva)  # RVA
            pe += struct.pack("<I", len(edata))  # Size
        else:
            pe += struct.pack("<I", 0)  # RVA
            pe += struct.pack("<I", 0)  # Size

    # Section Headers
    # .text
    pe += b".text\x00\x00\x00"  # Name (8 bytes)
    pe += struct.pack("<I", len(text_code))  # VirtualSize
    pe += struct.pack("<I", text_rva)  # VirtualAddress
    pe += struct.pack("<I", _align(len(text_code), FILE_ALIGNMENT))  # SizeOfRawData
    pe += struct.pack("<I", text_file_offset)  # PointerToRawData
    pe += struct.pack("<I", 0)  # PointerToRelocations
    pe += struct.pack("<I", 0)  # PointerToLineNumbers
    pe += struct.pack("<H", 0)  # NumberOfRelocations
    pe += struct.pack("<H", 0)  # NumberOfLineNumbers
    pe += struct.pack("<I", 0x60000020)  # Characteristics: CODE|EXECUTE|READ

    # .edata
    pe += b".edata\x00\x00"  # Name (8 bytes)
    pe += struct.pack("<I", len(edata))  # VirtualSize
    pe += struct.pack("<I", edata_rva)  # VirtualAddress
    pe += struct.pack("<I", edata_raw_size)  # SizeOfRawData
    pe += struct.pack("<I", edata_file_offset)  # PointerToRawData
    pe += struct.pack("<I", 0)  # PointerToRelocations
    pe += struct.pack("<I", 0)  # PointerToLineNumbers
    pe += struct.pack("<H", 0)  # NumberOfRelocations
    pe += struct.pack("<H", 0)  # NumberOfLineNumbers
    pe += struct.pack("<I", 0x40000040)  # Characteristics: INITIALIZED_DATA|READ

    # Pad headers to file alignment
    pe += b"\x00" * (headers_raw_size - len(pe))

    # .text section data
    pe += text_code
    pe += b"\x00" * (_align(len(text_code), FILE_ALIGNMENT) - len(text_code))

    # .edata section data
    pe += bytes(edata)
    pe += b"\x00" * (edata_raw_size - len(edata))

    return bytes(pe)


def create_missing_system_stubs(target_dir):
    """
    Create stub DLLs for system DLLs missing on Windows Server Core.
    These stubs satisfy the Windows loader without providing real functionality,
    since the underlying features (VfW capture, multimedia) are unused on Server Core.

    Uses a pure-Python PE builder - no compiler or linker required.
    """
    stubs_needed = {
        # avdevice imports these 2 functions from AVICAP32.dll
        "AVICAP32.dll": [
            "capCreateCaptureWindowA",
            "capGetDriverDescriptionA",
        ],
        # AVICAP32.dll imports these from MSVFW32.dll
        "MSVFW32.dll": [
            "DrawDibClose",
            "DrawDibDraw",
            "DrawDibOpen",
            "DrawDibSetPalette",
            "ICClose",
            "ICDecompress",
            "ICGetInfo",
            "ICLocate",
            "ICOpen",
            "ICSendMessage",
        ],
        # AVICAP32.dll imports these from WINMM.dll
        "WINMM.dll": [
            "timeBeginPeriod",
            "timeEndPeriod",
            "timeGetTime",
        ],
    }

    created = []
    for dll_name, exports in stubs_needed.items():
        system_path = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", dll_name)
        if os.path.exists(system_path):
            print(f"  {dll_name}: already exists in System32, skipping")
            continue

        target_path = os.path.join(target_dir, dll_name)
        if os.path.exists(target_path):
            print(f"  {dll_name}: already exists in target dir, skipping")
            continue

        print(f"  Creating stub for {dll_name} ({len(exports)} exports)...")
        try:
            dll_bytes = build_pe_dll(dll_name, exports)
            with open(target_path, "wb") as f:
                f.write(dll_bytes)
            print(f"  Created: {target_path} ({len(dll_bytes):,} bytes)")
            created.append(dll_name)
        except Exception as e:
            print(f"  ERROR: Failed to create stub for {dll_name}: {e}")
            raise e

    return created


def patch_av_init_ctypes_preload():
    """
    Patch av/__init__.py to pre-load bundled DLLs via ctypes.CDLL.

    The default delvewheel patch uses os.add_dll_directory(), which is
    ineffective in AKS Windows containers. We replace it with explicit
    ctypes.CDLL loading which puts the DLLs in the process image before
    _core.pyd asks for them.
    """
    init_path = get_av_init_path()
    code = pathlib.Path(init_path).read_text()

    # Check if already patched: look for our specific CDLL preload marker
    if "_ctypes.CDLL(" in code:
        print(f"  av/__init__.py already patched, skipping")
        return False

    old_patch = "os.add_dll_directory(libs_dir)"
    if old_patch not in code:
        print(f"  Warning: Expected delvewheel patch not found in av/__init__.py")
        print(f"  File content may have changed. Manual patching may be needed.")
        return False

    # New patch: keep add_dll_directory but also pre-load via ctypes,
    # preserving the original line indentation.
    old_index = code.index(old_patch)
    line_start = code.rfind("\n", 0, old_index) + 1
    indent = code[line_start:old_index]

    new_patch_lines = [
        "os.add_dll_directory(libs_dir)",
        f"{indent}import ctypes as _ctypes",
        f"{indent}import glob as _glob",
        f"{indent}import sys as _sys",
        f"{indent}for _dll in sorted(_glob.glob(os.path.join(libs_dir, '*.dll'))):",
        f"{indent}    try:",
        f"{indent}        _ctypes.CDLL(_dll)",
        f"{indent}    except OSError as _exc:",
        f'{indent}        print(f"Failed to preload DLL: {{_dll}}: {{_exc}}", file=_sys.stderr)',
    ]
    new_patch = "\n".join(new_patch_lines)
    new_code = code[:old_index] + new_patch + code[old_index + len(old_patch) :]
    pathlib.Path(init_path).write_text(new_code)
    print(f"  Patched {init_path}")
    return True


def verify_import():
    """Verify that av can now be imported."""
    for mod_name in list(sys.modules.keys()):
        if mod_name == "av" or mod_name.startswith("av."):
            del sys.modules[mod_name]

    try:
        import av

        print(f"  av {av.__version__} imported successfully!")
        return True
    except ImportError as e:
        print(f"  Import still fails: {e}")
        return False


def main():
    if platform.system() != "Windows":
        print("This script is only needed on Windows. Skipping.")
        return 0

    print("=" * 60)
    print("PyAV Server Core Workaround")
    print("=" * 60)

    # Step 1: Find av.libs directory
    print("\n[1/4] Locating av.libs directory...")
    try:
        libs_dir = get_av_libs_dir()
        print(f"  Found: {libs_dir}")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return 1

    # Step 2: Create stub DLLs for missing system DLLs
    print("\n[2/4] Creating stub DLLs for missing system DLLs...")
    try:
        created = create_missing_system_stubs(libs_dir)
        if created:
            print(f"  Created stubs: {', '.join(created)}")
        else:
            print("  No stubs needed (all system DLLs present or already stubbed)")
    except Exception as e:
        print(f"  Warning: Stub creation failed: {e}")
        print("  Continuing with ctypes preloading only...")

    # Step 3: Patch av/__init__.py
    print("\n[3/4] Patching av/__init__.py for ctypes preloading...")
    try:
        patch_av_init_ctypes_preload()
    except Exception as e:
        print(f"  Warning: Patching failed: {e}")

    # Step 4: Verify import
    print("\n[4/4] Verifying av import...")
    success = verify_import()

    print("\n" + "=" * 60)
    if success:
        print("SUCCESS: PyAV is now importable on this system.")
    else:
        print("PARTIAL: Workaround applied but import still fails.")
        print("Additional investigation needed - check error above.")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
