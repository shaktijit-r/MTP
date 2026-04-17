#!/usr/bin/env python3
"""
Fix corrupted ELF .dynsym sections for LLD 18+ compatibility.

Problem: Old PyTorch Android prebuilt .so files have STB_LOCAL symbols
(like __bss_start, _end, _edata) placed in the "global" partition of 
.dynsym (at indices >= sh_info). LLD 18 (NDK 27) rejects this as invalid.

Fix: For any STB_LOCAL symbol found in the global partition of .dynsym,
change its binding to STB_GLOBAL. These are just linker-generated section
markers and the binding change has no functional impact at runtime.
"""
import struct
import sys
import os


def read_le_u16(f):
    return struct.unpack('<H', f.read(2))[0]

def read_le_u32(f):
    return struct.unpack('<I', f.read(4))[0]

def read_le_u64(f):
    return struct.unpack('<Q', f.read(8))[0]


def fix_dynsym(filepath):
    """Fix misplaced STB_LOCAL symbols in .dynsym global partition."""
    file_size = os.path.getsize(filepath)
    if file_size < 64:
        return False

    with open(filepath, 'r+b') as f:
        # Verify ELF magic
        magic = f.read(4)
        if magic != b'\x7fELF':
            return False

        ei_class = struct.unpack('B', f.read(1))[0]  # 1=32bit, 2=64bit
        ei_data = struct.unpack('B', f.read(1))[0]    # 1=LE, 2=BE

        # Android is always little-endian
        if ei_data != 1:
            return False

        is64 = (ei_class == 2)

        # --- Read ELF header fields ---
        if is64:
            f.seek(40)
            e_shoff = read_le_u64(f)
            f.seek(58)
            e_shentsize = read_le_u16(f)
            e_shnum = read_le_u16(f)
            e_shstrndx = read_le_u16(f)
        else:
            f.seek(32)
            e_shoff = read_le_u32(f)
            f.seek(46)
            e_shentsize = read_le_u16(f)
            e_shnum = read_le_u16(f)
            e_shstrndx = read_le_u16(f)

        if e_shnum == 0 or e_shstrndx >= e_shnum:
            return False

        # --- Read section name string table ---
        shstrtab_hdr_off = e_shoff + e_shstrndx * e_shentsize
        if is64:
            f.seek(shstrtab_hdr_off + 24)
            shstrtab_offset = read_le_u64(f)
            shstrtab_size = read_le_u64(f)
        else:
            f.seek(shstrtab_hdr_off + 16)
            shstrtab_offset = read_le_u32(f)
            shstrtab_size = read_le_u32(f)

        f.seek(shstrtab_offset)
        shstrtab = f.read(shstrtab_size)

        # --- Find .dynsym section ---
        for i in range(e_shnum):
            sh_hdr_off = e_shoff + i * e_shentsize
            f.seek(sh_hdr_off)
            sh_name_idx = read_le_u32(f)
            sh_type = read_le_u32(f)

            # Get section name
            name_end = shstrtab.index(b'\x00', sh_name_idx)
            name = shstrtab[sh_name_idx:name_end].decode('ascii', errors='replace')

            if name != '.dynsym' or sh_type != 11:  # SHT_DYNSYM = 11
                continue

            # Read .dynsym section details
            if is64:
                f.seek(sh_hdr_off + 24)
                sh_offset = read_le_u64(f)
                sh_size = read_le_u64(f)
                f.seek(sh_hdr_off + 44)
                sh_info = read_le_u32(f)
                f.seek(sh_hdr_off + 56)
                sh_entsize = read_le_u64(f)
            else:
                f.seek(sh_hdr_off + 16)
                sh_offset = read_le_u32(f)
                sh_size = read_le_u32(f)
                f.seek(sh_hdr_off + 28)
                sh_info = read_le_u32(f)
                f.seek(sh_hdr_off + 36)
                sh_entsize = read_le_u32(f)

            if sh_entsize == 0:
                sh_entsize = 24 if is64 else 16

            num_symbols = sh_size // sh_entsize
            # st_info is at offset 4 in Elf64_Sym, offset 12 in Elf32_Sym
            st_info_field_offset = 4 if is64 else 12

            fixed_count = 0
            # Scan the "global" partition (indices >= sh_info)
            for j in range(sh_info, num_symbols):
                sym_off = sh_offset + j * sh_entsize + st_info_field_offset
                f.seek(sym_off)
                st_info = struct.unpack('B', f.read(1))[0]
                bind = (st_info >> 4) & 0xF
                sym_type = st_info & 0xF

                if bind == 0:  # STB_LOCAL in global partition — invalid!
                    # Change binding to STB_GLOBAL (1), preserve type
                    new_st_info = (1 << 4) | sym_type
                    f.seek(sym_off)
                    f.write(struct.pack('B', new_st_info))
                    fixed_count += 1

            if fixed_count > 0:
                print(f"  Fixed {fixed_count} misplaced local symbols in {os.path.basename(filepath)}")
                return True
            break

    return False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: fix_elf_dynsym.py <file1.so> [file2.so ...]")
        sys.exit(1)

    total_fixed = 0
    for path in sys.argv[1:]:
        if os.path.isfile(path):
            if fix_dynsym(path):
                total_fixed += 1

    print(f"Patched {total_fixed} files.")
