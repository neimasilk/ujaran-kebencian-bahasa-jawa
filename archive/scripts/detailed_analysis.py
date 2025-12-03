#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script untuk analisis detail struktur template jurnal dan panduan penyesuaian

Author: Assistant
Date: 2024
"""

import os
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
import re

def analyze_template_format(template_path):
    """
    Menganalisis format detail dari template jurnal
    
    Args:
        template_path (str): Path ke file template
    """
    try:
        doc = Document(template_path)
        
        print(f"\n{'='*70}")
        print("ANALISIS DETAIL FORMAT TEMPLATE JURNAL")
        print(f"{'='*70}")
        
        # Analisis margin dan page setup
        sections = doc.sections
        if sections:
            section = sections[0]
            print(f"\nPAGE SETUP:")
            print(f"  Margin Top: {section.top_margin.inches:.2f} inch")
            print(f"  Margin Bottom: {section.bottom_margin.inches:.2f} inch")
            print(f"  Margin Left: {section.left_margin.inches:.2f} inch")
            print(f"  Margin Right: {section.right_margin.inches:.2f} inch")
            print(f"  Page Width: {section.page_width.inches:.2f} inch")
            print(f"  Page Height: {section.page_height.inches:.2f} inch")
        
        # Analisis styles detail
        print(f"\nSTYLES DETAIL:")
        for style in doc.styles:
            if style.type == WD_STYLE_TYPE.PARAGRAPH:
                print(f"\n  Style: {style.name}")
                if hasattr(style, 'font') and style.font:
                    if style.font.name:
                        print(f"    Font: {style.font.name}")
                    if style.font.size:
                        print(f"    Size: {style.font.size.pt} pt")
                    if style.font.bold is not None:
                        print(f"    Bold: {style.font.bold}")
                    if style.font.italic is not None:
                        print(f"    Italic: {style.font.italic}")
                
                if hasattr(style, 'paragraph_format') and style.paragraph_format:
                    pf = style.paragraph_format
                    if pf.alignment is not None:
                        alignment_map = {
                            0: 'Left',
                            1: 'Center', 
                            2: 'Right',
                            3: 'Justify'
                        }
                        print(f"    Alignment: {alignment_map.get(pf.alignment, 'Unknown')}")
                    if pf.space_before:
                        print(f"    Space Before: {pf.space_before.pt} pt")
                    if pf.space_after:
                        print(f"    Space After: {pf.space_after.pt} pt")
                    if pf.line_spacing:
                        print(f"    Line Spacing: {pf.line_spacing}")
        
        # Analisis struktur konten
        print(f"\n{'='*50}")
        print("STRUKTUR KONTEN TEMPLATE")
        print(f"{'='*50}")
        
        sections_found = []
        current_section = None
        
        for i, para in enumerate(doc.paragraphs):
            text = para.text.strip()
            if not text:
                continue
                
            # Identifikasi bagian-bagian utama
            text_lower = text.lower()
            
            # Header/Title patterns
            if any(keyword in text_lower for keyword in ['title', 'judul']) and len(text) < 100:
                sections_found.append({
                    'type': 'TITLE',
                    'text': text,
                    'style': para.style.name,
                    'position': i
                })
            
            # Author patterns
            elif any(keyword in text_lower for keyword in ['author', 'penulis']) and len(text) < 200:
                sections_found.append({
                    'type': 'AUTHOR',
                    'text': text,
                    'style': para.style.name,
                    'position': i
                })
            
            # Abstract patterns
            elif text_lower.startswith('abstract') or text_lower.startswith('intisari'):
                sections_found.append({
                    'type': 'ABSTRACT',
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'style': para.style.name,
                    'position': i
                })
            
            # Keywords patterns
            elif text_lower.startswith('keywords') or text_lower.startswith('kata kunci'):
                sections_found.append({
                    'type': 'KEYWORDS',
                    'text': text,
                    'style': para.style.name,
                    'position': i
                })
            
            # Main sections
            elif any(text_lower.startswith(keyword) for keyword in ['introduction', 'pendahuluan']):
                sections_found.append({
                    'type': 'INTRODUCTION',
                    'text': text,
                    'style': para.style.name,
                    'position': i
                })
            
            elif any(keyword in text_lower for keyword in ['method', 'metodologi', 'research method']):
                sections_found.append({
                    'type': 'METHOD',
                    'text': text,
                    'style': para.style.name,
                    'position': i
                })
            
            elif any(keyword in text_lower for keyword in ['result', 'hasil', 'discussion', 'pembahasan']):
                sections_found.append({
                    'type': 'RESULTS',
                    'text': text,
                    'style': para.style.name,
                    'position': i
                })
            
            elif any(keyword in text_lower for keyword in ['conclusion', 'kesimpulan']):
                sections_found.append({
                    'type': 'CONCLUSION',
                    'text': text,
                    'style': para.style.name,
                    'position': i
                })
            
            elif any(keyword in text_lower for keyword in ['reference', 'daftar pustaka', 'bibliography']):
                sections_found.append({
                    'type': 'REFERENCES',
                    'text': text,
                    'style': para.style.name,
                    'position': i
                })
        
        # Tampilkan struktur yang ditemukan
        print("\nSTRUKTUR DOKUMEN YANG TERIDENTIFIKASI:")
        for section in sections_found:
            print(f"  {section['position']:2d}. [{section['type']}] {section['text']}")
            print(f"      Style: {section['style']}")
        
        return sections_found
        
    except Exception as e:
        print(f"Error menganalisis template: {str(e)}")
        return []

def generate_detailed_guide(template_structure):
    """
    Membuat panduan detail untuk penyesuaian format
    
    Args:
        template_structure (list): Struktur template yang teridentifikasi
    """
    print(f"\n{'='*70}")
    print("PANDUAN DETAIL PENYESUAIAN FORMAT")
    print(f"{'='*70}")
    
    print("\n1. URUTAN STRUKTUR DOKUMEN YANG DISARANKAN:")
    print("   Berdasarkan analisis template, struktur dokumen harus mengikuti urutan:")
    
    expected_order = ['TITLE', 'AUTHOR', 'ABSTRACT', 'KEYWORDS', 'INTRODUCTION', 'METHOD', 'RESULTS', 'CONCLUSION', 'REFERENCES']
    
    for i, section_type in enumerate(expected_order, 1):
        # Cari section yang sesuai di template
        found_section = next((s for s in template_structure if s['type'] == section_type), None)
        if found_section:
            print(f"   {i:2d}. {section_type}: Style '{found_section['style']}'")
            print(f"       Contoh: {found_section['text'][:80]}...")
        else:
            print(f"   {i:2d}. {section_type}: (Tidak ditemukan di template, gunakan style standar)")
    
    print("\n2. PENYESUAIAN STYLE KHUSUS:")
    
    # Analisis style yang paling sering digunakan
    style_usage = {}
    for section in template_structure:
        style = section['style']
        style_usage[style] = style_usage.get(style, 0) + 1
    
    print("   Styles yang paling sering digunakan dalam template:")
    for style, count in sorted(style_usage.items(), key=lambda x: x[1], reverse=True):
        print(f"     - {style}: digunakan {count} kali")
    
    print("\n3. CHECKLIST PENYESUAIAN:")
    checklist = [
        "□ Sesuaikan margin halaman (biasanya 1 inch semua sisi)",
        "□ Pastikan font utama sesuai template (biasanya Times New Roman 12pt)",
        "□ Sesuaikan format judul (biasanya bold, center, 14-16pt)",
        "□ Format nama penulis dan afiliasi sesuai template",
        "□ Abstract dalam bahasa Inggris dengan format yang benar",
        "□ Intisari dalam bahasa Indonesia (jika diperlukan)",
        "□ Keywords/kata kunci dengan format yang sesuai",
        "□ Heading dan subheading menggunakan style yang tepat",
        "□ Paragraf menggunakan justify alignment",
        "□ Spasi antar paragraf sesuai template",
        "□ Format referensi sesuai style guide jurnal",
        "□ Nomor halaman di posisi yang benar",
        "□ Header/footer sesuai template jurnal"
    ]
    
    for item in checklist:
        print(f"   {item}")
    
    print("\n4. LANGKAH PRAKTIS IMPLEMENTASI:")
    steps = [
        "Buka draft paper di Microsoft Word",
        "Buka template jurnal di window terpisah sebagai referensi",
        "Copy styles dari template ke draft (Home > Styles > Manage Styles)",
        "Sesuaikan Page Layout (Layout > Margins, Size, Orientation)",
        "Terapkan styles yang sesuai untuk setiap bagian dokumen",
        "Periksa format font dan spacing di setiap paragraf",
        "Sesuaikan format header dan footer",
        "Periksa format tabel dan gambar",
        "Review keseluruhan dokumen untuk konsistensi format",
        "Simpan dokumen dengan nama yang sesuai"
    ]
    
    for i, step in enumerate(steps, 1):
        print(f"   {i:2d}. {step}")

def create_format_template_guide():
    """
    Membuat panduan template format yang dapat digunakan sebagai referensi
    """
    print(f"\n{'='*70}")
    print("TEMPLATE FORMAT REFERENSI")
    print(f"{'='*70}")
    
    template_guide = """
    STRUKTUR DOKUMEN JURNAL STANDAR:
    
    1. JUDUL ARTIKEL
       - Font: Times New Roman, 14pt, Bold
       - Alignment: Center
       - Spacing: 0pt before, 12pt after
    
    2. NAMA PENULIS
       - Font: Times New Roman, 12pt, Regular
       - Alignment: Center
       - Spacing: 6pt before, 6pt after
    
    3. AFILIASI
       - Font: Times New Roman, 10pt, Italic
       - Alignment: Center
       - Spacing: 0pt before, 12pt after
    
    4. ABSTRACT
       - Heading: Times New Roman, 12pt, Bold
       - Content: Times New Roman, 11pt, Regular
       - Alignment: Justify
       - Spacing: 12pt before, 6pt after
    
    5. KEYWORDS
       - Font: Times New Roman, 11pt, Regular
       - Format: "Keywords: word1, word2, word3"
       - Spacing: 6pt before, 12pt after
    
    6. BODY TEXT (Introduction, Method, Results, etc.)
       - Font: Times New Roman, 12pt, Regular
       - Alignment: Justify
       - Line Spacing: 1.15 atau 1.5
       - Spacing: 0pt before, 6pt after
    
    7. HEADINGS
       - Level 1: Times New Roman, 12pt, Bold, ALL CAPS
       - Level 2: Times New Roman, 12pt, Bold, Title Case
       - Level 3: Times New Roman, 12pt, Bold Italic, Title Case
    
    8. REFERENCES
       - Font: Times New Roman, 11pt, Regular
       - Alignment: Justify
       - Hanging indent: 0.5 inch
       - Spacing: 0pt before, 3pt after
    """
    
    print(template_guide)

def main():
    """
    Fungsi utama untuk analisis detail
    """
    base_path = "d:/documents/ujaran-kebencian-bahasa-jawa/paper"
    template_path = os.path.join(base_path, "New_Template_Jurnal_JITK_Nusa_Mandiri_V11 No 1 August 2025.docx")
    
    print("ANALISIS DETAIL TEMPLATE JURNAL")
    print("=" * 70)
    print(f"Template: {template_path}")
    
    if not os.path.exists(template_path):
        print(f"\nError: File template tidak ditemukan: {template_path}")
        return
    
    # Analisis template
    template_structure = analyze_template_format(template_path)
    
    # Generate panduan detail
    generate_detailed_guide(template_structure)
    
    # Buat template referensi
    create_format_template_guide()
    
    print(f"\n{'='*70}")
    print("ANALISIS DETAIL SELESAI")
    print(f"{'='*70}")
    print("\nGunakan panduan di atas untuk menyesuaikan format draft paper dengan template jurnal.")
    print("Simpan panduan ini sebagai referensi selama proses penyesuaian.")

if __name__ == "__main__":
    main()