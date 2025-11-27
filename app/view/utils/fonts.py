from PyQt6.QtGui import QFontDatabase

def set_font(font_path):
    font_id = QFontDatabase.addApplicationFont(font_path)
    if font_id == -1:
        print(f"[ERROR] Không thể load font: {font_path}")
        return "Arial"  # fallback
    families = QFontDatabase.applicationFontFamilies(font_id)
    if not families:
        print(f"[ERROR] Font không có family: {font_path}")
        return "Arial"
    return families[0]


# EduNSWACTCursive = set_font("assets/fonts/EduNSWACTCursive-VariableFont_wght.ttf")
# OpenSans = set_font("assets/fonts/OpenSans-VariableFont_wdth,wght.ttf")