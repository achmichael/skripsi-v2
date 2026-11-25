#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# Data dari training results
data = {
    "Kapasitas": ["900 VA", "900 VA", "1300 VA", "1300 VA", "2200 VA", "2200 VA"],
    "Metode": ["All Features", "Feature Selection", "All Features", "Feature Selection", "All Features", "Feature Selection"],
    "Jumlah Fitur": [45, 36, 45, 36, 45, 36],
    "MAE (hari)": ["3.46 ± 0.59", "3.10 ± 0.42", "3.31 ± 0.71", "2.79 ± 0.64", "3.55 ± 1.16", "3.21 ± 0.69"],
    "RMSE (hari)": ["4.52 ± 0.76", "3.88 ± 0.57", "4.57 ± 1.04", "3.65 ± 0.66", "4.67 ± 1.94", "4.09 ± 1.01"],
    "MAPE (%)": ["26.86 ± 6.40", "26.20 ± 3.17", "25.31 ± 9.29", "19.99 ± 3.96", "30.92 ± 13.05", "25.30 ± 6.25"],
    "R² Score": ["0.5518 ± 0.2083", "0.6634 ± 0.1661", "0.3858 ± 0.2462", "0.5949 ± 0.1725", "0.5626 ± 0.2118", "0.6506 ± 0.1412"],
}

df = pd.DataFrame(data)

# Simpan ke CSV
output_csv = Path("comparison_table_all_vs_selection.csv")
df.to_csv(output_csv, index=False)
print(f"CSV saved: {output_csv.absolute()}")

# Simpan ke Excel dengan formatting
output_excel = Path("comparison_table_all_vs_selection.xlsx")
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Comparison', index=False)
    
    # Formatting
    worksheet = writer.sheets['Comparison']
    
    # Set column widths
    worksheet.column_dimensions['A'].width = 12
    worksheet.column_dimensions['B'].width = 18
    worksheet.column_dimensions['C'].width = 14
    worksheet.column_dimensions['D'].width = 16
    worksheet.column_dimensions['E'].width = 16
    worksheet.column_dimensions['F'].width = 16
    worksheet.column_dimensions['G'].width = 16
    
    # Header styling
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF", size=11)
    
    for cell in worksheet[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal='center', vertical='center')
    
    # Data styling
    thin_border = Border(
        left=Side(style='thin', color='D0D0D0'),
        right=Side(style='thin', color='D0D0D0'),
        top=Side(style='thin', color='D0D0D0'),
        bottom=Side(style='thin', color='D0D0D0')
    )
    
    # Alternate row colors and centering
    light_fill = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
    
    for idx, row in enumerate(worksheet.iter_rows(min_row=2, max_row=len(df)+1), start=2):
        for cell in row:
            cell.border = thin_border
            cell.alignment = Alignment(horizontal='center', vertical='center')
            if idx % 2 == 0:
                cell.fill = light_fill

print(f"Excel saved: {output_excel.absolute()}")

# Print tabel untuk terminal
print("\n" + "="*120)
print("PERBANDINGAN PERFORMA MODEL: ALL FEATURES VS FEATURE SELECTION")
print("="*120)
print(df.to_string(index=False))
print("="*120)
