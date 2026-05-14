from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 16)
        self.cell(0, 10, "Lab 2 Assignment Links", ln=True, align="C")
        self.ln(10)

pdf = PDF()
pdf.add_page()
pdf.set_font("helvetica", size=12)

# Student Info
pdf.set_font("helvetica", "B", 12)
pdf.cell(0, 10, "Student ID: 220119", ln=True)
pdf.ln(10)

# Links
pdf.set_font("helvetica", "B", 12)
pdf.cell(0, 10, "1. GitHub Profile:", ln=True)
pdf.set_font("helvetica", "", 12)
pdf.set_text_color(0, 0, 255)
pdf.cell(0, 10, "https://github.com/ahsanjust", ln=True, link="https://github.com/ahsanjust")
pdf.set_text_color(0, 0, 0)
pdf.ln(5)

pdf.set_font("helvetica", "B", 12)
pdf.cell(0, 10, "2. Project Repository:", ln=True)
pdf.set_font("helvetica", "", 12)
pdf.set_text_color(0, 0, 255)
pdf.cell(0, 10, "https://github.com/ahsanjust/Artificial-Intelligence-and-Machine-Learning-Lab", ln=True, link="https://github.com/ahsanjust/Artificial-Intelligence-and-Machine-Learning-Lab")
pdf.set_text_color(0, 0, 0)
pdf.ln(5)

pdf.set_font("helvetica", "B", 12)
pdf.cell(0, 10, "3. Google Colab Notebooks:", ln=True)
pdf.ln(2)

pdf.set_font("helvetica", "I", 12)
pdf.cell(0, 8, "Linear Regression:", ln=True)
pdf.set_font("helvetica", "", 11)
pdf.set_text_color(0, 0, 255)
pdf.cell(0, 8, "https://colab.research.google.com/drive/1X8NY6jvNWh2SouooaZQV0Ax2bdSDBUGO", ln=True, link="https://colab.research.google.com/drive/1X8NY6jvNWh2SouooaZQV0Ax2bdSDBUGO")
pdf.set_text_color(0, 0, 0)
pdf.ln(3)

pdf.set_font("helvetica", "I", 12)
pdf.cell(0, 8, "Logistic Regression:", ln=True)
pdf.set_font("helvetica", "", 11)
pdf.set_text_color(0, 0, 255)
pdf.cell(0, 8, "https://colab.research.google.com/drive/1FKUkqYmXxY6ZBvyYFs2YxcqMM3KQbNyl", ln=True, link="https://colab.research.google.com/drive/1FKUkqYmXxY6ZBvyYFs2YxcqMM3KQbNyl")
pdf.set_text_color(0, 0, 0)

pdf.output("220119_links.pdf")
print("PDF created successfully: 220119_links.pdf")
