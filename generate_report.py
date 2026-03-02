#!/usr/bin/env python3
"""
Generate a detailed PDF report for the Fake News Detection project.
Run:  python generate_report.py
Output: FakeNewsDetection_Report.pdf
"""

import os
import joblib
import numpy as np
from fpdf import FPDF
from datetime import datetime


# -- Paths -----------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
OUTPUT_PDF = os.path.join(BASE_DIR, "FakeNewsDetection_Report.pdf")

# -- Load model artifacts --------------------------------------------------
model = joblib.load(os.path.join(MODEL_DIR, "logistic_regression_model.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
feature_names = vectorizer.get_feature_names_out()
coef = model.coef_[0]
params = model.get_params()


# -- Custom PDF class ------------------------------------------------------
class ReportPDF(FPDF):
    # Professional color palette
    PRIMARY     = (55, 48, 163)       # Deep indigo
    PRIMARY_L   = (99, 102, 241)      # Lighter indigo
    ACCENT      = (16, 185, 129)      # Emerald green
    DANGER      = (239, 68, 68)       # Rose red
    WHITE       = (255, 255, 255)
    BG_PAGE     = (248, 250, 252)     # Soft off-white
    CARD_BG     = (255, 255, 255)
    HEADING     = (30, 27, 75)        # Very dark indigo
    TEXT_BODY   = (51, 65, 85)        # Slate
    TEXT_MUTED  = (100, 116, 139)     # Lighter slate
    TABLE_HEAD  = (55, 48, 163)       # Indigo
    TABLE_EVEN  = (238, 242, 255)     # Very light indigo
    TABLE_ODD   = (255, 255, 255)
    BORDER      = (203, 213, 225)     # Light border
    COVER_TOP   = (55, 48, 163)       # Cover gradient top
    COVER_BOT   = (99, 102, 241)      # Cover gradient bottom

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=22)

    def _page_bg(self):
        self.set_fill_color(*self.BG_PAGE)
        self.rect(0, 0, 210, 297, "F")

    def header(self):
        if self.page_no() == 1:
            return
        # Top accent bar
        self.set_fill_color(*self.PRIMARY)
        self.rect(0, 0, 210, 3, "F")
        # Header text
        self.set_y(6)
        self.set_font("Helvetica", "I", 7.5)
        self.set_text_color(*self.TEXT_MUTED)
        self.cell(95, 5, "Fake News Detection - Project Report", align="L")
        self.cell(95, 5, f"Generated: {datetime.now().strftime('%B %d, %Y')}", align="R")
        self.ln(10)

    def footer(self):
        self.set_y(-16)
        self.set_draw_color(*self.BORDER)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.set_y(-13)
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*self.TEXT_MUTED)
        self.cell(95, 5, "Fake News Detection System", align="L")
        self.cell(95, 5, f"Page {self.page_no()}/{{nb}}", align="R")

    # -- Helpers -----------------------------------------------------------
    def section_title(self, num, title):
        self.ln(6)
        # Colored bar behind section number
        y = self.get_y()
        self.set_fill_color(*self.PRIMARY)
        self.rect(10, y, 8, 8, "F")
        self.set_xy(10, y)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*self.WHITE)
        self.cell(8, 8, str(num), align="C")
        # Title text
        self.set_xy(21, y)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*self.HEADING)
        self.cell(0, 8, title)
        self.ln(12)
        # Accent underline
        self.set_draw_color(*self.PRIMARY_L)
        self.set_line_width(0.6)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 10.5)
        self.set_text_color(*self.PRIMARY)
        self.cell(0, 7, title)
        self.ln(8)

    def body_text(self, text):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*self.TEXT_BODY)
        self.multi_cell(0, 5.5, text)
        self.ln(3)

    def key_value(self, key, value):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*self.TEXT_MUTED)
        self.cell(55, 7, key)
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*self.HEADING)
        self.cell(0, 7, str(value))
        self.ln(7)

    def info_box(self, text):
        """Light indigo info box."""
        y = self.get_y()
        self.set_fill_color(238, 242, 255)
        self.set_draw_color(*self.PRIMARY_L)
        self.set_line_width(0.4)
        self.rect(10, y, 190, 14, "DF")
        # Left accent bar
        self.set_fill_color(*self.PRIMARY_L)
        self.rect(10, y, 3, 14, "F")
        self.set_xy(16, y + 3)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*self.PRIMARY)
        self.multi_cell(180, 5, text)
        self.set_y(y + 17)

    def table_header(self, cols, widths):
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(*self.WHITE)
        self.set_fill_color(*self.TABLE_HEAD)
        for col, w in zip(cols, widths):
            self.cell(w, 8, col, border=0, fill=True, align="C")
        self.ln(8)

    def table_row(self, cells, widths, fill=False):
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*self.TEXT_BODY)
        if fill:
            self.set_fill_color(*self.TABLE_EVEN)
        else:
            self.set_fill_color(*self.TABLE_ODD)
        for cell_text, w in zip(cells, widths):
            self.cell(w, 7, str(cell_text), border=0, fill=True, align="C")
        self.ln(7)


# ========================================================================
# BUILD THE REPORT
# ========================================================================
pdf = ReportPDF()
pdf.alias_nb_pages()

# ====================== COVER PAGE ======================================
pdf.add_page()
# Deep indigo cover background
pdf.set_fill_color(*ReportPDF.COVER_TOP)
pdf.rect(0, 0, 210, 297, "F")

# Lighter accent strip at bottom
pdf.set_fill_color(*ReportPDF.COVER_BOT)
pdf.rect(0, 220, 210, 77, "F")

# Visual Elements - Circles and Lines
pdf.set_fill_color(79, 70, 229) # Removed the alpha parameter to fix the error
pdf.ellipse(140, -40, 150, 150, "F")
pdf.set_fill_color(67, 56, 202)
pdf.ellipse(-30, 230, 100, 100, "F")

# Project Title and Subtitle
pdf.set_y(80)
pdf.set_font("Helvetica", "B", 42)
pdf.set_text_color(*ReportPDF.WHITE)
pdf.cell(0, 18, "FAKE NEWS", align="C")
pdf.ln(18)
pdf.cell(0, 18, "DETECTION", align="C")
pdf.ln(25)

pdf.set_font("Helvetica", "", 18)
pdf.set_text_color(210, 214, 255)
pdf.cell(0, 10, "AI-Powered Verification System", align="C")
pdf.ln(45)

# Detailed Project Report - Refined
pdf.set_font("Helvetica", "B", 12)
pdf.set_text_color(*ReportPDF.WHITE)
pdf.cell(0, 8, "DETAILED PROJECT REPORT", align="C")
pdf.ln(12)

# Horizontal line on cover
pdf.set_draw_color(255, 255, 255)
pdf.set_line_width(0.8)
pdf.line(75, pdf.get_y(), 135, pdf.get_y())
pdf.ln(35)

# Tech badges at the bottom
pdf.set_y(245)
pdf.set_font("Helvetica", "B", 10)
pdf.set_text_color(238, 242, 255)
badges = ["PYTHON", "FLASK", "SCIKIT-LEARN", "TF-IDF", "LOGISTIC REGRESSION"]
badge_text = "  |  ".join(badges)
pdf.cell(0, 8, badge_text, align="C")

# ====================== PAGE 2+ =========================================
pdf.add_page()
pdf._page_bg()

# -- 1. Executive Summary -------------------------------------------------
pdf.section_title("1", "Executive Summary")
pdf.body_text(
    "This project implements an AI-powered Fake News Detection web application "
    "capable of analyzing news articles and classifying them as Real or Fake "
    "with a confidence score. The system uses a Logistic Regression classifier "
    "trained on a TF-IDF vectorized text corpus, deployed through a Flask web "
    "server with a premium glassmorphism user interface."
)
pdf.ln(1)
pdf.info_box(
    "Key capabilities: Real-time classification, word contribution analysis, "
    "interactive statistics dashboard, and session analytics."
)
pdf.ln(2)

# -- 2. Project Architecture -----------------------------------------------
pdf.section_title("2", "Project Architecture")
pdf.sub_title("System Overview")
pdf.body_text(
    "The application follows a client-server architecture:\n\n"
    "1. Frontend (HTML/CSS/JS) - Premium dark-themed UI with glassmorphism\n"
    "2. Backend (Flask) - REST API serving model predictions\n"
    "3. ML Pipeline - TF-IDF vectorization + Logistic Regression classification"
)

pdf.sub_title("Project Structure")
pdf.set_font("Courier", "", 8.5)
pdf.set_text_color(*ReportPDF.TEXT_BODY)
structure = (
    "FakeNewsDetection/\n"
    "+-- app.py                   (Flask backend server)\n"
    "+-- requirements.txt         (Python dependencies)\n"
    "+-- generate_report.py       (PDF report generator)\n"
    "+-- model/\n"
    "|   +-- logistic_regression_model.pkl\n"
    "|   +-- tfidf_vectorizer.pkl\n"
    "+-- templates/\n"
    "|   +-- index.html           (Main UI template)\n"
    "+-- static/\n"
    "    +-- css/style.css        (Styling)\n"
    "    +-- js/app.js            (Frontend logic)"
)
# Code block background
y = pdf.get_y()
pdf.set_fill_color(241, 245, 249)
pdf.set_draw_color(*ReportPDF.BORDER)
pdf.rect(10, y, 190, 62, "DF")
pdf.set_xy(14, y + 3)
pdf.multi_cell(182, 4.5, structure)
pdf.set_y(y + 66)

pdf.sub_title("Data Flow")
pdf.body_text(
    "User Input -> Frontend (POST /predict) -> Flask Backend -> "
    "TF-IDF Vectorizer -> Logistic Regression Model -> "
    "Prediction + Confidence + Word Contributions -> Frontend Display"
)

# -- 3. Model Details -------------------------------------------------------
pdf.section_title("3", "Machine Learning Model")

pdf.sub_title("3.1 Model Configuration")
pdf.key_value("Algorithm:", "Logistic Regression (Binary Classifier)")
pdf.key_value("Solver:", f"{params.get('solver', 'N/A').upper()} (Quasi-Newton optimizer)")
pdf.key_value("Regularization (C):", str(params.get("C", "N/A")))
pdf.key_value("Max Iterations:", str(params.get("max_iter", "N/A")))
pdf.key_value("Penalty:", str(params.get("penalty", "N/A")))
pdf.key_value("Classes:", "0 (Fake), 1 (Real)")
pdf.ln(2)

pdf.sub_title("3.2 Feature Extraction - TF-IDF")
pdf.key_value("Vectorizer:", "TfidfVectorizer")
pdf.key_value("Vocabulary Size:", f"{len(vectorizer.vocabulary_):,}")
pdf.key_value("Total Features:", f"{len(feature_names):,}")
pdf.body_text(
    "The TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer converts "
    "raw text into numerical feature vectors. Each word in the vocabulary is assigned "
    "a weight based on its frequency in the document relative to its frequency across "
    "all documents, emphasizing distinctive terms."
)

# -- Top Fake Indicator Words --
pdf.add_page()
pdf._page_bg()
pdf.sub_title("3.3 Top 20 Fake News Indicator Words")
pdf.body_text(
    "These words have the most negative coefficients in the Logistic Regression model, "
    "meaning their presence most strongly pushes a prediction toward 'Fake'."
)

fake_indices = np.argsort(coef)[:20]
cols = ["Rank", "Word", "Weight"]
widths = [20, 60, 50]
pdf.table_header(cols, widths)
for i, idx in enumerate(fake_indices):
    pdf.table_row(
        [str(i + 1), feature_names[idx], f"{coef[idx]:.4f}"],
        widths,
        fill=(i % 2 == 0),
    )
pdf.ln(6)

# -- Top Real Indicator Words --
pdf.sub_title("3.4 Top 20 Real News Indicator Words")
pdf.body_text(
    "These words have the most positive coefficients, meaning their presence "
    "most strongly pushes a prediction toward 'Real'."
)

pdf.add_page()
pdf._page_bg()

real_indices = np.argsort(coef)[-20:][::-1]
pdf.table_header(cols, widths)
for i, idx in enumerate(real_indices):
    pdf.table_row(
        [str(i + 1), feature_names[idx], f"{coef[idx]:.4f}"],
        widths,
        fill=(i % 2 == 0),
    )
pdf.ln(6)

# -- 4. API Documentation ---------------------------------------------------
pdf.section_title("4", "API Documentation")

pdf.sub_title("4.1  GET /")
pdf.body_text("Serves the main HTML page with the detection UI and statistics dashboard.")

pdf.sub_title("4.2  POST /predict")
pdf.body_text("Accepts a JSON body with a 'text' field containing the news article to analyze.")

# Code block
code_req = (
    'Request:  { "text": "(article text)" }\n\n'
    'Response: {\n'
    '  "prediction": "Real" | "Fake",\n'
    '  "confidence": 93.48,\n'
    '  "top_words": [\n'
    '    { "word": "trump", "contribution": -0.515 },\n'
    '    ...\n'
    '  ]\n'
    '}'
)
y = pdf.get_y()
pdf.set_fill_color(241, 245, 249)
pdf.set_draw_color(*ReportPDF.BORDER)
pdf.rect(10, y, 190, 48, "DF")
pdf.set_xy(14, y + 3)
pdf.set_font("Courier", "", 8.5)
pdf.set_text_color(*ReportPDF.TEXT_BODY)
pdf.multi_cell(182, 4.5, code_req)
pdf.set_y(y + 52)

pdf.sub_title("4.3  GET /stats")
pdf.body_text(
    "Returns comprehensive model metadata (type, solver, features, top indicator words) "
    "and session analytics (total predictions, fake/real counts, average confidence, "
    "prediction history)."
)

# -- 5. Frontend UI ----------------------------------------------------------
pdf.section_title("5", "Frontend User Interface")
pdf.sub_title("5.1 Detect Tab")
pdf.body_text(
    "- Large textarea for pasting news articles\n"
    "- Animated 'Analyze Article' button with loading spinner\n"
    "- Result card with Real/Fake verdict and animated confidence bar\n"
    "- Per-article word contribution analysis with bar chart\n"
    "- Keyboard shortcut: Cmd/Ctrl + Enter to analyze\n"
    "- Error toast notifications for validation"
)

pdf.sub_title("5.2 Model Stats Tab")
pdf.body_text(
    "- Model Information: algorithm, solver, regularization, features count\n"
    "- Session Analytics: total predictions, real/fake counts, avg confidence\n"
    "- Animated SVG donut chart for real vs fake distribution\n"
    "- Top 20 Fake Indicator Words with horizontal bar chart\n"
    "- Top 20 Real Indicator Words with horizontal bar chart\n"
    "- Prediction History table with result pills"
)

pdf.sub_title("5.3 Design System")
pdf.body_text(
    "- Theme: Premium dark mode with glassmorphism cards\n"
    "- Colors: Indigo/Violet primary, Green (Real), Red (Fake)\n"
    "- Typography: Inter (body), JetBrains Mono (data/code)\n"
    "- Animations: Floating background orbs, slide-up results, pop-in icons\n"
    "- Responsive: Fully responsive for mobile/tablet/desktop"
)

# -- 6. Technology Stack -----------------------------------------------------
pdf.add_page()
pdf._page_bg()

pdf.section_title("6", "Technology Stack")
tech_cols = ["Component", "Technology", "Version / Details"]
tech_widths = [45, 50, 75]
pdf.table_header(tech_cols, tech_widths)
techs = [
    ("Backend",       "Flask",        "3.1.0 - WSGI micro-framework"),
    ("ML Model",      "Scikit-Learn", "1.6.1 - Logistic Regression"),
    ("Vectorizer",    "Scikit-Learn", "TfidfVectorizer"),
    ("Serialization", "Joblib",       "1.4.2 - Model persistence"),
    ("CORS",          "Flask-CORS",   "5.0.1 - Cross-origin support"),
    ("Frontend",      "HTML/CSS/JS",  "Vanilla - no framework"),
    ("Typography",    "Google Fonts", "Inter + JetBrains Mono"),
    ("Language",      "Python",       "3.x"),
]
for i, (comp, tech, detail) in enumerate(techs):
    pdf.table_row([comp, tech, detail], tech_widths, fill=(i % 2 == 0))
pdf.ln(6)

# -- 7. How to Run -----------------------------------------------------------
pdf.section_title("7", "Installation and Usage")
pdf.sub_title("Prerequisites")
pdf.body_text("- Python 3.8 or later\n- pip package manager")

pdf.sub_title("Setup")
setup_code = (
    "# Clone / navigate to the project directory\n"
    "cd FakeNewsDetection\n\n"
    "# Install dependencies\n"
    "pip install -r requirements.txt\n\n"
    "# Start the server\n"
    "python app.py\n\n"
    "# Open in browser\n"
    "http://127.0.0.1:5001"
)
y = pdf.get_y()
pdf.set_fill_color(241, 245, 249)
pdf.set_draw_color(*ReportPDF.BORDER)
pdf.rect(10, y, 190, 52, "DF")
pdf.set_xy(14, y + 3)
pdf.set_font("Courier", "", 8.5)
pdf.set_text_color(*ReportPDF.TEXT_BODY)
pdf.multi_cell(182, 4.5, setup_code)
pdf.set_y(y + 56)

# -- 8. Future Enhancements ---------------------------------------------------
pdf.section_title("8", "Future Enhancements")
pdf.body_text(
    "1. Deep Learning Models - Integrate BERT or DistilBERT for improved accuracy\n"
    "2. Multi-language Support - Extend detection to Hindi, Spanish, etc.\n"
    "3. URL Scraping - Paste a URL to auto-extract and analyze article text\n"
    "4. Database - Store prediction history persistently with SQLite/PostgreSQL\n"
    "5. User Authentication - Add login to track individual user analytics\n"
    "6. Browser Extension - Chrome/Firefox extension for in-page verification\n"
    "7. Model Retraining - Periodic retraining pipeline with fresh data\n"
    "8. Explainability - LIME/SHAP integration for detailed feature explanations"
)

# -- 9. Conclusion ------------------------------------------------------------
pdf.section_title("9", "Conclusion")
pdf.body_text(
    "The Fake News Detection application successfully demonstrates the practical "
    "deployment of a machine learning model for real-world text classification. "
    f"With {len(feature_names):,} TF-IDF features and a Logistic Regression classifier, "
    "the system provides instant predictions with confidence scoring and transparent "
    "word-level explanations. The premium UI with interactive statistics dashboard "
    "makes the tool accessible and insightful for end users."
)

# -- Save ------------------------------------------------------------------
pdf.output(OUTPUT_PDF)
print(f"\nReport generated successfully!")
print(f"Location: {OUTPUT_PDF}")
print(f"Pages: {pdf.page_no()}")
