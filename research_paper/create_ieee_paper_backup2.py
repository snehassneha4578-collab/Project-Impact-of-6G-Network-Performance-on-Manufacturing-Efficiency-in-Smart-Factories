from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
    Table, TableStyle, KeepTogether
)
from pathlib import Path

OUT = Path("6G_Smart_Factory_IEEE_Research_Paper.pdf")

PAGE_W, PAGE_H = A4
LEFT = 14 * mm
RIGHT = 14 * mm
TOP = 14 * mm
BOTTOM = 15 * mm
GAP = 5 * mm

column_width = (PAGE_W - LEFT - RIGHT - GAP) / 2

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "PaperTitle",
    parent=styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=14,
    leading=16.5,
    alignment=TA_CENTER,
    spaceAfter=6
)

author_style = ParagraphStyle(
    "Author",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=8,
    leading=10,
    alignment=TA_CENTER,
    spaceAfter=6
)

abstract_style = ParagraphStyle(
    "Abstract",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=7.2,
    leading=9.1,
    alignment=TA_JUSTIFY,
    spaceAfter=3
)

section_style = ParagraphStyle(
    "Section",
    parent=styles["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=8.3,
    leading=10,
    spaceBefore=4,
    spaceAfter=2
)

subsection_style = ParagraphStyle(
    "Subsection",
    parent=styles["Heading3"],
    fontName="Helvetica-Bold",
    fontSize=7.6,
    leading=9,
    spaceBefore=2,
    spaceAfter=1.5
)

body_style = ParagraphStyle(
    "Body",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=7.2,
    leading=9.1,
    alignment=TA_JUSTIFY,
    spaceAfter=2.5
)

caption_style = ParagraphStyle(
    "Caption",
    parent=styles["Normal"],
    fontName="Helvetica-Oblique",
    fontSize=6.6,
    leading=7.8,
    alignment=TA_CENTER,
    spaceBefore=2,
    spaceAfter=3
)

reference_style = ParagraphStyle(
    "Reference",
    parent=body_style,
    fontSize=6.7,
    leading=8.2,
    leftIndent=7,
    firstLineIndent=-7
)

story = []

def P(text, style=body_style):
    story.append(Paragraph(text, style))

def section(title):
    story.append(Paragraph(title, section_style))

def subsection(title):
    story.append(Paragraph(title, subsection_style))

def fig_placeholder(number, caption):
    box = Table(
        [[Paragraph(
            f"<b>FIGURE {number}</b><br/><br/>"
            "Insert the corresponding project screenshot, analysis plot, "
            "or architecture diagram here.",
            ParagraphStyle(
                "FigBox",
                parent=body_style,
                alignment=TA_CENTER,
                fontSize=7,
                leading=9
            )
        )]],
        colWidths=[column_width - 3*mm],
        rowHeights=[27*mm]
    )
    box.setStyle(TableStyle([
        ("BOX", (0,0), (-1,-1), 0.5, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(box)
    story.append(Paragraph(f"Fig. {number}. {caption}", caption_style))

# ---------------- TITLE ----------------

story.append(Paragraph(
    "Impact of 6G Network Performance on Manufacturing Efficiency "
    "in Smart Factories: A Machine Learning-Based Analytical Framework",
    title_style
))

story.append(Paragraph(
    "<b>Sneha S</b><br/>"
    "Department of Electronics and Communication Engineering<br/>"
    "University BDT College of Engineering, Davanagere, Karnataka, India",
    author_style
))

# ---------------- ABSTRACT ----------------

P(
    "<b>Abstract—</b> Smart factories increasingly depend on connected "
    "machines, sensors, industrial communication networks and intelligent "
    "automation systems. As future industrial networks move toward 6G "
    "technologies, understanding the relationship between communication "
    "performance and manufacturing efficiency becomes an important research "
    "problem. This work presents a machine-learning-based analytical "
    "framework for studying network performance and manufacturing efficiency "
    "in a smart-factory environment. The proposed workflow includes data "
    "preparation, exploratory data analysis, feature engineering, network "
    "KPI analysis, manufacturing KPI analysis, relationship analysis and "
    "supervised machine learning. Logistic Regression, Decision Tree and "
    "Random Forest models are considered for efficiency-status prediction. "
    "The implementation also generates model results, predictions, "
    "prediction errors and analytical findings as project artifacts. "
    "The study is data-driven and observational; therefore, statistical "
    "relationships are interpreted as associations rather than causal effects. "
    "The proposed framework provides a foundation for future real-time "
    "industrial monitoring, edge intelligence and digital-twin applications."
, abstract_style)

P(
    "<b>Keywords—</b> 6G, Smart Factory, Industrial IoT, Manufacturing "
    "Efficiency, Machine Learning, Network KPI, Predictive Analytics, "
    "Industry 4.0",
    abstract_style
)

# ---------------- I ----------------

section("I. INTRODUCTION")

P(
    "The rapid development of smart manufacturing has increased the "
    "dependence of industrial systems on reliable communication between "
    "machines, sensors, robots, controllers and cloud or edge platforms. "
    "Industrial automation applications generate continuous data and may "
    "require communication with high reliability, low latency and sufficient "
    "capacity."
)

P(
    "Sixth-generation wireless communication, commonly referred to as 6G, "
    "is expected to extend the capabilities of future industrial networks "
    "through improved connectivity, intelligent network management, sensing "
    "capabilities and support for demanding applications. At the same time, "
    "manufacturing efficiency depends on several operational characteristics "
    "such as production performance, quality, utilization and process stability."
)

P(
    "This work investigates these two domains using a structured smart-factory "
    "dataset and a machine-learning-based analytical workflow. Network "
    "performance indicators are analysed together with manufacturing "
    "indicators to identify measurable relationships and evaluate whether "
    "machine-learning models can predict an efficiency-status target."
)

fig_placeholder(1, "Overall analytical framework for connecting network performance with manufacturing efficiency.")

# ---------------- II ----------------

section("II. RELATED WORK")

P(
    "Several studies have investigated low-power, reliable and high-performance "
    "communication systems for industrial automation. Recent research on 6G "
    "wireless communication for industrial automation has discussed future "
    "industrial scenarios, communication requirements, reliability, latency "
    "and challenges associated with highly connected manufacturing systems."
)

P(
    "Machine learning has also been applied to industrial monitoring, "
    "prediction, anomaly detection and optimization. Data-driven methods can "
    "identify patterns in complex industrial datasets that may not be easily "
    "captured using fixed analytical rules. However, the relationship between "
    "network-level indicators and manufacturing-level outcomes depends on "
    "the characteristics and quality of the available data."
)

P(
    "Based on these observations, the present work combines network KPI "
    "analysis, manufacturing KPI analysis and supervised machine learning "
    "within a single reproducible project workflow."
)

# ---------------- III ----------------

section("III. METHODOLOGY")

P(
    "The proposed work follows a structured data-analysis and machine-learning "
    "methodology. The complete project includes data preparation, exploratory "
    "analysis, network analysis, manufacturing analysis, relationship analysis "
    "and predictive modelling."
)

subsection("A. Dataset and Project Specifications")

P(
    "The project uses a structured smart-factory dataset containing variables "
    "representing network performance and manufacturing behaviour. The dataset "
    "is prepared for statistical analysis and supervised machine learning. "
    "The efficiency-status variable is used as the prediction target."
)

subsection("B. Data Preprocessing")

P(
    "The preprocessing stage includes inspection of the dataset, identification "
    "of relevant variables, preparation of data types and selection of suitable "
    "features for subsequent analysis. Data preparation is performed before "
    "model training to maintain a consistent modelling workflow."
)

subsection("C. Exploratory Data Analysis")

P(
    "Exploratory data analysis is performed to understand variable distributions, "
    "patterns and relationships. Descriptive statistics and visual analysis "
    "are used to identify useful characteristics of the dataset before applying "
    "machine-learning algorithms."
)

subsection("D. Feature Engineering")

P(
    "Relevant network and manufacturing variables are selected and prepared as "
    "model inputs. Feature engineering is performed according to the structure "
    "of the available dataset so that the resulting features can be used by "
    "the classification models."
)

subsection("E. Machine Learning Pipeline")

P(
    "The machine-learning workflow follows the sequence: Dataset -> "
    "Preprocessing -> Feature Selection -> Train/Test Split -> Model Training "
    "-> Prediction -> Evaluation -> Error Analysis."
)

fig_placeholder(2, "Machine-learning workflow used for efficiency-status prediction.")

# ---------------- IV ----------------

section("IV. NETWORK PERFORMANCE ANALYSIS")

P(
    "The network analysis stage examines communication-related indicators "
    "contained in the smart-factory dataset. Network KPIs provide information "
    "about the communication conditions associated with industrial operation."
)

P(
    "The generated network findings and relationship files provide a structured "
    "representation of the observed network characteristics. These outputs "
    "can be extended with additional 6G indicators when real industrial "
    "measurements become available."
)

fig_placeholder(3, "Network KPI analysis and visualization.")

# ---------------- V ----------------

section("V. MANUFACTURING EFFICIENCY ANALYSIS")

P(
    "Manufacturing-related variables are analysed to understand the operational "
    "performance represented in the dataset. These variables form the basis "
    "for interpreting the efficiency-status target used in the predictive "
    "analysis."
)

P(
    "The manufacturing analysis is treated as a multidimensional problem. "
    "Individual indicators may describe different aspects of manufacturing "
    "performance, and their interpretation depends on the definitions used "
    "in the dataset."
)

fig_placeholder(4, "Manufacturing efficiency analysis.")

# ---------------- VI ----------------

section("VI. NETWORK–MANUFACTURING RELATIONSHIP ANALYSIS")

P(
    "The relationship-analysis stage examines statistical associations between "
    "network performance indicators and manufacturing indicators. Correlation "
    "and relationship outputs are stored as separate project artifacts."
)

P(
    "These relationships should not be interpreted as proof of causation. "
    "An observed association may result from common underlying factors, "
    "dataset construction or variables that are not represented in the analysis. "
    "Controlled experiments and real industrial measurements are required "
    "to establish causal relationships."
)

fig_placeholder(5, "Relationship analysis between network and manufacturing indicators.")

# ---------------- VII ----------------

section("VII. MACHINE LEARNING MODELS")

subsection("A. Logistic Regression")

P(
    "Logistic Regression is considered as a baseline classification model. "
    "It provides a comparatively simple approach for estimating the probability "
    "of the efficiency-status classes from the selected input features."
)

subsection("B. Decision Tree")

P(
    "The Decision Tree model represents classification decisions through a "
    "hierarchical set of feature-based conditions. Its structure can provide "
    "an interpretable representation of the classification process."
)

subsection("C. Random Forest")

P(
    "Random Forest combines multiple decision trees to obtain an ensemble "
    "classifier. The method can represent nonlinear relationships and complex "
    "feature interactions present in the dataset."
)

P(
    "The three models are trained and evaluated using the project modelling "
    "pipeline. Predictions and prediction errors are retained for further "
    "inspection."
)

fig_placeholder(6, "Machine-learning model comparison and prediction workflow.")

# ---------------- VIII ----------------

section("VIII. RESULTS AND DISCUSSION")

P(
    "The project generates dedicated result files containing model results, "
    "predictions and prediction errors. Additional outputs include network "
    "findings, manufacturing findings and network–manufacturing relationship "
    "summaries."
)

P(
    "The numerical performance values are intentionally not fabricated in this "
    "paper. Exact accuracy, precision, recall, F1-score or other metrics should "
    "be reported directly from the generated project result files after "
    "verification of the final experiment."
)

P(
    "The analytical workflow demonstrates how communication indicators and "
    "manufacturing indicators can be studied together and subsequently supplied "
    "to machine-learning models. Predictive performance, however, should be "
    "interpreted together with target balance, data quality and the intended "
    "industrial application."
)

# ---------------- IX ----------------

section("IX. LIMITATIONS")

P(
    "The present study is based on a structured dataset rather than live "
    "measurements from a physical 6G smart-factory deployment. Therefore, "
    "the observed relationships describe the analysed data and should not "
    "automatically be generalized to every industrial environment."
)

P(
    "Other limitations include dependence on feature selection, target "
    "definition, dataset representativeness and the evaluation protocol. "
    "Real industrial systems may additionally contain missing telemetry, "
    "equipment-specific behaviour, changing network conditions and concept drift."
)

# ---------------- X ----------------

section("X. FUTURE SCOPE")

P(
    "The framework can be extended toward real-time industrial IoT monitoring "
    "using sensors, edge devices and wireless communication infrastructure. "
    "Future development can include online anomaly detection, time-series "
    "forecasting, explainable machine learning and adaptive resource allocation."
)

P(
    "Integration with a digital twin can further enable simulation of network "
    "and manufacturing conditions before operational decisions are applied. "
    "Hardware-assisted experiments can also provide real measurements for "
    "stronger validation of the relationships identified by the current study."
)

# ---------------- XI ----------------

section("XI. CONCLUSION")

P(
    "This paper presented a machine-learning-based analytical framework for "
    "studying 6G network performance and manufacturing efficiency in smart "
    "factories. The proposed workflow integrates data preparation, exploratory "
    "analysis, network KPI analysis, manufacturing analysis, relationship "
    "analysis and supervised machine learning."
)

P(
    "The framework provides a reproducible foundation for connecting future "
    "industrial wireless-network measurements with manufacturing intelligence. "
    "Although the current study is observational and dataset-dependent, the "
    "approach can be extended toward real-time monitoring, edge intelligence "
    "and digital-twin-based industrial decision support."
)

# ---------------- TABLE ----------------

section("TABLE I. ANALYTICAL COMPONENTS")

table_data = [
    ["Component", "Purpose"],
    ["Network KPI analysis", "Analyse communication-performance indicators."],
    ["Manufacturing analysis", "Study manufacturing and efficiency indicators."],
    ["Relationship analysis", "Identify statistical associations."],
    ["Classification", "Predict efficiency-status classes."],
    ["Error analysis", "Inspect prediction errors."],
]

table = Table(
    table_data,
    colWidths=[42*mm, 113*mm],
    repeatRows=1
)

table.setStyle(TableStyle([
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 6.5),
    ("LEADING", (0,0), (-1,-1), 7.8),
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E8E8E8")),
    ("GRID", (0,0), (-1,-1), 0.35, colors.grey),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("LEFTPADDING", (0,0), (-1,-1), 3),
    ("RIGHTPADDING", (0,0), (-1,-1), 3),
    ("TOPPADDING", (0,0), (-1,-1), 2),
    ("BOTTOMPADDING", (0,0), (-1,-1), 2),
]))

story.append(table)
story.append(Paragraph(
    "Table I summarizes the principal analytical components implemented in the project.",
    caption_style
))

# ---------------- REFERENCES ----------------

section("REFERENCES")

references = [
    '[1] E. Zeydan, S. Arslan, and Y. Turk, “6G wireless communications for industrial automation: Scenarios, requirements and challenges,” <i>Journal of Industrial Information Integration</i>, vol. 42, Art. no. 100732, Nov. 2024, doi: 10.1016/j.jii.2024.100732.',
    '[2] S. S, “Impact of 6G Network Performance on Manufacturing Efficiency in Smart Factories,” project implementation and analytical artifacts, 2026.',
    '[3] S. S, “6G Smart Factory Network Analysis,” network findings, manufacturing findings, relationship analysis and machine-learning result artifacts, 2026.'
]

for ref in references:
    story.append(Paragraph(ref, reference_style))

P(
    "<b>Project Repository:</b> "
    "github.com/snehassneha4578-collab/"
    "Project-Impact-of-6G-Network-Performance-on-Manufacturing-Efficiency-in-Smart-Factories",
    body_style)

# ---------------- TWO-COLUMN DOCUMENT ----------------

class IEEEPage(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        BaseDocTemplate.__init__(self, filename, **kwargs)

        frame1 = Frame(
            LEFT, BOTTOM,
            column_width, PAGE_H - TOP - BOTTOM,
            id="left",
            leftPadding=0, rightPadding=0,
            topPadding=0, bottomPadding=0
        )

        frame2 = Frame(
            LEFT + column_width + GAP, BOTTOM,
            column_width, PAGE_H - TOP - BOTTOM,
            id="right",
            leftPadding=0, rightPadding=0,
            topPadding=0, bottomPadding=0
        )

        self.addPageTemplates([
            PageTemplate(id="TwoColumn", frames=[frame1, frame2])
        ])

paper = IEEEPage(
    str(OUT),
    pagesize=A4,
    leftMargin=LEFT,
    rightMargin=RIGHT,
    topMargin=TOP,
    bottomMargin=BOTTOM,
    title="Impact of 6G Network Performance on Manufacturing Efficiency in Smart Factories",
    author="Sneha S"
)

def draw_page(canvas, doc):
    canvas.saveState()

    canvas.setFont("Helvetica", 6.5)
    canvas.drawCentredString(
        PAGE_W / 2,
        7.5 * mm,
        str(doc.page)
    )

    canvas.restoreState()

paper.build(story)

print("")
print("==============================================")
print("IEEE-STYLE RESEARCH PAPER CREATED")
print("==============================================")
print(f"File: {OUT}")
print(f"Size: {OUT.stat().st_size:,} bytes")
print("==============================================")


