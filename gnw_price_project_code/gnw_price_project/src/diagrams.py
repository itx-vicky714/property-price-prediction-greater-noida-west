"""diagrams.py - draws the architecture and mechanism diagrams used in the report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "../outputs/charts/"
NAVY, TEAL, ORANGE, LIGHT, GREY = "#1F3A5F", "#2A9D8F", "#E76F51", "#EAF4F3", "#8D99AE"


def box(ax, x, y, w, h, text, fc=NAVY, tc="white", fs=8.5, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, color=GREY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                 color=color, lw=1.6))


# ---------- Diagram 1: end-to-end architecture ----------
fig, ax = plt.subplots(figsize=(7.0, 5.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")

box(ax, 0.3, 6.4, 2.6, 1.2, "1. DATA SOURCE\nProperty listings CSV\n(600 records)", TEAL)
box(ax, 3.7, 6.4, 2.6, 1.2, "2. DATA CLEANING\nMissing values (median)\nPrice per sq ft", NAVY)
box(ax, 7.1, 6.4, 2.6, 1.2, "3. EDA\nStats, charts,\ncorrelation", NAVY)
arrow(ax, 2.9, 7.0, 3.7, 7.0); arrow(ax, 6.3, 7.0, 7.1, 7.0)

box(ax, 7.1, 4.2, 2.6, 1.2, "4. FEATURE\nENGINEERING\nOne-hot encoding", NAVY)
arrow(ax, 8.4, 6.4, 8.4, 5.4)

box(ax, 3.7, 4.2, 2.6, 1.2, "5. TRAIN / TEST SPLIT\n80% train\n20% test", NAVY)
arrow(ax, 7.1, 4.8, 6.3, 4.8)

box(ax, 0.3, 4.2, 2.6, 1.2, "6. MODEL TRAINING\nLinear Regression\nRandom Forest", ORANGE)
arrow(ax, 3.7, 4.8, 2.9, 4.8)

box(ax, 0.3, 2.0, 2.6, 1.2, "7. EVALUATION\nR2, MAE, RMSE, MAPE", NAVY)
arrow(ax, 1.6, 4.2, 1.6, 3.2)

box(ax, 3.7, 2.0, 2.6, 1.2, "8. PREDICTION &\nINSIGHTS\nPrice estimator", TEAL)
arrow(ax, 2.9, 2.6, 3.7, 2.6)

box(ax, 7.1, 2.0, 2.6, 1.2, "9. GITHUB\nDEPLOYMENT\nCode + README + report", NAVY)
arrow(ax, 6.3, 2.6, 7.1, 2.6)

ax.add_patch(FancyBboxPatch((0.3, 0.3), 9.4, 1.0, boxstyle="round,pad=0.02,rounding_size=0.12",
                            fc=LIGHT, ec=TEAL, lw=1.2))
ax.text(5, 0.8, "Tools: Python | Pandas | NumPy | Matplotlib | Scikit-learn | Git & GitHub",
        ha="center", va="center", fontsize=9, color=NAVY, fontweight="bold")
ax.set_title("Fig 3.1  System Architecture / Workflow of the Project", fontsize=10.5, color=NAVY, pad=6)
fig.tight_layout(); fig.savefig(OUT + "d1_architecture.png", dpi=200); plt.close(fig)

# ---------- Diagram 2: mechanism of prediction ----------
fig, ax = plt.subplots(figsize=(7.0, 4.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")

feats = ["Locality", "BHK", "Area (sq ft)", "Floor", "Age (years)", "Amenities score",
         "Furnishing", "Metro distance"]
ys = [6.3 - i * 0.78 for i in range(len(feats))]
for f, y in zip(feats, ys):
    box(ax, 0.2, y - 0.28, 2.2, 0.56, f, TEAL, fs=8)
    arrow(ax, 2.4, y, 3.9, 3.5, color="#B8C0CC")

box(ax, 3.9, 2.3, 2.2, 2.4, "ML MODEL\n\nLearns the\nrelationship\nbetween features\nand price", ORANGE, fs=8.5)
arrow(ax, 6.1, 3.5, 7.4, 3.5)
box(ax, 7.4, 2.9, 2.4, 1.2, "Predicted Price\n(INR Lakh)", NAVY, fs=9)

ax.text(1.3, 0.3, "INPUT: property features", ha="center", fontsize=8.5, color=NAVY, fontweight="bold")
ax.text(5.0, 1.5, "PROCESS: fit on 80% data,\nvalidate on 20%", ha="center", fontsize=8.5,
        color=NAVY, fontweight="bold")
ax.text(8.6, 2.2, "OUTPUT: price estimate", ha="center", fontsize=8.5, color=NAVY, fontweight="bold")
ax.set_title("Fig 3.2  Working Mechanism of the Price Prediction Model", fontsize=10.5, color=NAVY, pad=6)
fig.tight_layout(); fig.savefig(OUT + "d2_mechanism.png", dpi=200); plt.close(fig)

# ---------- Diagram 3: GitHub deployment flow ----------
fig, ax = plt.subplots(figsize=(7.0, 1.9))
ax.set_xlim(0, 10); ax.set_ylim(0, 2.6); ax.axis("off")
steps = ["Local project\nfolder", "git init\ngit add .", "git commit\n-m message", "Create repo\non GitHub",
         "git push\norigin main"]
w = 1.7
for i, s in enumerate(steps):
    x = 0.1 + i * 2.0
    box(ax, x, 0.7, w, 1.2, s, NAVY if i % 2 == 0 else TEAL, fs=8)
    if i < len(steps) - 1:
        arrow(ax, x + w, 1.3, x + 2.0, 1.3)
ax.set_title("Fig 6.1  GitHub Deployment Flow", fontsize=10.5, color=NAVY, pad=2)
fig.tight_layout(); fig.savefig(OUT + "d3_github.png", dpi=200); plt.close(fig)
print("diagrams done")
