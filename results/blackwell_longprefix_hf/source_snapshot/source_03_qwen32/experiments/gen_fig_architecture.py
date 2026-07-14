"""Generate ShadowKV++ architecture diagram using matplotlib."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(6, 5.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors
C_DARK = '#2c3e50'
C_BLUE = '#3498db'
C_GREEN = '#2ecc71'
C_ORANGE = '#e67e22'
C_RED = '#e74c3c'
C_PURPLE = '#9b59b6'
C_GRAY = '#ecf0f1'
C_WHITE = '#ffffff'

def box(x, y, w, h, color, text, text_color='white', fontsize=9, ha='center'):
    rect = mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.08",
                                     facecolor=color, edgecolor='none', zorder=3)
    ax.add_patch(rect)
    ax.text(x, y, text, ha=ha, va='center', fontsize=fontsize, color=text_color,
            fontweight='bold', zorder=4)

def dashed_box(x, y, w, h, color='#bdc3c7'):
    rect = mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.15",
                                     facecolor='none', edgecolor=color, linestyle='--',
                                     linewidth=1.5, zorder=2)
    ax.add_patch(rect)

def arrow(x1, y1, x2, y2, label='', color='#7f8c8d', lw=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw), zorder=3)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my, label, ha='center', va='bottom' if y2 > y1 else 'top',
                fontsize=7, color='#555', fontstyle='italic')

def darrow(x1, y1, x2, y2, label='', color='#95a5a6', lw=1.5):
    """Dashed arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, linestyle='dashed'), zorder=3)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my, label, ha='center', va='bottom', fontsize=7, color='#555', fontstyle='italic')

# Row positions (y-coordinates)
Y_TOP = 8.5
Y_CTRL = 6.5
Y_BANK = 7.8
Y_SEM = 5.2
Y_BACK = 6.5
Y_PLAN = 4.0
Y_LEARN = 2.5

# Request
box(1.0, Y_TOP, 1.2, 0.6, C_DARK, 'Request x')

# Raw Gate
box(1.0, 7.2, 1.2, 0.6, C_ORANGE, 'Raw Gate')

# Controller dashed box
dashed_box(4.3, Y_CTRL, 4.2, 3.0, '#3498db')
ax.text(4.3, Y_CTRL + 1.35, 'Adaptive Controller', ha='center', va='center',
        fontsize=8, color=C_BLUE, fontweight='bold', style='italic')

# Controller sub-boxes
# Bypass
box(3.2, Y_CTRL + 0.7, 1.6, 0.5, C_RED, 'Bypass')
ax.text(4.4, Y_CTRL + 0.7, 'U < 0', ha='left', va='center', fontsize=6.5, color='#555')

# Exact
box(3.2, Y_CTRL - 0.2, 1.6, 0.5, C_GREEN, 'Exact')
ax.text(4.4, Y_CTRL - 0.2, 'U\u2091 \u2265 0', ha='left', va='center', fontsize=6.5, color='#555')

# Semantic
box(3.2, Y_CTRL - 1.1, 1.6, 0.5, C_ORANGE, 'Semantic')
ax.text(4.4, Y_CTRL - 1.1, '\u03c3 \u2265 0.58', ha='left', va='center', fontsize=6.5, color='#555')

# Coupling penalty note
ax.text(3.2, Y_CTRL - 1.65, '\u03bb \u00b7 \u03ba \u00b7 B \u00b7 max(\u00ea_w, 0.02)',
        ha='left', va='center', fontsize=5.5, color='#888', fontstyle='italic')

# State Bank
box(1.0, Y_BANK, 1.4, 0.6, C_GREEN, 'State\nBank')

# Semantic Index
box(1.0, Y_SEM, 1.4, 0.6, C_RED, 'Semantic\nIndex')

# Backend
box(8.0, Y_BACK, 1.4, 0.6, C_BLUE, 'Backend')

# ReusePlan
box(8.0, Y_PLAN, 1.4, 0.6, C_PURPLE, 'ReusePlan')

# Offline Learner
box(8.0, Y_LEARN, 1.4, 0.6, C_DARK, 'Offline\nLearner')

# Arrows
arrow(1.6, Y_TOP, 1.6, 7.5, 'admit?', C_DARK)
arrow(1.0, 6.9, 2.4, Y_CTRL + 0.7, '', C_DARK)  # gate to controller
arrow(1.7, Y_BANK, 2.4, Y_CTRL + 0.3, 'match', C_DARK)  # bank to controller
arrow(1.0, Y_SEM + 0.3, 2.4, Y_CTRL - 0.5, 'sim', C_DARK)  # semidx to controller
arrow(5.4, Y_CTRL, 7.3, Y_BACK, 'plan', C_DARK)  # controller to backend
arrow(8.0, Y_BACK - 0.3, 8.0, Y_PLAN + 0.3, '', C_DARK)  # backend to plan
darrow(8.0, Y_PLAN - 0.3, 8.0, Y_LEARN + 0.3, '', C_DARK)  # plan to learner
arrow(5.4, Y_CTRL - 1.1, 1.7, Y_SEM + 0.3, 'query', C_DARK)  # controller to semidx

# EWMA feedback: plan → controller (curved)
ax.annotate('', xy=(6.2, Y_CTRL - 0.2), xytext=(6.2, Y_PLAN + 0.3),
            arrowprops=dict(arrowstyle='->', color='#95a5a6', lw=1.5, linestyle='dashed'),
            zorder=3)
ax.text(6.3, (Y_PLAN + Y_CTRL)/2, '\u00ea_h, \u00ea_w', ha='left', va='center',
        fontsize=6.5, color='#555', fontstyle='italic')

# Title
plt.suptitle('ShadowKV++ Serving-Time Architecture', fontsize=13, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('v10/experiments/fig_architecture_v2.png', dpi=200, bbox_inches='tight')
print('Saved: v10/experiments/fig_architecture_v2.png')
