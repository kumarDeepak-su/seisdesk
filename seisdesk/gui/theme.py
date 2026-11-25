"""Application colours. Scientific canvases keep their own amplitude colour maps."""

import os
from pathlib import Path
from PySide6 import QtGui, QtWidgets


def apply_theme(name: str) -> str:
    name = name if name in ("dark", "light") else "light"
    dark = name == "dark"
    bg, panel, field = ("#111923", "#192430", "#101923") if dark else ("#eef2f6", "#ffffff", "#f5f7fa")
    text, muted, border = ("#e7edf5", "#a5b6c8", "#334456") if dark else ("#182b40", "#53677b", "#cbd5e1")
    hover = "#273a4f" if dark else "#e4edf6"
    accent, selected = ("#55c9dc", "#234c61") if dark else ("#006f87", "#d6edf4")
    app = QtWidgets.QApplication.instance()
    if os.name == "nt" and app.platformName() == "offscreen" and not app.property("seisdeskFontsLoaded"):
        fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        for filename in ("segoeui.ttf", "segoeuib.ttf", "seguisb.ttf", "arial.ttf", "arialbd.ttf"):
            if (fonts / filename).is_file():
                QtGui.QFontDatabase.addApplicationFont(str(fonts / filename))
        app.setProperty("seisdeskFontsLoaded", True)
    app.setFont(QtGui.QFont("Segoe UI" if os.name == "nt" else "Arial", 10))
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    for role, value in {
        "Window": bg, "WindowText": text, "Base": field, "AlternateBase": panel,
        "Text": text, "Button": panel, "ButtonText": text, "BrightText": "#ffffff",
        "Highlight": selected, "HighlightedText": text, "ToolTipBase": panel,
        "ToolTipText": text, "Link": accent, "PlaceholderText": muted,
    }.items():
        palette.setColor(getattr(QtGui.QPalette.ColorRole, role), QtGui.QColor(value))
    for role in ("Text", "ButtonText", "WindowText"):
        palette.setColor(QtGui.QPalette.ColorGroup.Disabled, getattr(QtGui.QPalette.ColorRole, role),
                         QtGui.QColor(muted))
    app.setPalette(palette)
    app.setStyleSheet(f"""
        QWidget {{ font-family: 'Segoe UI', 'Arial', sans-serif; font-size: 10pt; }}
        QMainWindow, QDialog {{ background: {bg}; }}
        QToolTip {{ color: {text}; background: {panel}; border: 1px solid {border}; padding: 6px; }}
        QToolBar {{ background: {panel}; border: none; border-bottom: 1px solid {border}; spacing: 8px; padding: 8px; }}
        QMenuBar {{ background: {panel}; color: {text}; }}
        QMenuBar::item {{ padding: 5px 9px; }}
        QMenuBar::item:selected, QMenu::item:selected {{ background: {hover}; }}
        QMenu {{ background: {panel}; border: 1px solid {border}; padding: 5px; }}
        QMenu::item {{ padding: 7px 24px; }}
        QTabWidget::pane {{ border: 1px solid {border}; border-radius: 6px; }}
        QTabBar::tab {{ background: {bg}; color: {muted}; padding: 10px 18px; border-bottom: 3px solid transparent; }}
        QTabBar::tab:selected {{ background: {panel}; color: {text}; border-bottom: 3px solid {accent}; }}
        QTabBar::tab:hover {{ background: {hover}; color: {text}; }}
        QDockWidget {{ font-weight: 600; }}
        QDockWidget::title {{ background: {panel}; padding: 9px; }}
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {{
            background: {field}; color: {text}; border: 1px solid {border}; border-radius: 5px; padding: 5px;
            selection-background-color: {selected}; selection-color: {text};
        }}
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {accent}; }}
        QComboBox::drop-down {{ width: 20px; border: none; }}
        QComboBox {{ padding-right: 24px; }}
        QPushButton, QToolButton {{ color: {text}; background: {panel}; border: 1px solid {border};
            border-radius: 5px; padding: 6px 10px; }}
        QPushButton:hover, QToolButton:hover {{ background: {hover}; border-color: {accent}; }}
        QPushButton:focus, QToolButton:focus {{ border: 1px solid {accent}; }}
        QPushButton:checked, QToolButton:checked {{ background: {selected}; border-color: {accent}; }}
        QPushButton:disabled, QToolButton:disabled {{ color: {muted}; background: {bg}; border-color: {border}; }}
        QPushButton[primary="true"] {{ background: #087e96; color: #ffffff; border: 1px solid #1596ac; font-weight: 600; }}
        QPushButton[primary="true"]:hover {{ background: #0c91aa; }}
        QPushButton[primary="true"]:disabled {{ background: {panel}; color: {muted}; border-color: {border}; }}
        QListWidget, QTreeWidget, QTableWidget {{ background: {field}; color: {text}; border: 1px solid {border};
            border-radius: 6px; outline: none; alternate-background-color: {panel}; }}
        QListWidget::item {{ padding: 10px 8px; border-radius: 4px; }}
        QTreeWidget::item {{ padding: 5px 3px; }}
        QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {{ background: {selected}; color: {text}; }}
        QListWidget::item:hover, QTreeWidget::item:hover {{ background: {hover}; }}
        QHeaderView::section {{ background: {panel}; color: {text}; padding: 6px; border: 1px solid {border}; }}
        QScrollArea {{ border: none; }}
        QGroupBox {{ border: 1px solid {border}; border-radius: 6px; margin-top: 14px; padding-top: 12px; }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 5px; }}
        QSplitter::handle {{ background: {bg}; width: 8px; height: 8px; }}
        QSplitter::handle:hover {{ background: {border}; }}
        QProgressBar {{ background: {field}; border: 1px solid {border}; border-radius: 5px; min-height: 20px;
            color: {text}; text-align: center; }}
        QProgressBar::chunk {{ background: {selected}; border-radius: 4px; }}
        QStatusBar {{ background: {panel}; color: {muted}; }}
        QLabel[muted="true"] {{ color: {muted}; }}
        QLabel[heading="true"] {{ font-size: 22pt; font-weight: 600; }}
        QLabel[section="true"] {{ color: {accent}; font-size: 10pt; font-weight: 600; }}
        QLabel[brand="true"] {{ font-size: 15pt; font-weight: 700; padding-right: 12px; }}
        QLabel[metric="true"] {{ font-size: 18pt; font-weight: 600; color: {accent}; }}
        QFrame[card="true"] {{ background: {panel}; border: 1px solid {border}; border-radius: 10px; }}
        QFrame[card="true"] QLabel {{ background: transparent; border: none; }}
    """)
    app.setProperty("seisdeskTheme", name)
    return name
