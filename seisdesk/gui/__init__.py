"""Qt (PySide6 + pyqtgraph) front end of SeisDesk."""

import os

# pyqtgraph must use the same Qt binding as the rest of the GUI, whatever else is installed
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
