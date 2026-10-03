import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from config import LINE_GAIN


class LiveOverlay(QWidget):
    """Transparent desktop overlay for FOE, tracked points, and flow lines."""

    def __init__(self):
        self.app = QApplication.instance() or QApplication([])
        super().__init__()
        self._foe = None
        self._old_points = np.empty((0, 2), dtype=np.float32)
        self._new_points = np.empty((0, 2), dtype=np.float32)
        self._source_size = (1, 1)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        screen = QGuiApplication.primaryScreen()
        self.setGeometry(screen.geometry())
        self.show()
        self.raise_()

    def update_data(self, foe, old_points, new_points, source_size):
        self._foe = foe
        self._old_points = old_points.copy()
        self._new_points = new_points.copy()
        self._source_size = source_size
        self.update()
        self.app.processEvents()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        scale_x = self.width() / self._source_size[0]
        scale_y = self.height() / self._source_size[1]

        flow_pen = QPen(QColor(0, 255, 0, 220), 2)
        painter.setPen(flow_pen)
        for old, new in zip(self._old_points, self._new_points):
            end = old + (new - old) * LINE_GAIN
            painter.drawLine(
                int(old[0] * scale_x),
                int(old[1] * scale_y),
                int(end[0] * scale_x),
                int(end[1] * scale_y),
            )

        point_pen = QPen(QColor(255, 0, 0, 230), 6)
        painter.setPen(point_pen)
        for point in self._new_points:
            painter.drawPoint(int(point[0] * scale_x), int(point[1] * scale_y))

        if self._foe is not None:
            x, y = self._foe
            x *= scale_x
            y *= scale_y
            foe_pen = QPen(QColor(255, 255, 255, 255), 3)
            painter.setPen(foe_pen)
            painter.drawLine(int(x - 12), int(y), int(x + 12), int(y))
            painter.drawLine(int(x), int(y - 12), int(x), int(y + 12))

    def close_overlay(self):
        self.close()
