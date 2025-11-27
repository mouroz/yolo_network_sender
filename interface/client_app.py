#!/usr/bin/env python3
"""
YOLO Defect Detection Client Interface
A PyQt6 GUI for the AsyncSender client with batch operations.
"""

import sys
import os
import asyncio
from pathlib import Path
from datetime import datetime
from functools import partial
import traceback

# Setup paths
FILE = Path(__file__).resolve()
INTERFACE_DIR = FILE.parent
ROOT = FILE.parents[1]  # yolo_network_sender
COMM_DIR = ROOT / 'communication_submodule'

if str(COMM_DIR) not in sys.path:
    sys.path.insert(0, str(COMM_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QProgressBar,
    QStackedWidget, QListWidget, QListWidgetItem, QStatusBar, QFrame,
    QSplitter, QScrollArea, QComboBox, QGroupBox, QGridLayout,
    QFileDialog, QMessageBox, QSpinBox, QDialog, QDialogButtonBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QFont, QColor, QTextCursor

# Import the sender modules
from async_communication import AsyncSender
import sender as sender_module
import image_input as img


# ══════════════════════════════════════════════════════════════════════════════
# STYLE CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

STYLE_SHEET = """
QMainWindow {
    background-color: #1a1b26;
}

QWidget {
    background-color: #1a1b26;
    color: #a9b1d6;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 13px;
}

QGroupBox {
    border: 1px solid #3b4261;
    border-radius: 8px;
    margin-top: 12px;
    padding: 16px;
    background-color: #24283b;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 16px;
    padding: 0 8px;
    color: #7aa2f7;
    font-weight: bold;
    font-size: 14px;
}

QLabel {
    color: #a9b1d6;
    background: transparent;
}

QLineEdit, QSpinBox {
    background-color: #1f2335;
    border: 1px solid #3b4261;
    border-radius: 6px;
    padding: 10px 14px;
    color: #a9b1d6;
    selection-background-color: #7aa2f7;
}

QLineEdit:focus, QSpinBox:focus {
    border-color: #7aa2f7;
    background-color: #24283b;
}

QSpinBox::up-button, QSpinBox::down-button {
    background-color: #3b4261;
    border: none;
    width: 20px;
}

QPushButton {
    background-color: #7aa2f7;
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
    color: #1a1b26;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #89b4fa;
}

QPushButton:pressed {
    background-color: #7aa2f7;
}

QPushButton:disabled {
    background-color: #3b4261;
    color: #565f89;
}

QPushButton#secondary {
    background-color: #3b4261;
    border: 1px solid #565f89;
    color: #a9b1d6;
}

QPushButton#secondary:hover {
    background-color: #414868;
    border-color: #7aa2f7;
}

QPushButton#success {
    background-color: #9ece6a;
    color: #1a1b26;
}

QPushButton#success:hover {
    background-color: #a6e070;
}

QPushButton#warning {
    background-color: #e0af68;
    color: #1a1b26;
}

QPushButton#warning:hover {
    background-color: #e8b96e;
}

QPushButton#danger {
    background-color: #f7768e;
    color: #1a1b26;
}

QPushButton#danger:hover {
    background-color: #ff8fa0;
}

QPushButton#operation {
    background-color: #3b4261;
    color: #a9b1d6;
    text-align: left;
    padding: 14px 18px;
    border-radius: 8px;
}

QPushButton#operation:hover {
    background-color: #414868;
    border: 1px solid #7aa2f7;
}

QTextEdit {
    background-color: #1f2335;
    border: 1px solid #3b4261;
    border-radius: 8px;
    padding: 12px;
    color: #a9b1d6;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}

QListWidget {
    background-color: #1f2335;
    border: 1px solid #3b4261;
    border-radius: 8px;
    padding: 4px;
}

QListWidget::item {
    padding: 10px;
    border-radius: 4px;
    margin: 2px;
}

QListWidget::item:selected {
    background-color: #7aa2f740;
}

QListWidget::item:hover {
    background-color: #3b4261;
}

QComboBox {
    background-color: #1f2335;
    border: 1px solid #3b4261;
    border-radius: 6px;
    padding: 8px 14px;
    color: #a9b1d6;
    min-width: 150px;
}

QComboBox:hover {
    border-color: #7aa2f7;
}

QComboBox::drop-down {
    border: none;
    width: 30px;
}

QComboBox QAbstractItemView {
    background-color: #1f2335;
    border: 1px solid #3b4261;
    selection-background-color: #7aa2f7;
}

QProgressBar {
    background-color: #1f2335;
    border: 1px solid #3b4261;
    border-radius: 6px;
    height: 24px;
    text-align: center;
    color: #a9b1d6;
}

QProgressBar::chunk {
    background-color: #7aa2f7;
    border-radius: 4px;
}

QScrollBar:vertical {
    background-color: #1f2335;
    width: 10px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background-color: #3b4261;
    border-radius: 5px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #565f89;
}

QStatusBar {
    background-color: #24283b;
    border-top: 1px solid #3b4261;
    color: #565f89;
    padding: 4px;
}

QFrame#infoCard {
    background-color: #24283b;
    border: 1px solid #3b4261;
    border-radius: 12px;
    padding: 16px;
}

QLabel#pathLabel {
    color: #7aa2f7;
    font-size: 12px;
}

QLabel#countLabel {
    color: #9ece6a;
    font-size: 16px;
    font-weight: bold;
}

QDialog {
    background-color: #1a1b26;
}
"""


# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENT ASYNC LOOP THREAD
# ══════════════════════════════════════════════════════════════════════════════

class AsyncLoopThread(QThread):
    """
    Persistent thread running an asyncio event loop.
    All async operations should be submitted to this loop.
    """
    
    task_finished = pyqtSignal(str, bool, str)  # task_id, success, message
    
    def __init__(self):
        super().__init__()
        self.loop = None
        self._running = False
        self._ready = False
        
    def run(self):
        """Run the event loop forever until stopped."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._running = True
        self._ready = True
        
        try:
            self.loop.run_forever()
        finally:
            # Clean up pending tasks
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()
            
            self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self.loop.close()
            self._running = False
            
    def stop(self):
        """Stop the event loop."""
        if self.loop and self._running:
            self.loop.call_soon_threadsafe(self.loop.stop)
            
    def is_ready(self):
        """Check if the loop is ready to accept tasks."""
        return self._ready and self._running and self.loop is not None
            
    def run_coroutine(self, coro, task_id: str = "default"):
        """
        Schedule a coroutine to run on the event loop.
        Results are emitted via task_finished signal.
        """
        if not self.is_ready():
            self.task_finished.emit(task_id, False, "Event loop not ready")
            return
            
        async def wrapper():
            try:
                result = await coro
                # Use call_soon_threadsafe to emit signal from the correct thread
                self.task_finished.emit(task_id, True, str(result) if result else "Operação concluída")
            except Exception as e:
                traceback.print_exc()
                self.task_finished.emit(task_id, False, str(e))
                
        future = asyncio.run_coroutine_threadsafe(wrapper(), self.loop)
        return future


# ══════════════════════════════════════════════════════════════════════════════
# RECONNECT DIALOG
# ══════════════════════════════════════════════════════════════════════════════

class ReconnectDialog(QDialog):
    """Dialog for reconnecting to a different server."""
    
    def __init__(self, current_host: str, current_port: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reconectar")
        self.setMinimumWidth(350)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        
        # Host
        host_layout = QHBoxLayout()
        host_label = QLabel("Host:")
        self.host_input = QLineEdit(current_host)
        host_layout.addWidget(host_label)
        host_layout.addWidget(self.host_input)
        
        # Port
        port_layout = QHBoxLayout()
        port_label = QLabel("Porta:")
        self.port_input = QLineEdit(str(current_port))
        port_layout.addWidget(port_label)
        port_layout.addWidget(self.port_input)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        
        layout.addLayout(host_layout)
        layout.addLayout(port_layout)
        layout.addWidget(buttons)
        
    def get_values(self):
        return self.host_input.text().strip(), self.port_input.text().strip()


# ══════════════════════════════════════════════════════════════════════════════
# LOG WIDGET
# ══════════════════════════════════════════════════════════════════════════════

class LogWidget(QTextEdit):
    """Read-only log display widget with auto-scroll."""
    
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setMinimumHeight(150)
        
    def log(self, message: str, level: str = "info"):
        """Add a log message with timestamp and color."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        colors = {
            "info": "#a9b1d6",
            "success": "#9ece6a",
            "warning": "#e0af68",
            "error": "#f7768e",
            "debug": "#565f89"
        }
        
        icons = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "debug": "🔍"
        }
        
        color = colors.get(level, colors["info"])
        icon = icons.get(level, icons["info"])
        
        html = f'<span style="color: #565f89;">[{timestamp}]</span> ' \
               f'{icon} <span style="color: {color};">{message}</span><br>'
        
        self.moveCursor(QTextCursor.MoveOperation.End)
        self.insertHtml(html)
        self.moveCursor(QTextCursor.MoveOperation.End)
        
    def clear_log(self):
        self.clear()
        self.log("Log limpo", "debug")


# ══════════════════════════════════════════════════════════════════════════════
# CONNECTION FORM WIDGET
# ══════════════════════════════════════════════════════════════════════════════

class ConnectionForm(QWidget):
    """Initial connection form for client configuration."""
    
    connect_requested = pyqtSignal(str, int, int, int, int)  # host, port, folder_batch, yolo_batch, workers
    
    def __init__(self):
        super().__init__()
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)
        
        # Logo/Title area
        title_container = QWidget()
        title_layout = QVBoxLayout(title_container)
        title_layout.setSpacing(8)
        
        title = QLabel("📡 YOLO Defect Client")
        title.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #7aa2f7;
            background: transparent;
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        subtitle = QLabel("Cliente de Envio de Imagens")
        subtitle.setStyleSheet("""
            font-size: 14px;
            color: #565f89;
            background: transparent;
        """)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        
        # Connection form
        conn_group = QGroupBox("Conexão com Servidor")
        conn_layout = QGridLayout(conn_group)
        conn_layout.setSpacing(16)
        conn_layout.setContentsMargins(24, 24, 24, 24)
        
        # Host field
        host_label = QLabel("IP do Servidor:")
        self.host_input = QLineEdit("127.0.0.1")
        self.host_input.setPlaceholderText("Ex: 192.168.1.100")
        self.host_input.setMinimumWidth(280)
        
        # Port field
        port_label = QLabel("Porta:")
        self.port_input = QLineEdit("8888")
        self.port_input.setPlaceholderText("Ex: 8888")
        
        conn_layout.addWidget(host_label, 0, 0)
        conn_layout.addWidget(self.host_input, 0, 1)
        conn_layout.addWidget(port_label, 1, 0)
        conn_layout.addWidget(self.port_input, 1, 1)
        
        # Advanced settings
        adv_group = QGroupBox("Configurações Avançadas")
        adv_layout = QGridLayout(adv_group)
        adv_layout.setSpacing(16)
        adv_layout.setContentsMargins(24, 24, 24, 24)
        
        # Folder batch size
        folder_batch_label = QLabel("Batch de Pasta:")
        self.folder_batch_spin = QSpinBox()
        self.folder_batch_spin.setRange(1, 100)
        self.folder_batch_spin.setValue(4)
        self.folder_batch_spin.setToolTip("Mínimo de imagens antes de enviar automaticamente")
        
        # YOLO batch size
        yolo_batch_label = QLabel("Batch YOLO:")
        self.yolo_batch_spin = QSpinBox()
        self.yolo_batch_spin.setRange(1, 32)
        self.yolo_batch_spin.setValue(4)
        self.yolo_batch_spin.setToolTip("Tamanho do batch para detecção")
        
        # Workers
        workers_label = QLabel("Workers:")
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 8)
        self.workers_spin.setValue(2)
        self.workers_spin.setToolTip("Número de threads para processamento")
        
        adv_layout.addWidget(folder_batch_label, 0, 0)
        adv_layout.addWidget(self.folder_batch_spin, 0, 1)
        adv_layout.addWidget(yolo_batch_label, 1, 0)
        adv_layout.addWidget(self.yolo_batch_spin, 1, 1)
        adv_layout.addWidget(workers_label, 2, 0)
        adv_layout.addWidget(self.workers_spin, 2, 1)
        
        # Connect button
        self.connect_btn = QPushButton("🔗  Conectar ao Servidor")
        self.connect_btn.setMinimumHeight(48)
        self.connect_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.connect_btn.clicked.connect(self._on_connect)
        
        # Error label
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #f7768e; background: transparent;")
        self.error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Add all to main layout
        layout.addStretch()
        layout.addWidget(title_container)
        layout.addSpacing(20)
        layout.addWidget(conn_group)
        layout.addWidget(adv_group)
        layout.addWidget(self.connect_btn)
        layout.addWidget(self.error_label)
        layout.addStretch()
        
        # Set fixed width for forms
        conn_group.setMaximumWidth(450)
        adv_group.setMaximumWidth(450)
        self.connect_btn.setMaximumWidth(450)
        
    def _on_connect(self):
        """Validate and emit connection request."""
        host = self.host_input.text().strip()
        port_str = self.port_input.text().strip()
        
        if not host:
            self.error_label.setText("❌ IP não pode estar vazio")
            return
            
        if not port_str.isdigit():
            self.error_label.setText("❌ Porta deve ser um número")
            return
            
        port = int(port_str)
        if port < 1 or port > 65535:
            self.error_label.setText("❌ Porta deve estar entre 1 e 65535")
            return
            
        self.error_label.setText("")
        self.connect_btn.setEnabled(False)
        self.connect_btn.setText("Conectando...")
        
        self.connect_requested.emit(
            host, 
            port,
            self.folder_batch_spin.value(),
            self.yolo_batch_spin.value(),
            self.workers_spin.value()
        )
        
    def reset(self):
        """Reset form state."""
        self.connect_btn.setEnabled(True)
        self.connect_btn.setText("🔗  Conectar ao Servidor")
        self.error_label.setText("")
        
    def set_error(self, message: str):
        """Display error message."""
        self.error_label.setText(f"❌ {message}")
        self.reset()


# ══════════════════════════════════════════════════════════════════════════════
# OPERATIONS PANEL
# ══════════════════════════════════════════════════════════════════════════════

class OperationsPanel(QWidget):
    """Panel with operation buttons replacing terminal commands."""
    
    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
        self.selected_folder = None
        self._setup_ui()
        
    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(16, 16, 16, 16)
        
        # ─── LEFT: Folder Selection & Operations ──────────────────────────────
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(12)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Folder selection group
        folder_group = QGroupBox("Seleção de Pasta de Imagens")
        folder_layout = QVBoxLayout(folder_group)
        
        folder_btn_layout = QHBoxLayout()
        self.select_folder_btn = QPushButton("📁  Selecionar Pasta")
        self.select_folder_btn.setObjectName("secondary")
        self.select_folder_btn.clicked.connect(self._select_folder)
        folder_btn_layout.addWidget(self.select_folder_btn)
        folder_btn_layout.addStretch()
        
        self.folder_path_label = QLabel("Nenhuma pasta selecionada")
        self.folder_path_label.setObjectName("pathLabel")
        self.folder_path_label.setWordWrap(True)
        
        self.image_count_label = QLabel("")
        self.image_count_label.setObjectName("countLabel")
        
        folder_layout.addLayout(folder_btn_layout)
        folder_layout.addWidget(self.folder_path_label)
        folder_layout.addWidget(self.image_count_label)
        
        # Operations group
        ops_group = QGroupBox("Operações")
        ops_layout = QVBoxLayout(ops_group)
        ops_layout.setSpacing(8)
        
        # Operation buttons matching terminal options
        self.btn_reconnect = QPushButton("🔄  1. Reconectar")
        self.btn_reconnect.setObjectName("operation")
        self.btn_reconnect.clicked.connect(self._on_reconnect)
        
        self.btn_add_image = QPushButton("🖼️  2. Adicionar Imagem")
        self.btn_add_image.setObjectName("operation")
        self.btn_add_image.clicked.connect(self._on_add_image)
        
        self.btn_add_folder = QPushButton("📂  3. Adicionar Pasta")
        self.btn_add_folder.setObjectName("operation")
        self.btn_add_folder.clicked.connect(self._on_add_folder)
        
        self.btn_force_run = QPushButton("🚀  4. Forçar Detecção")
        self.btn_force_run.setObjectName("operation")
        self.btn_force_run.setStyleSheet("""
            QPushButton#operation {
                background-color: #9ece6a30;
                border: 1px solid #9ece6a;
            }
            QPushButton#operation:hover {
                background-color: #9ece6a50;
            }
        """)
        self.btn_force_run.clicked.connect(self._on_force_run)
        
        self.btn_retry_pending = QPushButton("🔁  5. Reenviar Pendentes")
        self.btn_retry_pending.setObjectName("operation")
        self.btn_retry_pending.clicked.connect(self._on_retry_pending)
        
        self.btn_list_runs = QPushButton("📋  6. Listar Runs")
        self.btn_list_runs.setObjectName("operation")
        self.btn_list_runs.clicked.connect(self._on_list_runs)
        
        self.btn_resend_run = QPushButton("📤  7. Reenviar Run")
        self.btn_resend_run.setObjectName("operation")
        self.btn_resend_run.clicked.connect(self._on_resend_run)
        
        for btn in [self.btn_reconnect, self.btn_add_image, self.btn_add_folder,
                    self.btn_force_run, self.btn_retry_pending, 
                    self.btn_list_runs, self.btn_resend_run]:
            ops_layout.addWidget(btn)
            
        # Current run info
        info_group = QGroupBox("Informações")
        info_layout = QVBoxLayout(info_group)
        
        self.current_run_label = QLabel("Run atual: --")
        self.buffer_count_label = QLabel("Imagens no buffer: 0")
        self.pending_count_label = QLabel("Transferências pendentes: 0")
        
        info_layout.addWidget(self.current_run_label)
        info_layout.addWidget(self.buffer_count_label)
        info_layout.addWidget(self.pending_count_label)
        
        left_layout.addWidget(folder_group)
        left_layout.addWidget(ops_group)
        left_layout.addWidget(info_group)
        left_layout.addStretch()
        
        # ─── CENTER: Runs List ────────────────────────────────────────────────
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        
        runs_group = QGroupBox("Runs Locais")
        runs_layout = QVBoxLayout(runs_group)
        
        self.runs_list = QListWidget()
        self.runs_list.itemDoubleClicked.connect(self._on_run_double_click)
        
        refresh_runs_btn = QPushButton("🔄  Atualizar Lista")
        refresh_runs_btn.setObjectName("secondary")
        refresh_runs_btn.clicked.connect(self._refresh_runs_list)
        
        runs_layout.addWidget(self.runs_list)
        runs_layout.addWidget(refresh_runs_btn)
        
        center_layout.addWidget(runs_group)
        
        # ─── RIGHT: Log Panel ─────────────────────────────────────────────────
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        log_group = QGroupBox("Log de Operações")
        log_layout = QVBoxLayout(log_group)
        
        self.log_widget = LogWidget()
        
        log_btn_layout = QHBoxLayout()
        clear_log_btn = QPushButton("🗑️  Limpar")
        clear_log_btn.setObjectName("secondary")
        clear_log_btn.clicked.connect(self.log_widget.clear_log)
        log_btn_layout.addStretch()
        log_btn_layout.addWidget(clear_log_btn)
        
        log_layout.addWidget(self.log_widget, 1)
        log_layout.addLayout(log_btn_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        log_layout.addWidget(self.progress_bar)
        
        right_layout.addWidget(log_group)
        
        # ─── Add to splitter ──────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 280, 450])
        
        main_layout.addWidget(splitter)
        
    def _select_folder(self):
        """Open folder selection dialog."""
        folder = QFileDialog.getExistingDirectory(
            self, 
            "Selecionar Pasta de Imagens",
            str(ROOT / "defect-free")
        )
        
        if folder:
            self.selected_folder = Path(folder)
            self.folder_path_label.setText(str(self.selected_folder))
            
            # Count images
            image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            count = sum(1 for f in self.selected_folder.iterdir() 
                       if f.is_file() and f.suffix.lower() in image_exts)
            
            self.image_count_label.setText(f"📷 {count} imagens encontradas")
            self.log_widget.log(f"Pasta selecionada: {folder} ({count} imagens)", "info")
            
    def _refresh_runs_list(self):
        """Refresh the list of local runs."""
        self.runs_list.clear()
        
        yolo_in = ROOT / "yolo_in"
        yolo_out = ROOT / "yolo_out"
        
        if not yolo_in.exists() and not yolo_out.exists():
            self.log_widget.log("Nenhuma pasta de runs encontrada", "warning")
            return
            
        runs = set()
        
        if yolo_in.exists():
            for d in yolo_in.iterdir():
                if d.is_dir() and d.name.startswith('run'):
                    runs.add(d.name)
                    
        if yolo_out.exists():
            for d in yolo_out.iterdir():
                if d.is_dir() and d.name.startswith('run'):
                    runs.add(d.name)
                    
        # Sort runs by number
        sorted_runs = sorted(runs, key=lambda x: int(x.replace('run', '')) 
                            if x.replace('run', '').isdigit() else 0, reverse=True)
        
        for run_name in sorted_runs:
            in_count = 0
            out_count = 0
            
            in_path = yolo_in / run_name
            out_path = yolo_out / run_name
            
            if in_path.exists():
                in_count = img.get_image_count(in_path)
            if out_path.exists():
                out_count = img.get_image_count(out_path)
                
            item = QListWidgetItem(f"📁 {run_name}  |  In: {in_count}  |  Out: {out_count}")
            item.setData(Qt.ItemDataRole.UserRole, run_name)
            self.runs_list.addItem(item)
            
        self.log_widget.log(f"Encontrados {len(runs)} runs locais", "info")
        
    def update_info(self, manager):
        """Update info labels from manager state."""
        if manager:
            self.current_run_label.setText(f"Run atual: {manager.current_run_name}")
            buffer_count = img.get_image_count(manager.current_folder) if manager.current_folder.exists() else 0
            self.buffer_count_label.setText(f"Imagens no buffer: {buffer_count}")
            self.pending_count_label.setText(f"Transferências pendentes: {len(manager.transfer_queue)}")
            
    # ─── Operation handlers ───────────────────────────────────────────────────
    
    def _on_reconnect(self):
        """Handle reconnect button click."""
        self.parent_window._show_reconnect_dialog()
        
    def _on_add_image(self):
        """Handle add single image button click."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar Imagem",
            str(ROOT / "defect-free"),
            "Images (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )
        
        if file_path:
            self.parent_window._add_image(file_path)
            
    def _on_add_folder(self):
        """Handle add folder button click."""
        if not self.selected_folder:
            self.log_widget.log("Selecione uma pasta primeiro!", "warning")
            return
            
        self.parent_window._add_folder(str(self.selected_folder))
        
    def _on_force_run(self):
        """Handle force detection button click."""
        self.parent_window._force_run()
        
    def _on_retry_pending(self):
        """Handle retry pending transfers button click."""
        self.parent_window._retry_pending()
        
    def _on_list_runs(self):
        """Handle list runs button click."""
        self._refresh_runs_list()
        
    def _on_resend_run(self):
        """Handle resend run button click."""
        selected = self.runs_list.selectedItems()
        if not selected:
            self.log_widget.log("Selecione um run na lista primeiro!", "warning")
            return
            
        run_name = selected[0].data(Qt.ItemDataRole.UserRole)
        self.parent_window._resend_run(run_name)
        
    def _on_run_double_click(self, item):
        """Handle double-click on run to resend."""
        run_name = item.data(Qt.ItemDataRole.UserRole)
        reply = QMessageBox.question(
            self,
            "Reenviar Run",
            f"Deseja reenviar o run '{run_name}' para o servidor?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.parent_window._resend_run(run_name)
            
    def set_operations_enabled(self, enabled: bool):
        """Enable or disable operation buttons."""
        for btn in [self.btn_add_image, self.btn_add_folder, self.btn_force_run,
                    self.btn_retry_pending, self.btn_resend_run]:
            btn.setEnabled(enabled)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class ClientMainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.sender = None
        self.manager = None
        self.async_loop = None
        self._pending_task = None
        self._connection_params = None
        
        self._setup_window()
        self._setup_ui()
        self._setup_async_loop()
        
    def _setup_window(self):
        """Configure main window properties."""
        self.setWindowTitle("YOLO Defect Detection Client")
        self.setMinimumSize(1280, 800)
        self.resize(1400, 900)
        
    def _setup_async_loop(self):
        """Setup the persistent async event loop thread."""
        self.async_loop = AsyncLoopThread()
        self.async_loop.task_finished.connect(self._on_task_finished)
        self.async_loop.start()
        
        # Wait for loop to be ready
        QTimer.singleShot(100, self._check_loop_ready)
        
    def _check_loop_ready(self):
        """Check if async loop is ready."""
        if self.async_loop and self.async_loop.is_ready():
            print("Async loop ready")
        else:
            QTimer.singleShot(100, self._check_loop_ready)
        
    def _setup_ui(self):
        """Setup the main UI structure."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Stacked widget for form/operations
        self.stack = QStackedWidget()
        
        # Connection form
        self.connection_form = ConnectionForm()
        self.connection_form.connect_requested.connect(self._connect_to_server)
        
        # Operations panel
        self.operations_panel = OperationsPanel(self)
        
        self.stack.addWidget(self.connection_form)
        self.stack.addWidget(self.operations_panel)
        
        layout.addWidget(self.stack)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Aguardando conexão...")
        
        # Disconnect button (hidden initially)
        self.disconnect_btn = QPushButton("⏹  Desconectar")
        self.disconnect_btn.setObjectName("danger")
        self.disconnect_btn.clicked.connect(self._disconnect)
        self.disconnect_btn.hide()
        self.status_bar.addPermanentWidget(self.disconnect_btn)
        
    def _connect_to_server(self, host: str, port: int, folder_batch: int, 
                           yolo_batch: int, workers: int):
        """Connect to the server."""
        self.sender = AsyncSender(host, port)
        self._connection_params = (host, port, folder_batch, yolo_batch, workers)
        
        # Run connection on the async loop
        self.async_loop.run_coroutine(self.sender.connect(), "connect")
        
    def _on_task_finished(self, task_id: str, success: bool, message: str):
        """Handle async task completion."""
        if task_id == "connect":
            self._on_connected(success, message)
        elif task_id == "disconnect":
            self._on_disconnected()
        elif task_id == "reconnect":
            self._on_reconnected(success, message)
        elif task_id == "add_image":
            self._on_operation_finished(success, message, "Adicionar imagem")
        elif task_id == "add_folder":
            self._on_operation_finished(success, message, "Adicionar pasta")
        elif task_id == "force_run":
            self._on_operation_finished(success, message, "Forçar detecção")
        elif task_id.startswith("resend_"):
            run_name = task_id.replace("resend_", "")
            self._on_operation_finished(success, message, f"Reenviar {run_name}")
        
    def _on_connected(self, success: bool, message: str):
        """Handle connection result."""
        if success and self._connection_params:
            host, port, folder_batch, yolo_batch, workers = self._connection_params
            
            # Create batch manager
            self.manager = sender_module.BatchManager(
                sender=self.sender,
                batch_size=yolo_batch,
                workers=workers,
                transfer_batch_size=folder_batch
            )
            
            self.stack.setCurrentIndex(1)  # Switch to operations
            self.status_bar.showMessage(f"✅ Conectado a {host}:{port}")
            self.disconnect_btn.show()
            
            self.operations_panel.log_widget.log(
                f"Conectado ao servidor {host}:{port}", "success"
            )
            self.operations_panel.update_info(self.manager)
            self.operations_panel._refresh_runs_list()
            
            # Setup periodic info update
            self.info_timer = QTimer()
            self.info_timer.timeout.connect(
                lambda: self.operations_panel.update_info(self.manager)
            )
            self.info_timer.start(2000)
        else:
            self.connection_form.set_error(message)
            self.sender = None
            
    def _on_disconnected(self):
        """Handle disconnect completion."""
        self.sender = None
        self.manager = None
        self.stack.setCurrentIndex(0)
        self.connection_form.reset()
        self.disconnect_btn.hide()
        self.status_bar.showMessage("Desconectado")
            
    def _disconnect(self):
        """Disconnect from server."""
        if hasattr(self, 'info_timer'):
            self.info_timer.stop()
            
        if self.sender and self.async_loop and self.async_loop.is_ready():
            self.async_loop.run_coroutine(self.sender.disconnect(), "disconnect")
        else:
            self._on_disconnected()
        
    def _show_reconnect_dialog(self):
        """Show reconnect dialog."""
        if not self.sender:
            return
            
        dialog = ReconnectDialog(self.sender.host, self.sender.port, self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_host, new_port = dialog.get_values()
            
            if new_port.isdigit():
                self._do_reconnect(new_host, int(new_port))
            else:
                self.operations_panel.log_widget.log("Porta inválida!", "error")
                
    def _do_reconnect(self, host: str, port: int):
        """Perform reconnection."""
        self.operations_panel.log_widget.log(f"Reconectando a {host}:{port}...", "info")
        self.operations_panel.set_operations_enabled(False)
        
        async def reconnect():
            await self.sender.disconnect()
            self.sender.host = host
            self.sender.port = port
            await self.sender.connect()
            
        self.async_loop.run_coroutine(reconnect(), "reconnect")
        
    def _on_reconnected(self, success: bool, message: str):
        """Handle reconnection result."""
        self.operations_panel.set_operations_enabled(True)
        if success:
            self.operations_panel.log_widget.log(
                f"Reconectado a {self.sender.host}:{self.sender.port}", "success"
            )
            self.status_bar.showMessage(f"✅ Conectado a {self.sender.host}:{self.sender.port}")
        else:
            self.operations_panel.log_widget.log(f"Falha ao reconectar: {message}", "error")
        
    def _add_image(self, path: str):
        """Add a single image to the batch."""
        if not self.manager or not self.async_loop.is_ready():
            return
            
        self.operations_panel.log_widget.log(f"Adicionando imagem: {Path(path).name}", "info")
        self.operations_panel.set_operations_enabled(False)
        
        self.async_loop.run_coroutine(self.manager.async_add_file(path), "add_image")
        
    def _add_folder(self, path: str):
        """Add all images from a folder to the batch."""
        if not self.manager or not self.async_loop.is_ready():
            return
            
        self.operations_panel.log_widget.log(f"Adicionando pasta: {path}", "info")
        self.operations_panel.set_operations_enabled(False)
        self.operations_panel.progress_bar.setVisible(True)
        self.operations_panel.progress_bar.setRange(0, 0)  # Indeterminate
        
        self.async_loop.run_coroutine(self.manager.async_add_folder(path), "add_folder")
        
    def _force_run(self):
        """Force detection run."""
        if not self.manager or not self.async_loop.is_ready():
            return
            
        self.operations_panel.log_widget.log("Forçando detecção...", "info")
        self.operations_panel.set_operations_enabled(False)
        self.operations_panel.progress_bar.setVisible(True)
        self.operations_panel.progress_bar.setRange(0, 0)
        
        self.async_loop.run_coroutine(self.manager.async_force_run(), "force_run")
        
    def _retry_pending(self):
        """Retry pending transfers."""
        if not self.manager:
            return
            
        if not self.manager.transfer_queue:
            self.operations_panel.log_widget.log("Fila de pendentes está vazia", "info")
            return
            
        self.operations_panel.log_widget.log(
            f"Reenviando {len(self.manager.transfer_queue)} transferências pendentes...", 
            "info"
        )
        
        # The resend_pending_transfer is synchronous in current implementation
        sender_module.BatchManager.resend_pending_transfer(self.manager.transfer_queue)
        self.operations_panel.log_widget.log("Tentativa de reenvio concluída", "success")
        
    def _resend_run(self, run_name: str):
        """Resend a specific run folder."""
        if not self.sender or not self.async_loop.is_ready():
            return
            
        yolo_out = ROOT / "yolo_out" / run_name
        yolo_in = ROOT / "yolo_in" / run_name
        
        # Check which folders exist
        folders_to_send = []
        if yolo_in.exists():
            folders_to_send.append(str(yolo_in))
        if yolo_out.exists():
            folders_to_send.append(str(yolo_out))
            
        if not folders_to_send:
            self.operations_panel.log_widget.log(f"Run '{run_name}' não encontrado!", "error")
            return
            
        self.operations_panel.log_widget.log(f"Reenviando run: {run_name}", "info")
        self.operations_panel.set_operations_enabled(False)
        self.operations_panel.progress_bar.setVisible(True)
        self.operations_panel.progress_bar.setRange(0, 0)
        
        async def send_folders():
            for folder in folders_to_send:
                await self.sender.send_folder_recursive(folder)
                
        self.async_loop.run_coroutine(send_folders(), f"resend_{run_name}")
        
    def _on_operation_finished(self, success: bool, message: str, operation: str):
        """Handle operation completion."""
        self.operations_panel.set_operations_enabled(True)
        self.operations_panel.progress_bar.setVisible(False)
        self.operations_panel.progress_bar.setRange(0, 100)
        
        if success:
            self.operations_panel.log_widget.log(f"{operation}: Concluído", "success")
        else:
            self.operations_panel.log_widget.log(f"{operation}: Falhou - {message}", "error")
            
            # Add to pending queue if it was a transfer failure
            if self.manager and "transfer" in operation.lower():
                self.operations_panel.log_widget.log(
                    "Adicionado à fila de pendentes", "warning"
                )
                
        self.operations_panel.update_info(self.manager)
        self.operations_panel._refresh_runs_list()
        
    def closeEvent(self, event):
        """Clean up on window close."""
        # Stop the async loop thread
        if self.async_loop:
            self.async_loop.stop()
            self.async_loop.wait(2000)
            
        if hasattr(self, 'info_timer'):
            self.info_timer.stop()
            
        event.accept()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE_SHEET)
    
    # Set application properties
    app.setApplicationName("YOLO Defect Client")
    app.setOrganizationName("TI6")
    
    window = ClientMainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

