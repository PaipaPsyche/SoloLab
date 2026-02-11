import sys
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import numpy as np
from matplotlib.colors import LogNorm
from solo_epd_loader import epd_load
from qtpy.QtCore import QDate, QDateTime, Qt
from qtpy.QtWidgets import QListWidget, QListWidgetItem
from qtpy.QtGui import QColor, QPalette, QIcon, QPixmap
# sys.path.insert(0, '/home/dpaipa/Documents/doctorat_lesia_obspm/DoctoratLesia_EnergeticParticlesInSolarFlares/scripts/')
# import sololab as sololab

from .quicklooks import *
from .stix_read import *
from .rpw_read import *
from .values import *
 

from matplotlib.dates import DateFormatter
from qtpy.QtCore import QDate, QDateTime, Qt

"""
sololab_app.py

A simple QtPy6 application to manage importing STIX / RPW / EPD inputs and plotting
with basic plotting preferences. The GUI is minimal and demonstrates the requested
controls and behaviors.

Requirements:
- qtpy
- PySide6 or PyQt6 (qtpy will select available backend)
- matplotlib
- numpy
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QFileDialog,
    QFormLayout,
    QComboBox,
    QCheckBox,
    QTabWidget,
    QDateEdit,
    QDateTimeEdit,
    QDialogButtonBox,
    QMessageBox,
    QGroupBox,
    QRadioButton,
    QButtonGroup,
    QLayout,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QListWidget,
    QListWidgetItem,
)


class InstrumentSelectionDialog(QDialog):
    def __init__(self, parent=None, available=None, selected=None):
        super().__init__(parent)
        self.setWindowTitle("Select instruments to plot")
        self.resize(400, 300)

        self.available = available or {}
        self.selected_instruments = selected[:] if selected else []

        layout = QVBoxLayout(self)

        self.instrument_list = QListWidget()
        layout.addWidget(self.instrument_list)

        for name in self.selected_instruments:
            self.instrument_list.addItem(QListWidgetItem(name))

        buttons_layout = QHBoxLayout()

        self.add_stix_btn = QPushButton("Add STIX")
        self.add_stix_btn.clicked.connect(lambda: self._add_instrument("STIX"))
        self.add_stix_btn.setEnabled(self.available.get("STIX", False))
        buttons_layout.addWidget(self.add_stix_btn)

        self.add_hfr_btn = QPushButton("Add RPW-HFR")
        self.add_hfr_btn.clicked.connect(lambda: self._add_instrument("RPW-HFR"))
        self.add_hfr_btn.setEnabled(self.available.get("RPW-HFR", False))
        buttons_layout.addWidget(self.add_hfr_btn)

        self.add_tnr_btn = QPushButton("Add RPW-TNR")
        self.add_tnr_btn.clicked.connect(lambda: self._add_instrument("RPW-TNR"))
        self.add_tnr_btn.setEnabled(self.available.get("RPW-TNR", False))
        buttons_layout.addWidget(self.add_tnr_btn)

        self.add_epd_btn = QPushButton("Add EPD")
        self.add_epd_btn.clicked.connect(lambda: self._add_instrument("EPD"))
        self.add_epd_btn.setEnabled(self.available.get("EPD", False))
        buttons_layout.addWidget(self.add_epd_btn)

        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self._remove_selected)
        self.remove_btn.setEnabled(False)
        buttons_layout.addWidget(self.remove_btn)

        self.done_btn = QPushButton("Done")
        self.done_btn.clicked.connect(self._finish)
        buttons_layout.addWidget(self.done_btn)

        layout.addLayout(buttons_layout)

        self.instrument_list.itemSelectionChanged.connect(self._update_remove_state)

    def _add_instrument(self, name):
        existing = [self.instrument_list.item(i).text() for i in range(self.instrument_list.count())]
        if name in existing:
            return
        self.instrument_list.addItem(QListWidgetItem(name))

    def _remove_selected(self):
        for item in self.instrument_list.selectedItems():
            row = self.instrument_list.row(item)
            self.instrument_list.takeItem(row)
        self._update_remove_state()

    def _update_remove_state(self):
        self.remove_btn.setEnabled(len(self.instrument_list.selectedItems()) > 0)

    def _finish(self):
        self.selected_instruments = [
            self.instrument_list.item(i).text() for i in range(self.instrument_list.count())
        ]
        self.accept()

    def get_selected_instruments(self):
        return self.selected_instruments


class CombinedPlotDialog(QDialog):
    def __init__(self, parent=None, prefs_dialog=None):
        super().__init__(parent)
        self.setWindowTitle("Combined plot")
        self.resize(500, 320)

        self.prefs_dialog = prefs_dialog
        self.main_window = prefs_dialog.parent if prefs_dialog else None
        self.display_instruments = []

        layout = QVBoxLayout(self)

        self.select_instruments_btn = QPushButton("Select instruments to plot")
        self.select_instruments_btn.clicked.connect(self._open_instrument_selector)
        layout.addWidget(self.select_instruments_btn)

        self.display_label = QLabel("Selected instruments: None")
        layout.addWidget(self.display_label)

        self.date_range_checkbox = QCheckBox("Set date range")
        self.date_range_checkbox.toggled.connect(self._toggle_date_range)
        layout.addWidget(self.date_range_checkbox)

        date_layout = QHBoxLayout()
        date_layout.addWidget(QLabel("  "))
        date_layout.addWidget(QLabel("From:"))
        self.date_start = QDateTimeEdit(QDateTime.currentDateTime())
        self.date_start.setCalendarPopup(True)
        self.date_start.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        self.date_start.setEnabled(False)
        date_layout.addWidget(self.date_start)
        date_layout.addWidget(QLabel("to:"))
        self.date_end = QDateTimeEdit(QDateTime.currentDateTime())
        self.date_end.setCalendarPopup(True)
        self.date_end.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        self.date_end.setEnabled(False)
        date_layout.addWidget(self.date_end)
        layout.addLayout(date_layout)

        line_layout = QFormLayout()
        self.linewidth_spin = QDoubleSpinBox()
        self.linewidth_spin.setRange(0.1, 5.0)
        self.linewidth_spin.setSingleStep(0.1)
        self.linewidth_spin.setValue(1.5)
        line_layout.addRow("Line width:", self.linewidth_spin)

        self.fontsize_spin = QSpinBox()
        self.fontsize_spin.setRange(3, 48)
        self.fontsize_spin.setValue(6)
        line_layout.addRow("Font size:", self.fontsize_spin)
        layout.addLayout(line_layout)

        plot_btn = QPushButton("PLOT")
        plot_btn.clicked.connect(self._plot_combined)
        layout.addWidget(plot_btn)

    def _toggle_date_range(self, enabled):
        self.date_start.setEnabled(enabled)
        self.date_end.setEnabled(enabled)

    def _open_instrument_selector(self):
        if not self.main_window:
            QMessageBox.warning(self, "Error", "Main window reference not available.")
            return

        epd_loaded = (
            (self.main_window.df_protons_ept is not None or self.main_window.df_electrons_ept is not None)
            and self.main_window.energies_ept is not None
        )
        available = {
            "STIX": self.main_window.stix_counts_data is not None,
            "RPW-HFR": self.main_window.rpw_hfr_data is not None,
            "RPW-TNR": self.main_window.rpw_tnr_data is not None,
            "EPD": epd_loaded,
        }

        dlg = InstrumentSelectionDialog(self, available=available, selected=self.display_instruments)
        if dlg.exec_() == QDialog.Accepted:
            self.display_instruments = dlg.get_selected_instruments()
            self._update_display_label()

    def _update_display_label(self):
        if self.display_instruments:
            self.display_label.setText("Selected instruments: " + ", ".join(self.display_instruments))
        else:
            self.display_label.setText("Selected instruments: None")

    def retrieve_display_plots(self):
        mapping = {
            "STIX": "stix",
            "RPW-HFR": "hfr",
            "RPW-TNR": "tnr",
            "EPD": "epd",
        }
        return [mapping[name] for name in self.display_instruments if name in mapping]

    def _get_date_range(self):
        if not self.date_range_checkbox.isChecked():
            return None
        start_dt = self.date_start.dateTime().toPython()
        end_dt = self.date_end.dateTime().toPython()

        month_map = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        start_str = (
            f"{start_dt.day:02d}-{month_map[start_dt.month - 1]}-"
            f"{start_dt.year:04d} {start_dt.hour:02d}:{start_dt.minute:02d}:{start_dt.second:02d}"
        )
        end_str = (
            f"{end_dt.day:02d}-{month_map[end_dt.month - 1]}-"
            f"{end_dt.year:04d} {end_dt.hour:02d}:{end_dt.minute:02d}:{end_dt.second:02d}"
        )
        return [start_str, end_str]

    def _plot_combined(self):
        if not self.prefs_dialog or not self.main_window:
            QMessageBox.warning(self, "Error", "Plot preferences not available.")
            return

        display = self.retrieve_display_plots()
        if not display:
            QMessageBox.warning(self, "No Instruments", "Please select at least one instrument to plot.")
            return

        prefs = self.prefs_dialog.get_values()
        self.main_window.plot_prefs = prefs

        stix_type_map = {
            "spectrogram": "spec",
            "time profiles": "curve",
            "overlay": "overlay",
        }
        rpw_type_map = {
            "spectrogram": "spec",
            "time profiles": "curve",
            "overlay": "overlay",
        }
        rpw_overlap_map = {
            "Only HFR": "hfr",
            "Only TNR": "tnr",
            "Both": "both",
        }

        stix_logy = prefs["stix"]["logy"]
        stix_spec_ylog = False
        stix_curve_ylog = False
        if prefs["stix"]["type"] == "spectrogram":
            stix_spec_ylog = bool(stix_logy)
        elif prefs["stix"]["type"] == "time profiles":
            stix_curve_ylog = bool(stix_logy)
        elif prefs["stix"]["type"] == "overlay":
            if isinstance(stix_logy, dict):
                stix_spec_ylog = stix_logy.get("energy", False)
                stix_curve_ylog = stix_logy.get("countrate", False)
            else:
                stix_spec_ylog = bool(stix_logy)
                stix_curve_ylog = bool(stix_logy)

        rpw_overlap_choice = prefs["rpw"].get("overlay", "Both")
        rpw_overlap = rpw_overlap_map.get(rpw_overlap_choice, "both")

        stix_energy_range = prefs["stix"].get("energy_range") or [4, 28]
        stix_energy_bins = prefs["stix"].get("energy_ranges") or [[4, 12], [16, 28]]

        rpw_freqs = prefs["rpw"].get("selected_frequencies") or []
        date_range = self._get_date_range()

        try:
            quicklook_plot(
                stix_counts=self.main_window.stix_counts_data,
                hfr_psd=self.main_window.rpw_hfr_data,
                tnr_psd=self.main_window.rpw_tnr_data,
                epd_data=(self.main_window.df_electrons_ept if self.main_window.epd_particle == "Electron" else self.main_window.df_protons_ept),
                epd_energies=self.main_window.energies_ept,
                display=display,
                date_range=date_range,
                stix_energy_range=stix_energy_range,
                stix_energy_bins=stix_energy_bins,
                stix_mode=stix_type_map[prefs["stix"]["type"]],
                stix_smoothing_points=prefs["stix"].get("smoothing_points", 1),
                stix_cmap="bone",
                stix_curves_ylogscale=stix_curve_ylog,
                stix_spec_ylogscale=stix_spec_ylog,
                stix_spec_zlogscale=prefs["stix"]["logz"],
                stix_lcolor=None,
                rpw_frequency_range=prefs["rpw"].get("freq_range"),
                hfr_frequencies=rpw_freqs,
                tnr_frequencies=rpw_freqs,
                rpw_mode=rpw_type_map[prefs["rpw"]["type"]],
                rpw_overlap=rpw_overlap,
                rpw_plot_bias=None,
                rpw_units="wmhz",
                rpw_invert_yaxis=prefs["rpw"].get("invert_y", True),
                rpw_cmap="nipy_spectral",
                rpw_smoothing_points=prefs["rpw"].get("smoothing_points", 1),
                rpw_lcolor=None,
                rpw_guidelines=False,
                epd_channels=prefs["epd"].get("selected_channels", []),
                epd_particle=self.main_window.epd_particle or "Electron",
                epd_resample=self.main_window.epd_resample or "1min",
                epd_round_label=True,
                cmap=None,
                date_fmt="%H:%M",
                figsize=(14, 10),
                timegrid=True,
                fontsize=self.fontsize_spin.value(),
                markers={},
                imaging_intervals=None,
                linewidth=self.linewidth_spin.value(),
                savename=None,
            )
            plt.show(block=False)
            self.accept()
        except Exception as exc:
            QMessageBox.critical(self, "Plot Error", f"Error generating combined plot:\n{exc}")

class ImportStixDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import STIX data")
        self.resize(800, 750)
        
        # Store the processed data
        self.stix_counts = None
        self.processed_stix_counts = None

        layout = QVBoxLayout(self)

        # Main data section
        main_form = QFormLayout()
        
        # STIX file widget
        self.stix_edit = QLineEdit()
        stix_btn = QPushButton("Browse...")
        stix_btn.clicked.connect(lambda: self._browse(self.stix_edit))
        
        # Preview button
        self.preview_btn = QPushButton("Preview Data")
        self.preview_btn.setEnabled(False)
        self.preview_btn.clicked.connect(self._preview_stix_data)
        
        h1 = QHBoxLayout()
        h1.addWidget(self.stix_edit)
        h1.addWidget(stix_btn)
        h1.addWidget(self.preview_btn)
        main_form.addRow("STIX spectrogram file:", h1)
        
        # Connect text change to enable/disable preview button
        self.stix_edit.textChanged.connect(self._on_stix_file_changed)
        
        layout.addLayout(main_form)

        # Preview section
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        # Create matplotlib figure and canvas
        self.preview_figure = Figure(figsize=(8, 4))
        self.preview_canvas = FigureCanvas(self.preview_figure)
        preview_layout.addWidget(self.preview_canvas)
        
        # Initially hide the preview section
        preview_group.setVisible(False)
        self.preview_group = preview_group
        
        layout.addWidget(preview_group)

        # Background section
        bkg_group = QGroupBox("Background")
        bkg_layout = QVBoxLayout(bkg_group)
        
        # Checkbox for BKG file
        self.bkg_file_checkbox = QCheckBox("Import BKG file")
        bkg_layout.addWidget(self.bkg_file_checkbox)
        
        # BKG file selection
        self.bkg_file_edit = QLineEdit()
        self.bkg_file_edit.setEnabled(False)
        bkg_file_btn = QPushButton("Browse...")
        bkg_file_btn.clicked.connect(lambda: self._browse(self.bkg_file_edit))
        bkg_file_btn.setEnabled(False)
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("  "))  # Indent
        h2.addWidget(self.bkg_file_edit)
        h2.addWidget(bkg_file_btn)
        bkg_layout.addLayout(h2)
        
        # Checkbox for time range
        self.bkg_time_checkbox = QCheckBox("Estimate background from time range")
        bkg_layout.addWidget(self.bkg_time_checkbox)
        
        # DateTime range selection
        datetime_layout = QHBoxLayout()
        datetime_layout.addWidget(QLabel("  "))  # Indent
        datetime_layout.addWidget(QLabel("From:"))
        now = QDateTime.currentDateTime()
        self.bkg_start_datetime = QDateTimeEdit(now)
        self.bkg_start_datetime.setCalendarPopup(True)
        self.bkg_start_datetime.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        self.bkg_start_datetime.setEnabled(False)
        datetime_layout.addWidget(self.bkg_start_datetime)
        
        datetime_layout.addWidget(QLabel("To:"))
        self.bkg_end_datetime = QDateTimeEdit(now)
        self.bkg_end_datetime.setCalendarPopup(True)
        self.bkg_end_datetime.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        self.bkg_end_datetime.setEnabled(False)
        datetime_layout.addWidget(self.bkg_end_datetime)
        bkg_layout.addLayout(datetime_layout)
        
        # Background polling function selection
        bkg_poll_layout = QHBoxLayout()
        bkg_poll_layout.addWidget(QLabel("Background polling function:"))
        self.bkg_poll_combo = QComboBox()
        self.bkg_poll_combo.addItems(["mean", "median", "min", "max", "P_25", "P_75"])
        self.bkg_poll_combo.setCurrentText("mean")  # Default
        bkg_poll_layout.addWidget(self.bkg_poll_combo)
        bkg_layout.addLayout(bkg_poll_layout)
        
        # Background preview and load buttons
        bkg_btn_layout = QHBoxLayout()
        self.preview_bkg_btn = QPushButton("Preview with Background Subtraction")
        self.preview_bkg_btn.setEnabled(False)
        self.preview_bkg_btn.clicked.connect(self._preview_with_background)
        bkg_btn_layout.addWidget(self.preview_bkg_btn)
        
        self.load_btn = QPushButton("LOAD")
        self.load_btn.setEnabled(False)
        self.load_btn.clicked.connect(self._load_stix_data)
        bkg_btn_layout.addWidget(self.load_btn)
        
        # Add Plot Background button
        self.plot_bkg_btn = QPushButton("Plot Background")
        self.plot_bkg_btn.setEnabled(False)
        self.plot_bkg_btn.clicked.connect(self._plot_background)
        bkg_btn_layout.addWidget(self.plot_bkg_btn)
        
        bkg_layout.addLayout(bkg_btn_layout)
        
        # Connect checkboxes to enable/disable controls
        self.bkg_file_checkbox.toggled.connect(lambda: self._toggle_bkg_controls())
        self.bkg_time_checkbox.toggled.connect(lambda: self._toggle_bkg_controls())
        
        # Store reference to bkg_file_btn for enabling/disabling
        self.bkg_file_btn = bkg_file_btn
        
        layout.addWidget(bkg_group)

        # buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self, line_edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(self, "Select file")
        if path:
            line_edit.setText(path)

    def _on_stix_file_changed(self):
        """Enable/disable preview button based on file path"""
        has_file = bool(self.stix_edit.text().strip())
        self.preview_btn.setEnabled(has_file)
        self.preview_bkg_btn.setEnabled(has_file)
        self.load_btn.setEnabled(has_file)
        
        # Hide preview if file is changed/cleared
        if not has_file:
            self.preview_group.setVisible(False)
            self.stix_counts = None
            self.processed_stix_counts = None

    def _preview_stix_data(self):
        """Preview raw STIX data in the dialog window"""
        stix_file = self.stix_edit.text().strip()
        if not stix_file:
            QMessageBox.warning(self, "No File", "Please select a STIX file first.")
            return
        
        try:
            # Call the preview function
            self._show_stix_preview(stix_file)
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", f"Error previewing STIX data:\n{str(e)}")

    def _show_stix_preview(self, filepath):
        """Show raw STIX data preview plot in the dialog"""
        try:
            # Clear the previous plot
            self.preview_figure.clear()
            
            # Load raw STIX data using stix_create_counts
            self.stix_counts = stix_create_counts(filepath)
            
            # Extract min and max times from the data
            min_time = self.stix_counts['time'][0]
            max_time = self.stix_counts['time'][-1]
            
            # Auto-populate time range from data - convert datetime to QDateTime
            if hasattr(min_time, 'year'):  # Check if it's already a datetime object
                min_qdatetime = QDateTime(min_time)
                max_qdatetime = QDateTime(max_time)
            else:
                # Handle case where times might be strings or other formats
                try:
                    from datetime import datetime
                    if isinstance(min_time, str):
                        # Try parsing different date formats
                        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d-%b-%Y %H:%M:%S"]:
                            try:
                                dt_min = datetime.strptime(min_time, fmt)
                                dt_max = datetime.strptime(max_time, fmt)
                                min_qdatetime = QDateTime(dt_min)
                                max_qdatetime = QDateTime(dt_max)
                                break
                            except ValueError:
                                continue
                        else:
                            # If no format worked, use current time as fallback
                            min_qdatetime = QDateTime.currentDateTime()
                            max_qdatetime = QDateTime.currentDateTime()
                    else:
                        # Try direct conversion
                        min_qdatetime = QDateTime(min_time)
                        max_qdatetime = QDateTime(max_time)
                except:
                    # Fallback to current time if all conversion attempts fail
                    min_qdatetime = QDateTime.currentDateTime()
                    max_qdatetime = QDateTime.currentDateTime()
            
            # Set the datetime widgets with the data time range
            self.bkg_start_datetime.setDateTime(min_qdatetime)
            self.bkg_end_datetime.setDateTime(max_qdatetime)
            
            # Create axes in the preview figure
            ax = self.preview_figure.add_subplot(111)
            stix_plot_spectrogram(self.stix_counts, ax=ax, x_axis=True)
            
            # Adjust the plot title to show filename
            ax.set_title(f'STIX Raw Data: {filepath.split("/")[-1]}', fontsize=10)
            
            # Tight layout and refresh canvas
            self.preview_figure.tight_layout()
            self.preview_canvas.draw()
            
            # Show the preview section
            self.preview_group.setVisible(True)
            
            # Show success message with time range info
            time_info = f"Data time range: {min_time} to {max_time}"
            QMessageBox.information(self, "Preview Generated", 
                                  f"STIX raw data preview loaded successfully.\n\nFile: {filepath.split('/')[-1]}\n{time_info}\n\nBackground time range has been set to data limits.")
            
        except Exception as e:
            # Hide preview section on error
            self.preview_group.setVisible(False)
            raise Exception(f"Failed to generate preview: {str(e)}")

    def _preview_with_background(self):
        """Preview STIX data with background subtraction"""
        if not self.stix_edit.text().strip():
            QMessageBox.warning(self, "No File", "Please select a STIX file first.")
            return
        
        # Check if at least one background option is selected
        if not (self.bkg_file_checkbox.isChecked() or self.bkg_time_checkbox.isChecked()):
            QMessageBox.warning(self, "No Background", "Please select at least one background option.")
            return
        
        try:
            # Process background subtraction using stix_remove_bkg_counts
            processed_counts = self._apply_background_subtraction()
            
            # Clear the previous plot
            self.preview_figure.clear()
            
            # Create axes in the preview figure
            ax = self.preview_figure.add_subplot(111)
            stix_plot_spectrogram(processed_counts, ax=ax,x_axis=True)
            if self.bkg_time_checkbox.isChecked():
                # add vertical lines for background time range in ax
                start_time = self.bkg_start_datetime.dateTime().toPython()
                end_time = self.bkg_end_datetime.dateTime().toPython()
                ax.axvline(start_time, color='red', linestyle='--')
                ax.axvline(end_time, color='red', linestyle='--')
       

            # Adjust the plot title
            bkg_info = self._get_background_info()
            ax.set_title(f'STIX Data (Background Subtracted)\n{bkg_info}', fontsize=9)
            
            # Tight layout and refresh canvas
            self.preview_figure.tight_layout()
            self.preview_canvas.draw()
            
            # Show the preview section
            self.preview_group.setVisible(True)
            
            # Store the processed data
            self.processed_stix_counts = processed_counts
            
            QMessageBox.information(self, "Background Preview", 
                                  f"STIX data with background subtraction preview generated.\n\n{bkg_info}")
            
            self.plot_bkg_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Background Error", f"Error applying background subtraction:\n{str(e)}")

    def _apply_background_subtraction(self):
        """Apply background subtraction to STIX data using stix_remove_bkg_counts"""
        stix_file = self.stix_edit.text().strip()
        if not stix_file:
            raise ValueError("No STIX file specified")
        
        # Get polling function
        bkg_poll_function = self.bkg_poll_combo.currentText()
        
        # Prepare arguments for stix_remove_bkg_counts
        kwargs = {
            'energy_shift': 0,
            'bkg_poll_function': bkg_poll_function
        }
        
        # Add background file if selected
        if self.bkg_file_checkbox.isChecked():
            bkg_file = self.bkg_file_edit.text().strip()
            if not bkg_file:
                raise ValueError("Background file checkbox is checked but no file is specified")
            kwargs['pathbkg'] = bkg_file
        
        # Add time range if selected
        if self.bkg_time_checkbox.isChecked():
            start_datetime = self.bkg_start_datetime.dateTime().toPython()
            end_datetime = self.bkg_end_datetime.dateTime().toPython()
            
            # Format datetime strings as expected by sololab
            start_str = start_datetime.strftime("%d-%b-%Y %H:%M:%S")
            end_str = end_datetime.strftime("%d-%b-%Y %H:%M:%S")
            bkg_range = (start_str, end_str)
            kwargs['stix_bkg_range'] = bkg_range
        
        # Call stix_remove_bkg_counts with appropriate arguments
        if self.bkg_file_checkbox.isChecked() or self.bkg_time_checkbox.isChecked():
            processed_counts = stix_remove_bkg_counts(stix_file, **kwargs)
        else:
            # If no background options selected, just load raw data
            processed_counts = stix_create_counts(stix_file)
        
        return processed_counts

    def _get_background_info(self):
        """Get background information string for display"""
        info_parts = []
        
        if self.bkg_file_checkbox.isChecked():
            bkg_file = self.bkg_file_edit.text().strip()
            if bkg_file:
                info_parts.append(f"BKG File: {bkg_file.split('/')[-1]}")
        
        if self.bkg_time_checkbox.isChecked():
            start_time = self.bkg_start_datetime.dateTime().toString("yyyy-MM-dd hh:mm:ss")
            end_time = self.bkg_end_datetime.dateTime().toString("yyyy-MM-dd hh:mm:ss")
            info_parts.append(f"BKG time Range: {start_time} to {end_time}")
        
        if info_parts:
            poll_func = self.bkg_poll_combo.currentText()
            info_parts.append(f"BKG poll funct.: {poll_func}")
            return " | ".join(info_parts)
        else:
            return "No background subtraction applied"
    def _plot_background(self):
        """Plot STIX background in a separate window"""
        if not self.processed_stix_counts:
            QMessageBox.warning(self, "No Background Data", "Please preview with background subtraction first.")
            return
        
        try:
            # Create new figure for background plot
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(111)
            
            # Plot background using sololab function
            stix_plot_bkg(self.processed_stix_counts, ax=ax)
            
            # Set window title
            fig.suptitle("STIX Background Plot", fontsize=12)
            
            plt.tight_layout()
            plt.show(block=False)  # Non-blocking

        except Exception as e:
            QMessageBox.critical(self, "Background Plot Error", f"Error plotting STIX background:\n{str(e)}")

    
    def _load_stix_data(self):
        """Load and process STIX data, then close dialog"""
        if not self.stix_edit.text().strip():
            QMessageBox.warning(self, "No File", "Please select a STIX file first.")
            return
        
        try:
            # Apply background subtraction if any is selected, otherwise load raw data
            if self.bkg_file_checkbox.isChecked() or self.bkg_time_checkbox.isChecked():
                self.processed_stix_counts = self._apply_background_subtraction()
            else:
                # Load raw data if no background options are selected
                self.processed_stix_counts = stix_create_counts(self.stix_edit.text().strip())
            
            # Enable plot background button if background was applied    
            if self.bkg_file_checkbox.isChecked() or self.bkg_time_checkbox.isChecked():
                self.plot_bkg_btn.setEnabled(True)
            
            # Accept the dialog with processed data
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Error loading STIX data:\n{str(e)}")

    def _toggle_bkg_controls(self):
        is_file_checked = self.bkg_file_checkbox.isChecked()
        is_time_checked = self.bkg_time_checkbox.isChecked()
        
        # Enable/disable file controls
        self.bkg_file_edit.setEnabled(is_file_checked)
        self.bkg_file_btn.setEnabled(is_file_checked)
        
        # Enable/disable datetime controls
        self.bkg_start_datetime.setEnabled(is_time_checked)
        self.bkg_end_datetime.setEnabled(is_time_checked)

    def get_values(self):
        result = {
            "stix_file": self.stix_edit.text().strip() or None,
            "bkg_file_enabled": self.bkg_file_checkbox.isChecked(),
            "bkg_time_enabled": self.bkg_time_checkbox.isChecked(),
            "bkg_poll_function": self.bkg_poll_combo.currentText(),
            "stix_counts": self.processed_stix_counts,  # Include processed data
        }
        
        if self.bkg_file_checkbox.isChecked():
            result["bkg_file"] = self.bkg_file_edit.text().strip() or None
            
        if self.bkg_time_checkbox.isChecked():
            result["bkg_start_datetime"] = self.bkg_start_datetime.dateTime().toPython()
            result["bkg_end_datetime"] = self.bkg_end_datetime.dateTime().toPython()
        
        return result
class ImportRpwHfrDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import RPW-HFR data")
        self.resize(800, 700)
        
        # Store the processed data
        self.rpw_data = None
        self.processed_rpw_data = None

        layout = QVBoxLayout(self)

        # Main data section
        main_form = QFormLayout()
        
        # RPW HFR file widget
        self.rpw_hfr_edit = QLineEdit()
        hfr_btn = QPushButton("Browse...")
        hfr_btn.clicked.connect(lambda: self._browse(self.rpw_hfr_edit))
        
        # Preview button
        self.preview_btn = QPushButton("Preview Data")
        self.preview_btn.setEnabled(False)
        self.preview_btn.clicked.connect(self._preview_rpw_data)
        
        h2 = QHBoxLayout()
        h2.addWidget(self.rpw_hfr_edit)
        h2.addWidget(hfr_btn)
        h2.addWidget(self.preview_btn)
        main_form.addRow("RPW-HFR file:", h2)
        
        # Connect text change to enable/disable preview button
        self.rpw_hfr_edit.textChanged.connect(self._on_file_changed)
        
        layout.addLayout(main_form)

        # Preview section
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        # Create matplotlib figure and canvas
        self.preview_figure = Figure(figsize=(8, 4))
        self.preview_canvas = FigureCanvas(self.preview_figure)
        preview_layout.addWidget(self.preview_canvas)
        
        # Initially hide the preview section
        preview_group.setVisible(False)
        self.preview_group = preview_group
        
        layout.addWidget(preview_group)

        # Background section
        bkg_group = QGroupBox("Background")
        bkg_layout = QVBoxLayout(bkg_group)
        
        # Radio buttons for background options
        self.bkg_button_group = QButtonGroup()
        
        self.no_bkg_radio = QRadioButton("No background subtraction")
        self.no_bkg_radio.setChecked(True)  # Default option
        self.bkg_button_group.addButton(self.no_bkg_radio, 0)
        bkg_layout.addWidget(self.no_bkg_radio)
        
        self.bkg_time_radio = QRadioButton("Estimate background from time range")
        self.bkg_button_group.addButton(self.bkg_time_radio, 1)
        bkg_layout.addWidget(self.bkg_time_radio)
        
        # DateTime range selection
        datetime_layout = QHBoxLayout()
        datetime_layout.addWidget(QLabel("  "))  # Indent
        datetime_layout.addWidget(QLabel("From:"))
        now = QDateTime.currentDateTime()
        self.bkg_start_datetime = QDateTimeEdit(now)
        self.bkg_start_datetime.setCalendarPopup(True)
        self.bkg_start_datetime.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        self.bkg_start_datetime.setEnabled(False)
        datetime_layout.addWidget(self.bkg_start_datetime)
        
        datetime_layout.addWidget(QLabel("To:"))
        self.bkg_end_datetime = QDateTimeEdit(now)
        self.bkg_end_datetime.setCalendarPopup(True)
        self.bkg_end_datetime.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        self.bkg_end_datetime.setEnabled(False)
        datetime_layout.addWidget(self.bkg_end_datetime)
        bkg_layout.addLayout(datetime_layout)
        
        # Background polling function selection
        bkg_poll_layout = QHBoxLayout()
        bkg_poll_layout.addWidget(QLabel("Background polling function:"))
        self.bkg_poll_combo = QComboBox()
        self.bkg_poll_combo.addItems(["max", "mean", "median", "min", "P_25", "P_75"])
        self.bkg_poll_combo.setCurrentText("max")  # Default for RPW
        bkg_poll_layout.addWidget(self.bkg_poll_combo)
        bkg_layout.addLayout(bkg_poll_layout)
        
        # Background preview and load buttons
        bkg_btn_layout = QHBoxLayout()
        self.preview_bkg_btn = QPushButton("Preview with Background Subtraction")
        self.preview_bkg_btn.setEnabled(False)
        self.preview_bkg_btn.clicked.connect(self._preview_with_background)
        bkg_btn_layout.addWidget(self.preview_bkg_btn)
        
        self.load_btn = QPushButton("LOAD")
        self.load_btn.setEnabled(True)  # Always enabled
        self.load_btn.clicked.connect(self._load_rpw_data)
        bkg_btn_layout.addWidget(self.load_btn)
        
        # Add Plot Background button
        self.plot_bkg_btn = QPushButton("Plot Background")
        self.plot_bkg_btn.setEnabled(False)
        self.plot_bkg_btn.clicked.connect(self._plot_background)
        bkg_btn_layout.addWidget(self.plot_bkg_btn)
        
        bkg_layout.addLayout(bkg_btn_layout)
        
        # Connect radio buttons to enable/disable controls
        self.no_bkg_radio.toggled.connect(lambda: self._toggle_bkg_controls())
        self.bkg_time_radio.toggled.connect(lambda: self._toggle_bkg_controls())
        
        layout.addWidget(bkg_group)

        # buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self, line_edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(self, "Select file")
        if path:
            line_edit.setText(path)

    def _on_file_changed(self):
        """Enable/disable buttons based on file path"""
        has_file = bool(self.rpw_hfr_edit.text().strip())
        self.preview_btn.setEnabled(has_file)
        self.preview_bkg_btn.setEnabled(has_file and self.bkg_time_radio.isChecked())
        
        # Hide preview if file is changed/cleared
        if not has_file:
            self.preview_group.setVisible(False)
            self.rpw_data = None
            self.processed_rpw_data = None

    def _preview_rpw_data(self):
        """Preview raw RPW-HFR data"""
        rpw_file = self.rpw_hfr_edit.text().strip()
        if not rpw_file:
            QMessageBox.warning(self, "No File", "Please select an RPW-HFR file first.")
            return
        
        try:
            self._show_rpw_preview(rpw_file)
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", f"Error previewing RPW-HFR data:\n{str(e)}")

    
    def _show_rpw_preview(self, filepath):
        """Show raw RPW-HFR data preview plot"""
        try:
            # Clear the previous plot
            self.preview_figure.clear()
            
            # Load raw RPW data
            rpw_data = rpw_get_data(filepath)
            self.rpw_data = rpw_create_PSD(rpw_data, which_freqs="non_zero")
            
            # Extract min and max times from the data
            min_time = self.rpw_data['time'][0]
            max_time = self.rpw_data['time'][-1]
            
            # Auto-populate time range from data - convert datetime to QDateTime
            if hasattr(min_time, 'year'):  # Check if it's already a datetime object
                min_qdatetime = QDateTime(min_time)
                max_qdatetime = QDateTime(max_time)
            else:
                # Handle case where times might be strings or other formats
                try:
                    from datetime import datetime
                    if isinstance(min_time, str):
                        # Try parsing different date formats
                        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d-%b-%Y %H:%M:%S"]:
                            try:
                                dt_min = datetime.strptime(min_time, fmt)
                                dt_max = datetime.strptime(max_time, fmt)
                                min_qdatetime = QDateTime(dt_min)
                                max_qdatetime = QDateTime(dt_max)
                                break
                            except ValueError:
                                continue
                        else:
                            # If no format worked, use current time as fallback
                            min_qdatetime = QDateTime.currentDateTime()
                            max_qdatetime = QDateTime.currentDateTime()
                    else:
                        # Try direct conversion
                        min_qdatetime = QDateTime(min_time)
                        max_qdatetime = QDateTime(max_time)
                except:
                    # Fallback to current time if all conversion attempts fail
                    min_qdatetime = QDateTime.currentDateTime()
                    max_qdatetime = QDateTime.currentDateTime()
            
            # Set the datetime widgets with the data time range
            self.bkg_start_datetime.setDateTime(min_qdatetime)
            self.bkg_end_datetime.setDateTime(max_qdatetime)
            
            # Create axes in the preview figure
            ax = self.preview_figure.add_subplot(111)
            rpw_plot_psd(self.rpw_data, ax=ax, xlabel=True, frequency_range=[0,17000], 
                               t_format="%H:%M", rpw_cbar_units="wmhz")
            
            # Adjust the plot title
            ax.set_title(f'RPW-HFR Raw Data: {filepath.split("/")[-1]}', fontsize=10)
            
            # Tight layout and refresh canvas
            self.preview_figure.tight_layout()
            self.preview_canvas.draw()
            
            # Show the preview section
            self.preview_group.setVisible(True)
            
            # Show success message with time range info
            time_info = f"Data time range: {min_time} to {max_time}"
            QMessageBox.information(self, "Preview Generated", 
                                  f"RPW-HFR raw data preview loaded successfully.\n\nFile: {filepath.split('/')[-1]}\n{time_info}\n\nBackground time range has been set to data limits.")
            
        except Exception as e:
            self.preview_group.setVisible(False)
            raise Exception(f"Failed to generate preview: {str(e)}")

    def _preview_with_background(self):
        """Preview RPW-HFR data with background subtraction"""
        if not self.rpw_hfr_edit.text().strip():
            QMessageBox.warning(self, "No File", "Please select an RPW-HFR file first.")
            return
        
        if not self.bkg_time_radio.isChecked():
            QMessageBox.warning(self, "No Background", "Please select background time range option.")
            return
        
        try:
            # Process background subtraction
            processed_data = self._apply_background_subtraction()
            
            # Clear the previous plot
            self.preview_figure.clear()
            
            # Create axes in the preview figure
            ax = self.preview_figure.add_subplot(111)
            rpw_plot_psd(processed_data, ax=ax, xlabel=True, frequency_range=[0,17000], 
                               t_format="%H:%M", rpw_cbar_units="wmhz")
            
            # Add vertical lines for background time range
            start_time = self.bkg_start_datetime.dateTime().toPython()
            end_time = self.bkg_end_datetime.dateTime().toPython()
            ax.axvline(start_time, color='red', linestyle='--', label='Background range')
            ax.axvline(end_time, color='red', linestyle='--')
            
            # Adjust the plot title
            bkg_info = self._get_background_info()
            ax.set_title(f'RPW-HFR Data (Background Subtracted)\n{bkg_info}', fontsize=9)
            
            # Tight layout and refresh canvas
            self.preview_figure.tight_layout()
            self.preview_canvas.draw()
            
            # Show the preview section
            self.preview_group.setVisible(True)
            
            # Store the processed data
            self.processed_rpw_data = processed_data
            
            QMessageBox.information(self, "Background Preview", 
                                  f"RPW-HFR data with background subtraction preview generated.\n\n{bkg_info}")
            self.plot_bkg_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Background Error", f"Error applying background subtraction:\n{str(e)}")
    
    
    def _plot_background(self):
        """Plot RPW-HFR background in a separate window"""
        if not self.processed_rpw_data:
            QMessageBox.warning(self, "No Background Data", "Please preview with background subtraction first.")
            return
        
        try:
            # Create new figure for background plot
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(111)
            
            # Plot background using sololab function
            rpw_plot_bkg(self.processed_rpw_data, ax=ax)
            
            # Set window title
            fig.suptitle("RPW-HFR Background Plot", fontsize=12)
            
            plt.tight_layout()
            plt.show(block=False)  # Non-blocking
            
        except Exception as e:
            QMessageBox.critical(self, "Background Plot Error", f"Error plotting RPW-HFR background:\n{str(e)}")


    def _apply_background_subtraction(self):
        """Apply background subtraction to RPW-HFR data"""
        rpw_file = self.rpw_hfr_edit.text().strip()
        if not rpw_file:
            raise ValueError("No RPW-HFR file specified")
        
        # Get polling function
        bkg_poll_function = self.bkg_poll_combo.currentText()
        
        # Load raw RPW data
        rpw_data = rpw_get_data(rpw_file)
        
        if self.bkg_time_radio.isChecked():
            # Apply time range background subtraction
            start_datetime = self.bkg_start_datetime.dateTime().toPython()
            end_datetime = self.bkg_end_datetime.dateTime().toPython()
            
            # Format datetime strings as expected by sololab
            start_str = start_datetime.strftime("%d-%b-%Y %H:%M:%S")
            end_str = end_datetime.strftime("%d-%b-%Y %H:%M:%S")
            bkg_range = (start_str, end_str)
            
            # Create PSD with background subtraction
            processed_data = rpw_create_PSD(rpw_data, which_freqs="non_zero", 
                                                  rpw_bkg_interval=bkg_range, 
                                                  bkg_poll_function=bkg_poll_function)
        else:
            # No background subtraction
            processed_data = rpw_create_PSD(rpw_data, which_freqs="non_zero")
        
        return processed_data

    def _get_background_info(self):
        """Get background information string for display"""
        if self.bkg_time_radio.isChecked():
            start_time = self.bkg_start_datetime.dateTime().toString("yyyy-MM-dd hh:mm:ss")
            end_time = self.bkg_end_datetime.dateTime().toString("yyyy-MM-dd hh:mm:ss")
            poll_func = self.bkg_poll_combo.currentText()
            return f"BKG time range: {start_time} to {end_time} | BKG poll funct.: {poll_func}"
        else:
            return "No background subtraction applied"

    def _load_rpw_data(self):
        """Load and process RPW-HFR data, then close dialog"""
        if not self.rpw_hfr_edit.text().strip():
            QMessageBox.warning(self, "No File", "Please select an RPW-HFR file first.")
            return
        
        try:
            # Apply background subtraction if selected, otherwise load raw data
            if self.bkg_time_radio.isChecked():
                self.processed_rpw_data = self._apply_background_subtraction()
            else:
                # Load raw data
                rpw_data = rpw_get_data(self.rpw_hfr_edit.text().strip())
                self.processed_rpw_data = rpw_create_PSD(rpw_data, which_freqs="non_zero")
            
            if self.bkg_time_radio.isChecked():
                self.plot_bkg_btn.setEnabled(True)
            # Accept the dialog with processed data
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Error loading RPW-HFR data:\n{str(e)}")

    def _toggle_bkg_controls(self):
        is_time_selected = self.bkg_time_radio.isChecked()
        
        # Enable/disable datetime controls
        self.bkg_start_datetime.setEnabled(is_time_selected)
        self.bkg_end_datetime.setEnabled(is_time_selected)
        
        # Update background preview button
        has_file = bool(self.rpw_hfr_edit.text().strip())
        self.preview_bkg_btn.setEnabled(has_file and is_time_selected)

    def get_values(self):
        bkg_option = self.bkg_button_group.checkedId()
        result = {
            "rpw_hfr_file": self.rpw_hfr_edit.text().strip() or None,
            "bkg_option": bkg_option,  # 0: no bkg, 1: time range
            "bkg_poll_function": self.bkg_poll_combo.currentText(),
            "rpw_data": self.processed_rpw_data,  # Include processed data
        }
        
        if bkg_option == 1:  # Time range
            result["bkg_start_datetime"] = self.bkg_start_datetime.dateTime().toPython()
            result["bkg_end_datetime"] = self.bkg_end_datetime.dateTime().toPython()
        
        return result


class ImportRpwTnrDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import RPW-TNR data")
        self.resize(800, 700)
        
        # Store the processed data
        self.rpw_data = None
        self.processed_rpw_data = None

        layout = QVBoxLayout(self)

        # Main data section
        main_form = QFormLayout()
        
        # RPW TNR file widget
        self.rpw_tnr_edit = QLineEdit()
        tnr_btn = QPushButton("Browse...")
        tnr_btn.clicked.connect(lambda: self._browse(self.rpw_tnr_edit))
        
        # Preview button
        self.preview_btn = QPushButton("Preview Data")
        self.preview_btn.setEnabled(False)
        self.preview_btn.clicked.connect(self._preview_rpw_data)
        
        h3 = QHBoxLayout()
        h3.addWidget(self.rpw_tnr_edit)
        h3.addWidget(tnr_btn)
        h3.addWidget(self.preview_btn)
        main_form.addRow("RPW-TNR file:", h3)
        
        # Connect text change to enable/disable preview button
        self.rpw_tnr_edit.textChanged.connect(self._on_file_changed)
        
        layout.addLayout(main_form)

        # Preview section
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        # Create matplotlib figure and canvas
        self.preview_figure = Figure(figsize=(8, 4))
        self.preview_canvas = FigureCanvas(self.preview_figure)
        preview_layout.addWidget(self.preview_canvas)
        
        # Initially hide the preview section
        preview_group.setVisible(False)
        self.preview_group = preview_group
        
        layout.addWidget(preview_group)

        # Background section
        bkg_group = QGroupBox("Background")
        bkg_layout = QVBoxLayout(bkg_group)
        
        # Radio buttons for background options
        self.bkg_button_group = QButtonGroup()
        
        self.no_bkg_radio = QRadioButton("No background subtraction")
        self.no_bkg_radio.setChecked(True)  # Default option
        self.bkg_button_group.addButton(self.no_bkg_radio, 0)
        bkg_layout.addWidget(self.no_bkg_radio)
        
        self.bkg_time_radio = QRadioButton("Estimate background from time range")
        self.bkg_button_group.addButton(self.bkg_time_radio, 1)
        bkg_layout.addWidget(self.bkg_time_radio)
        
        # DateTime range selection
        datetime_layout = QHBoxLayout()
        datetime_layout.addWidget(QLabel("  "))  # Indent
        datetime_layout.addWidget(QLabel("From:"))
        now = QDateTime.currentDateTime()
        self.bkg_start_datetime = QDateTimeEdit(now)
        self.bkg_start_datetime.setCalendarPopup(True)
        self.bkg_start_datetime.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        self.bkg_start_datetime.setEnabled(False)
        datetime_layout.addWidget(self.bkg_start_datetime)
        
        datetime_layout.addWidget(QLabel("To:"))
        self.bkg_end_datetime = QDateTimeEdit(now)
        self.bkg_end_datetime.setCalendarPopup(True)
        self.bkg_end_datetime.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        self.bkg_end_datetime.setEnabled(False)
        datetime_layout.addWidget(self.bkg_end_datetime)
        bkg_layout.addLayout(datetime_layout)
        
        # Background polling function selection
        bkg_poll_layout = QHBoxLayout()
        bkg_poll_layout.addWidget(QLabel("Background polling function:"))
        self.bkg_poll_combo = QComboBox()
        self.bkg_poll_combo.addItems(["max", "mean", "median", "min", "P_25", "P_75"])
        self.bkg_poll_combo.setCurrentText("max")  # Default for RPW
        bkg_poll_layout.addWidget(self.bkg_poll_combo)
        bkg_layout.addLayout(bkg_poll_layout)
        
        # Background preview and load buttons
        bkg_btn_layout = QHBoxLayout()
        self.preview_bkg_btn = QPushButton("Preview with Background Subtraction")
        self.preview_bkg_btn.setEnabled(False)
        self.preview_bkg_btn.clicked.connect(self._preview_with_background)
        bkg_btn_layout.addWidget(self.preview_bkg_btn)
        
        self.load_btn = QPushButton("LOAD")
        self.load_btn.setEnabled(True)  # Always enabled
        self.load_btn.clicked.connect(self._load_rpw_data)
        bkg_btn_layout.addWidget(self.load_btn)
        
        bkg_layout.addLayout(bkg_btn_layout)
        
        # Connect radio buttons to enable/disable controls
        self.no_bkg_radio.toggled.connect(lambda: self._toggle_bkg_controls())
        self.bkg_time_radio.toggled.connect(lambda: self._toggle_bkg_controls())
        
        layout.addWidget(bkg_group)

        # buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self, line_edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(self, "Select file")
        if path:
            line_edit.setText(path)

    def _on_file_changed(self):
        """Enable/disable buttons based on file path"""
        has_file = bool(self.rpw_tnr_edit.text().strip())
        self.preview_btn.setEnabled(has_file)
        self.preview_bkg_btn.setEnabled(has_file and self.bkg_time_radio.isChecked())
        
        # Hide preview if file is changed/cleared
        if not has_file:
            self.preview_group.setVisible(False)
            self.rpw_data = None
            self.processed_rpw_data = None

    def _preview_rpw_data(self):
        """Preview raw RPW-TNR data"""
        rpw_file = self.rpw_tnr_edit.text().strip()
        if not rpw_file:
            QMessageBox.warning(self, "No File", "Please select an RPW-TNR file first.")
            return
        
        try:
            self._show_rpw_preview(rpw_file)
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", f"Error previewing RPW-TNR data:\n{str(e)}")

    def _show_rpw_preview(self, filepath):
        """Show raw RPW-TNR data preview plot"""
        try:
            # Clear the previous plot
            self.preview_figure.clear()
            
            # Load raw RPW data
            rpw_data = rpw_get_data(filepath)
            self.rpw_data = rpw_create_PSD(rpw_data, which_freqs="non_zero")
            
            # Extract min and max times from the data
            min_time = self.rpw_data['time'][0]
            max_time = self.rpw_data['time'][-1]
            
            # Auto-populate time range from data - convert datetime to QDateTime
            if hasattr(min_time, 'year'):  # Check if it's already a datetime object
                min_qdatetime = QDateTime(min_time)
                max_qdatetime = QDateTime(max_time)
            else:
                # Handle case where times might be strings or other formats
                try:
                    from datetime import datetime
                    if isinstance(min_time, str):
                        # Try parsing different date formats
                        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d-%b-%Y %H:%M:%S"]:
                            try:
                                dt_min = datetime.strptime(min_time, fmt)
                                dt_max = datetime.strptime(max_time, fmt)
                                min_qdatetime = QDateTime(dt_min)
                                max_qdatetime = QDateTime(dt_max)
                                break
                            except ValueError:
                                continue
                        else:
                            # If no format worked, use current time as fallback
                            min_qdatetime = QDateTime.currentDateTime()
                            max_qdatetime = QDateTime.currentDateTime()
                    else:
                        # Try direct conversion
                        min_qdatetime = QDateTime(min_time)
                        max_qdatetime = QDateTime(max_time)
                except:
                    # Fallback to current time if all conversion attempts fail
                    min_qdatetime = QDateTime.currentDateTime()
                    max_qdatetime = QDateTime.currentDateTime()
            
            # Set the datetime widgets with the data time range
            self.bkg_start_datetime.setDateTime(min_qdatetime)
            self.bkg_end_datetime.setDateTime(max_qdatetime)
            
            # Create axes in the preview figure
            ax = self.preview_figure.add_subplot(111)
            rpw_plot_psd(self.rpw_data, ax=ax, xlabel=True, frequency_range=[0,17000], 
                               t_format="%H:%M", rpw_cbar_units="wmhz")
            
            # Adjust the plot title
            ax.set_title(f'RPW-TNR Raw Data: {filepath.split("/")[-1]}', fontsize=10)
            
            # Tight layout and refresh canvas
            self.preview_figure.tight_layout()
            self.preview_canvas.draw()
            
            # Show the preview section
            self.preview_group.setVisible(True)
            
            # Show success message with time range info
            time_info = f"Data time range: {min_time} to {max_time}"
            QMessageBox.information(self, "Preview Generated", 
                                  f"RPW-TNR raw data preview loaded successfully.\n\nFile: {filepath.split('/')[-1]}\n{time_info}\n\nBackground time range has been set to data limits.")
            
        except Exception as e:
            self.preview_group.setVisible(False)
            raise Exception(f"Failed to generate preview: {str(e)}")
    def _preview_with_background(self):
        """Preview RPW-TNR data with background subtraction"""
        if not self.rpw_tnr_edit.text().strip():
            QMessageBox.warning(self, "No File", "Please select an RPW-TNR file first.")
            return
        
        if not self.bkg_time_radio.isChecked():
            QMessageBox.warning(self, "No Background", "Please select background time range option.")
            return
        
        try:
            # Process background subtraction
            processed_data = self._apply_background_subtraction()
            
            # Clear the previous plot
            self.preview_figure.clear()
            
            # Create axes in the preview figure
            ax = self.preview_figure.add_subplot(111)
            rpw_plot_psd(processed_data, ax=ax, xlabel=True, frequency_range=[0,17000], 
                               t_format="%H:%M", rpw_cbar_units="wmhz")
            
            # Add vertical lines for background time range
            start_time = self.bkg_start_datetime.dateTime().toPython()
            end_time = self.bkg_end_datetime.dateTime().toPython()
            ax.axvline(start_time, color='red', linestyle='--', label='Background range')
            ax.axvline(end_time, color='red', linestyle='--')
            
            # Adjust the plot title
            bkg_info = self._get_background_info()
            ax.set_title(f'RPW-TNR Data (Background Subtracted)\n{bkg_info}', fontsize=9)
            
            # Tight layout and refresh canvas
            self.preview_figure.tight_layout()
            self.preview_canvas.draw()
            
            # Show the preview section
            self.preview_group.setVisible(True)
            
            # Store the processed data
            self.processed_rpw_data = processed_data
            
            QMessageBox.information(self, "Background Preview", 
                                  f"RPW-TNR data with background subtraction preview generated.\n\n{bkg_info}")
            
        except Exception as e:
            QMessageBox.critical(self, "Background Error", f"Error applying background subtraction:\n{str(e)}")

    def _apply_background_subtraction(self):
        """Apply background subtraction to RPW-TNR data"""
        rpw_file = self.rpw_tnr_edit.text().strip()
        if not rpw_file:
            raise ValueError("No RPW-TNR file specified")
        
        # Get polling function
        bkg_poll_function = self.bkg_poll_combo.currentText()
        
        # Load raw RPW data
        rpw_data = rpw_get_data(rpw_file)
        
        if self.bkg_time_radio.isChecked():
            # Apply time range background subtraction
            start_datetime = self.bkg_start_datetime.dateTime().toPython()
            end_datetime = self.bkg_end_datetime.dateTime().toPython()
            
            # Format datetime strings as expected by sololab
            start_str = start_datetime.strftime("%d-%b-%Y %H:%M:%S")
            end_str = end_datetime.strftime("%d-%b-%Y %H:%M:%S")
            bkg_range = (start_str, end_str)
            
            # Create PSD with background subtraction
            processed_data = rpw_create_PSD(rpw_data, which_freqs="non_zero", 
                                                  rpw_bkg_interval=bkg_range, 
                                                  bkg_poll_function=bkg_poll_function)
        else:
            # No background subtraction
            processed_data = rpw_create_PSD(rpw_data, which_freqs="non_zero")
        
        return processed_data

    def _get_background_info(self):
        """Get background information string for display"""
        if self.bkg_time_radio.isChecked():
            start_time = self.bkg_start_datetime.dateTime().toString("yyyy-MM-dd hh:mm:ss")
            end_time = self.bkg_end_datetime.dateTime().toString("yyyy-MM-dd hh:mm:ss")
            poll_func = self.bkg_poll_combo.currentText()
            return f"BKG time range: {start_time} to {end_time} | BKG poll funct.: {poll_func}"
        else:
            return "No background subtraction applied"

    def _load_rpw_data(self):
        """Load and process RPW-TNR data, then close dialog"""
        if not self.rpw_tnr_edit.text().strip():
            QMessageBox.warning(self, "No File", "Please select an RPW-TNR file first.")
            return
        
        try:
            # Apply background subtraction if selected, otherwise load raw data
            if self.bkg_time_radio.isChecked():
                self.processed_rpw_data = self._apply_background_subtraction()
            else:
                # Load raw data
                rpw_data = rpw_get_data(self.rpw_tnr_edit.text().strip())
                self.processed_rpw_data = rpw_create_PSD(rpw_data, which_freqs="non_zero")
            
            # Accept the dialog with processed data
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Error loading RPW-TNR data:\n{str(e)}")

    def _toggle_bkg_controls(self):
        is_time_selected = self.bkg_time_radio.isChecked()
        
        # Enable/disable datetime controls
        self.bkg_start_datetime.setEnabled(is_time_selected)
        self.bkg_end_datetime.setEnabled(is_time_selected)
        
        # Update background preview button
        has_file = bool(self.rpw_tnr_edit.text().strip())
        self.preview_bkg_btn.setEnabled(has_file and is_time_selected)

    def get_values(self):
        bkg_option = self.bkg_button_group.checkedId()
        result = {
            "rpw_tnr_file": self.rpw_tnr_edit.text().strip() or None,
            "bkg_option": bkg_option,  # 0: no bkg, 1: time range
            "bkg_poll_function": self.bkg_poll_combo.currentText(),
            "rpw_data": self.processed_rpw_data,  # Include processed data
        }
        
        if bkg_option == 1:  # Time range
            result["bkg_start_datetime"] = self.bkg_start_datetime.dateTime().toPython()
            result["bkg_end_datetime"] = self.bkg_end_datetime.dateTime().toPython()
        
        return result


# ... ImportEpdDialog and PlotPrefsDialog classes remain the same ...

class ImportEpdDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import EPD data")
        self.resize(800, 600)
        
        # Store the downloaded data
        self.df_protons_ept = None
        self.df_electrons_ept = None
        self.energies_ept = None

        layout = QVBoxLayout(self)

        # Main data section
        main_form = QFormLayout()
        
        # Download path selection
        self.path_edit = QLineEdit()
        path_btn = QPushButton("Browse...")
        path_btn.clicked.connect(self._browse_path)
        h1 = QHBoxLayout()
        h1.addWidget(self.path_edit)
        h1.addWidget(path_btn)
        main_form.addRow("Download path:", h1)
        
        # Observation date
        today = QDate.currentDate()
        self.obs_date = QDateEdit(today)
        self.obs_date.setCalendarPopup(True)
        self.obs_date.setDisplayFormat("yyyy-MM-dd")
        main_form.addRow("Observation date:", self.obs_date)
        
        # Particle selection
        self.particle_combo = QComboBox()
        self.particle_combo.addItems(["Electron", "Proton"])
        main_form.addRow("Particle type:", self.particle_combo)
        
        # Resample selection
        self.resample_combo = QComboBox()
        self.resample_combo.addItems(["30sec", "1min", "2min", "5min", "10min"])
        self.resample_combo.setCurrentText("1min")  # Default
        main_form.addRow("Resample:", self.resample_combo)
        
        layout.addLayout(main_form)
        
        # Download button
        download_layout = QHBoxLayout()
        self.download_btn = QPushButton("Download EPD Data")
        self.download_btn.setEnabled(False)
        self.download_btn.clicked.connect(self._download_epd_data)
        download_layout.addWidget(self.download_btn)
        layout.addLayout(download_layout)
        
        # Connect path change to enable download button
        self.path_edit.textChanged.connect(self._on_path_changed)

        # Preview section
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        # Create matplotlib figure and canvas
        self.preview_figure = Figure(figsize=(11, 5))
        self.preview_canvas = FigureCanvas(self.preview_figure)
        preview_layout.addWidget(self.preview_canvas)
        
        # Initially hide the preview section
        preview_group.setVisible(False)
        self.preview_group = preview_group
        
        layout.addWidget(preview_group)
        
        # Preview and Load buttons
        btn_layout = QHBoxLayout()
        
        self.preview_btn = QPushButton("Plot Preview")
        self.preview_btn.setEnabled(False)
        self.preview_btn.clicked.connect(self._plot_preview)
        btn_layout.addWidget(self.preview_btn)
        
        self.load_btn = QPushButton("LOAD")
        self.load_btn.setEnabled(False)
        self.load_btn.clicked.connect(self._load_epd_data)
        btn_layout.addWidget(self.load_btn)
        
        layout.addLayout(btn_layout)

        # Cancel button
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_path(self):
        """Browse for download directory"""
        path = QFileDialog.getExistingDirectory(self, "Select Download Directory")
        if path:
            self.path_edit.setText(path)

    def _on_path_changed(self):
        """Enable/disable download button based on path"""
        has_path = bool(self.path_edit.text().strip())
        self.download_btn.setEnabled(has_path)
        
        # Reset data and buttons if path changes
        if not has_path:
            self.df_protons_ept = None
            self.df_electrons_ept = None
            self.energies_ept = None
            self.preview_btn.setEnabled(False)
            self.load_btn.setEnabled(False)
            self.preview_group.setVisible(False)

    def _download_epd_data(self):
        """Download EPD data using epd_load"""
        download_path = self.path_edit.text().strip()
        if not download_path:
            QMessageBox.warning(self, "No Path", "Please select a download path first.")
            return
        
        # Get the selected date
        selected_date = self.obs_date.date().toPython()
        date_format = int(selected_date.strftime("%Y%m%d"))
        
        try:
            # Show progress message
            QMessageBox.information(self, "Downloading", "Downloading EPD data... This may take a moment.")
            
            # Download EPD data
            self.df_protons_ept, self.df_electrons_ept, self.energies_ept = epd_load(
                sensor='ept', 
                level='l2', 
                startdate=date_format,
                enddate=date_format, 
                viewing='sun', 
                path=download_path, 
                autodownload=True
            )
            
            # Enable preview and load buttons
            self.preview_btn.setEnabled(True)
            self.load_btn.setEnabled(True)
            
            QMessageBox.information(self, "Download Complete", 
                                  f"EPD data downloaded successfully!\n\nDate: {selected_date}\nPath: {download_path}")
            
            print(f"EPD data downloaded for {selected_date}")
            print(f"Protons data shape: {self.df_protons_ept.shape if self.df_protons_ept is not None else 'None'}")
            print(f"Electrons data shape: {self.df_electrons_ept.shape if self.df_electrons_ept is not None else 'None'}")
            print(f"Energies loaded: {self.energies_ept is not None}")
            
        except Exception as e:
            QMessageBox.critical(self, "Download Error", f"Error downloading EPD data:\n{str(e)}")
            print(f"EPD download error: {e}")

    def _plot_preview(self):
        """Plot preview of EPD data"""
        if self.df_protons_ept is None or self.df_electrons_ept is None or self.energies_ept is None:
            QMessageBox.warning(self, "No Data", "Please download EPD data first.")
            return
        
        try:
            # Get selected parameters
            particle = self.particle_combo.currentText()
            resample = self.resample_combo.currentText()
            
            # Select the appropriate data based on particle choice
            if particle == "Electron":
                epd_data = self.df_electrons_ept
            else:  # Proton
                epd_data = self.df_protons_ept
            
            # Create date range string from the observation date
            selected_date = self.obs_date.date().toPython()
            date_range_str = [
                selected_date.strftime("%d-%b-%Y %H:%M:%S").replace("00:00:00", "00:00:00"),
                selected_date.strftime("%d-%b-%Y %H:%M:%S").replace("00:00:00", "23:59:59")
            ]
            
            # Clear the previous plot
            self.preview_figure.clear()
            
            # Create axes in the preview figure
            ax = self.preview_figure.add_subplot(111)
            
            # Plot EPD data using sololab function
            plot_ept_data(
                epd_data, 
                self.energies_ept, 
                ax=ax, 
                particle=particle,
                channels=[2, 6, 14, 18, 26],
                resample=resample,
                date_range=date_range_str,
                round_epd_label=True
            )
            
            # Format x-axis to show time as HH:MM
            
            time_formatter = DateFormatter('%H:%M')
            ax.xaxis.set_major_formatter(time_formatter)
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Adjust the plot title
            ax.set_title(f'EPD {particle} Data Preview\nDate: {selected_date} | Resample: {resample}', fontsize=10)
            
            # Tight layout and refresh canvas
            self.preview_figure.tight_layout()
            self.preview_canvas.draw()
            
            # Show the preview section
            self.preview_group.setVisible(True)
            
            QMessageBox.information(self, "Preview Generated", 
                                f"EPD {particle} data preview generated successfully.\n\nDate: {selected_date}\nResample: {resample}")
            
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", f"Error generating EPD preview:\n{str(e)}")
            print(f"EPD preview error: {e}")

    def _load_epd_data(self):
        """Load EPD data and close dialog"""
        if self.df_protons_ept is None or self.df_electrons_ept is None or self.energies_ept is None:
            QMessageBox.warning(self, "No Data", "Please download EPD data first.")
            return
        
        # Accept the dialog with loaded data
        self.accept()

    def get_values(self):
        """Return the downloaded EPD data and parameters"""
        selected_date = self.obs_date.date().toPython()
        return {
            "epd_date": selected_date,
            "download_path": self.path_edit.text().strip(),
            "particle": self.particle_combo.currentText(),
            "resample": self.resample_combo.currentText(),
            "df_protons_ept": self.df_protons_ept,
            "df_electrons_ept": self.df_electrons_ept,
            "energies_ept": self.energies_ept,
        }

class EpdChannelSelectionDialog(QDialog):
    def __init__(self, parent=None, energies_ept=None, particle="Electron", initial_channels=None):
        super().__init__(parent)
        self.setWindowTitle("Select EPD Energy Channels")
        self.resize(500, 400)
        
        self.energies_ept = energies_ept
        self.particle = particle
        self.selected_channels = initial_channels[:] if initial_channels else []
        
        layout = QVBoxLayout(self)
        
        # Title label
        title_label = QLabel(f"Selected Energy Channels for {particle}")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title_label)
        
        # Main horizontal layout
        main_layout = QHBoxLayout()
        
        # Left side: Selected channels list
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Selected Channels:"))
        
        self.channels_list = QListWidget()
        
        
        main_layout.addLayout(left_layout)
        
        # Right side: Control buttons
        buttons_layout = QVBoxLayout()
        
        self.add_btn = QPushButton("Add Channel")
        self.add_btn.clicked.connect(self._add_channel)
        buttons_layout.addWidget(self.add_btn)
        
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.setEnabled(False)
        self.remove_btn.clicked.connect(self._remove_channel)
        buttons_layout.addWidget(self.remove_btn)
        
        # Add Remove All button
        self.remove_all_btn = QPushButton("Remove All")
        self.remove_all_btn.clicked.connect(self._remove_all_channels)
        self.remove_all_btn.setEnabled(False)  # Initially disabled
        buttons_layout.addWidget(self.remove_all_btn)
        
        buttons_layout.addStretch()
        
        self.done_btn = QPushButton("Done")
        self.done_btn.clicked.connect(self.accept)
        buttons_layout.addWidget(self.done_btn)
        
        main_layout.addLayout(buttons_layout)
        layout.addLayout(main_layout)
        
        
        # Populate the channels list with initial channels
        self._update_channels_list()
        left_layout.addWidget(self.channels_list)
        
        # Connect selection change to enable/disable remove button
        self.channels_list.itemSelectionChanged.connect(self._on_selection_changed)
    
    def _update_channels_list(self):
        """Update the channels list widget"""
        self.channels_list.clear()
        
        if not self.energies_ept:
            self.remove_all_btn.setEnabled(False)
            return
        
        # Get energy list for current particle
        en_list = self.energies_ept[f"{self.particle}_Bins_Text"].tolist()
        
        for channel_idx in self.selected_channels:
            if channel_idx < len(en_list):
                channel_text = f"{channel_idx} : {en_list[channel_idx][0]}"
                item = QListWidgetItem(channel_text)
                self.channels_list.addItem(item)
        
        # Update Remove All button state
        self.remove_all_btn.setEnabled(len(self.selected_channels) > 0)
    
    def _on_selection_changed(self):
        """Enable/disable remove button based on selection"""
        has_selection = len(self.channels_list.selectedItems()) > 0
        self.remove_btn.setEnabled(has_selection)
    
    def _add_channel(self):
        """Open dialog to add a new channel"""
        if not self.energies_ept:
            QMessageBox.warning(self, "No Data", "No EPD energy data available.")
            return
        
        dlg = AddChannelDialog(self, self.energies_ept, self.particle, self.selected_channels)
        if dlg.exec_() == QDialog.Accepted:
            new_channel = dlg.get_selected_channel()
            if new_channel is not None and new_channel not in self.selected_channels:
                self.selected_channels.append(new_channel)
                self.selected_channels.sort()  # Keep channels sorted
                self._update_channels_list()
    
    def _remove_channel(self):
        """Remove selected channel from list"""
        current_row = self.channels_list.currentRow()
        if current_row >= 0 and current_row < len(self.selected_channels):
            del self.selected_channels[current_row]
            self._update_channels_list()
    
    def _remove_all_channels(self):
        """Remove all selected channels with confirmation"""
        if len(self.selected_channels) == 0:
            return
        
        # Ask for confirmation
        reply = QMessageBox.question(self, "Remove All Channels", 
                                   "Are you sure you want to remove all selected channels?",
                                   QMessageBox.Yes | QMessageBox.No, 
                                   QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.selected_channels.clear()
            self._update_channels_list()
            QMessageBox.information(self, "Removed", "All selected channels have been removed.")
    
    def get_selected_channels(self):
        """Return list of selected channel indices as integers"""
        return self.selected_channels[:]


class AddChannelDialog(QDialog):
    def __init__(self, parent=None, energies_ept=None, particle="Electron", existing_channels=None):
        super().__init__(parent)
        self.setWindowTitle("Add Energy Channel")
        self.resize(400, 300)
        
        self.energies_ept = energies_ept
        self.particle = particle
        self.existing_channels = existing_channels or []
        
        layout = QVBoxLayout(self)
        
        # Title
        layout.addWidget(QLabel(f"Available {particle} Energy Channels:"))
        
        # Channel selection list
        self.channel_list = QListWidget()
        self._populate_channel_list()
        layout.addWidget(self.channel_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        self.done_btn = QPushButton("Done")
        self.done_btn.clicked.connect(self._add_selected_channel)
        self.done_btn.setEnabled(False)
        button_layout.addWidget(self.done_btn)
        
        layout.addLayout(button_layout)
        
        # Connect selection change to enable/disable done button
        self.channel_list.itemSelectionChanged.connect(self._on_selection_changed)
        
        # Connect double-click to add channel directly
        self.channel_list.itemDoubleClicked.connect(self._add_selected_channel)
    
    def _populate_channel_list(self):
        """Populate the channel selection list"""
        if not self.energies_ept:
            return
        
        # Get energy list for current particle
        en_list = self.energies_ept[f"{self.particle}_Bins_Text"].tolist()
        
        # Create channel list
        for k in range(len(en_list)):
            if k not in self.existing_channels:  # Only show channels not already selected
                channel_text = f"{k} : {en_list[k][0]}"
                item = QListWidgetItem(channel_text)
                item.setData(Qt.UserRole, k)  # Store the channel index
                self.channel_list.addItem(item)
    
    def _on_selection_changed(self):
        """Enable/disable done button based on selection"""
        has_selection = len(self.channel_list.selectedItems()) > 0
        self.done_btn.setEnabled(has_selection)
    
    def _add_selected_channel(self):
        """Add the selected channel and close dialog"""
        selected_items = self.channel_list.selectedItems()
        if selected_items:
            self.selected_channel = selected_items[0].data(Qt.UserRole)
            self.accept()
    
    def get_selected_channel(self):
        """Return the selected channel index"""
        return getattr(self, 'selected_channel', None)

class EnergyRangeDialog(QDialog):
    def __init__(self, parent=None, energy_range=None):
        super().__init__(parent)
        self.setWindowTitle("Energy Range")
        self.resize(300, 150)
        
        layout = QVBoxLayout(self)
        
        # Form layout for energy inputs
        form_layout = QFormLayout()
        
        # Min energy (4-150 keV)
        self.min_energy = QLineEdit()
        self.min_energy.setText(str(energy_range[0]) if energy_range else "4")
        form_layout.addRow("Min Energy (keV):", self.min_energy)
        
        # Max energy (4-150 keV)
        self.max_energy = QLineEdit()
        self.max_energy.setText(str(energy_range[1]) if energy_range else "12")
        form_layout.addRow("Max Energy (keV):", self.max_energy)
        
        layout.addLayout(form_layout)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        buttons.rejected.connect(self.reject)
        
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self._save_energy_range)
        buttons.addButton(self.save_btn, QDialogButtonBox.ActionRole)
        
        layout.addWidget(buttons)
    
    def _save_energy_range(self):
        """Validate and save energy range"""
        try:
            min_energy = int(self.min_energy.text())
            max_energy = int(self.max_energy.text())
            
            # Validate range
            if not (4 <= min_energy <= 150) or not (4 <= max_energy <= 150):
                QMessageBox.warning(self, "Invalid Range", "Energy values must be between 4 and 150 keV.")
                return
            
            if min_energy >= max_energy:
                QMessageBox.warning(self, "Invalid Range", "Minimum energy must be smaller than maximum energy.")
                return
            
            self.energy_range = [min_energy, max_energy]
            self.accept()
            
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid integer values.")
    
    def get_energy_range(self):
        return getattr(self, 'energy_range', None)


class EnergyRangesDialog(QDialog):
    def __init__(self, parent=None, initial_ranges=None):
        super().__init__(parent)
        self.setWindowTitle("Set Energy Ranges")
        self.resize(400, 300)
        
        self.energy_ranges = initial_ranges[:] if initial_ranges else []
        
        layout = QVBoxLayout(self)
        
        # List widget for energy ranges
        self.ranges_list = QWidget()
        list_layout = QVBoxLayout(self.ranges_list)
        list_layout.addWidget(QLabel("Energy Ranges (keV):"))
        
        self.list_widget = QListWidget()
        
        
        layout.addWidget(self.ranges_list)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.add_btn = QPushButton("Add")
        self.add_btn.clicked.connect(self._add_range)
        btn_layout.addWidget(self.add_btn)
        
        self.edit_btn = QPushButton("Edit")
        self.edit_btn.clicked.connect(self._edit_range)
        self.edit_btn.setEnabled(False)
        btn_layout.addWidget(self.edit_btn)
        
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._delete_range)
        self.delete_btn.setEnabled(False)
        btn_layout.addWidget(self.delete_btn)
        
        # Add Remove All button
        self.remove_all_btn = QPushButton("Remove All")
        self.remove_all_btn.clicked.connect(self._remove_all_ranges)
        self.remove_all_btn.setEnabled(False)  # Initially disabled
        btn_layout.addWidget(self.remove_all_btn)
        
        layout.addLayout(btn_layout)
        
        # Populate the list widget with initial energy ranges
        self._update_list()
        list_layout.addWidget(self.list_widget)
        
        # Connect selection change
        self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        
        # Done button
        done_layout = QHBoxLayout()
        done_btn = QPushButton("Done")
        done_btn.clicked.connect(self.accept)
        done_layout.addWidget(done_btn)
        layout.addLayout(done_layout)
    
    def _update_list(self):
        """Update the list widget with current energy ranges"""
        self.list_widget.clear()
        for energy_range in self.energy_ranges:
            item_text = f"{energy_range[0]} - {energy_range[1]} keV"
            item = QListWidgetItem(item_text)
            self.list_widget.addItem(item)
        
        # Update Remove All button state
        self.remove_all_btn.setEnabled(len(self.energy_ranges) > 0)
    
    def _on_selection_changed(self):
        """Enable/disable edit and delete buttons based on selection"""
        has_selection = len(self.list_widget.selectedItems()) > 0
        self.edit_btn.setEnabled(has_selection)
        self.delete_btn.setEnabled(has_selection)
    
    def _add_range(self):
        """Add new energy range"""
        dlg = EnergyRangeDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            energy_range = dlg.get_energy_range()
            if energy_range:
                self.energy_ranges.append(energy_range)
                self._update_list()
    
    def _edit_range(self):
        """Edit selected energy range"""
        current_row = self.list_widget.currentRow()
        if current_row >= 0:
            current_range = self.energy_ranges[current_row]
            dlg = EnergyRangeDialog(self, current_range)
            if dlg.exec_() == QDialog.Accepted:
                energy_range = dlg.get_energy_range()
                if energy_range:
                    self.energy_ranges[current_row] = energy_range
                    self._update_list()
    
    def _delete_range(self):
        """Delete selected energy range"""
        current_row = self.list_widget.currentRow()
        if current_row >= 0:
            del self.energy_ranges[current_row]
            self._update_list()
    
    def _remove_all_ranges(self):
        """Remove all energy ranges with confirmation"""
        if len(self.energy_ranges) == 0:
            return
        
        # Ask for confirmation
        reply = QMessageBox.question(self, "Remove All Ranges", 
                                   "Are you sure you want to remove all energy ranges?",
                                   QMessageBox.Yes | QMessageBox.No, 
                                   QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.energy_ranges.clear()
            self._update_list()
            QMessageBox.information(self, "Removed", "All energy ranges have been removed.")
    
    def get_energy_ranges(self):
        return self.energy_ranges
class RpwFrequencySelectionDialog(QDialog):
    def __init__(self, parent=None, rpw_data=None, data_type="HFR", initial_frequencies=None):
        super().__init__(parent)
        self.setWindowTitle(f"Select RPW-{data_type} Frequencies")
        self.resize(500, 400)
        
        self.rpw_data = rpw_data
        self.data_type = data_type
        self.selected_frequencies = initial_frequencies[:] if initial_frequencies else []
        
        layout = QVBoxLayout(self)
        
        # Title label
        title_label = QLabel(f"Selected Frequencies for RPW-{data_type}")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title_label)
        
        # Main horizontal layout
        main_layout = QHBoxLayout()
        
        # Left side: Selected frequencies list
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Frequencies (kHz):"))
        
        self.frequencies_list = QListWidget()
       
        
        main_layout.addLayout(left_layout)
        
        # Right side: Control buttons
        buttons_layout = QVBoxLayout()
        
        self.add_btn = QPushButton("Add Frequency")
        self.add_btn.clicked.connect(self._add_frequency)
        buttons_layout.addWidget(self.add_btn)
        
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.setEnabled(False)
        self.remove_btn.clicked.connect(self._remove_frequency)
        buttons_layout.addWidget(self.remove_btn)
        
        # Add Remove All button
        self.remove_all_btn = QPushButton("Remove All")
        self.remove_all_btn.clicked.connect(self._remove_all_frequencies)
        self.remove_all_btn.setEnabled(False)  # Initially disabled
        buttons_layout.addWidget(self.remove_all_btn)
        
        buttons_layout.addStretch()
        
        self.done_btn = QPushButton("Done")
        self.done_btn.clicked.connect(self.accept)
        buttons_layout.addWidget(self.done_btn)
        
        
        # Populate the frequencies list with initial frequencies
        self._update_frequencies_list()
        left_layout.addWidget(self.frequencies_list)
        
        main_layout.addLayout(buttons_layout)
        layout.addLayout(main_layout)
        
        # Connect selection change to enable/disable remove button
        self.frequencies_list.itemSelectionChanged.connect(self._on_selection_changed)
    
    def _update_frequencies_list(self):
        """Update the frequencies list widget"""
        self.frequencies_list.clear()
        
        for freq in self.selected_frequencies:
            freq_text = f"{freq:.2f} kHz"
            item = QListWidgetItem(freq_text)
            self.frequencies_list.addItem(item)
        
        # Update Remove All button state
        self.remove_all_btn.setEnabled(len(self.selected_frequencies) > 0)
    
    def _on_selection_changed(self):
        """Enable/disable remove button based on selection"""
        has_selection = len(self.frequencies_list.selectedItems()) > 0
        self.remove_btn.setEnabled(has_selection)
    
    def _add_frequency(self):
        """Open dialog to add a new frequency"""
        if not self.rpw_data:
            QMessageBox.warning(self, "No Data", "No RPW data available.")
            return
        
        dlg = AddFrequencyDialog(self, self.rpw_data, self.data_type, self.selected_frequencies)
        if dlg.exec_() == QDialog.Accepted:
            new_frequency = dlg.get_selected_frequency()
            if new_frequency is not None and new_frequency not in self.selected_frequencies:
                self.selected_frequencies.append(new_frequency)
                self.selected_frequencies.sort()  # Keep frequencies sorted
                self._update_frequencies_list()
    
    def _remove_frequency(self):
        """Remove selected frequency from list"""
        current_row = self.frequencies_list.currentRow()
        if current_row >= 0 and current_row < len(self.selected_frequencies):
            del self.selected_frequencies[current_row]
            self._update_frequencies_list()
    
    def _remove_all_frequencies(self):
        """Remove all selected frequencies with confirmation"""
        if len(self.selected_frequencies) == 0:
            return
        
        # Ask for confirmation
        reply = QMessageBox.question(self, "Remove All Frequencies", 
                                   "Are you sure you want to remove all selected frequencies?",
                                   QMessageBox.Yes | QMessageBox.No, 
                                   QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.selected_frequencies.clear()
            self._update_frequencies_list()
            QMessageBox.information(self, "Removed", "All selected frequencies have been removed.")
    
    def get_selected_frequencies(self):
        """Return list of selected frequencies as floats"""
        return self.selected_frequencies[:]


class AddFrequencyDialog(QDialog):
    def __init__(self, parent=None, rpw_data=None, data_type="HFR", existing_frequencies=None):
        super().__init__(parent)
        self.setWindowTitle("Add Frequency")
        self.resize(400, 300)
        
        self.rpw_data = rpw_data
        self.data_type = data_type
        self.existing_frequencies = existing_frequencies or []
        
        layout = QVBoxLayout(self)
        
        # Title
        layout.addWidget(QLabel(f"Available RPW-{data_type} Frequencies:"))
        
        # Frequency selection list
        self.frequency_list = QListWidget()
        self._populate_frequency_list()
        layout.addWidget(self.frequency_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        self.done_btn = QPushButton("Done")
        self.done_btn.clicked.connect(self._add_selected_frequency)
        self.done_btn.setEnabled(False)
        button_layout.addWidget(self.done_btn)
        
        layout.addLayout(button_layout)
        
        # Connect selection change to enable/disable done button
        self.frequency_list.itemSelectionChanged.connect(self._on_selection_changed)
        
        # Connect double-click to add frequency directly
        self.frequency_list.itemDoubleClicked.connect(self._add_selected_frequency)
    
    def _populate_frequency_list(self):
        """Populate the frequency selection list"""
        if not self.rpw_data or 'frequency' not in self.rpw_data:
            return
        
        # Get frequencies from RPW data
        frequencies = self.rpw_data['frequency']# in kHz
        print('populating',frequencies)
        
        # Create frequency list
        for freq in frequencies:
            freq_khz = freq# / 1000.0  # Convert Hz to kHz
            
            if freq_khz not in self.existing_frequencies:  # Only show frequencies not already selected
                
                freq_text = f"{freq_khz:.2f} kHz"
                #print('freq text ',freq_text )
                item = QListWidgetItem(freq_text)
                item.setData(Qt.UserRole, freq_khz)  # Store the frequency value in kHz
                self.frequency_list.addItem(item)
    
    def _on_selection_changed(self):
        """Enable/disable done button based on selection"""
        has_selection = len(self.frequency_list.selectedItems()) > 0
        self.done_btn.setEnabled(has_selection)
    
    def _add_selected_frequency(self):
        """Add the selected frequency and close dialog"""
        selected_items = self.frequency_list.selectedItems()
        if selected_items:
            self.selected_frequency = selected_items[0].data(Qt.UserRole)
            self.accept()
    
    def get_selected_frequency(self):
        """Return the selected frequency value"""
        return getattr(self, 'selected_frequency', None)
class PlotPrefsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plotting preferences")
        self.resize(520, 520)
        self.parent = parent  # Store reference to main window
        
        # Store energy ranges
        self.stix_energy_ranges = []

        v = QVBoxLayout(self)
        v.setSizeConstraint(QLayout.SetFixedSize)

        tabs = QTabWidget()
        v.addWidget(tabs)

        # STIX tab
        stix_widget = QWidget()
        stix_layout = QVBoxLayout(stix_widget)
        
        # Basic STIX controls
        stix_form = QFormLayout()
        self.stix_choice = QComboBox()
        self.stix_choice.addItems(["spectrogram", "time profiles", "overlay"])
        self.stix_choice.currentTextChanged.connect(self._on_stix_type_changed)
        stix_form.addRow("Type:", self.stix_choice)
        
        # Context-specific Log Y axis checkboxes for STIX
        self.stix_logy_energy = QCheckBox("Log Y axis (energy)")
        self.stix_logy_countrate = QCheckBox("Log Y axis (count rate)")
        self.stix_logy_energy_overlay = QCheckBox("Log Y axis (energy in spectrogram)")
        self.stix_logy_countrate_overlay = QCheckBox("Log Y axis (count rate in time profiles)")
        
        # Initially show only energy checkbox (spectrogram is default)
        self.stix_logy_countrate.setVisible(False)
        self.stix_logy_energy_overlay.setVisible(False)
        self.stix_logy_countrate_overlay.setVisible(False)
        
        stix_form.addRow(self.stix_logy_energy)
        stix_form.addRow(self.stix_logy_countrate)
        stix_form.addRow(self.stix_logy_energy_overlay)
        stix_form.addRow(self.stix_logy_countrate_overlay)
        
        # Log Z axis (only for spectrogram/overlay)
        self.stix_logz = QCheckBox("Log Z axis")
        stix_form.addRow(self.stix_logz)
        
        stix_layout.addLayout(stix_form)
        
        # Curve smoothing section (initially hidden)
        self.stix_smoothing_group = QGroupBox("Curve Smoothing")
        smoothing_layout = QFormLayout(self.stix_smoothing_group)
        
        # Smoothing points with spinbox
        smoothing_control_layout = QHBoxLayout()
        self.stix_smoothing_spinbox = QSpinBox()
        self.stix_smoothing_spinbox.setMinimum(1)
        self.stix_smoothing_spinbox.setMaximum(100)
        self.stix_smoothing_spinbox.setValue(1)
        smoothing_control_layout.addWidget(self.stix_smoothing_spinbox)
        smoothing_layout.addRow("Curve smoothing (points):", smoothing_control_layout)
        
        self.stix_smoothing_group.setVisible(False)
        stix_layout.addWidget(self.stix_smoothing_group)
        
        # Set energy range section (for spectrogram/overlay)
        self.stix_energy_range_group = QGroupBox("Energy Range")
        energy_range_layout = QVBoxLayout(self.stix_energy_range_group)
        
        self.stix_energy_range_checkbox = QCheckBox("Set energy range")
        self.stix_energy_range_checkbox.toggled.connect(self._toggle_stix_energy_range)
        energy_range_layout.addWidget(self.stix_energy_range_checkbox)
        
        # Energy range inputs
        energy_inputs_layout = QHBoxLayout()
        energy_inputs_layout.addWidget(QLabel("  "))  # Indent
        energy_inputs_layout.addWidget(QLabel("From:"))
        self.stix_energy_min = QSpinBox()
        self.stix_energy_min.setRange(4, 150)
        self.stix_energy_min.setValue(4)
        self.stix_energy_min.setSuffix(" keV")
        self.stix_energy_min.setEnabled(False)
        energy_inputs_layout.addWidget(self.stix_energy_min)
        
        energy_inputs_layout.addWidget(QLabel("to:"))
        self.stix_energy_max = QSpinBox()
        self.stix_energy_max.setRange(4, 150)
        self.stix_energy_max.setValue(28)
        self.stix_energy_max.setSuffix(" keV")
        self.stix_energy_max.setEnabled(False)
        energy_inputs_layout.addWidget(self.stix_energy_max)
        
        energy_range_layout.addLayout(energy_inputs_layout)
        
        self.stix_energy_range_group.setVisible(False)
        stix_layout.addWidget(self.stix_energy_range_group)
        
        # Energy ranges section (for time profiles/overlay - existing functionality)
        self.stix_energy_group = QGroupBox("Energy Ranges")
        energy_layout = QVBoxLayout(self.stix_energy_group)
        self.energy_ranges_btn = QPushButton("Set Energy Ranges")
        self.energy_ranges_btn.clicked.connect(self._set_energy_ranges)
        energy_layout.addWidget(self.energy_ranges_btn)
        self.stix_energy_group.setVisible(False)
        stix_layout.addWidget(self.stix_energy_group)
        
        # STIX preview button
        stix_preview_btn = QPushButton("Plot STIX Preview")
        stix_preview_btn.clicked.connect(self._plot_stix_preview)
        stix_layout.addWidget(stix_preview_btn)
        
        # Add STIX background plot button
        self.stix_bkg_btn = QPushButton("Plot STIX Background")
        self.stix_bkg_btn.clicked.connect(self._plot_stix_background)
        self.stix_bkg_btn.setEnabled(self._has_stix_background_data())
        stix_layout.addWidget(self.stix_bkg_btn)
        
        tabs.addTab(stix_widget, "STIX data")

        # RPW tab
        rpw_widget = QWidget()
        rpw_layout = QVBoxLayout(rpw_widget)
        
        # Basic RPW controls
        rpw_form = QFormLayout()
        self.rpw_choice = QComboBox()
        self.rpw_choice.addItems(["spectrogram", "time profiles", "overlay"])
        self.rpw_choice.currentTextChanged.connect(self._on_rpw_type_changed)
        rpw_form.addRow("Type:", self.rpw_choice)
        
        # Context-specific Log Y axis checkboxes for RPW
        self.rpw_logy_frequency = QCheckBox("Log Y axis (frequency)")
        self.rpw_logy_intensity = QCheckBox("Log Y axis (intensity)")
        self.rpw_logy_frequency_overlay = QCheckBox("Log Y axis (frequency in spectrogram)")
        self.rpw_logy_intensity_overlay = QCheckBox("Log Y axis (intensity in time profiles)")
        
        # Initially show only frequency checkbox (spectrogram is default)
        self.rpw_logy_intensity.setVisible(False)
        self.rpw_logy_frequency_overlay.setVisible(False)
        self.rpw_logy_intensity_overlay.setVisible(False)
        
        rpw_form.addRow(self.rpw_logy_frequency)
        rpw_form.addRow(self.rpw_logy_intensity)
        rpw_form.addRow(self.rpw_logy_frequency_overlay)
        rpw_form.addRow(self.rpw_logy_intensity_overlay)
        
        # Log Z axis (only for spectrogram/overlay)
        self.rpw_logz = QCheckBox("Log Z axis")
        rpw_form.addRow(self.rpw_logz)
        
        # Invert Y axis (only for spectrogram/overlay)
        self.rpw_invert_y = QCheckBox("Invert Y axis")
        self.rpw_invert_y.setChecked(True)  # Default True
        rpw_form.addRow(self.rpw_invert_y)
        
        
        # Add the missing RPW overlay choice
        self.rpw_overlay_choice = QComboBox()
        self.rpw_overlay_choice.addItems(["Only TNR", "Only HFR", "Both"])
        rpw_form.addRow("HFR-TNR overlay:", self.rpw_overlay_choice)

        rpw_layout.addLayout(rpw_form)
        
        # RPW Curve smoothing section (initially hidden)
        self.rpw_smoothing_group = QGroupBox("Curve Smoothing")
        rpw_smoothing_layout = QFormLayout(self.rpw_smoothing_group)
        
        # Smoothing points with spinbox
        rpw_smoothing_control_layout = QHBoxLayout()
        self.rpw_smoothing_spinbox = QSpinBox()
        self.rpw_smoothing_spinbox.setMinimum(1)
        self.rpw_smoothing_spinbox.setMaximum(100)
        self.rpw_smoothing_spinbox.setValue(1)
        rpw_smoothing_control_layout.addWidget(self.rpw_smoothing_spinbox)
        rpw_smoothing_layout.addRow("Curve smoothing (points):", rpw_smoothing_control_layout)
        
        self.rpw_smoothing_group.setVisible(False)
        rpw_layout.addWidget(self.rpw_smoothing_group)
        
        # Frequency range section
        self.freq_range_checkbox = QCheckBox("Set frequency range")
        self.freq_range_checkbox.toggled.connect(self._toggle_freq_range)
        rpw_layout.addWidget(self.freq_range_checkbox)
        
        freq_range_layout = QHBoxLayout()
        freq_range_layout.addWidget(QLabel("  "))  # Indent
        freq_range_layout.addWidget(QLabel("From:"))
        self.freq_min = QLineEdit("100")
        self.freq_min.setEnabled(False)
        freq_range_layout.addWidget(self.freq_min)
        freq_range_layout.addWidget(QLabel("to:"))
        self.freq_max = QLineEdit("8000")
        self.freq_max.setEnabled(False)
        freq_range_layout.addWidget(self.freq_max)
        freq_range_layout.addWidget(QLabel("kHz"))
        rpw_layout.addLayout(freq_range_layout)
        
        
        # Frequency selection section (for time profiles)
        self.rpw_freq_selection_group = QGroupBox("Frequency Selection")
        freq_sel_layout = QVBoxLayout(self.rpw_freq_selection_group)
        
        self.rpw_select_freq_btn = QPushButton("Select Frequencies")
        self.rpw_select_freq_btn.clicked.connect(self._select_rpw_frequencies)
        freq_sel_layout.addWidget(self.rpw_select_freq_btn)
        
        self.rpw_freq_selection_group.setVisible(False)
        rpw_layout.addWidget(self.rpw_freq_selection_group)
        
        # Store selected frequencies
        self.rpw_selected_frequencies = []  # Will store frequency values
        
        
        
        # RPW preview buttons
        rpw_hfr_preview_btn = QPushButton("Plot RPW-HFR Preview")
        rpw_hfr_preview_btn.clicked.connect(self._plot_rpw_hfr_preview)
        rpw_layout.addWidget(rpw_hfr_preview_btn)
        
        rpw_tnr_preview_btn = QPushButton("Plot RPW-TNR Preview")
        rpw_tnr_preview_btn.clicked.connect(self._plot_rpw_tnr_preview)
        rpw_layout.addWidget(rpw_tnr_preview_btn)
        
        # Add RPW background plot buttons
        self.rpw_hfr_bkg_btn = QPushButton("Plot RPW-HFR Background")
        self.rpw_hfr_bkg_btn.clicked.connect(self._plot_rpw_hfr_background)
        self.rpw_hfr_bkg_btn.setEnabled(self._has_rpw_hfr_background_data())
        rpw_layout.addWidget(self.rpw_hfr_bkg_btn)
        
        self.rpw_tnr_bkg_btn = QPushButton("Plot RPW-TNR Background")
        self.rpw_tnr_bkg_btn.clicked.connect(self._plot_rpw_tnr_background)
        self.rpw_tnr_bkg_btn.setEnabled(self._has_rpw_tnr_background_data())
        rpw_layout.addWidget(self.rpw_tnr_bkg_btn)
        
        tabs.addTab(rpw_widget, "RPW data")

        
        # Store selected EPD channels
        self.epd_selected_channels = [2, 6, 14, 18, 26]  # Default channels
        
        # EPD tab 
        epd_widget = QWidget()
        epd_layout = QVBoxLayout(epd_widget)
        
        epd_form = QFormLayout()
        self.epd_logy = QCheckBox("Log Y axis")
        epd_form.addRow(self.epd_logy)
        epd_layout.addLayout(epd_form)
        
        # EPD channel selection button
        self.epd_channels_btn = QPushButton("Select Energy Channels")
        self.epd_channels_btn.clicked.connect(self._select_epd_channels)
        self.epd_channels_btn.setEnabled(self._has_epd_data())
        epd_layout.addWidget(self.epd_channels_btn)
        
        # EPD preview button
        epd_preview_btn = QPushButton("Plot EPD Preview")
        epd_preview_btn.clicked.connect(self._plot_epd_preview)
        epd_layout.addWidget(epd_preview_btn)
        
        tabs.addTab(epd_widget, "EPD data")

        # buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        buttons.rejected.connect(self.reject)
        
        # Add Combined Plot button
        self.plot_btn = QPushButton("Combined Plot")
        self.plot_btn.clicked.connect(self._do_combined_plot)
        buttons.addButton(self.plot_btn, QDialogButtonBox.ActionRole)
        
        v.addWidget(buttons)
        
        

    
    def _on_stix_type_changed(self):
        """Show/hide controls based on STIX type"""
        stix_type = self.stix_choice.currentText()
        
        # Show/hide log Z axis (only for spectrogram/overlay)
        show_logz = stix_type in ("spectrogram", "overlay")
        self.stix_logz.setVisible(show_logz)
        
        # Show/hide appropriate log Y axis checkboxes based on type
        if stix_type == "spectrogram":
            # Show only energy checkbox
            self.stix_logy_energy.setVisible(True)
            self.stix_logy_countrate.setVisible(False)
            self.stix_logy_energy_overlay.setVisible(False)
            self.stix_logy_countrate_overlay.setVisible(False)
        elif stix_type == "time profiles":
            # Show only count rate checkbox
            self.stix_logy_energy.setVisible(False)
            self.stix_logy_countrate.setVisible(True)
            self.stix_logy_energy_overlay.setVisible(False)
            self.stix_logy_countrate_overlay.setVisible(False)
        elif stix_type == "overlay":
            # Show both overlay checkboxes
            self.stix_logy_energy.setVisible(False)
            self.stix_logy_countrate.setVisible(False)
            self.stix_logy_energy_overlay.setVisible(True)
            self.stix_logy_countrate_overlay.setVisible(True)
        
        # Show/hide curve smoothing (only for time profiles)
        show_smoothing = stix_type in ("time profiles", "overlay")
        self.stix_smoothing_group.setVisible(show_smoothing)
        
        # Show/hide energy range controls (for spectrogram/overlay)
        show_energy_range = stix_type in ("spectrogram", "overlay")
        self.stix_energy_range_group.setVisible(show_energy_range)
        
        # Show/hide energy ranges section (for time profiles/overlay - existing functionality)
        show_energy_ranges = stix_type in ("overlay", "time profiles")
        self.stix_energy_group.setVisible(show_energy_ranges)

    def _on_rpw_type_changed(self):
        """Show/hide controls based on RPW type"""
        rpw_type = self.rpw_choice.currentText()
        
        # Show/hide log Z axis (only for spectrogram/overlay)
        show_logz = rpw_type in ("spectrogram", "overlay")
        self.rpw_logz.setVisible(show_logz)
        
        # Show/hide invert Y axis (only for spectrogram/overlay)
        show_invert_y = rpw_type in ("spectrogram", "overlay")
        self.rpw_invert_y.setVisible(show_invert_y)
        
        # Show/hide appropriate log Y axis checkboxes based on type
        if rpw_type == "spectrogram":
            # Show only frequency checkbox
            self.rpw_logy_frequency.setVisible(True)
            self.rpw_logy_intensity.setVisible(False)
            self.rpw_logy_frequency_overlay.setVisible(False)
            self.rpw_logy_intensity_overlay.setVisible(False)
        elif rpw_type == "time profiles":
            # Show only intensity checkbox
            self.rpw_logy_frequency.setVisible(False)
            self.rpw_logy_intensity.setVisible(True)
            self.rpw_logy_frequency_overlay.setVisible(False)
            self.rpw_logy_intensity_overlay.setVisible(False)
        elif rpw_type == "overlay":
            # Show both overlay checkboxes
            self.rpw_logy_frequency.setVisible(False)
            self.rpw_logy_intensity.setVisible(False)
            self.rpw_logy_frequency_overlay.setVisible(True)
            self.rpw_logy_intensity_overlay.setVisible(True)
        
        # Show/hide curve smoothing (only for time profiles)
        show_smoothing = rpw_type in ("time profiles", "overlay")
        self.rpw_smoothing_group.setVisible(show_smoothing)
        
        # Show/hide frequency selection (time profiles and overlay)
        show_freq_selection = rpw_type in ("time profiles", "overlay")
        self.rpw_freq_selection_group.setVisible(show_freq_selection)
    
    def _select_rpw_frequencies(self):
        """Open RPW frequency selection dialog"""
        # Determine which RPW data to use (HFR or TNR based on current context or user preference)
        rpw_data = None
        data_type = "HFR"  # Default
        
        if self.parent and self.parent.rpw_hfr_data:
            rpw_data = self.parent.rpw_hfr_data
            data_type = "HFR"
            print("HFR data")
        elif self.parent and self.parent.rpw_tnr_data:
            rpw_data = self.parent.rpw_tnr_data
            data_type = "TNR"
            print("TNR data")
        
        if not rpw_data:
            QMessageBox.warning(self, "No Data", "No RPW data available for frequency selection.")
            return
        
        dlg = RpwFrequencySelectionDialog(self, rpw_data, data_type, self.rpw_selected_frequencies)
        dlg._update_frequencies_list()
        if dlg.exec_() == QDialog.Accepted:
            self.rpw_selected_frequencies = dlg.get_selected_frequencies()
            QMessageBox.information(self, "Frequencies Updated", 
                                  f"Selected frequencies: {self.rpw_selected_frequencies}")

    def retrieve_hfr_frequency_list(self):
        """Retrieve selected HFR frequencies"""
        return self.rpw_selected_frequencies

    def retrieve_tnr_frequency_list(self):
        """Retrieve selected TNR frequencies"""
        return self.rpw_selected_frequencies
    
    def _has_epd_data(self):
        """Check if EPD data is available"""
        if not self.parent:
            return False
        return (self.parent.df_protons_ept is not None or self.parent.df_electrons_ept is not None) and self.parent.energies_ept is not None

    def _select_epd_channels(self):
        """Open EPD channel selection dialog"""
        if not self.parent or not self.parent.energies_ept:
            QMessageBox.warning(self, "No Data", "No EPD energy data available.")
            return
        
        # Use the current particle selection from parent
        particle = self.parent.epd_particle if self.parent.epd_particle else "Electron"
        
        dlg = EpdChannelSelectionDialog(self, self.parent.energies_ept, particle, self.epd_selected_channels)
        if dlg.exec_() == QDialog.Accepted:
            self.epd_selected_channels = dlg.get_selected_channels()
            QMessageBox.information(self, "Channels Updated", 
                                  f"Selected channels: {self.epd_selected_channels}")

    def _toggle_stix_energy_range(self):
        """Enable/disable STIX energy range inputs"""
        enabled = self.stix_energy_range_checkbox.isChecked()
        self.stix_energy_min.setEnabled(enabled)
        self.stix_energy_max.setEnabled(enabled)

    def _toggle_freq_range(self):
        """Enable/disable frequency range inputs"""
        enabled = self.freq_range_checkbox.isChecked()
        self.freq_min.setEnabled(enabled)
        self.freq_max.setEnabled(enabled)

    def retrieve_stix_energy_range(self):
        """Retrieve STIX energy range as a list [min, max]"""
        if self.stix_energy_range_checkbox.isChecked():
            return [self.stix_energy_min.value(), self.stix_energy_max.value()]
        else:
            return None

    def _set_energy_ranges(self):
        """Open energy ranges dialog"""
        dlg = EnergyRangesDialog(self, self.stix_energy_ranges)
        if dlg.exec_() == QDialog.Accepted:
            self.stix_energy_ranges = dlg.get_energy_ranges()
            print(f"Energy ranges set: {self.stix_energy_ranges}")
            
            
    # Update the get_values method to include new parameters
    def get_values(self):
        # Get frequency range
        freq_range = None
        if self.freq_range_checkbox.isChecked():
            try:
                freq_min = float(self.freq_min.text())
                freq_max = float(self.freq_max.text())
                if 1 <= freq_min < freq_max <= 16400:
                    freq_range = [freq_min, freq_max]
            except ValueError:
                pass  # Use default if invalid
        
        # Determine which log Y axis to use based on STIX type
        stix_type = self.stix_choice.currentText()
        if stix_type == "spectrogram":
            stix_logy = self.stix_logy_energy.isChecked()
        elif stix_type == "time profiles":
            stix_logy = self.stix_logy_countrate.isChecked()
        elif stix_type == "overlay":
            # For overlay, we might need both values
            stix_logy = {
                'energy': self.stix_logy_energy_overlay.isChecked(),
                'countrate': self.stix_logy_countrate_overlay.isChecked()
            }
        
        # Determine which log Y axis to use based on RPW type
        rpw_type = self.rpw_choice.currentText()
        if rpw_type == "spectrogram":
            rpw_logy = self.rpw_logy_frequency.isChecked()
        elif rpw_type == "time profiles":
            rpw_logy = self.rpw_logy_intensity.isChecked()
        elif rpw_type == "overlay":
            # For overlay, we might need both values
            rpw_logy = {
                'frequency': self.rpw_logy_frequency_overlay.isChecked(),
                'intensity': self.rpw_logy_intensity_overlay.isChecked()
            }
        
        return {
            "stix": {
                "type": self.stix_choice.currentText(),
                "logy": stix_logy,
                "logz": self.stix_logz.isChecked(),
                "energy_ranges": self.stix_energy_ranges,
                "energy_range": self.retrieve_stix_energy_range(),
                "smoothing_points": self.stix_smoothing_spinbox.value(),
            },
            "rpw": {
                "type": self.rpw_choice.currentText(),
                "logy": rpw_logy,
                "logz": self.rpw_logz.isChecked(),
                "invert_y": self.rpw_invert_y.isChecked(),  # Add invert Y axis
                "overlay": self.rpw_overlay_choice.currentText(),
                "freq_range": freq_range,
                "smoothing_points": self.rpw_smoothing_spinbox.value(),
                "selected_frequencies": self.rpw_selected_frequencies,  # Add selected frequencies
            },
            "epd": {
                "logy": self.epd_logy.isChecked(),
                "selected_channels": self.epd_selected_channels
            },
        }


    # Update preview methods to use appropriate log Y axis settings
  
    def _plot_stix_preview(self):
        """Plot STIX preview"""
        if not self.parent or not self.parent.stix_counts_data:
            QMessageBox.warning(self, "No Data", "No STIX data loaded.")
            return
        
        try:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            
            stix_type = self.stix_choice.currentText()
            
            # Get appropriate log Y axis setting based on type
            if stix_type == "spectrogram":
                logy = self.stix_logy_energy.isChecked()
            elif stix_type == "time profiles":
                logy = self.stix_logy_countrate.isChecked()
            elif stix_type == "overlay":
                # For overlay, use the spectrogram setting for energy axis
                logy = self.stix_logy_energy_overlay.isChecked()
            
            # Rest of the method remains the same...
            if stix_type == "spectrogram":
                # Check if energy range is set
                energy_range = None
                if self.stix_energy_range_checkbox.isChecked():
                    energy_range = [self.stix_energy_min.value(), self.stix_energy_max.value()]
                
                stix_plot_spectrogram(
                    self.parent.stix_counts_data, 
                    ax=ax, 
                    x_axis=True,
                    logscale=self.stix_logz.isChecked(), 
                    ylogscale=logy,
                    e_range=energy_range
                )
                title = "STIX Spectrogram Preview"
                if energy_range:
                    title += f"\nEnergy range: {energy_range[0]}-{energy_range[1]} keV"
                ax.set_title(title)
                
            elif stix_type == "time profiles":
                # Use energy ranges if available, otherwise None
                integrate_bins = self.stix_energy_ranges if self.stix_energy_ranges else None
                smoothing_pts = self.stix_smoothing_spinbox.value()
                logy_timecurves = self.stix_logy_countrate.isChecked()

                # Plot using stix_plot_counts
                stix_plot_counts(
                    self.parent.stix_counts_data,
                    title="STIX Time Profiles Preview",
                    e_range=None,  # Let function use default energy range
                    ax=ax,
                    date_range=None,  # Let function use full time range from data
                    legend=True,
                    lw=1.5,
                    ls="-",
                    smoothing_pts=smoothing_pts,
                    integrate_bins=integrate_bins,
                    zlogscale=self.stix_logz.isChecked(),
                    ylogscale=logy_timecurves,
                    verbose=True,
                    axis_fontsize=13,
                )

                # Add energy ranges info to title if available
                if self.stix_energy_ranges:
                    current_title = ax.get_title()
                    ax.set_title(f"{current_title}\nEnergy ranges: {self.stix_energy_ranges}")

            elif stix_type == "overlay":
                energy_range = None
                if self.stix_energy_range_checkbox.isChecked():
                    energy_range = [self.stix_energy_min.value(), self.stix_energy_max.value()]

                integrate_bins = self.stix_energy_ranges if self.stix_energy_ranges else None

                stix_plot_overlay(
                    self.parent.stix_counts_data,
                    ax=ax,
                    energy_range=energy_range,
                    date_range=None,
                    cmap="bone",
                    linewidth=1.5,
                    stix_smoothing_points=self.stix_smoothing_spinbox.value(),
                    stix_energy_bins=integrate_bins,
                    stix_lcolor=None,
                    stix_spec_zlogscale=self.stix_logz.isChecked(),
                    stix_spec_ylogscale=self.stix_logy_energy_overlay.isChecked(),
                    stix_curves_ylogscale=self.stix_logy_countrate_overlay.isChecked(),
                )

                title = "STIX Overlay Preview"
                if energy_range:
                    title += f"\nEnergy range: {energy_range[0]}-{energy_range[1]} keV"
                if self.stix_energy_ranges:
                    title += f"\nEnergy ranges: {self.stix_energy_ranges}"
                ax.set_title(title)
            
            plt.tight_layout()
            plt.show(block=False)
            
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", f"Error plotting STIX preview:\n{str(e)}")
    def _plot_rpw_hfr_preview(self):
        """Plot RPW-HFR preview"""
        if not self.parent or not self.parent.rpw_hfr_data:
            QMessageBox.warning(self, "No Data", "No RPW-HFR data loaded.")
            return
        
        try:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            
            # Get appropriate log Y axis setting based on RPW type
            rpw_type = self.rpw_choice.currentText()
            
            if rpw_type == "time profiles":
                # Use rpw_plot_curves for time profiles
                logy = self.rpw_logy_intensity.isChecked()
                
                # Get selected frequencies for plotting
                if hasattr(self, 'rpw_selected_frequencies') and self.rpw_selected_frequencies:
                    freqs = self.rpw_selected_frequencies
                else:
                    # Default frequencies if none selected
                    freqs = [500, 3500, 13000]  # Default HFR frequencies in kHz
                    # Filter to only include frequencies available in the data
                    if self.parent.rpw_hfr_data and 'freq' in self.parent.rpw_hfr_data:
                        available_freqs = self.parent.rpw_hfr_data['freq']
                        freqs = [f for f in freqs if min(available_freqs) <= f <= max(available_freqs)]
                
                # Get date range if set
                date_range = None
                # You can add date range logic here if needed
                
                # Get smoothing points
                smoothing_pts = None
                if hasattr(self, 'rpw_smoothing_spinbox'):
                    smoothing_pts = self.rpw_smoothing_spinbox.value() if self.rpw_smoothing_spinbox.value() > 1 else None
                
                # Plot using rpw_plot_curves
                rpw_plot_curves(
                    self.parent.rpw_hfr_data,
                    ax=ax,
                    date_range=date_range,
                    lw=1.5,
                    ls="-",
                    freqs=freqs,
                    ylogscale=logy,
                    smoothing_pts=smoothing_pts
                )
                
                ax.set_title(f"RPW-HFR Time Profiles Preview\nFrequencies: {freqs} kHz")
                
            else:
                if rpw_type == "spectrogram":
                    logy = self.rpw_logy_frequency.isChecked()

                # Get frequency range if set
                freq_range = None
                if hasattr(self, 'freq_range_checkbox') and self.freq_range_checkbox.isChecked():
                    try:
                        freq_min = float(self.freq_min.text())
                        freq_max = float(self.freq_max.text())
                        if 1 <= freq_min < freq_max <= 16400:
                            freq_range = [freq_min, freq_max]
                        else:
                            QMessageBox.warning(self, "Invalid Range", "Frequency range must be between 1 and 16400 kHz, with min < max.")
                            return
                    except ValueError:
                        QMessageBox.warning(self, "Invalid Input", "Please enter valid frequency values.")
                        return

                if rpw_type == "overlay":
                    if hasattr(self, 'rpw_selected_frequencies') and self.rpw_selected_frequencies:
                        freqs = self.rpw_selected_frequencies
                    else:
                        freqs = [500, 3500, 13000]
                        if self.parent.rpw_hfr_data and 'freq' in self.parent.rpw_hfr_data:
                            available_freqs = self.parent.rpw_hfr_data['freq']
                            freqs = [f for f in freqs if min(available_freqs) <= f <= max(available_freqs)]

                    rpw_plot_overlay(
                        self.parent.rpw_hfr_data,
                        freqs=freqs,
                        ax=ax,
                        frequency_range=freq_range,
                        date_range=None,
                        cmap="nipy_spectral",
                        rpw_units="wmhz",
                        linewidth=1.5,
                        smoothing_pts=self.rpw_smoothing_spinbox.value(),
                        lcolor=None,
                        rpw_plot_bias=None,
                        rpw_guidelines=False,
                        rpw_invert_yaxis=self.rpw_invert_y.isChecked(),
                        axis_fontsize=13,
                    )
                    ax.set_title(f"RPW-HFR Overlay Preview\nFrequencies: {freqs} kHz")
                else:
                    # Plot RPW-HFR spectrogram
                    if freq_range:
                        rpw_plot_psd(self.parent.rpw_hfr_data, ax=ax, xlabel=True, 
                                        frequency_range=freq_range, t_format="%H:%M", rpw_cbar_units="wmhz")
                        ax.set_title(f"RPW-HFR Preview\nFreq range: {freq_range[0]}-{freq_range[1]} kHz")
                    else:
                        rpw_plot_psd(self.parent.rpw_hfr_data, ax=ax, xlabel=True, 
                                        frequency_range=[0,17000], t_format="%H:%M", rpw_cbar_units="wmhz")
                        ax.set_title("RPW-HFR Preview")
                    
                    if logy:
                        ax.set_yscale("log")
                    
                    # Apply invert Y axis if enabled and visible
                    if hasattr(self, 'rpw_invert_y') and self.rpw_invert_y.isVisible() and self.rpw_invert_y.isChecked():
                        ax.invert_yaxis()
            
            plt.tight_layout()
            plt.show(block=False)
            
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", f"Error plotting RPW-HFR preview:\n{str(e)}")

    def _plot_rpw_tnr_preview(self):
        """Plot RPW-TNR preview"""
        if not self.parent or not self.parent.rpw_tnr_data:
            QMessageBox.warning(self, "No Data", "No RPW-TNR data loaded.")
            return
        
        try:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            
            # Get appropriate log Y axis setting based on RPW type
            rpw_type = self.rpw_choice.currentText()
            
            if rpw_type == "time profiles":
                # Use rpw_plot_curves for time profiles
                logy = self.rpw_logy_intensity.isChecked()
                
                # Get selected frequencies for plotting
                if hasattr(self, 'rpw_selected_frequencies') and self.rpw_selected_frequencies:
                    freqs = self.rpw_selected_frequencies
                else:
                    # Default frequencies if none selected
                    freqs = [50, 100, 400]  # Default TNR frequencies in kHz
                    # Filter to only include frequencies available in the data
                    if self.parent.rpw_tnr_data and 'freq' in self.parent.rpw_tnr_data:
                        available_freqs = self.parent.rpw_tnr_data['freq']
                        freqs = [f for f in freqs if min(available_freqs) <= f <= max(available_freqs)]
                
                # Get date range if set
                date_range = None
                # You can add date range logic here if needed
                
                # Get smoothing points
                smoothing_pts = None
                if hasattr(self, 'rpw_smoothing_spinbox'):
                    smoothing_pts = self.rpw_smoothing_spinbox.value() if self.rpw_smoothing_spinbox.value() > 1 else None
                
                # Plot using rpw_plot_curves
                rpw_plot_curves(
                    self.parent.rpw_tnr_data,
                    ax=ax,
                    date_range=date_range,
                    lw=1.5,
                    ls="-",
                    freqs=freqs,
                    ylogscale=logy,
                    smoothing_pts=smoothing_pts
                )
                
                ax.set_title(f"RPW-TNR Time Profiles Preview\nFrequencies: {freqs} kHz")
                
            else:
                if rpw_type == "spectrogram":
                    logy = self.rpw_logy_frequency.isChecked()

                # Get frequency range if set
                freq_range = None
                if hasattr(self, 'freq_range_checkbox') and self.freq_range_checkbox.isChecked():
                    try:
                        freq_min = float(self.freq_min.text())
                        freq_max = float(self.freq_max.text())
                        if 1 <= freq_min < freq_max <= 16400:
                            freq_range = [freq_min, freq_max]
                        else:
                            QMessageBox.warning(self, "Invalid Range", "Frequency range must be between 1 and 16400 kHz, with min < max.")
                            return
                    except ValueError:
                        QMessageBox.warning(self, "Invalid Input", "Please enter valid frequency values.")
                        return

                if rpw_type == "overlay":
                    if hasattr(self, 'rpw_selected_frequencies') and self.rpw_selected_frequencies:
                        freqs = self.rpw_selected_frequencies
                    else:
                        freqs = [50, 100, 400]
                        if self.parent.rpw_tnr_data and 'freq' in self.parent.rpw_tnr_data:
                            available_freqs = self.parent.rpw_tnr_data['freq']
                            freqs = [f for f in freqs if min(available_freqs) <= f <= max(available_freqs)]

                    rpw_plot_overlay(
                        self.parent.rpw_tnr_data,
                        freqs=freqs,
                        ax=ax,
                        frequency_range=freq_range,
                        date_range=None,
                        cmap="nipy_spectral",
                        rpw_units="wmhz",
                        linewidth=1.5,
                        smoothing_pts=self.rpw_smoothing_spinbox.value(),
                        lcolor=None,
                        rpw_plot_bias=None,
                        rpw_guidelines=False,
                        rpw_invert_yaxis=self.rpw_invert_y.isChecked(),
                        axis_fontsize=13,
                    )
                    ax.set_title(f"RPW-TNR Overlay Preview\nFrequencies: {freqs} kHz")
                else:
                    # Plot RPW-TNR spectrogram
                    if freq_range:
                        rpw_plot_psd(self.parent.rpw_tnr_data, ax=ax, xlabel=True, 
                                        frequency_range=freq_range, t_format="%H:%M", rpw_cbar_units="wmhz")
                        ax.set_title(f"RPW-TNR Preview\nFreq range: {freq_range[0]}-{freq_range[1]} kHz")
                    else:
                        rpw_plot_psd(self.parent.rpw_tnr_data, ax=ax, xlabel=True, 
                                        frequency_range=[0,17000], t_format="%H:%M", rpw_cbar_units="wmhz")
                        ax.set_title("RPW-TNR Preview")
                    
                    if logy:
                        ax.set_yscale("log")
                    
                    # Apply invert Y axis if enabled and visible
                    if hasattr(self, 'rpw_invert_y') and self.rpw_invert_y.isVisible() and self.rpw_invert_y.isChecked():
                        ax.invert_yaxis()
            
            plt.tight_layout()
            plt.show(block=False)
            
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", f"Error plotting RPW-TNR preview:\n{str(e)}")
    def _plot_epd_preview(self):
        """Plot EPD preview"""
        if not self.parent or not (self.parent.df_protons_ept is not None or self.parent.df_electrons_ept is not None):
            QMessageBox.warning(self, "No Data", "No EPD data loaded.")
            return
        
        try:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            
            # Use the selected particle type from EPD data
            if self.parent.epd_particle == "Electron":
                epd_data = self.parent.df_electrons_ept
            else:
                epd_data = self.parent.df_protons_ept
            
            # Create date range string
            if self.parent.epd_date:
                date_range_str = [
                    self.parent.epd_date.strftime("%d-%b-%Y %H:%M:%S").replace("00:00:00", "00:00:00"),
                    self.parent.epd_date.strftime("%d-%b-%Y %H:%M:%S").replace("00:00:00", "23:59:59")
                ]
            else:
                date_range_str = None
            
            # Plot EPD data using selected channels
            plot_ept_data(
                epd_data, 
                self.parent.energies_ept, 
                ax=ax, 
                particle=self.parent.epd_particle,
                channels=self.epd_selected_channels,  # Use selected channels
                resample=self.parent.epd_resample,
                date_range=date_range_str,
                round_epd_label=True
            )
            
            # Format x-axis
            time_formatter = DateFormatter('%H:%M')
            ax.xaxis.set_major_formatter(time_formatter)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            channels_str = ', '.join(map(str, self.epd_selected_channels))
            ax.set_title(f"EPD {self.parent.epd_particle} Preview\nChannels: {channels_str}")
            
            if self.epd_logy.isChecked():
                ax.set_yscale("log")
            
            plt.tight_layout()
            plt.show(block=False)
            
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", f"Error plotting EPD preview:\n{str(e)}")

    def get_selected_epd_channels(self):
        """Return the selected EPD channels as list of integers"""
        return self.epd_selected_channels

            
    def _has_stix_background_data(self):
        """Check if STIX data with background subtraction is available"""
        if not self.parent or not self.parent.stix_counts_data:
            return False
        return (self.parent.stix_bkg_file_enabled or self.parent.stix_bkg_time_enabled)

    def _has_rpw_hfr_background_data(self):
        """Check if RPW-HFR data with background subtraction is available"""
        if not self.parent or not self.parent.rpw_hfr_data:
            return False
        return (self.parent.rpw_hfr_bkg_option == 1)  # Time range background

    def _has_rpw_tnr_background_data(self):
        """Check if RPW-TNR data with background subtraction is available"""
        if not self.parent or not self.parent.rpw_tnr_data:
            return False
        return (self.parent.rpw_tnr_bkg_option == 1)  # Time range background

    def _plot_stix_background(self):
        """Plot STIX background in a separate window"""
        if not self.parent or not self.parent.stix_counts_data:
            QMessageBox.warning(self, "No Data", "No STIX data with background subtraction loaded.")
            return
        
        if not (self.parent.stix_bkg_file_enabled or self.parent.stix_bkg_time_enabled):
            QMessageBox.warning(self, "No Background", "STIX data does not have background subtraction applied.")
            return
        
        try:
            # Create new figure for background plot
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(111)
            
            # Plot background using sololab function
            stix_plot_bkg(self.parent.stix_counts_data, ax=ax)
            
            # Set window title
            fig.suptitle("STIX Background Plot", fontsize=12)
            
            plt.tight_layout()
            plt.show(block=False)  # Non-blocking
            
        except Exception as e:
            QMessageBox.critical(self, "Background Plot Error", f"Error plotting STIX background:\n{str(e)}")

    def _plot_rpw_hfr_background(self):
        """Plot RPW-HFR background in a separate window"""
        if not self.parent or not self.parent.rpw_hfr_data:
            QMessageBox.warning(self, "No Data", "No RPW-HFR data loaded.")
            return
        
        if self.parent.rpw_hfr_bkg_option != 1:
            QMessageBox.warning(self, "No Background", "RPW-HFR data does not have background subtraction applied.")
            return
        
        try:
            # Create new figure for background plot
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(111)
            
            # Plot background using sololab function
            rpw_plot_bkg(self.parent.rpw_hfr_data, ax=ax)
            
            # Set window title
            fig.suptitle("RPW-HFR Background Plot", fontsize=12)
            
            plt.tight_layout()
            plt.show(block=False)  # Non-blocking
            
        except Exception as e:
            QMessageBox.critical(self, "Background Plot Error", f"Error plotting RPW-HFR background:\n{str(e)}")

    def _plot_rpw_tnr_background(self):
        """Plot RPW-TNR background in a separate window"""
        if not self.parent or not self.parent.rpw_tnr_data:
            QMessageBox.warning(self, "No Data", "No RPW-TNR data loaded.")
            return
        
        if self.parent.rpw_tnr_bkg_option != 1:
            QMessageBox.warning(self, "No Background", "RPW-TNR data does not have background subtraction applied.")
            return
        
        try:
            # Create new figure for background plot
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(111)
            
            # Plot background using sololab function
            rpw_plot_bkg(self.parent.rpw_tnr_data, ax=ax)
            
            # Set window title
            fig.suptitle("RPW-TNR Background Plot", fontsize=12)
            
            plt.tight_layout()
            plt.show(block=False)  # Non-blocking
            
        except Exception as e:
            QMessageBox.critical(self, "Background Plot Error", f"Error plotting RPW-TNR background:\n{str(e)}")

    def _do_combined_plot(self):
        """Execute combined plotting with current preferences"""
        if not self.parent:
            QMessageBox.warning(self, "Error", "No parent window reference available.")
            return

        dlg = CombinedPlotDialog(self, prefs_dialog=self)
        dlg.exec_()
        


    # def get_values(self):
    #     # Get frequency range
    #     freq_range = None
    #     if self.freq_range_checkbox.isChecked():
    #         try:
    #             freq_min = float(self.freq_min.text())
    #             freq_max = float(self.freq_max.text())
    #             if 1 <= freq_min < freq_max <= 16400:
    #                 freq_range = [freq_min, freq_max]
    #         except ValueError:
    #             pass  # Use default if invalid
        
    #     return {
    #         "stix": {
    #             "type": self.stix_choice.currentText(),
    #             "logy": stix_logy,
    #             "logz": self.stix_logz.isChecked(),
    #             "energy_ranges": self.stix_energy_ranges,
    #             "energy_range": self.retrieve_stix_energy_range(),
    #             "smoothing_points": self.stix_smoothing_spinbox.value(),
    #         },
    #         "rpw": {
    #             "type": self.rpw_choice.currentText(),
    #             "logy": rpw_logy,
    #             "logz": self.rpw_logz.isChecked(),
    #             "overlay": self.rpw_overlay_choice.currentText(),
    #             "freq_range": freq_range,
    #             "smoothing_points": self.rpw_smoothing_spinbox.value(),
    #         },
    #         "epd": {
    #             "logy": self.epd_logy.isChecked(),
    #             "selected_channels": self.epd_selected_channels
    #         },
    #     }
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SoloLab - Temporal Analysis Tool")
        icon_path = Path(__file__).resolve().parent / "src" / "sololab_icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        else:
            print(f"Warning: Icon file not found at {icon_path}")
        self.resize(400, 450)  # Keep same size

        self.stix_file = None
        self.stix_counts_data = None
        self.stix_bkg_file_enabled = False
        self.stix_bkg_time_enabled = False
        self.stix_bkg_file = None
        self.stix_bkg_start_datetime = None
        self.stix_bkg_end_datetime = None
        
        # Separate RPW-HFR and RPW-TNR data storage
        self.rpw_hfr_file = None
        self.rpw_hfr_data = None
        self.rpw_hfr_bkg_option = 0
        self.rpw_hfr_bkg_start_datetime = None
        self.rpw_hfr_bkg_end_datetime = None
        
        self.rpw_tnr_file = None
        self.rpw_tnr_data = None
        self.rpw_tnr_bkg_option = 0
        self.rpw_tnr_bkg_start_datetime = None
        self.rpw_tnr_bkg_end_datetime = None
        
        # EPD data storage
        self.epd_date = None
        self.epd_download_path = None
        self.epd_particle = None
        self.epd_resample = None
        self.df_protons_ept = None
        self.df_electrons_ept = None
        self.energies_ept = None
        
        self.plot_prefs = {
            "stix": {"type": "spectrogram", "logy": False, "logz": False},
            "rpw": {"type": "spectrogram", "logy": False, "logz": False, "invert_y": True, "overlay": "Both", "selected_frequencies": []},
            "epd": {"logy": False},
        }

        central = QWidget()
        self.setCentralWidget(central)
        v = QVBoxLayout(central)
        v.setAlignment(Qt.AlignTop)

        self._banner_pixmap = None
        self.banner_label = QLabel()
        self.banner_label.setAlignment(Qt.AlignCenter)
        self.banner_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        banner_path = Path(__file__).resolve().parent / "src" / "sololab_banner.png"
        if banner_path.exists():
            self._banner_pixmap = QPixmap(str(banner_path))
            self._update_banner_pixmap()
        v.addWidget(self.banner_label)

        # Data import buttons section
        import_group = QGroupBox("Data Import")
        import_layout = QVBoxLayout(import_group)

        self.btn_import_stix = QPushButton("Import STIX data")
        self.btn_import_stix.clicked.connect(self.open_import_stix)
        import_layout.addWidget(self.btn_import_stix)

        self.btn_import_rpw_hfr = QPushButton("Import RPW-HFR data")
        self.btn_import_rpw_hfr.clicked.connect(self.open_import_rpw_hfr)
        import_layout.addWidget(self.btn_import_rpw_hfr)

        self.btn_import_rpw_tnr = QPushButton("Import RPW-TNR data")
        self.btn_import_rpw_tnr.clicked.connect(self.open_import_rpw_tnr)
        import_layout.addWidget(self.btn_import_rpw_tnr)

        self.btn_import_epd = QPushButton("Import EPD data")
        self.btn_import_epd.clicked.connect(self.open_import_epd)
        import_layout.addWidget(self.btn_import_epd)

        # Data status section
        status_group = QGroupBox("Loaded Data Status")
        status_layout = QVBoxLayout(status_group)
        
        # Create status labels
        self.stix_status_label = QLabel("STIX: No data loaded")
        self.stix_status_label.setStyleSheet("color: red;")
        status_layout.addWidget(self.stix_status_label)
        
        self.rpw_hfr_status_label = QLabel("RPW-HFR: No data loaded")
        self.rpw_hfr_status_label.setStyleSheet("color: red;")
        status_layout.addWidget(self.rpw_hfr_status_label)
        
        self.rpw_tnr_status_label = QLabel("RPW-TNR: No data loaded")
        self.rpw_tnr_status_label.setStyleSheet("color: red;")
        status_layout.addWidget(self.rpw_tnr_status_label)
        
        self.epd_status_label = QLabel("EPD: No data loaded")
        self.epd_status_label.setStyleSheet("color: red;")
        status_layout.addWidget(self.epd_status_label)

        panels_row = QHBoxLayout()
        panels_row.addWidget(import_group)
        panels_row.addWidget(status_group)
        v.addLayout(panels_row)

        pack_group = QGroupBox("Data Pack")
        pack_layout = QHBoxLayout(pack_group)

        self.btn_save_pack = QPushButton("Save Data Pack")
        self.btn_save_pack.clicked.connect(self._save_data_pack)
        self.btn_save_pack.setEnabled(False)
        pack_layout.addWidget(self.btn_save_pack)

        self.btn_load_pack = QPushButton("Load Data Pack")
        self.btn_load_pack.clicked.connect(self._load_data_pack)
        pack_layout.addWidget(self.btn_load_pack)

        v.addWidget(pack_group)

        btn_plot = QPushButton("PLOT DATA")
        btn_plot.clicked.connect(self.open_plot_prefs)  # Now opens plot preferences
        btn_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        v.addWidget(btn_plot)

        self._update_pack_buttons()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_banner_pixmap()

    def _update_banner_pixmap(self):
        if not self._banner_pixmap:
            return
        target_width = int(self.width() * 0.9)
        if target_width > 0:
            scaled = self._banner_pixmap.scaledToWidth(target_width, Qt.SmoothTransformation)
            self.banner_label.setPixmap(scaled)

    # ... all the status update and import methods remain the same ...
    def _update_stix_status(self):
        """Update STIX status label with data information"""
        if self.stix_counts_data is not None:
            try:
                min_time = self.stix_counts_data['time'][0]
                max_time = self.stix_counts_data['time'][-1]
                
                # Format datetime for display
                if hasattr(min_time, 'strftime'):
                    min_str = min_time.strftime("%Y-%m-%d %H:%M")
                    max_str = max_time.strftime("%Y-%m-%d %H:%M")
                else:
                    min_str = str(min_time)[:16]  # Truncate if string
                    max_str = str(max_time)[:16]
                
                self.stix_status_label.setText(f"STIX: Loaded | {min_str} to {max_str}")
                self.stix_status_label.setStyleSheet("color: green; font-weight: bold;")
            except:
                self.stix_status_label.setText("STIX: Loaded (time range unavailable)")
                self.stix_status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.stix_status_label.setText("STIX: No data loaded")
            self.stix_status_label.setStyleSheet("color: red;")

    def _update_rpw_hfr_status(self):
        """Update RPW-HFR status label with data information"""
        if self.rpw_hfr_data is not None:
            try:
                min_time = self.rpw_hfr_data['time'][0]
                max_time = self.rpw_hfr_data['time'][-1]
                
                # Format datetime for display
                if hasattr(min_time, 'strftime'):
                    min_str = min_time.strftime("%Y-%m-%d %H:%M")
                    max_str = max_time.strftime("%Y-%m-%d %H:%M")
                else:
                    min_str = str(min_time)[:16]  # Truncate if string
                    max_str = str(max_time)[:16]
                
                self.rpw_hfr_status_label.setText(f"RPW-HFR: Loaded | {min_str} to {max_str}")
                self.rpw_hfr_status_label.setStyleSheet("color: green; font-weight: bold;")
            except:
                self.rpw_hfr_status_label.setText("RPW-HFR: Loaded (time range unavailable)")
                self.rpw_hfr_status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.rpw_hfr_status_label.setText("RPW-HFR: No data loaded")
            self.rpw_hfr_status_label.setStyleSheet("color: red;")

    def _update_rpw_tnr_status(self):
        """Update RPW-TNR status label with data information"""
        if self.rpw_tnr_data is not None:
            try:
                min_time = self.rpw_tnr_data['time'][0]
                max_time = self.rpw_tnr_data['time'][-1]
                
                # Format datetime for display
                if hasattr(min_time, 'strftime'):
                    min_str = min_time.strftime("%Y-%m-%d %H:%M")
                    max_str = max_time.strftime("%Y-%m-%d %H:%M")
                else:
                    min_str = str(min_time)[:16]  # Truncate if string
                    max_str = str(max_time)[:16]
                
                self.rpw_tnr_status_label.setText(f"RPW-TNR: Loaded | {min_str} to {max_str}")
                self.rpw_tnr_status_label.setStyleSheet("color: green; font-weight: bold;")
            except:
                self.rpw_tnr_status_label.setText("RPW-TNR: Loaded (time range unavailable)")
                self.rpw_tnr_status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.rpw_tnr_status_label.setText("RPW-TNR: No data loaded")
            self.rpw_tnr_status_label.setStyleSheet("color: red;")

    def _update_epd_status(self):
        """Update EPD status label with data information"""
        if self.epd_date is not None and (self.df_protons_ept is not None or self.df_electrons_ept is not None):
            date_str = self.epd_date.strftime("%Y-%m-%d")
            particle_str = f" | {self.epd_particle}" if self.epd_particle else ""
            resample_str = f" | {self.epd_resample}" if self.epd_resample else ""
            self.epd_status_label.setText(f"EPD: Loaded | {date_str}{particle_str}{resample_str}")
            self.epd_status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.epd_status_label.setText("EPD: No data loaded")
            self.epd_status_label.setStyleSheet("color: red;")

    def _update_stix_button_color(self, loaded=False):
        """Update STIX button color based on load status"""
        if loaded:
            # Set green color for loaded data
            self.btn_import_stix.setStyleSheet("color: green; font-weight: bold;")
        else:
            # Reset to default color
            self.btn_import_stix.setStyleSheet("")
        self._update_stix_status()
        self._update_pack_buttons()

    def _update_rpw_hfr_button_color(self, loaded=False):
        """Update RPW-HFR button color based on load status"""
        if loaded:
            # Set green color for loaded data
            self.btn_import_rpw_hfr.setStyleSheet("color: green; font-weight: bold;")
        else:
            # Reset to default color
            self.btn_import_rpw_hfr.setStyleSheet("")
        self._update_rpw_hfr_status()
        self._update_pack_buttons()

    def _update_rpw_tnr_button_color(self, loaded=False):
        """Update RPW-TNR button color based on load status"""
        if loaded:
            # Set green color for loaded data
            self.btn_import_rpw_tnr.setStyleSheet("color: green; font-weight: bold;")
        else:
            # Reset to default color
            self.btn_import_rpw_tnr.setStyleSheet("")
        self._update_rpw_tnr_status()
        self._update_pack_buttons()
            
    def _update_epd_button_color(self, loaded=False):
        """Update EPD button color based on load status"""
        if loaded:
            # Set green color for loaded data
            self.btn_import_epd.setStyleSheet("color: green; font-weight: bold;")
        else:
            # Reset to default color
            self.btn_import_epd.setStyleSheet("")
        self._update_epd_status()
        self._update_pack_buttons()

    def _has_any_loaded_data(self):
        return any(
            [
                self.stix_counts_data is not None,
                self.rpw_hfr_data is not None,
                self.rpw_tnr_data is not None,
                (self.df_protons_ept is not None or self.df_electrons_ept is not None)
                and self.energies_ept is not None,
            ]
        )

    def _loaded_instruments_tag(self):
        tags = []
        if self.stix_counts_data is not None:
            tags.append("stix")
        if self.rpw_hfr_data is not None:
            tags.append("hfr")
        if self.rpw_tnr_data is not None:
            tags.append("tnr")
        if (
            (self.df_protons_ept is not None or self.df_electrons_ept is not None)
            and self.energies_ept is not None
        ):
            tags.append("epd")
        return "_".join(tags) if tags else "none"

    def _update_pack_buttons(self):
        self.btn_save_pack.setEnabled(self._has_any_loaded_data())

    def _save_data_pack(self):
        if not self._has_any_loaded_data():
            QMessageBox.warning(self, "No Data", "No instrument data loaded to save.")
            return

        timestamp = datetime.now().strftime("%Y%m%dT%H%M")
        instruments_tag = self._loaded_instruments_tag()
        default_name = f"sololab_{timestamp}_{instruments_tag}.pkl"

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Data Pack",
            default_name,
            "Data Pack (*.pkl)"
        )
        if not save_path:
            return

        payload = {
            "stix": {
                "counts": self.stix_counts_data,
                "bkg_file_enabled": self.stix_bkg_file_enabled,
                "bkg_time_enabled": self.stix_bkg_time_enabled,
                "bkg_file": self.stix_bkg_file,
                "bkg_start": self.stix_bkg_start_datetime,
                "bkg_end": self.stix_bkg_end_datetime,
            },
            "rpw_hfr": {
                "data": self.rpw_hfr_data,
                "bkg_option": self.rpw_hfr_bkg_option,
                "bkg_start": self.rpw_hfr_bkg_start_datetime,
                "bkg_end": self.rpw_hfr_bkg_end_datetime,
            },
            "rpw_tnr": {
                "data": self.rpw_tnr_data,
                "bkg_option": self.rpw_tnr_bkg_option,
                "bkg_start": self.rpw_tnr_bkg_start_datetime,
                "bkg_end": self.rpw_tnr_bkg_end_datetime,
            },
            "epd": {
                "date": self.epd_date,
                "particle": self.epd_particle,
                "resample": self.epd_resample,
                "df_protons": self.df_protons_ept,
                "df_electrons": self.df_electrons_ept,
                "energies": self.energies_ept,
            },
        }

        try:
            with open(save_path, "wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            QMessageBox.information(self, "Saved", f"Data pack saved to:\n{save_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", f"Error saving data pack:\n{exc}")

    def _load_data_pack(self):
        load_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Data Pack",
            "",
            "Data Pack (*.pkl)"
        )
        if not load_path:
            return

        try:
            with open(load_path, "rb") as handle:
                payload = pickle.load(handle)

            stix_payload = payload.get("stix", {})
            self.stix_counts_data = stix_payload.get("counts")
            self.stix_bkg_file_enabled = stix_payload.get("bkg_file_enabled", False)
            self.stix_bkg_time_enabled = stix_payload.get("bkg_time_enabled", False)
            self.stix_bkg_file = stix_payload.get("bkg_file")
            self.stix_bkg_start_datetime = stix_payload.get("bkg_start")
            self.stix_bkg_end_datetime = stix_payload.get("bkg_end")

            rpw_hfr_payload = payload.get("rpw_hfr", {})
            self.rpw_hfr_data = rpw_hfr_payload.get("data")
            self.rpw_hfr_bkg_option = rpw_hfr_payload.get("bkg_option", 0)
            self.rpw_hfr_bkg_start_datetime = rpw_hfr_payload.get("bkg_start")
            self.rpw_hfr_bkg_end_datetime = rpw_hfr_payload.get("bkg_end")

            rpw_tnr_payload = payload.get("rpw_tnr", {})
            self.rpw_tnr_data = rpw_tnr_payload.get("data")
            self.rpw_tnr_bkg_option = rpw_tnr_payload.get("bkg_option", 0)
            self.rpw_tnr_bkg_start_datetime = rpw_tnr_payload.get("bkg_start")
            self.rpw_tnr_bkg_end_datetime = rpw_tnr_payload.get("bkg_end")

            epd_payload = payload.get("epd", {})
            self.epd_date = epd_payload.get("date")
            self.epd_particle = epd_payload.get("particle")
            self.epd_resample = epd_payload.get("resample")
            self.df_protons_ept = epd_payload.get("df_protons")
            self.df_electrons_ept = epd_payload.get("df_electrons")
            self.energies_ept = epd_payload.get("energies")

            self._update_stix_button_color(loaded=self.stix_counts_data is not None)
            self._update_rpw_hfr_button_color(loaded=self.rpw_hfr_data is not None)
            self._update_rpw_tnr_button_color(loaded=self.rpw_tnr_data is not None)
            epd_loaded = (
                (self.df_protons_ept is not None or self.df_electrons_ept is not None)
                and self.energies_ept is not None
            )
            self._update_epd_button_color(loaded=epd_loaded)

            QMessageBox.information(self, "Loaded", f"Data pack loaded from:\n{load_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Error loading data pack:\n{exc}")

    # ... all the import methods remain exactly the same ...
    def open_import_stix(self):
        dlg = ImportStixDialog(self)
        # preload if available
        if self.stix_file:
            dlg.stix_edit.setText(self.stix_file)
        if self.stix_bkg_file:
            dlg.bkg_file_edit.setText(self.stix_bkg_file)
        if self.stix_bkg_start_datetime:
            dlg.bkg_start_datetime.setDateTime(QDateTime(self.stix_bkg_start_datetime))
        if self.stix_bkg_end_datetime:
            dlg.bkg_end_datetime.setDateTime(QDateTime(self.stix_bkg_end_datetime))
        
        dlg.bkg_file_checkbox.setChecked(self.stix_bkg_file_enabled)
        dlg.bkg_time_checkbox.setChecked(self.stix_bkg_time_enabled)

        if dlg.exec_() == QDialog.Accepted:
            vals = dlg.get_values()
            self.stix_file = vals["stix_file"]
            self.stix_counts_data = vals["stix_counts"]  # Store the processed counts object
            self.stix_bkg_file_enabled = vals["bkg_file_enabled"]
            self.stix_bkg_time_enabled = vals["bkg_time_enabled"]
            
            if vals["bkg_file_enabled"]:
                self.stix_bkg_file = vals.get("bkg_file")
            if vals["bkg_time_enabled"]:
                self.stix_bkg_start_datetime = vals.get("bkg_start_datetime")
                self.stix_bkg_end_datetime = vals.get("bkg_end_datetime")
            
            # Update button color to indicate data is loaded
            self._update_stix_button_color(loaded=True)
            
            QMessageBox.information(self, "Loaded", "STIX data has been successfully loaded and processed!")
            print("STIX file:", self.stix_file)
            print("STIX counts data loaded:", self.stix_counts_data is not None)
            print("STIX background file enabled:", self.stix_bkg_file_enabled)
            print("STIX background time enabled:", self.stix_bkg_time_enabled)
            if self.stix_bkg_file_enabled:
                print("STIX background file:", self.stix_bkg_file)
            if self.stix_bkg_time_enabled:
                print("STIX background time range:", self.stix_bkg_start_datetime, "to", self.stix_bkg_end_datetime)

    def open_import_rpw_hfr(self):
        dlg = ImportRpwHfrDialog(self)
        # preload if available
        if self.rpw_hfr_file:
            dlg.rpw_hfr_edit.setText(self.rpw_hfr_file)
        if self.rpw_hfr_bkg_start_datetime:
            dlg.bkg_start_datetime.setDateTime(QDateTime(self.rpw_hfr_bkg_start_datetime))
        if self.rpw_hfr_bkg_end_datetime:
            dlg.bkg_end_datetime.setDateTime(QDateTime(self.rpw_hfr_bkg_end_datetime))

        # Set the background option radio button
        if self.rpw_hfr_bkg_option == 1:
            dlg.bkg_time_radio.setChecked(True)
        else:
            dlg.no_bkg_radio.setChecked(True)

        if dlg.exec_() == QDialog.Accepted:
            vals = dlg.get_values()
            self.rpw_hfr_file = vals["rpw_hfr_file"]
            self.rpw_hfr_data = vals["rpw_data"]  # Store the processed RPW-HFR data
            self.rpw_hfr_bkg_option = vals["bkg_option"]
            
            if vals["bkg_option"] == 1:  # Time range
                self.rpw_hfr_bkg_start_datetime = vals.get("bkg_start_datetime")
                self.rpw_hfr_bkg_end_datetime = vals.get("bkg_end_datetime")
            
            # Update button color to indicate data is loaded
            self._update_rpw_hfr_button_color(loaded=True)
            
            QMessageBox.information(self, "Loaded", "RPW-HFR data has been successfully loaded and processed!")
            print("RPW HFR file:", self.rpw_hfr_file)
            print("RPW HFR data loaded:", self.rpw_hfr_data is not None)
            print("RPW HFR background option:", self.rpw_hfr_bkg_option)
            if self.rpw_hfr_bkg_option == 1:
                print("RPW HFR background time range:", self.rpw_hfr_bkg_start_datetime, "to", self.rpw_hfr_bkg_end_datetime)

    def open_import_rpw_tnr(self):
        dlg = ImportRpwTnrDialog(self)
        # preload if available
        if self.rpw_tnr_file:
            dlg.rpw_tnr_edit.setText(self.rpw_tnr_file)
        if self.rpw_tnr_bkg_start_datetime:
            dlg.bkg_start_datetime.setDateTime(QDateTime(self.rpw_tnr_bkg_start_datetime))
        if self.rpw_tnr_bkg_end_datetime:
            dlg.bkg_end_datetime.setDateTime(QDateTime(self.rpw_tnr_bkg_end_datetime))

        # Set the background option radio button
        if self.rpw_tnr_bkg_option == 1:
            dlg.bkg_time_radio.setChecked(True)
        else:
            dlg.no_bkg_radio.setChecked(True)

        if dlg.exec_() == QDialog.Accepted:
            vals = dlg.get_values()
            self.rpw_tnr_file = vals["rpw_tnr_file"]
            self.rpw_tnr_data = vals["rpw_data"]  # Store the processed RPW-TNR data
            self.rpw_tnr_bkg_option = vals["bkg_option"]
            
            if vals["bkg_option"] == 1:  # Time range
                self.rpw_tnr_bkg_start_datetime = vals.get("bkg_start_datetime")
                self.rpw_tnr_bkg_end_datetime = vals.get("bkg_end_datetime")
            
            # Update button color to indicate data is loaded
            self._update_rpw_tnr_button_color(loaded=True)
            
            QMessageBox.information(self, "Loaded", "RPW-TNR data has been successfully loaded and processed!")
            print("RPW TNR file:", self.rpw_tnr_file)
            print("RPW TNR data loaded:", self.rpw_tnr_data is not None)
            print("RPW TNR background option:", self.rpw_tnr_bkg_option)
            if self.rpw_tnr_bkg_option == 1:
                print("RPW TNR background time range:", self.rpw_tnr_bkg_start_datetime, "to", self.rpw_tnr_bkg_end_datetime)

    def open_import_epd(self):
        dlg = ImportEpdDialog(self)
        
        # Preload if available
        if self.epd_download_path:
            dlg.path_edit.setText(self.epd_download_path)
        if self.epd_date:
            dlg.obs_date.setDate(QDate(self.epd_date.year, self.epd_date.month, self.epd_date.day))
        if self.epd_particle:
            dlg.particle_combo.setCurrentText(self.epd_particle)
        if self.epd_resample:
            dlg.resample_combo.setCurrentText(self.epd_resample)
        
        if dlg.exec_() == QDialog.Accepted:
            vals = dlg.get_values()
            
            # Store EPD data and parameters
            self.epd_date = vals["epd_date"]
            self.epd_download_path = vals["download_path"]
            self.epd_particle = vals["particle"]
            self.epd_resample = vals["resample"]
            self.df_protons_ept = vals["df_protons_ept"]
            self.df_electrons_ept = vals["df_electrons_ept"]
            self.energies_ept = vals["energies_ept"]
            
            # Update button color to indicate data is loaded
            self._update_epd_button_color(loaded=True)
            
            QMessageBox.information(self, "Loaded", "EPD data has been successfully loaded!")
            print(f"EPD data loaded - Date: {self.epd_date}, Particle: {self.epd_particle}, Resample: {self.epd_resample}")
            print(f"Download path: {self.epd_download_path}")
            print(f"Protons data loaded: {self.df_protons_ept is not None}")
            print(f"Electrons data loaded: {self.df_electrons_ept is not None}")
            print(f"Energies loaded: {self.energies_ept is not None}")

    def open_plot_prefs(self):
        dlg = PlotPrefsDialog(self)
        # preload current preferences
        p = self.plot_prefs
        dlg.stix_choice.setCurrentText(p["stix"]["type"])
        
        # Use the appropriate log Y axis checkbox based on current type
        stix_type = p["stix"]["type"]
        if stix_type == "spectrogram":
            dlg.stix_logy_energy.setChecked(p["stix"]["logy"])
        elif stix_type == "time profiles":
            dlg.stix_logy_countrate.setChecked(p["stix"]["logy"])
        elif stix_type == "overlay":
            if isinstance(p["stix"]["logy"], dict):
                dlg.stix_logy_energy_overlay.setChecked(p["stix"]["logy"].get("energy", False))
                dlg.stix_logy_countrate_overlay.setChecked(p["stix"]["logy"].get("countrate", False))
            else:
                # Backward compatibility
                dlg.stix_logy_energy_overlay.setChecked(p["stix"]["logy"])
                dlg.stix_logy_countrate_overlay.setChecked(p["stix"]["logy"])
        
        dlg.stix_logz.setChecked(p["stix"]["logz"])

        dlg.rpw_choice.setCurrentText(p["rpw"]["type"])
        
        # Use the appropriate RPW log Y axis checkbox based on current type
        rpw_type = p["rpw"]["type"]
        if rpw_type == "spectrogram":
            dlg.rpw_logy_frequency.setChecked(p["rpw"]["logy"])
        elif rpw_type == "time profiles":
            dlg.rpw_logy_intensity.setChecked(p["rpw"]["logy"])
        elif rpw_type == "overlay":
            if isinstance(p["rpw"]["logy"], dict):
                dlg.rpw_logy_frequency_overlay.setChecked(p["rpw"]["logy"].get("frequency", False))
                dlg.rpw_logy_intensity_overlay.setChecked(p["rpw"]["logy"].get("intensity", False))
            else:
                # Backward compatibility
                dlg.rpw_logy_frequency_overlay.setChecked(p["rpw"]["logy"])
                dlg.rpw_logy_intensity_overlay.setChecked(p["rpw"]["logy"])
        
        dlg.rpw_logz.setChecked(p["rpw"]["logz"])
        dlg.rpw_invert_y.setChecked(p["rpw"].get("invert_y", True))  # Default True
        dlg.rpw_overlay_choice.setCurrentText(p["rpw"]["overlay"])

        # Load selected frequencies if available
        if "selected_frequencies" in p["rpw"]:
            dlg.rpw_selected_frequencies = p["rpw"]["selected_frequencies"]

        dlg.epd_logy.setChecked(p["epd"]["logy"])

        # Execute dialog (plotting happens within the dialog)
        dlg.exec_()

    def do_plot(self):
        # Use the loaded STIX data if available
        if self.stix_counts_data:
            # Plot using the actual loaded and processed STIX data
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            
            try:
                stix_plot_spectrogram(self.stix_counts_data, ax=ax)
                ax.set_title("STIX Data (Loaded and Processed)")
                plt.tight_layout()
                plt.show(block=False)
                return
            except Exception as e:
                print(f"Error plotting STIX data: {e}")
                QMessageBox.warning(self, "Plot Error", f"Error plotting STIX data: {str(e)}")
        
        # Fallback to original dummy plotting logic
        prefs = self.plot_prefs
        fig = plt.figure(figsize=(8, 6))

        stix_pref = prefs.get("stix", {})
        rpw_pref = prefs.get("rpw", {})
        epd_pref = prefs.get("epd", {})

        plots_made = 0

        if stix_pref["type"] in ("spectrogram", "overlay"):
            plots_made += 1
            ax = fig.add_subplot(2, 1, plots_made) if (stix_pref["type"] == "overlay" or rpw_pref["type"] != "spectrogram") else fig.add_subplot(1, 1, 1)
            data = np.abs(np.random.rand(100, 200))
            if stix_pref["logz"]:
                im = ax.imshow(data + 1e-6, aspect="auto", origin="lower", norm=LogNorm())
            else:
                im = ax.imshow(data, aspect="auto", origin="lower")
            ax.set_title("STIX spectrogram (dummy)")
            ax.set_ylabel("Freq (log)" if stix_pref["logy"] else "Freq")
            if stix_pref["logy"]:
                ax.set_yscale("log")
            fig.colorbar(im, ax=ax)

        elif stix_pref["type"] == "time profiles":
            plots_made += 1
            ax = fig.add_subplot(1, 1, plots_made)
            t = np.linspace(0, 100, 500)
            ax.plot(t, np.sin(t / 5.0) + np.random.normal(scale=0.2, size=t.shape), label="STIX channel 1")
            ax.plot(t, 0.5 * np.cos(t / 7.0) + np.random.normal(scale=0.2, size=t.shape), label="STIX channel 2")
            ax.set_title("STIX time profiles (dummy)")
            if stix_pref["logy"]:
                ax.set_yscale("log")
            ax.legend()

        # RPW plotting (unchanged)
        if rpw_pref["type"] in ("spectrogram", "overlay"):
            plots_made += 1
            ax = fig.add_subplot(2, 1, plots_made)
            data = np.abs(np.random.rand(80, 180))
            if rpw_pref["logz"]:
                im = ax.imshow(data + 1e-6, aspect="auto", origin="lower", norm=LogNorm())
            else:
                im = ax.imshow(data, aspect="auto", origin="lower")
            ax.set_title(f"RPW spectrogram (dummy) - overlay choice: {rpw_pref.get('overlay')}")
            if rpw_pref["logy"]:
                ax.set_yscale("log")
            fig.colorbar(im, ax=ax)

        elif rpw_pref["type"] == "time profiles":
            plots_made += 1
            ax = fig.add_subplot(1, 1, plots_made)
            t = np.linspace(0, 100, 300)
            overlay = rpw_pref.get("overlay", "Both")
            if overlay in ("Only TNR", "Both"):
                ax.plot(t, np.sin(t / 10.0) + np.random.normal(scale=0.1, size=t.shape), label="TNR")
            if overlay in ("Only HFR", "Both"):
                ax.plot(t, np.cos(t / 12.0) + np.random.normal(scale=0.1, size=t.shape), label="HFR")
            ax.set_title("RPW time profiles (dummy)")
            if rpw_pref["logy"]:
                ax.set_yscale("log")
            ax.legend()

        # EPD plotting (unchanged)
        if epd_pref.get("logy"):
            fig2 = plt.figure(figsize=(6, 3))
            ax2 = fig2.add_subplot(1, 1, 1)
            t = np.linspace(0, 50, 200)
            ax2.plot(t, np.abs(np.random.randn(t.size)) * 1e-1 + 0.01)
            ax2.set_title("EPD (dummy) - log Y")
            ax2.set_yscale("log")
            fig2.tight_layout()
            plt.show(block=False)
        else:
            if plots_made == 0:
                ax = fig.add_subplot(1, 1, 1)
                t = np.linspace(0, 50, 200)
                ax.plot(t, np.abs(np.random.randn(t.size)) * 1e-1)
                ax.set_title("EPD (dummy)")
                plots_made = 1

        fig.tight_layout()
        plt.show(block=False)


def run_app():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


# if __name__ == "__main__":
#     run_app()