import queue
import threading

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore

# Import AppConfig from daq_config.py
from lib_ipmu_daq_config import AppConfig

class DAQGUI:
    """
    Handles all PyQtGraph-based visualization for the DAQ application.
    """
    def __init__(self, config: AppConfig, quad_q: queue.Queue, stop_event: threading.Event):
        self.cfg = config
        self.quad_q = quad_q
        self.stop_event = stop_event
        self.display_sec = self.cfg.gui.display_sec

    def run(self):
        """
        Initializes and runs the Qt application loop.
        This is a blocking call.
        """
        pg.setConfigOptions(useOpenGL=True, background="w", foreground="k")
        self.app = pg.mkQApp("Live Plots")
        self.win = pg.GraphicsLayoutWidget(title="DEMO")
        self.win.resize(1200, 800)
        
        self._setupPlots()
        self._initBuffers()

        self.win.show()

        # Timer for refreshing the plots
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._refresh)
        self.timer.start(self.cfg.gui.gui_interval_ms)

        # Timer for auto-stopping the application
        # A run_sec parameter should be added to config, assuming 30s for now
        # QtCore.QTimer.singleShot(int(self.display_sec * 1000), self._stop)
        
        self.app.exec()
    
    def _stop(self):
        """Signals the acquisition threads to stop and closes the GUI."""
        print("GUI auto-stop timer expired.")
        self.stop_event.set()
        self.app.quit()

    def _setupPlots(self):
        """Creates the layout and all the plot items."""
        layout = self.win.ci.layout
        layout.setColumnStretchFactor(0, 4)
        layout.setColumnStretchFactor(1, 4)

        # Plot for RAW A/B signals
        plt_ab = self.win.addPlot(row=0, col=0, title="Encoder Raw Signal (red: A / blue: B)")
        self.curve_A = plt_ab.plot(pen=pg.mkPen("#ff4b00", width=3), stepMode="right")
        self.curve_B = plt_ab.plot(pen=pg.mkPen("#005aff", width=3), stepMode="right")
        plt_ab.setLabel("left", "Amplitude [V]")
        plt_ab.setLabel("bottom", "Time [s]")
        plt_ab.setYRange(-0.5, self.cfg.debug_encoder.pulse_height + 0.5)

        # Plot for Delta velocity
        plt_cnt = self.win.addPlot(row=1, col=0, title="Diff of Velocity")
        self.curve_cnt = plt_cnt.plot(pen=pg.mkPen("#03af7a", width=3))
        plt_cnt.setLabel("left", "Diff of velocity [rps]")
        plt_cnt.setLabel("bottom", "Time [s]")
        self.plt_cnt = plt_cnt

        # Plot for Velocity
        plt_vel = self.win.addPlot(row=2, col=0, title="Measured and Command velocity")
        self.curve_vel = plt_vel.plot(pen=pg.mkPen("#00a0e9", width=3))
        self.curve_vel_ref = plt_vel.plot(pen=pg.mkPen("#a05aff", width=3), stepMode="right")
        plt_vel.setLabel("left", "Velocity [rps]")
        plt_vel.setLabel("bottom", "Time [s]")
        self.plt_vel = plt_vel

        # Plot for PhaseU I/V waveform
        plt_IV = self.win.addPlot(row=0, col=1, title="Current (Red) and Voltage (Blue) [Phase: U]")
        self.curve_I = plt_IV.plot(pen=pg.mkPen("#ff4b00", width=3))
        self.curve_V = plt_IV.plot(pen=pg.mkPen("#005aff", width=3))
        plt_IV.setLabel("left", "Amplitude [a.u.]")
        plt_IV.setLabel("bottom", "Time [s]")
        plt_IV.setYRange(-1.2, 1.2)
        self.plt_ab = plt_ab # Store for updating range
        self.plt_IV = plt_IV # Store for updating range

        # Plot for Power
        plt_pow = self.win.addPlot(row=1, col=1, title="Total Power")
        self.curve_pow_tot = plt_pow.plot(pen=pg.mkPen("#f6aa00", width=3))
        plt_pow.setLabel("left", "Power [W]")
        plt_pow.setLabel("bottom", "Time [s]")
        self.plt_pow = plt_pow

        # Plot for Squared current
        plt_squared = self.win.addPlot(row=2, col=1, title="Squared Current")
        self.curve_I2u = plt_squared.plot(pen=pg.mkPen("#f6aa00", width=3))
        self.curve_I2v = plt_squared.plot(pen=pg.mkPen("#a05aff", width=3))
        self.curve_I2w = plt_squared.plot(pen=pg.mkPen("#03af7a", width=3))
        plt_squared.setLabel("left", "Current^2 [A^2]")
        plt_squared.setLabel("bottom", "Time [s]")
        self.plt_squared = plt_squared

    def _initBuffers(self):
        """Initializes numpy arrays to store plot data."""
        self.xs = np.empty(0, dtype=np.float32)
        self.ya = np.empty(0, dtype=np.float32)
        self.yb = np.empty(0, dtype=np.float32)
        self.y_Iu = np.empty(0, dtype=np.float32)
        self.y_Vu = np.empty(0, dtype=np.float32)
        
        self.xs_cnt = np.empty(0, dtype=np.float32)
        self.y_cnt = np.empty(0, dtype=np.float32)
        
        self.xs_vel = np.empty(0, dtype=np.float32)
        self.y_vel = np.empty(0, dtype=np.float32)
        self.xr_vel = np.empty(0, dtype=np.float32)
        self.yr_vel = np.empty(0, dtype=np.float32)
        
        self.xs_pow = np.empty(0, dtype=np.float32)
        self.y_pow_tot = np.empty(0, dtype=np.float32)
        self.y_pow_w = np.empty(0, dtype=np.float32)

        self.xs_squared = np.empty(0, dtype=np.float32)
        self.y_I2u = np.empty(0, dtype=np.float32)
        self.y_I2v = np.empty(0, dtype=np.float32)
        self.y_I2w = np.empty(0, dtype=np.float32)


    def _refresh(self):
        """Called by the QTimer to update plot data."""
        try:
            while True:
                data = self.quad_q.get_nowait()
                t_ax, pA, pB, qsig, t_end, cum_cnt, vel, t_ref, v_ref, time_p, P_tot_sum, Iu_blk, Vu_blk, _I2u, _I2v, _I2w  = data

                pruning = self.cfg.gui.pruning if self.cfg.gui.pruning >= 1 else 1
                history = self.cfg.dependent.history

                # Update waveform buffers
                self.xs = np.concatenate((self.xs, t_ax[::pruning]))[-history:]
                self.ya = np.concatenate((self.ya, pA[::pruning]))[-history:]
                self.yb = np.concatenate((self.yb, pB[::pruning]))[-history:]
                self.y_Iu = np.concatenate((self.y_Iu, Iu_blk[::pruning]))[-history:]
                self.y_Vu = np.concatenate((self.y_Vu, Vu_blk[::pruning]))[-history:]
                
                # Update delta velocity buffers
                self.xs_cnt = np.append(self.xs_cnt, t_end)[-self.cfg.dependent.count_history:]
                v_ref_val = v_ref[-1] if v_ref.size > 0 else 0
                self.y_cnt = np.append(self.y_cnt, vel - v_ref_val)[-self.cfg.dependent.count_history:]
                
                # Update velocity buffers
                self.xs_vel = np.append(self.xs_vel, t_end)[-self.cfg.dependent.velo_history:]
                self.y_vel = np.append(self.y_vel, vel)[-self.cfg.dependent.velo_history:]
                self.xr_vel = np.concatenate((self.xr_vel, t_ref))[-self.cfg.dependent.velo_history:]
                self.yr_vel = np.concatenate((self.yr_vel, v_ref))[-self.cfg.dependent.velo_history:]
                
                # Update power buffers
                self.xs_pow = np.append(self.xs_pow, time_p)[-self.cfg.dependent.pow_history:]
                self.y_pow_tot = np.append(self.y_pow_tot, P_tot_sum)[-self.cfg.dependent.pow_history:]

                # Update squared current buffers
                self.xs_squared = np.append(self.xs_squared, time_p)[-self.cfg.dependent.count_history:]
                self.y_I2u = np.append(self.y_I2u, _I2u)[-self.cfg.dependent.count_history:]
                self.y_I2v = np.append(self.y_I2v, _I2v)[-self.cfg.dependent.count_history:]
                self.y_I2w = np.append(self.y_I2w, _I2w)[-self.cfg.dependent.count_history:]

                self.quad_q.task_done()
        except queue.Empty:
            pass # No new data

        # --- Update plots with new data ---
        
        # Scrolling window for waveforms for IV
        if self.xs.size:
            start = self.xs[-1] - self.cfg.gui.plot_sec
            #self.plt_ab.setXRange(start, self.xs[-1], padding=0)
            self.plt_IV.setXRange(start, self.xs[-1], padding=0)
        if self.xs.size:
            start = self.xs[-1] - self.cfg.gui.plot_sec/10
            self.plt_ab.setXRange(start, self.xs[-1], padding=0)
        
        # Scrolling window for time-series data
        history_sec = 10 # seconds
        if self.xs_cnt.size:
            self.plt_cnt.setXRange(self.xs_cnt[-1] - history_sec, self.xs_cnt[-1], padding=0)
        if self.xs_vel.size:
            self.plt_vel.setXRange(self.xs_vel[-1] - history_sec, self.xs_vel[-1], padding=0)
        if self.xs_pow.size:
            self.plt_pow.setXRange(self.xs_pow[-1] - history_sec, self.xs_pow[-1], padding=0)
        if self.xs_squared.size:
            self.plt_squared.setXRange(self.xs_squared[-1] - history_sec, self.xs_squared[-1], padding=0)
            
        # Set data on curves
        # Encoder
        self.curve_A.setData(self.xs, self.ya)
        self.curve_B.setData(self.xs, self.yb)
        # Phase U I/V
        self.curve_I.setData(self.xs, self.y_Iu)
        self.curve_V.setData(self.xs, self.y_Vu)
        # Delta velocity
        self.curve_cnt.setData(self.xs_cnt, self.y_cnt)
        self.curve_vel.setData(self.xs_vel, self.y_vel)
        self.curve_vel_ref.setData(self.xr_vel, self.yr_vel)
        # Power
        self.curve_pow_tot.setData(self.xs_pow, self.y_pow_tot)
        # Squared current
        self.curve_I2u.setData(self.xs_squared, self.y_I2u)
        self.curve_I2v.setData(self.xs_squared, self.y_I2v)
        self.curve_I2w.setData(self.xs_squared, self.y_I2w)