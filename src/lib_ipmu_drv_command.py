import threading
import time
import queue
import datetime
import numpy as np


# Import AppConfig from daq_config.py
from lib_ipmu_drv_config import AppConfig

# Driver
#import import_ipynb
#from db_class_AEdriver_rotation import AEdriver, stable_rotation
from class_AEdriver_rotation import AEdriver, ExtendedAEdriver

class Command:
    def __init__(self, config: AppConfig, comvel_q: queue.Queue, stop_event: threading.Event):
        """Initializes the Generator."""
        self.drv_cfg = config
        self.comvel_q = comvel_q
        self.stop_event = stop_event
        self.COM = self.drv_cfg.driver.device_com
        self.driver = None # first definition
        self.DEBUG=None
        self.DataStoreFlag = queue.Queue(maxsize=40)

    def setup(self, DEBUG=None):
        self.pps = self.drv_cfg.driver.pps
        self.pps_fin = self.drv_cfg.dependent.pps_fin
        self.step = self.drv_cfg.driver.step
        self.ppsps = self.drv_cfg.driver.ppsps
        self.initial_curr = self.drv_cfg.driver.current_init
        self.electric_angle_init = self.drv_cfg.driver.electrical_angle_init
        self.rotshift_time = self.drv_cfg.driver.rst
        self.stablerot_duration = self.drv_cfg.driver.t_stablerot
        self.excess_spindown_time = self.drv_cfg.driver.t_excess_spindown
        self.DC_duration = self.drv_cfg.driver.t_DC

        self.target_speed_rps = self.drv_cfg.driver.target_speed_rps
        self.t_total = self.drv_cfg.dependent.t_total
        self.dir_rotation = self.drv_cfg.driver.dir_rotation
        self.t_current_reduction_duration = self.drv_cfg.driver.t_current_reduction_duration
        self.step_current_reduction = self.drv_cfg.driver.step_current_reduction

        self.DEBUG=DEBUG

        if DEBUG:
            print("=====================================================================================")
            print('Rotational frequency                           : ', self.target_speed_rps, '[Hz]')
            print('Current pulse per second to finish acceleration: ', self.pps_fin)
            print('Initial current                                : ', '%1f'%(self.initial_curr*10) , '[mA]')
            print('Initial Pulse per second                       : ', self.pps )
            print('Maximum acceleration step                      : ', self.ppsps )
            print('Rotational shift after every pps               : ', self.rotshift_time , '[s]')
            print('Step                                           : ', self.step)
            print('Stable rotation duration                       : ', self.stablerot_duration/60, '[min]')
            print('excess time before stop the rotor command      : ', self.excess_spindown_time/60, '[min]')
            print('Total rotation duration time                   : ', self.t_total/60, '[min]' ) # added on 2025/05/27 by Taisei
            print("======================================================================================")
        else:
            pass
    
        self.driver = ExtendedAEdriver(self.COM)

    #def run(self):
        #StableRotation(self.driver, self.pps, self.pps_fin, self.step, self.ppsps, self.initial_curr, self.electric_angle_init, self.rotshift_time, self.stablerot_duration, self.excess_spindown_time, self.DC_duration, rot=self.dir_rotation, t_current_reduction_duration = self.t_current_reduction_duration, DEBUG = self.DEBUG).current_reduction()

    def run(self):
        # Write motor sequence here directly
        print ("All process start time", datetime.datetime.now(), "(*ゝ∀･)v")
        try:
            if self.initial_curr >= 50:
                raise
        except:
            print ("Look at the value of current!")
            self.driver.stop()
            self.driver.initialize()
            self.driver.ser.close()
            raise

        # Call function
        self.loop_limit = 40_000
        self.total_pulse = 36_000
        self.driver.initialize()
        self.driver.operation_setting()
        self.driver.pulse_setting()
        self.driver.initial_position(position=self.electric_angle_init)

        self.driver.highrate_speed_setting(self.pps)
        self.driver.startup_speed_setting(self.pps)
        self.driver.acc_rate_setting(self.ppsps)
        self.driver.complete_setting()

        self.driver.current_on(self.initial_curr)
        time.sleep(self.DC_duration) # DC Duration

        self.driver.rot_start(rot=self.dir_rotation)
        time.sleep(self.rotshift_time)
        
        # Spin Up
        n = 0
        while n < self.loop_limit and not self.stop_event.is_set():
            try:
                self.pps = self.pps + self.step
                if self.pps >= self.pps_fin:
                    self.driver.highrate_speed_setting(self.pps_fin)
                    self.comvel_q.put_nowait((self.pps_fin/self.total_pulse))
                    break
                self.driver.highrate_speed_setting(self.pps)
                self.comvel_q.put_nowait((self.pps/self.total_pulse))
                time.sleep(self.rotshift_time)
                n += 1
                #if self.DEBUG:
                #    self.driver.query()

            except self.stop_event.is_set():
                self._reset()
        
        # Current Reduction
        if self.stop_event.is_set():
            self._reset()
            pass
        else:
            print(f"Waiting start reducing current in {self.t_current_reduction_duration} s")
            time.sleep(self.t_current_reduction_duration)

            for i in self.step_current_reduction:
                try:
                    # current reduction
                    self.DataStoreFlag.put_nowait((i))
                    #print(self.DataStoreFlag)
                    print(f"Waiting start reducing current to {i}% in {5} s")
                    time.sleep(5)
                    self.driver.current_on(i)
                    time.sleep(self.t_current_reduction_duration)
                    if self.stop_event.is_set():
                        break
                    
                except Exception as e:
                    print(f"An error occurred during current reduction: {e}")
                    break

            self._reset()

    def _reset(self, COM=None):
        if self.driver is None:
            COM = self.COM
            try:
                driver = AEdriver(COM)
                driver.stop()
                driver.initialize()
                driver.ser.close()
                print("Driver has been shutdowned properly (o_ _)o")
            except:
                print("Aborting command... COM is not free")
        else:
            try:
                self.driver.stop()
                self.driver.initialize()
                self.driver.ser.close()
                print("Driver has been shutdowned properly (o_ _)o")
            except:
                print("Aborting command... Driver is exist but there are some issues")


