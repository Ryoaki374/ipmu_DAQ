import threading
import numpy as np
import time

# Import AppConfig from daq_config.py
from lib_ipmu_driver_config import AppConfig

# Driver
import import_ipynb
#from db_class_AEdriver_rotation import AEdriver, stable_rotation
from class_AEdriver_rotation import AEdriver, ExtendedAEdriver, StableRotation

class Command:
    def __init__(self, config: AppConfig,): #stop_event: threading.Event):
        """Initializes the Generator."""
        self.drv_cfg = config
        #self.stop_event = stop_event
        self.COM = self.drv_cfg.driver.device_com
        self.driver = None # first definition
        self.DEBUG=None

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

    def run(self):
        StableRotation(self.driver, self.pps, self.pps_fin, self.step, self.ppsps, self.initial_curr, self.electric_angle_init, self.rotshift_time, self.stablerot_duration, self.excess_spindown_time, self.DC_duration, rot=self.dir_rotation, DEBUG = self.DEBUG).current_reduction()
        
    def _reset(self, COM=None):
        if COM is None:
            COM = self.COM
        try:
            driver = AEdriver(COM)
            driver.stop()
            driver.initialize()
            driver.ser.close()
        except:
            pass
        