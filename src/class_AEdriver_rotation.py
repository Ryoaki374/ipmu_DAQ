import time
import numpy as np
import serial
import binascii
import datetime

class AEdriver():
    
    def __init__(self, COM, baudrate=38400):
        # serial 
        self.COM = COM
        self.ser = serial.Serial(port = self.COM, \
                                baudrate = 38400, \
                                parity = serial.PARITY_EVEN, \
                                stopbits = serial.STOPBITS_ONE, \
                                bytesize = serial.EIGHTBITS,
                                timeout=0.1)
        # parameters
        self.output = True
        self.acc = 0
        self.freq = 0
        self.curr_acc = 0
        self.curr_stable = 0
    
    def __del__(self):
        self.ser.close()
        print('closing serial port: ' + self.COM)

        
    def output_setting(self, output):
        if (output !=True) or (output !=False): 
            self.output = False
            print('ERROR: argment should be True or False, set as no output mode')
        else: self.output = output
        
    def query(self):
        self.ser.write("*IDN?")
        print(self.ser.readline())
    
    def hex1byte_to_8bit(self,hex1byte):
        return bin(int(hex1byte[0],16))[2:].zfill(4) + bin(int(hex1byte[1],16))[2:].zfill(4)

    def hex2byte_to_16bit(self,hex1byte):
        return  bin(int(hex1byte[2],16))[2:].zfill(4) + bin(int(hex1byte[3],16))[2:].zfill(4) + bin(int(hex1byte[0],16))[2:].zfill(4) + bin(int(hex1byte[1],16))[2:].zfill(4)
    
    def check_pulse_out(self,hex_read):
        status = self.hex1byte_to_8bit(hex_read[4:6])
        if int(status[-1]) == 1: return True
        else: return False
    
    def hex_to_signed_int(self,hexstr,bits):
        value = int(hexstr,16)
        if value & (1 << (bits-1)):
            value -= 1 << bits
        return value

    def initialize(self,address=2,return_bit=False):
        # address: 0 for Z4820, 1 for MCCA100A, 2 for both
        if (address == 0) or (address == 2):
            cmd = '04000105'
            self.ser.write(binascii.a2b_hex(cmd))
            read = binascii.b2a_hex(self.ser.readline()).decode("ascii").replace(cmd,'')
        elif (address == 1) or (address == 2):
            cmd = '04010106'
            self.ser.write(binascii.a2b_hex(cmd)) 
            read = binascii.b2a_hex(self.ser.readline()).decode("ascii").replace(cmd,'')
        else: 
            print('ERROR: argment should be 0, 1 or 2.')
            read = ''
        if return_bit: return read
            
    def stop(self,return_bit=False):
        cmd = '0401171c'
        self.ser.write(binascii.a2b_hex(cmd))
        read = binascii.b2a_hex(self.ser.readline()).decode("ascii").replace(cmd,'')
        if return_bit: return read
            
    def complete_setting(self,return_bit=False):
        cmd = '0401393e'
        self.ser.write(binascii.a2b_hex(cmd))
        read = binascii.b2a_hex(self.ser.readline()).decode("ascii").replace(cmd,'')
        if self.output: print('setting complete')
        if return_bit: return read
    
    def read_status(self, address, return_bit=False):
        if (address != 0) & (address != 1):
            print('ERROR: invalid address')
            return
        cmd = '040'+ str(address) + '020' + str(6+address) 
        self.driver.ser.write(binascii.a2b_hex(cmd))
        read = self.driver.ser.readline()
        read = binascii.b2a_hex(read).decode('ascii').replace(cmd,'')
        status = read[6:8]
        status = bin(int(status[0],16))[2:].zfill(4) + bin(int(status[1],16))[2:].zfill(4)
        label = []
        if address == 0: label = ['Pulse output','Imposition','Driver alarm','Gain SW','Excitation','None','Feedback type','Feedback status']
        elif address == 1: label = ['Pulse output','General input','Alarm signal','Imposition','ORG signal','None','+LS','-LS']
        if self.output: 
            if address == 0: print('===== read status from Z4820 =====')
            elif address == 1: print('===== read status from MCCA00A =====')
            print('status: ' + str(status))
            status = status[::-1]
            for i in range(len(label)):
                if i == 5: continue
                print(label[i] + ': ' + str(status[i]))
            print('==================================')
            print('')
        status = status[::-1]
        if return_bit: return read      
   
    def read_ascii(self,address,return_bit=False):
        if (address != 0) & (address != 1):
            print('error: invalid address')
            return
        cmd = '040' + str(address) + '040' + str(8+address)
        self.driver.ser.write(binascii.a2b_hex(cmd))
        read = self.driver.ser.readline()
        read = read[7:-2].decode('ascii').replace('\r','\n')
        if self.output: print(read)
        if return_bit: return read
        
    def read_position(self, return_bit=False):
        # position: -134,217,728 ~  +134,217,727
        # hex 4byte -> 28bit, two's complement
        cmd = '04014045'
        self.ser.write(binascii.a2b_hex(cmd))
        read = self.ser.readline()
        read = binascii.b2a_hex(read).decode("ascii").replace(cmd,'')
        position = read[6:13]
        position = self.hex_to_signed_int(position,28)
        if self.output: print('position: ' + str(position))
        if return_bit: return position
  
    def read_speed(self,return_bit=False):
        cmd = '04014146'
        self.ser.write(binascii.a2b_hex(cmd))
        read = self.ser.readline()
        read = binascii.b2a_hex(read).decode("ascii").replace(cmd,'')
        speed = int(read[6:10],16)
        if self.output: print('current speed: ' + str(speed) + ' [pps]')
        if return_bit: return speed

    def read_alarm_status(self,return_bit=False):
        cmd = '04014247'
        self.ser.write(binascii.a2b_hex(cmd))
        read = self.ser.readline()
        read = binascii.b2a_hex(read).decode("ascii").replace(cmd,'')
        status = read[6:10]
        status = bin(int(status[2],16))[2:].zfill(4) + bin(int(status[3],16))[2:].zfill(4) + bin(int(status[0],16))[2:].zfill(4) + bin(int(status[1],16))[2:].zfill(4)
        if self.output: print('alarm status: ' + str(status) + ' [pps]')
        if return_bit: return status
    
    def operation_setting(self, data_8bit='01010010',return_bit=False):
        cmd = hex(int(data_8bit,2))[2:].zfill(2)
        checksum = hex(int('05',16) + int('01',16) + int('34',16) + int(cmd,16))[2:]
        cmd = '050134' + cmd + checksum
        self.ser.write(binascii.a2b_hex(cmd))
        read = self.ser.readline()
        read = binascii.b2a_hex(read).decode("ascii").replace(cmd,'')
        if self.output: print('operation setting: ' + str(read))
        if return_bit: return read
        
    def pulse_setting(self,data_4bit='0110',return_bit=False):
        cmd = hex(int(data_4bit,2))[2:].zfill(2)
        checksum = hex(int('05',16) + int('01',16) + int('35',16) + int(cmd,16))[2:]
        cmd = '050135' + cmd + checksum
        self.ser.write(binascii.a2b_hex(cmd))
        read = self.ser.readline()
        read = binascii.b2a_hex(read).decode("ascii").replace(cmd,'')
        if self.output: print('pulse setting: ' + str(read))
        if return_bit: return read
    
    def highrate_speed_setting(self,pps,return_bit=False):
        # Editted by Shinya in 20210824
        # Original code's pps limit was ~65000. 
        # Updated to 16777215 pps.
        # pps: pulse per second
        hex_pps = hex(pps)[2:].zfill(6)
        checksum = hex(int('07',16) + int('01',16) + int('21',16) + int(hex_pps[:2],16) + int(hex_pps[2:4],16) + int(hex_pps[4:6],16))[2:]
        cmd = '070121' + hex_pps[4:6] + hex_pps[2:4] + hex_pps[:2] + checksum[-2:]
        #print (cmd)
        self.ser.write(binascii.a2b_hex(cmd))
        read = self.ser.readline()
        read = binascii.b2a_hex(read).decode("ascii").replace(cmd,'')
        if self.output: #print('highrate speed setting: ' + str(pps) + ' [pps], status: ' + str(read))
            print('\r'+'highrate speed setting: {0} [pps], status: {1}, cmd = {2}'.format(pps,read,cmd),end='')
        if return_bit: return read
       
        
    def startup_speed_setting(self,pps,return_bit=False): 
        # pps: pulse per second
        hex_pps = hex(pps)[2:].zfill(4)
        checksum = hex(int('06',16) + int('01',16) + int('22',16) + int(hex_pps[:2],16) + int(hex_pps[2:],16))[2:]
        cmd = '060122' + hex_pps[2:] + hex_pps[:2] + checksum[-2:]
        print (cmd)
        self.ser.write(binascii.a2b_hex(cmd))
        read = self.ser.readline()
        read = binascii.b2a_hex(read).decode("ascii").replace(cmd,'')
        if self.output: print('startup speed setting: ' + str(pps) + ' [pps], status: ' + str(read))
        if return_bit: return read
        
    def acc_rate_setting(self,ppsps,return_bit=False):  
        # pps: pulse per second / second
        hex_ppsps = hex(ppsps)[2:].zfill(4)
        checksum = hex(int('06',16) + int('01',16) + int('23',16) + int(hex_ppsps[:2],16) + int(hex_ppsps[2:],16))[2:]
        cmd = '060123' + hex_ppsps[2:] + hex_ppsps[:2] + checksum[-2:]
        print (cmd)
        self.ser.write(binascii.a2b_hex(cmd))
        read = binascii.b2a_hex(self.ser.readline()).decode("ascii").replace(cmd,'')
        if self.output: print('acceleration rate setting: ' + str(ppsps) + ' [pps/s], status: ' + str(read))
        if return_bit: return read
    
    def initial_position(self,position=51200,return_bit=False):
        # position: -134,217,728 ~  +134,217,727
        hex_position = hex(position)[2:]
        checksum = hex(int('06',16) + int('01',16) + int('12',16) + int(hex_position[:2],16) + int(hex_position[2:],16))[2:]
        cmd = '060112' + hex_position + checksum
        self.ser.write(binascii.a2b_hex(cmd))
        read = binascii.b2a_hex(self.ser.readline()).decode("ascii").replace(cmd,'')
        if self.output: print('initial position: ' + str(position) + ', status: ' + str(read))
        if return_bit: return read
    
    def current_on(self,current,return_bit=False):
        # current: percent from 1 A
        curr = hex(current)[2:].zfill(2)
        checksum = hex(int('05',16) + int('00',16) + int('30',16) + int(curr,16))[2:]
        cmd = '050030' + curr + checksum
        self.ser.write(binascii.a2b_hex(cmd))
        read = binascii.b2a_hex(self.ser.readline()).decode("ascii").replace(cmd,'')
        if self.output: print('current: ' + str(current*10) + ' [mA]' + ', status: ' + str(read))
        if return_bit: return read
    
    def rot_start(self,rot=0,return_bit=False):
        # rot == 0: CW
        # rot == 1: CCW
        label='CW'
        if rot == 0: cmd = '0501110017'
        elif rot == 1: 
            cmd = '0501110118'
            label='CCW'
        else: 
            print('ERROR: invalid address')
            return
        self.ser.write(binascii.a2b_hex(cmd))
        read = binascii.b2a_hex(self.ser.readline()).decode("ascii").replace(cmd,'')
        if self.output: print(label + ' rotation start')
        if return_bit: return read


class ExtendedAEdriver(AEdriver):

    def __init__(self, COM, baudrate=38400):
        super().__init__(COM,baudrate)

    def tx_cmd(self, tx_packet: bytes):
        """
        Sends a command byte string to the serial port.
        Clears the input buffer before sending.
        """
        if not self.ser: return
        #if self.output:
        #    print(f'TX: {tx_packet.hex().upper()}')
        self.ser.reset_input_buffer()
        self.ser.write(tx_packet)

    def rx_cmd(self, tx_packet: bytes, wait_ms: int = 20, strip_echo: bool = True) -> bytes:
        """
        Receives a response from the serial port using read_all.

        Args:
            tx_packet (bytes): The sent command. Used for echo removal.
            wait_ms (int): Wait time in milliseconds after sending before starting to receive.
            strip_echo (bool): If True, removes the local echo from the received data.

        Returns:
            bytes: The response data.
        """
        if not self.ser: return b''
        
        if wait_ms > 0:
            time.sleep(wait_ms / 1000)

        rx_packet: bytes = self.ser.read_all()
        
        #if self.output:
        #     print(f'RX (raw): {rx_packet.hex().upper()}')

        # Remove local echo
        if strip_echo and rx_packet.startswith(tx_packet):
            rx_packet = rx_packet[len(tx_packet):]
        
        return rx_packet
    
    def send_cmd(self, cmd: str, *, strip_echo: bool = True) -> bytes:
        """Sends a command in hex string format and returns the response as a byte string."""
        try:
            tx_packet = binascii.unhexlify(cmd)
        except binascii.Error:
            if self.output:
                print(f"Error: Invalid hex string provided: {cmd}")
            return b''

        self.tx_cmd(tx_packet)
        rx_response = self.rx_cmd(tx_packet, wait_ms=20, strip_echo=strip_echo)
        
        return rx_response

    # --- Individual command methods with a unified interface ---

    def initial_position(self, position=200, return_bit=False):
        
        hex_position = position.to_bytes(2, 'little')
        fixed = bytes([0x06, 0x01, 0x12])
        cmd_body = fixed + hex_position
        checksum = sum(cmd_body) & 0xFF
        tx_packet = cmd_body + bytes([checksum])

        self.tx_cmd(tx_packet)
        rx_response = self.rx_cmd(tx_packet, wait_ms=20)
        read = rx_response.hex().upper() 

        if self.output:
            print('initial position: ' + str(position) + ', status: ' + str(read))
        if return_bit:
            return read

    def write_electric_gear_numerator(self, numerator=40):
        hex_numerator = numerator.to_bytes(2, 'little')
        fixed = bytes([0x06, 0x00, 0x35])
        cmd_body = fixed + hex_numerator
        checksum = sum(cmd_body) & 0xFF
        tx_packet = cmd_body + bytes([checksum])
        
        self.tx_cmd(tx_packet)
        rx_response = self.rx_cmd(tx_packet, wait_ms=20)

    def read_electric_gear_numerator(self):
        cmd_body = bytes([0x05, 0x00, 0x80, 0x35])
        checksum = sum(cmd_body) & 0xFF
        tx_packet = cmd_body + bytes([checksum])

        self.tx_cmd(tx_packet)
        rx_response = self.rx_cmd(tx_packet, wait_ms=20)

        rx_hex = rx_response.hex().upper()
        if len(rx_hex) >= 10: # 5 bytes response expected
            data_hex = rx_hex[6:10]
            decimal_val = int.from_bytes(bytes.fromhex(data_hex), "little")
            return decimal_val
        else:
            return None

    def read_electric_gear_denominator(self):
        """Reads the denominator of the electric gear."""
        cmd_body = bytes([0x05, 0x00, 0x80, 0x36])
        checksum = sum(cmd_body) & 0xFF
        tx_packet = cmd_body + bytes([checksum])

        self.tx_cmd(tx_packet)
        rx_response = self.rx_cmd(tx_packet, wait_ms=20)

        rx_hex = rx_response.hex().upper()
        if len(rx_hex) >= 10: # 5 bytes response expected
            data_hex = rx_hex[6:10]
            decimal_val = int.from_bytes(bytes.fromhex(data_hex), "little")
            return decimal_val
        else:
            return None

    def read_current(self):
        cmd_body = bytes([0x05, 0x00, 0x80, 0x30])
        checksum = sum(cmd_body) & 0xFF
        tx_packet = cmd_body + bytes([checksum])

        self.tx_cmd(tx_packet)
        rx_response = self.rx_cmd(tx_packet, wait_ms=20)
        
        rx_hex = rx_response.hex().upper()
        if len(rx_hex) >= 8: # 5 bytes response expected
            data_hex = rx_hex[6:10]
            print(rx_hex,data_hex)
            decimal_val = int.from_bytes(bytes.fromhex(data_hex), "little")
            current_ma = decimal_val * 10
            return current_ma
        else:
            return None
    
    def read_internal_oscillator_speed(self):
        """Reads the speed of the internal oscillator."""
        cmd_body = bytes([0x05, 0x00, 0x80, 0x21])
        checksum = sum(cmd_body) & 0xFF
        tx_packet = cmd_body + bytes([checksum])

        self.tx_cmd(tx_packet)
        rx_response = self.rx_cmd(tx_packet, wait_ms=20)

        rx_hex = rx_response.hex().upper()
        if len(rx_hex) >= 10: # 5 bytes response expected
            data_hex = rx_hex[6:10]
            decimal_val = int.from_bytes(bytes.fromhex(data_hex), "little")
            return decimal_val
        else:
            return None

    def read_command_pulse_MCP100A(self):
        """Reads the command pulse for MCP100A as a 32-bit integer."""
        rx_packet = self.send_cmd(cmd='04014045', strip_echo=True)
        if len(rx_packet) >= 4:
            value = int.from_bytes(rx_packet[:4], 'little', signed=True)
            return value
        else:
            return None

    def read_command_pulse_D5700_position(self):
        """Reads the command pulse for D5700 (position) as a 32-bit integer."""
        rx_packet = self.send_cmd(cmd='04004044', strip_echo=True)
        if len(rx_packet) >= 4:
            value = int.from_bytes(rx_packet[:4], 'little', signed=True)
            return value
        else:
            return None


    def read_command_pulse_D5700_speed(self):
        """Reads the command pulse for D5700 (speed) as a 32-bit integer."""
        rx_packet = self.send_cmd(cmd='04004144', strip_echo=True)
        if len(rx_packet) >= 4:
            value = int.from_bytes(rx_packet[:4], 'little', signed=True)
            return value
        else:
            return None

    def read_idn_MCCA100A(self):
        """Reads the IDN string for MCCA100A."""
        rx_packet = self.send_cmd(cmd='04000408', strip_echo=True)
        response_str = rx_packet.decode('ascii', errors='ignore').strip()
        return response_str

    def read_idn_D5700(self):
        """Reads the IDN string for D5700."""
        rx_packet = self.send_cmd(cmd='04010409', strip_echo=True)
        response_str = rx_packet.decode('ascii', errors='ignore').strip()
        return response_str
        
    def highrate_speed_setting(self, pps: int, *, readback: bool = True, return_bit: bool = False):
        """
        Sets the High-rate Speed (CMD=0x21) and optionally reads it back.
        
        Parameters
        ----------
        pps : int
            The desired pulse frequency [pps].
        readback : bool, optional
            If True (default), sends a command to read the current speed and display it.
        return_bit : bool, optional
            If True, returns the ACK status as a hex string.
        """
        # Convert pps to a 3-byte little-endian value
        pps_bytes = pps.to_bytes(3, 'little')
        
        # Construct the command packet
        cmd_body = bytes([0x07, 0x01, 0x21]) + pps_bytes
        checksum = sum(cmd_body) & 0xFF
        tx_packet = cmd_body + bytes([checksum])

        # Send command and get response
        self.tx_cmd(tx_packet)
        rx_response = self.rx_cmd(tx_packet)
        ack_hex = rx_response.hex().upper()

        if self.output:
            print(f'High-rate speed setting: {pps} [pps], status: {ack_hex}')

        if return_bit:
            return ack_hex

    def query(self):        
        # Execute read commands
        #idn_mca = self.read_idn_MCCA100A()
        #idn_d5700 = self.read_idn_D5700()
        gear_num = self.read_electric_gear_numerator()
        gear_den = self.read_electric_gear_denominator()
        current = self.read_current()
        #osc_speed = self.read_internal_oscillator_speed()
        pulse_mcp = self.read_command_pulse_MCP100A()
        pulse_d5700_pos = self.read_command_pulse_D5700_position()
        pulse_d5700_speed = self.read_command_pulse_D5700_speed()

        # Print the collected values
        #print(f"  - IDN (MCCA100A): {idn_mca}")
        #print(f"  - IDN (D5700): {idn_d5700}")
        print(f"Gear {gear_num} / {gear_den}, {current} mA, pulse gen {pulse_mcp}, drv_pos: {pulse_d5700_pos}, drv_speed {pulse_d5700_speed}")

        
class StableRotation():
        
    def __init__(self, driver, pps, pps_fin, step, ppsps, initial_curr, electric_angle_init, rotshift_time, stablerot_duration, excess_spindown_time, DC_duration, rot=0, DEBUG=False):
        
        print ("All process start time", datetime.datetime.now())
        
        # Call the motor driver
        self.driver = driver
        
        # Parameters
        self.pps = pps
        self.pps_fin = pps_fin
        self.step = step
        self.ppsps = ppsps
        self.initial_curr = initial_curr
        self.electric_angle_init = electric_angle_init
        self.DC_duration = DC_duration
        try:
            if self.initial_curr >= 50:
                raise
        except:
            print ("Look at the value of current!")
            self.driver.stop()
            self.driver.initialize()
            self.driver.ser.close()
            raise
            
        self.rot = rot
        self.DEBUG = DEBUG
        
        self.rotshift_time = rotshift_time
        self.stablerot_duration = stablerot_duration
        self.excess_spindown_time = excess_spindown_time

        self.loop_limit = 40_000
        
        # Call function
        self.driver.initialize()
        self.driver.operation_setting()
        self.driver.pulse_setting()
        
        if self.rot == 0:
            self.driver.initial_position(position=self.electric_angle_init)
        elif self.rot == 1:
            self.driver.initial_position(position=self.electric_angle_init)

        self.driver.highrate_speed_setting(self.pps)
        self.driver.startup_speed_setting(self.pps)
        self.driver.acc_rate_setting(self.ppsps)
        self.driver.complete_setting()

        self.driver.current_on(self.initial_curr)
        time.sleep(self.DC_duration)

    #def __del__(self):
    #    self.driver.stop()
    #    self.driver.initialize()
    #    self.driver.ser.close()
            
    def rotation(self):
        
        self.driver.rot_start(rot=self.rot)
        time.sleep(self.rotshift_time)
        
        n = 0
        while n < self.loop_limit:
            try:
                self.pps = self.pps + self.step
                if self.pps >= self.pps_fin:
                    self.driver.highrate_speed_setting(self.pps_fin)
                    break
                self.driver.highrate_speed_setting(self.pps)
                #driver.complete_setting()
                time.sleep(self.rotshift_time)
                n = n + 1
            except KeyboardInterrupt:
                break
                
        print("\n")
        print ("========================================================================================")
        print ("Start",self.stablerot_duration, "[s] stable rotation at", datetime.datetime.now())
        time.sleep(self.stablerot_duration)
        print ("Finish stable rotation duration at", datetime.datetime.now())
        print ("For an open spindown: Please disconnect the current cables")
        print ("For an close spindown: Please run the reset() command, otherwise this will be an excess time before stop sending the current")
        print('And the time before stop the motor driver is: ', self.excess_spindown_time/60, '[min]')
        time.sleep(self.excess_spindown_time)
        
        self.driver.stop()
        time.sleep(1)
        self.driver.initialize()
        print ("Rot all done")
    
    def current_reduction(self):
        
        self.driver.rot_start(rot=self.rot)
        time.sleep(self.rotshift_time)
        
        n = 0
        while n < self.loop_limit:
            try:
                self.pps = self.pps + self.step
                if self.pps >= self.pps_fin:
                    self.driver.highrate_speed_setting(self.pps_fin)
                    break
                self.driver.highrate_speed_setting(self.pps)
                time.sleep(self.rotshift_time)
                if self.DEBUG:
                    self.driver.query()

                n = n + 1
            except KeyboardInterrupt:
                break

        print("\n")
        print ("========================================================================================")
        print ("Waiting extra command...")
        print ("if not aplicable, please enter the stop()")
    
    