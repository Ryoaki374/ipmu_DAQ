import time
import datetime as dt
import numpy as np
import pyvisa


def query():
    # look up instruments
    rm = pyvisa.ResourceManager()
    print("resource manager : ", rm)
    rl = rm.list_resources()
    print("List of resources: \n", rl)
    for i, name in enumerate(rl):
        if "GPIB" in name:
            my_instrument = rm.open_resource(name)
            print("Resource[", i, "]:", name, "\n", my_instrument.query("*IDN?"))
    return rm, rl


def create_csv(save_path, column_names):
    """
    Creates a CSV file for multimeter data recording.
    Generates a filename based on the current timestamp and writes a header line to the file.
    Parameters:
        save_path (str): The directory where the CSV file will be saved.
    Returns:
        tuple: A tuple containing the full path of the created CSV file and the refresh rate (in seconds).
    """
    # Generate a filename using the current date and time with the specified identifier
    filename = (
        save_path
        + dt.datetime.now().strftime("%Y%m%d%H%M%S")
        + "_MULTIMETER_KT2700.csv"
    )
    print("Generated filename:", filename)
    # Attempt to open the file and write the header line
    try:
        with open(filename, mode="a") as f:
            print(*column_names, sep=", ", file=f)
    except Exception as e:
        print("Error while writing the header:", e)
    # Set the refresh rate for the data acquisition (in seconds)
    return filename


def initialize(instrument):
    """
    Initializes the KT2700 multimeter by sending the required configuration commands.
    Parameters:
        KT2700 (list): A list where the first element is the multimeter instrument object,
                       which must support the .write() method.
    Returns:
        None
    """
    try:
        # Reset the instrument to its default settings
        instrument.write("*RST")
        # Disable continuous initiation and abort any ongoing measurement
        instrument.write(":INITiate:CONTinuous OFF;:ABORt")
        # Enable automatic range for DC voltage measurements
        instrument.write(":SENSe:VOLTage:DC:RANGe:AUTO ON")
        # Set the measurement function to DC voltage
        instrument.write(':SENSe:FUNCtion "VOLTage:DC"')
        # Set the NPLC (Number of Power Line Cycles) to 0.01 for the fastest reading speed
        instrument.write(":SENSe:VOLTage:DC:NPLC 0.01")
        # Disable autozero to further increase speed (note: this may cause drift over time)
        instrument.write(":SYSTem:AZERo:STATe OFF")
        # Turn off the averaging filter for faster measurements
        instrument.write(":SENSe:VOLTage:DC:AVERage:STATe OFF")
        # Set the trigger count to 1, so one measurement is taken per trigger command
        instrument.write(":TRIGger:COUNt 1")
        # Initiate a measurement by sending the READ? command
        instrument.write("READ?")
        # Debug call
        buf_str = instrument.read()
        print(buf_str)
        # Instead of reading the response, display an initialization complete message
        print("KT2700 initialization complete.")
    except Exception as e:
        print("An error occurred during initialization:", e)


def getVoltage(self):
    """
    Sends a 'READ?' command to the instrument, retrieves the voltage reading,
    and returns it as a float.
    The instrument's response is expected to be a string where the first value
    ends with 'VDC' (e.g., "12.345VDC"). If multiple values are returned, they
    are separated by commas.
    """
    # Send the command to read the voltage
    self.write("READ?")
    # Read the response and strip any trailing whitespace or newline characters
    response = self.read().strip()
    # Split the response by commas in case multiple values are returned
    parts = response.split(",")
    try:
        # Extract the voltage value from the first part by removing the trailing "VDC"
        voltage = float(parts[0][:-3])
    except (ValueError, IndexError) as e:
        raise ValueError(
            f"Unable to parse voltage reading from response: '{response}'"
        ) from e
    return voltage


def DataAcquisition(tp, filename, save_path, column_names, instrument):
    """
    Continuously records voltage data from the given instrument and appends it to a CSV file.

    Parameters:
        tp (generator): A time provider generator yielding relative time values.
        filename (str): The path to the CSV file where data will be saved.
        save_path (str): The directory path used to create a new file if the current file is not found.
        column_names (list): List of column names for the CSV header.
        sampling_rate (float): The number of samples per second.
        instrument: The instrument object (e.g., KT2700[0]) that supports the getVoltage() method.

    The function performs the following steps in an infinite loop:
      1. Retrieves the current relative time from the time provider.
      2. Retrieves the voltage measurement from the instrument.
      3. Attempts to append the data (relative time and voltage) to the CSV file.
         If the file is not found, it creates a new file with a header and writes the data.
    """
    while True:
        # Get the current relative time from the time provider
        current_relative_time = next(tp)
        # Build the data point starting with the relative time
        data_point = [current_relative_time]
        # Retrieve the voltage measurement using the instrument's getVoltage method
        voltage_value = getVoltage(instrument)
        data_point.append(voltage_value)

        # Optionally, print the data point for debugging
        # print(data_point)

        # Try to append the data point to the CSV file
        try:
            with open(filename, mode="a") as file:
                print(*data_point, sep=", ", file=file)
        except FileNotFoundError:
            print("FileNotFoundError occurred. Creating a new file...")
            # Create a new filename based on the current timestamp
            filename = save_path + dt.datetime.now().strftime("%Y%m%d%H%M%S") + ".csv"
            with open(filename, mode="a") as file:
                # Write the CSV header and then the current data point
                print(*column_names, sep=", ", file=file)
                print(*data_point, sep=", ", file=file)


def SingleDataAcquisition(tp, filename, save_path, column_names, instrument):
    """
    Acquires one data point by recording the current relative time and voltage from the instrument,
    and writes the data point to a CSV file.

    Parameters:
        tp (generator): A time provider generator yielding relative time values.
        filename (str): The path to the CSV file where data will be saved.
        save_path (str): The directory path used to create a new file if the current file is not found.
        column_names (list): List of column names for the CSV header.
        instrument: The instrument object (e.g., KT2700[0]) that supports the getVoltage() method.
    """
    # Retrieve the current relative time from the time provider
    current_relative_time = next(tp)
    # Build the data point starting with the relative time
    data_point = [current_relative_time]
    # Retrieve the voltage measurement using the instrument's getVoltage method
    voltage_value = getVoltage(instrument)
    data_point.append(voltage_value)

    # Optionally, print the data point for debugging
    # print(data_point)

    # Try to append the data point to the CSV file
    try:
        with open(filename, mode="a") as file:
            print(*data_point, sep=", ", file=file)
    except FileNotFoundError:
        print("FileNotFoundError occurred. Creating a new file...")
        # Create a new filename based on the current timestamp
        filename = save_path + dt.datetime.now().strftime("%Y%m%d%H%M%S") + ".csv"
        with open(filename, mode="a") as file:
            # Write the CSV header and then the current data point
            print(*column_names, sep=", ", file=file)
            print(*data_point, sep=", ", file=file)
