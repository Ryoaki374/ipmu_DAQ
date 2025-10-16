# ipmu_DAQ

This repository defines the arcitecture of the Python library for controlling NI-DAQ hardware, logging acquired data, and performing real-time processing and visualization.
Originally developed for the Breadboard Model (BBM) of our rotation mechanism. The software treats each component e.g. motor control, data acquisition/generation, post-processing, and visualization as an independent class. Each process runs asynchronously on its own thread, and thread-safe queues are used for inter-class data transfer.

## Directory layout
```
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ graph_preset.mplstyle # graph settings
├─ runs/ # data dir
├─ archive/ # previous versions
├─ src/
   ├─ _config_run.toml              # header file for each run
   ├─ _config_preset.toml           # header file for post processing and GUI
   ├─ lib_ipmu_daq_api.py           # lib for DAQApp
   ├─ lib_ipmu_daq_acquisition.py   # lib for NI-DAQ
   ├─ lib_ipmu_daq_config.py        # lib for I/O _config_preset.toml
   ├─ lib_ipmu_daq_generator.py     # lib for debug
   ├─ lib_ipmu_daq_graph.py         # lib for GUI
   ├─ lib_ipmu_daq_process.py       # lib for post-processing
   ├─ lib_ipmu_drv_command.py       # lib for motor control
   ├─ lib_ipmu_drv_config.py        # lib for I/O _config_preset.toml
   ├─ lib_plotdev.py                # lib for plot
   ├─ class_AEdriver_rotation.py    # lib for motor control
   ├─ dev_refresh_encoder_daq.ipynb # developing version
   ├─ hdf2csv.ipynb                 # transform hdf to csv
   └─ Integrated_motor_controller.ipynb # main notebook
```

## Installation
1. Clone the repository
```bash
   git clone https://github.com/Ryoaki374/ipmu_DAQ.git  
   cd ipmu_DAQ
```

2. (Optional) Create a virtual environment
```bash
   python -m venv .venv
   source .venv/bin/activate     (Windows: .venv\Scripts\activate)  
```

3. Install dependencies
```bash  
   pip install -r requirements.txt
```

4. Install ipmu_DAQ in editable mode
```bash  
   pip install -e .  
```

## Contributing
please add issues if you find bugs!

## License
MIT License (see LICENSE file)
