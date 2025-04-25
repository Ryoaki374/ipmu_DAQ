ipmu_DAQ

ipmu_DAQ is a lightweight Python library for controlling NI-DAQ hardware, logging acquired data, and performing real-time processing and visualization.  
Originally developed for the Breadboard Model (BBM) of our rotation mechanism, but its modular design makes it reusable in other laboratory setups.

Directory layout
----------------
README.md            - project documentation  
requirements.txt     - Python dependencies  
data/                - saved measurement files  
src/                 - library source code  
    daq_controller/    - NI-DAQ control modules  
    processing/        - post-processing utilities  
    visualization/     - real-time plotting widgets  
example_notebooks/   - Jupyter notebooks demonstrating workflows  

Installation
------------
1. Clone the repository  
   git clone https://github.com/Ryoaki374/ipmu_DAQ.git  
   cd ipmu_DAQ  

2. (Optional) Create a virtual environment  
   python -m venv .venv  
   source .venv/bin/activate     (Windows: .venv\Scripts\activate)  

3. Install dependencies  
   pip install -r requirements.txt  

4. Install ipmu_DAQ in editable mode  
   pip install -e .  

Quick start
-----------
Open any notebook in example_notebooks/ and run the cells. They demonstrate:  
1. Discovering connected NI-DAQ devices  
2. Configuring acquisition parameters  
3. Streaming data to disk (HDF5 or CSV)  
4. Live plotting and basic analysis   

Contributing
------------
please add issues if you find bugs!

License
-------
MIT License (see LICENSE file)
