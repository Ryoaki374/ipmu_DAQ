# ipmu_DAQ

ipmu_DAQ is a lightweight Python library for controlling NI-DAQ hardware, logging acquired data, and performing real-time processing and visualization.  
Originally developed for the Breadboard Model (BBM) of our rotation mechanism, but its modular design makes it reusable in other laboratory setups.

## Directory layout
README.md            - project documentation  
requirements.txt     - Python dependencies  
data/                - saved measurement files  
src/                 - library source code  
    daq_controller/    - NI-DAQ control modules  
    processing/        - post-processing utilities  
    visualization/     - real-time plotting widgets  
example_notebooks/   - Jupyter notebooks demonstrating workflows  

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

## Quick start
Open any notebook in example_notebooks/ and run the cells. They demonstrate:  
1. Discovering connected NI-DAQ devices  
2. Configuring acquisition parameters  
3. Streaming data to disk (HDF5 or CSV)  
4. Live plotting and basic analysis   

## Contributing
please add issues if you find bugs!

## License
MIT License (see LICENSE file)

## Memo
pip install pyqtgraph PyQt6


| **速度目標 [rps]** | **1秒当たり [pulses / s]** | **1iterあたり [pulses]** | **カウント** | **再構成速度 (UP / DOWN) \[rps]** | **偏差 \[%]** |
| ----------------- | -------------------------- | --------------------------- | ----------------------- | ------------------------------- | -------------------------- |
| 0.1               | 204.8                      | 25.6                        | 26<br>25                | 0.102<br>0.09765625             | +1.56<br>−2.34             |
| 0.2               | 409.6                      | 51.2                        | 52<br>51                | 0.203<br>0.19921875             | +1.56<br>−0.39             |
| 0.3               | 614.4                      | 76.8                        | 77<br>76                | 0.301<br>0.296875               | +0.26<br>−1.04             |
| 0.4               | 819.2                      | 102.4                       | 103<br>102              | 0.402<br>0.3984375              | +0.59<br>−0.39             |
| 0.5               | 1 024.0                    | 128.0                       | 128<br>128              | 0.500<br>0.500                  | 0.00<br>0.00               |
| 0.6               | 1 228.8                    | 153.6                       | 154<br>153              | 0.602<br>0.59765625             | +0.26<br>−0.39             |
| 0.7               | 1 433.6                    | 179.2                       | 180<br>179              | 0.703<br>0.69921875             | +0.45<br>−0.11             |
| 0.8               | 1 638.4                    | 204.8                       | 205<br>204              | 0.801<br>0.796875               | +0.10<br>−0.39             |
| 0.9               | 1 843.2                    | 230.4                       | 231<br>230              | 0.902<br>0.8984375              | +0.26<br>−0.17             |
| 1.0               | 2 048.0                    | 256.0                       | 256<br>256              | 1.000<br>1.000                  | 0.00<br>0.00               |


| **速度目標 [rps]** | **1秒当たり [pulses / s]** | **1iterあたり [pulses]** | **カウント** | **再構成速度 (UP / DOWN) \[rps]** | **偏差 \[%]** |
| ----------------- | -------------------------- | ------------------------------------ | ----------------------- | ------------------------------- | -------------------------- |
| 0.1               | 204.8                      | 51.2                                 | 52<br>51                | 0.102<br>0.100                  | +1.56<br>−0.39             |
| 0.2               | 409.6                      | 102.4                                | 103<br>102              | 0.201<br>0.199                  | +0.59<br>−0.39             |
| 0.3               | 614.4                      | 153.6                                | 154<br>153              | 0.301<br>0.299                  | +0.26<br>−0.39             |
| 0.4               | 819.2                      | 204.8                                | 205<br>204              | 0.400<br>0.398                  | +0.10<br>−0.39             |
| 0.5               | 1 024.0                    | 256.0                                | 256<br>256              | 0.500<br>0.500                  | 0.00<br>0.00               |
| 0.6               | 1 228.8                    | 307.2                                | 308<br>307              | 0.602<br>0.600                  | +0.26<br>−0.07             |
| 0.7               | 1 433.6                    | 358.4                                | 359<br>358              | 0.701<br>0.699                  | +0.17<br>−0.11             |
| 0.8               | 1 638.4                    | 409.6                                | 410<br>409              | 0.801<br>0.799                  | +0.10<br>−0.15             |
| 0.9               | 1 843.2                    | 460.8                                | 461<br>460              | 0.900<br>0.898                  | +0.04<br>−0.17             |
| 1.0               | 2 048.0                    | 512.0                                | 512<br>512              | 1.000<br>1.000                  | 0.00<br>0.00               |

