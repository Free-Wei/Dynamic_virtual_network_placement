# Dynamic_virtual_network_placement

This is the code attached to the following paper and variables in the code are the same as notations in the paper:

***Efficient Function Placement in Virtual Networks: An Online Learning Approach***, *Wei HUANG, Richard Combes, Hind Castel, Badii Jouaber*. (https://arxiv.org/abs/2410.13696)


# Requirements

- Python3

- numpy, matplotlib

- scipy


# How to run

1.To reproduce the results the same as numerical results in the paper, run the script `main.py`:
```
cd Dynamic_virtual_network_placement
python3 main.py
```
2.To reproduce the comparison between different algorithms (Fig.2 and Fig.3 in the paper) , run the script `plot_diff_algo.py`:
```
cd Dynamic_virtual_network_placement
python3 plot_diff_algo.py
```
3.To reproduce the comparison between different instances (Fig.4 and Fig.5 in the paper) , run the script `plot_diff_instances.py`:
```
cd Dynamic_virtual_network_placement
python3 plot_diff_instances.py
```

# Tips
- All performance measures are computed over 50 independent runs in the paper, feel free to change `test_times` in the above script.

- All date will be saved into the folder `dataOUTPUT` and all figures into `img` with format PDF.
