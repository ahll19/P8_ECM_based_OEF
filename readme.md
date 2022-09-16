## Background

This is the code repository of the article ["An Efficient Optimal Energy Flow Model for Integrated Energy Systems Based on Energy Circuit Modeling in the Frequency Domain"](https://arxiv.org/abs/2206.12799), which solves the **optimal energy flow** (OEF) problem of an integrated energy system by the **energy circuit method** (ECM).



## Quick Start

1. Install [Gurobi](https://www.gurobi.com/) (a license is required) and [Anaconda](https://www.anaconda.com/) (not necessary but recommended).

2. Prepare a Python environment: 

   Clone this repo to your local disk, and open a terminal/cmd in its root directory to enter the following codes.

   ```cmd
   conda create -n ecm_oef python=3.8
   conda activate ecm_oef
   pip install -r requirements.txt
   ```

3. Run the demo codes:

   ```cmd
   python demo_usage.py --instance_file "instance/small case/IES_E9H12G7-v1.xlsx" --model_type lazy_explicit
   ```

4. Do modifications to satisfy your customized needs.



## TODO List

- [x] Code cleaning and commenting.
- [ ] A document about input file structure, script arguments, involved classes and functions, etc.
- [ ] Support more functions, e.g. considering N-1 contingency.



Issues and Pull Requests are welcomed : )

