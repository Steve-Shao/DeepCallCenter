# DeepCallCenter

This project is an implementation of the paper:
- [Dynamic Scheduling of a Multiclass Queue in the Halfin-Whitt Regime: A Computational Approach for High-Dimensional Problems](https://arxiv.org/abs/2311.08818), 

authored by [Baris Ata](https://www.chicagobooth.edu/faculty/directory/a/baris-ata) and [Ebru Kasilarlar](https://ekasikaralar.github.io/Ebru-Kasikaralar/).

The paper discusses a dynamic scheduling problem for a call center system. 
The problem is modeled as a CTMDP, which yields a brownian control problem at the heavy-traffic limit. 
The paper shows that the brownian control problem can be solved using a deep learning approach, and with that, good control policies can be found for the CTMDP.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/steve-shao/DeepCallCenter.git
   cd DeepCallCenter
   ```

2. **Set Up a Conda Environment:**

   ```bash
   conda env create -f environment.yml
   conda activate deepcallcenter
   ```

3. **Run the code:**

    ```bash
    python main.py
    ```


## Dependencies
 
This project relies on two submodules:

- [TorchBSDE](https://github.com/steve-shao/TorchBSDE-Package)
    - This repository implements [Jiequn Han](https://users.flatironinstitute.org/~jhan/)'s [DeepBSDE solver](https://github.com/frankhan91/DeepBSDE) in `PyTorch`. The solver [solves high-dimensional PDEs using deep learning](https://doi.org/10.1073/pnas.1718942115). Various improvements are made to the original implementation.

- [TorchSimulator](https://github.com/steve-shao/TorchSimulator)
    - This repository is a demonstration framework designed to showcase how parallel, **tensorized simulations on GPUs** can accelerate performance. It provides a foundation for building custom simulation classes by allowing users to inherit from the provided `CTMCSimulator` or `CTMDPSimulator` classes.

The repositories can be cloned with the following command:
```
git submodule add https://github.com/steve-shao/TorchBSDE-Package.git torchbsde
git submodule add https://github.com/steve-shao/TorchSimulator.git torchsimulator
git submodule update --init --recursive
```

To update the submodules, run:
```
git submodule update --remote
```


## License

This project is licensed under the [MIT License](LICENSE).
