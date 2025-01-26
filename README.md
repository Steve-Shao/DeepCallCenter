# DeepCallCenter
 
This project requires two submodules:

- [TorchBSDE](https://github.com/steve-shao/TorchBSDE-Package)
- [TorchSimulator](https://github.com/steve-shao/TorchSimulator)

Please clone the repositories with the following command:
```
git submodule add https://github.com/steve-shao/TorchBSDE-Package.git torchbsde
git submodule add https://github.com/steve-shao/TorchSimulator.git torchsimulator
git submodule update --init --recursive
```

To update the submodules:
```
git submodule update --remote
```

nohup python main.py --ref_policy=minimal --bn_input=True --bn_hidden=True --bn_output=True > BN_TTT_minimal.log 2>&1 &
nohup python main.py --ref_policy=minimal --bn_input=True --bn_hidden=True --bn_output=False > BN_TTF_minimal.log 2>&1 &
nohup python main.py --ref_policy=minimal --bn_input=False --bn_hidden=True --bn_output=False > BN_FTF_minimal.log 2>&1 &
nohup python main.py --ref_policy=minimal --bn_input=True --bn_hidden=False --bn_output=False > BN_TFF_minimal.log 2>&1 &
nohup python main.py --ref_policy=minimal --bn_input=False --bn_hidden=False --bn_output=False > BN_FFF_minimal.log 2>&1 &


nohup python main.py --ref_policy=even --bn_input=True --bn_hidden=False --bn_output=False > BN_TFF_minimal.log 2>&1 &
nohup python main.py --ref_policy=weighted_split --bn_input=True --bn_hidden=False --bn_output=False > WS_TFF_minimal.log 2>&1 &