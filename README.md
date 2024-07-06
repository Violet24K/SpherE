# OpenKE
SpherE: Expressive and Interpretable Knowledge Graph Embedding for Set Retrieval, SIGIR'2024
## Environment Setup
To run our code, you need to have a Python environment and a C++ environment. The Python version we used is 3.11. Some requirements are needed, such as pytorch, sklearn, tqdm, numpy. To run our code, one needs to compile the C++ files first
```
cd openke
bash make.sh
```
Our code builds on the [OpenKE](https://github.com/thunlp/OpenKE) public benchmark repository for Knowledge Graph Embedding. The weights of our trained models that generate the test data reported in the paper can be downloaded at https://drive.google.com/file/d/1KmMORNqsQdtx8XUa-iqZX_rOBB12A4je/view?usp=drive_link. In this README, SS means we embed the entities as spheres instead of vectors/points. SSRotatE is SpherE-2D, SSRotatE3D is SpherE-3D, and k-dimensional SSHousE-r is Sphere-kD

## Command to train/test a SpherE model based on RotatE or RotatE 3D
The first line is to train the model, and the second line is to test the model
### command to run rotate model on dataset FB15K237
```
python train_rotate_FB15K237.py
python train_rotate_FB15K237.py --test
```
### command to run rotate model on dataset WN18RR
```
python train_rotate_WN18RR.py
python train_rotate_WN18RR.py --test
```
### command to run ssrotate model on dataset FB15K237
```
python train_ssrotate_FB15K237.py
python train_ssrotate_FB15K237.py --test
```
### command to run ssrotate model on dataset WN18RR
```
python train_ssrotate_WN18RR.py
python train_ssrotate_WN18RR.py --test
```
### command to run rotate3D model on dataset FB15K237
```
python train_rotate3D_FB15K237.py
python train_rotate3D_FB15K237.py --test
```
### command to run rotate3D model on dataset WN18RR
```
python train_rotate3D_WN18RR.py
python train_rotate3D_WN18RR.py --test
```
### command to run ssrotate3D model on dataset FB15K237
```
python train_ssrotate3D_FB15K237.py
python train_ssrotate3D_FB15K237.py --test
```
### command to run ssrotate3D model on dataset WN18RR
```
python train_ssrotate3D_WN18RR.py
python train_ssrotate3D_WN18RR.py --test
```


## Use our trained SpherE-2D or SpherE-3D weights instead of re-train
After downloading our model weights, extract the compressed file. You should have a foler with two sub-folders: models and sskgemb. Please place all the files in the "sskgemb" directory into the "./checkpoint" directory of the working directory of this README. Then, you can directly run the --test commands, for example 
```
python train_rotate_FB15K237.py --test
```


## Command to train/test a SpherE model based on HousE with k-dimensional rotation.
The code for SpherE and HousE are stored in the ./HousE directory.
```
cd HousE
```

### train HousE_r on FB15k237: remember to change the save path when re-running: -save models/HousE_r_FB15k-237_0
```
python codes/run.py --do_train --cuda --do_valid --do_test --data_path data/FB15k-237 --model HousE_r -n 500 -b 500 -d 600 -hd 20 -dn 6 -th 0.6 -g 5 -a 2 -adv -lr 0.0008 --max_steps 20000 --warm_up_steps 10000 -save models/HousE_r_FB15k-237_0 --test_batch_size 16 -r 0.003367
```

### train SSHousE_r on FB15k237: remember to change the save path when re-running: -save models/HousE_r_FB15k-237_0
```
python codes/run.py --do_train --cuda --do_valid --do_test --data_path data/FB15k-237 --model SSHousE_r -n 500 -b 500 -d 600 -hd 20 -dn 6 -th 0.6 -g 5 -a 2 -adv -lr 0.0008 --max_steps 20000 --warm_up_steps 10000 -save models/SSHousE_r_FB15k-237_0 --test_batch_size 16 -r 0.003367
```

### train HousE_r on WN18RR: remember to change the save path when re-running: -save models/HousE_r_wn18rr_0
```
python codes/run.py --do_train --cuda --do_valid --do_test --data_path data/wn18rr --model HousE_r -n 1000 -b 200 -d 800 -hd 20 -dn 1 -th 0.5 -g 6 -a 14940435933987 -adv -lr 0.00057 --max_steps 40000 --warm_up_steps 20000 -save models/HousE_r_wn18rr_0 --test_batch_size 8 -r 0.0960737047401994
```

### train SSHousE_r on WN18RR: remember to change the save path when re-running: -save models/SSHousE_r_wn18rr_0
```
python codes/run.py --do_train --cuda --do_valid --do_test --data_path data/wn18rr --model HousE_r -n 1000 -b 200 -d 800 -hd 20 -dn 1 -th 0.5 -g 6 -a 14940435933987 -adv -lr 0.00057 --max_steps 40000 --warm_up_steps 20000 -save models/SSHousE_r_wn18rr_0 --test_batch_size 8 -r 0.0960737047401994
```


### test HousE_r model: remember to check the model path to load --path dim_k_HousE_r_FB15k-237_0
```
python codes/test_sshouse_retrieval.py --path dim_k_HousE_r_FB15k-237_0 --model HousE_r
```

### test SSHousE_r model: remember to check the model path to load --path dim_k_SSHousE_r_FB15k-237_0
```
python codes/test_sshouse_retrieval.py --path dim_k_SSHousE_r_FB15k-237_0 --model SSHousE_r
```


### test HousE_r model: remember to check the model path to load --path dim_k_HousE_r_wn18rr_0
```
python codes/test_sshouse_retrieval.py --path dim_k_HousE_r_wn18rr_0 --model HousE_r
```

### test SSHousE_r model: remember to check the model path to load --path dim_k_SSHousE_r_wn18rr_0
```
python codes/test_sshouse_retrieval.py --path dim_k_SSHousE_r_wn18rr_0 --model SSHousE_r
```


## Use our trained SpherE-kD weights instead of re-train
After downloading our model weights, extract the compressed file. You should have a foler with two sub-folders: models and sskgemb. Please place all the files in the "model" directory into the "./checkpoint" directory of the working directory of this README. Then, you can directly run the --test commands, for example 
```
python codes/test_sshouse_retrieval.py --path dim_k_SSHousE_r_FB15k-237_0 --model SSHousE_r
```