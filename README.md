# Config total
There are two config files:
- ```config.json``` for general config
- ```train/loader/timestep.py``` for dataset edit

## General config
File name ```config.json```, located at the root folder.

### Change aggregate types & aggregate steps:
There are three types of aggregate ("alpha", "euclid", "cosine"):
```
config["aggregate_data"]["agg_type"]
```
In addition, you could change the aggregate steps:
```
config["aggregate_data"]["time_step"]
```

### Change model name: 
Change this value, to any value
```
config["normalized_data"]["file_name"]
```

## Train step config
File name ```train/loader/timestep.py```.

Each ```Dataset config``` has three main values: ```name, time_step, label```.

```name``` is for naming the model only.

```time_step``` is to define which time steps are used for training.

```label``` is to define which the labels of used time steps above. Each element of the label corresponds to the value of time step above. The length of the time step list needs to be equal to the length of the label list.

For example:
```
dataset_0 = DatasetConfig(
    name="pos_t_neg_t-20",
    time_step=[0,
     20, 21, 22, 23, 24, 
     25, 26, 27, 28, 29, 
     30, 31, 32, 33, 34, 
     35, 36, 37, 38, 39, 40],
    label=[1,
     0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 0])
```
means:

Name of model: ```pos_t_neg_t-20```

Time steps used: ```t-0``` and ```t-20 -> t-40```

Label of those time steps: ```t-0``` is ```1/True``` where the rest are ```0/False```

# Pipeline: Preprocess data, Train model, and Evaluate model
## Quick guide
At the root folder ```truongchu```, execute this to run everything
```
./run.sh preprocess train eval
```

## Preprocess data
At the root folder ```truongchu```, execute this
```
./run.sh preprocess
```

After running this path, two files of type ```.npy``` and ```.csv``` will appear. Those files would have the names that you specified in the ```config.json``` file above.

The ```.npy``` file would have the shape of ```(sample_size, 131, 17, 17)```.

The ```.csv``` file would contain ```sample_size``` rows and 4 columns: ```file_location, label, year, domain```.

For example, the filename in the config file is ```normalized_after_aggregated_euclid_5_time_step_POSITIVE_1_40_full_1999_2021```. Thus, two files of ```normalized_after_aggregated_euclid_5_time_step_POSITIVE_1_40_full_1999_2021.npy``` and ```normalized_after_aggregated_euclid_5_time_step_POSITIVE_1_40_full_1999_2021.csv``` will be created inside the ```data_total``` folder.

## Train model
At the root folder ```truongchu```, execute this
```
./run.sh train
```

After running preprocess pipeline, you will run this command to train the model.

The timesteps and labels used for training are specified in the ```train/loader/time_step.py``` file.

The model would be saved after each epoch, and the final models are saved in ```model_save/model``` folder of type ```.pt```. The description of the training process (epoch, loss, accuracy) would be saved on a ```json``` file inside the ```model_save/others``` folder with the corresponding name as the model file type ```.pt```.

In addition, after finish training, the file ```train/model_file.txt``` will contains all the models are trained.

For example, the model name is saved as ```model_save/model/first_model.pt```. Thus, the description file would be saved as ```model_save/others/first_model.json```, same name ```first_model``` as the model name.

## Evaluate model
At the root folder ```truongchu```, execute this
```
./run.sh eval
```

After training, all models appearing on the file ```model_file.txt``` in ```train``` folder will be evaluated.

Inside folder ```model_save/others```, another folder with the same name as the model will be created, which contains all the evaluation files.

For example, the model name is saved as ```model_save/model/first_model.pt```. Thus, the evaluation folder would be ```model_save/others/first_model```, same name ```first_model``` as the model name.

You could evaluate models using your command directly by specifying ```-data /data/path -eval /eval/folder/path -model /model/path```. 

For example, the model name is saved as ```model_save/model/first_model.pt```, and the data name is saved as ```data_total/first_data.npy```. You also want to save the evaluation in the folder ```./eval```. Thus, you will type: ```./eval.sh -data ./data_total/first_data.npy -model ./model_save/model/first_model.pt -eval ./eval```. Remember to use relative path before usage.

Inside each evaluation folder, there will be an output file which contains precision, recall, f1 score of the model for the corresponding data. Also, there will be a summary of score for each threshold and timestep, if you specify it on the config file (see above). Finally, there will be pictures of plotting also.
