class DatasetConfig():
    def __init__(self, name, time_step, label):
        self.name = name
        self.time_step = time_step
        self.label = label

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

dataset_1 = DatasetConfig(
    name="pos_t-2_neg_t-20",
    time_step=[0, 1, 2,
     20, 21, 22, 23, 24, 
     25, 26, 27, 28, 29, 
     30, 31, 32, 33, 34, 
     35, 36, 37, 38, 39, 40],
    label=[1, 1, 1,
     0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 0])

dataset_2 = DatasetConfig(
    name="pos_t-4_neg_t-20",
    time_step=[0, 1, 2, 3, 4,
     20, 21, 22, 23, 24, 
     25, 26, 27, 28, 29, 
     30, 31, 32, 33, 34, 
     35, 36, 37, 38, 39, 40],
    label=[1, 1, 1, 1, 1,
     0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 0])

dataset_3 = DatasetConfig(
    name="pos_t-6_neg_t-20",
    time_step=[0, 1, 2, 3, 4, 5, 6,
     20, 21, 22, 23, 24, 
     25, 26, 27, 28, 29, 
     30, 31, 32, 33, 34, 
     35, 36, 37, 38, 39, 40],
    label=[1, 1, 1, 1, 1, 1, 1,
     0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 0])

dataset_4 = DatasetConfig(
    name="pos_t_neg_t-20",
    time_step=[0, 1, 2, 3, 4, 5, 6, 7, 8,
     20, 21, 22, 23, 24, 
     25, 26, 27, 28, 29, 
     30, 31, 32, 33, 34, 
     35, 36, 37, 38, 39, 40],
    label=[1, 1, 1, 1, 1, 1, 1, 1, 1, 
     0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 0])

dataset_json = [dataset_0, dataset_1, dataset_2, dataset_3, dataset_4]
