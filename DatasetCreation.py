from prepare_dataset import dataloaders
from get_args import args
import torch
import os

#Parameter
batch_size_train = 64
InputSize = 5
Downscale_batch_size = 100_000

args.selected_targets = (0, 1)
args.batch_size = batch_size_train

#Options
args.shuffle = True
args.downscale = True
args.flatten_dataset = True
args.dataset_spin_form = False
args.pad_flattened_dataset = True  # must flatten_dataset be True to pad the dataset
args.remove_contradicting = True

#Adaptive
args.use_adaptive = True

args.downscale_batch_size = Downscale_batch_size
args.adaptive_avg_pool_shape = InputSize  # 5-> (5x5), 6-> (6x6)
if args.use_adaptive and args.downscale:
    args.shape_2d = (args.adaptive_avg_pool_shape, args.adaptive_avg_pool_shape) 


train_dataloader, test_dataloader = dataloaders(args)

Datasize  = train_dataloader.dataset[0][0].shape
Folder = f"Dataset/{InputSize}x{InputSize}"
try:
    os.makedirs(Folder)
    print(f"Folder '{Folder}' created successfully!")
except FileExistsError:
    print(f"Folder '{Folder}' already exists!")

torch.save(train_dataloader, Folder+"/Train.txt")
torch.save(test_dataloader, Folder+"/Test.txt")


file = open(Folder+"/Info.txt", "w")
file.write(f"Batch Size : {batch_size_train} \n" )
file.write(f"Picture Size : {InputSize}x{InputSize} \n" )
file.write(f"Data Size : {Datasize} \n" )
file.write(f"Shuffle : {args.shuffle} \n" )
file.write(f"Remove Contracdicting : {args.remove_contradicting} \n" )
file.write(f"Adpative : {args.use_adaptive}")
file.close()

# num = 0
# for _ in train_dataloader.dataset:
#     # print(int(_[0][0].detach().item()))
#     if int(_[0][0].detach().item()) == 1:
#         num = num + 1
# print(num)