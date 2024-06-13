import os, json
import matplotlib.pyplot as plt
import random
import pandas
import pickle, csv

import logging
import sys
from tqdm import tqdm
from mpi4py import MPI
from itertools import chain
import argparse
import time
import math

import hydragnn
from hydragnn.preprocess.load_data import split_dataset
from hydragnn.utils.print_utils import print_distributed, iterate_tqdm
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.pickledataset import SimplePickleWriter, SimplePickleDataset
from hydragnn.preprocess.utils import gather_deg
from hydragnn.utils.model import print_model
from hydragnn.utils.smiles_utils import (
    get_node_attribute_name,
    generate_graphdata_from_smilestr,
)
from hydragnn.utils import nsplit

import numpy as np

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

import torch_geometric.data
import torch
import torch.distributed as dist

# import warnings

# warnings.filterwarnings("error")

ogb_node_types = {
    "H": 0,
    "B": 1,
    "C": 2,
    "N": 3,
    "O": 4,
    "F": 5,
    "Si": 6,
    "P": 7,
    "S": 8,
    "Cl": 9,
    "Ca": 10,
    "Ge": 11,
    "As": 12,
    "Se": 13,
    "Br": 14,
    "I": 15,
    "Mg": 16,
    "Ti": 17,
    "Ga": 18,
    "Zn": 19,
    "Ar": 20,
    "Be": 21,
    "He": 22,
    "Al": 23,
    "Kr": 24,
    "V": 25,
    "Na": 26,
    "Li": 27,
    "Cu": 28,
    "Ne": 29,
    "Ni": 30,
}


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


from hydragnn.utils.abstractbasedataset import AbstractBaseDataset


def smiles_to_graph(datadir, files_list):

    subset = []

    for filename in files_list:

        df = pandas.read_csv(os.path.join(datadir, filename))
        rx = list(nsplit(range(len(df)), comm_size))[rank]

        for smile_id in tqdm(range(len(df))[rx.start : rx.stop]):
            ## get atomic positions and numbers
            dfrow = df.iloc[smile_id]

            if math.isnan(float(dfrow[-1])):
                continue

            smilestr = dfrow[0]
            ytarget = (
                torch.tensor(float(dfrow[-1]))
                .unsqueeze(0)
                .unsqueeze(1)
                .to(torch.float32)
            )  # HL gap

            try:
                data = generate_graphdata_from_smilestr(
                    smilestr,
                    ytarget,
                    ogb_node_types,
                    var_config,
                )

                subset.append(data)
            except KeyError:
                print("KeyError: ", smilestr)
                continue

    return subset

'''
# Example usage
num_layers = 18
num_stages = 4
result = layers_assignment(num_layers, num_stages)
print(result)  # Output: [4, 5, 5, 4]
'''
def layers_assignment(num_layers, num_stages):
    # Basic division
    base_layers_per_stage = num_layers // num_stages
    remainder_layers = num_layers % num_stages
    
    # Create a list with base layers for each stage
    layers_per_stage = [base_layers_per_stage] * num_stages
    
    # Distribute the remainder layers, prioritizing the center stages
    start_idx = (num_stages - remainder_layers) // 2
    for i in range(remainder_layers):
        layers_per_stage[start_idx + i] += 1
    
    return layers_per_stage

def workload_to_idx(workload):
    start_idx = []
    end_idx = []
    
    current_start = 0
    for work in workload:
        start_idx.append(current_start)
        current_start += work
        end_idx.append(current_start)
    
    return start_idx, end_idx

# A naive implementation of spliting the multiple layers within the same model into different stages on different gpus
from torch_geometric.nn import global_mean_pool
def model_parallel(layers, num_stages=4):
    # The conversion process is highly coupled with the Base model defination
    assert num_stages==4, "Hardcoded 4 stages (4 GPUs) for now"
    ## split graph_convs and feature_layers
    assert len(layers.graph_convs) == len(layers.feature_layers), "Number of blocks in graph_convs and feature_layers should be equal"
    stages_workload = layers_assignment(len(layers.graph_convs), num_stages)
    start_layer_idx, end_layer_idx = workload_to_idx(stages_workload)
    ### Assign graph_convs and feature_layers to different GPUs
    for i in range(num_stages):
        layers.graph_convs[start_layer_idx[i]:end_layer_idx[i]].to(f"cuda:{i}")
        layers.feature_layers[start_layer_idx[i]:end_layer_idx[i]].to(f"cuda:{i}")
    
    ### Assign decoder part to the last GPU
    layers.heads_NN.to(f"cuda:{num_stages-1}")
    layers.graph_shared.to(f"cuda:{num_stages-1}")

    ## Replace forward function
    def model_parallel_forward(self, data_queue, stage_inputs, stage_outputs):
        x = data_queue[-1].x
        pos = data_queue[-1].pos
        ### encoder part ####
        conv_args = self._conv_args(data_queue[-1])
        stage_inputs[0]["x"] = x
        stage_inputs[0]["pos"] = pos
        stage_inputs[0]["conv_args"] = conv_args

        layer_idx = 0
        gpu_idx = 0
        for conv, feat_layer in zip(self.graph_convs, self.feature_layers):
            if (layer_idx in start_layer_idx):
                # Update inputs for this gpu
                x = stage_inputs[gpu_idx]["x"]
                pos = stage_inputs[gpu_idx]["pos"]
                conv_args = stage_inputs[gpu_idx]["conv_args"]

                # Put input in the correct GPUs
                if x is not None:
                    x = x.to(f"cuda:{gpu_idx}")
                if pos is not None:
                    pos = pos.to(f"cuda:{gpu_idx}")
                if conv_args is not None:
                    conv_args["edge_index"] = conv_args["edge_index"].to(f"cuda:{gpu_idx}")
                    if self.use_edge_attr:
                        conv_args["edge_attr"] = conv_args["edge_attr"].to(f"cuda:{gpu_idx}")
            
            if x is not None:
                c, pos = conv(x=x, pos=pos, **conv_args)
                x = self.activation_function(feat_layer(c))

            if ((layer_idx+1) in end_layer_idx):
                # Update outpurs for this gpu (i.e., input for the next gpu in the next iter)
                stage_outputs[gpu_idx]["x"] = x
                stage_outputs[gpu_idx]["pos"] = pos
                stage_outputs[gpu_idx]["conv_args"] = conv_args
                gpu_idx += 1

            layer_idx += 1
        
        # Update stage_inputs based on stage_outputs:
        for gpu_idx in range(num_stages-1):
            stage_inputs[gpu_idx+1]["x"] = stage_outputs[gpu_idx]["x"]
            stage_inputs[gpu_idx+1]["pos"] = stage_outputs[gpu_idx]["pos"]
            stage_inputs[gpu_idx+1]["conv_args"] = stage_outputs[gpu_idx]["conv_args"]

        #### multi-head decoder part####
        # shared dense layers for graph level output
        if stage_outputs[num_stages-1]["x"] is not None:
            x = stage_outputs[num_stages-1]["x"]
            if data_queue[0].batch is None:
                x_graph = x.mean(dim=0, keepdim=True)
            else:
                x_graph = global_mean_pool(x, data_queue[0].batch.to(x.device))
            outputs = []
            for head_dim, headloc, type_head in zip(
                self.head_dims, self.heads_NN, self.head_type
            ):
                if type_head == "graph":
                    x_graph_head = self.graph_shared(x_graph)
                    outputs.append(headloc(x_graph_head))
                else:
                    if self.node_NN_type == "conv":
                        for conv, batch_norm in zip(headloc[0::2], headloc[1::2]):
                            c, pos = conv(x=x, pos=pos, **conv_args)
                            c = batch_norm(c)
                            x = self.activation_function(c)
                        x_node = x
                    else:
                        x_node = headloc(x=x, batch=data_queue[0].batch)
                    outputs.append(x_node)
            return outputs
        else:
            return None

    layers.forward = model_parallel_forward.__get__(layers, layers.__class__)

    return layers
    

class OGBDataset(AbstractBaseDataset):
    """OGBDataset dataset class"""

    def __init__(self, dirpath, var_config, dist=False):
        super().__init__()

        self.var_config = var_config
        self.dist = dist
        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

        if os.path.isdir(dirpath):
            dirfiles = sorted(os.listdir(dirpath))
        else:
            raise ValueError("OGBDataset takes dirpath as directory")

        setids_files = [x for x in dirfiles if x.endswith("csv")]

        self.dataset.extend(smiles_to_graph(dirpath, setids_files))

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]


def ogb_datasets_load(datafile, sampling=None, seed=None):
    if seed is not None:
        random.seed(seed)
    trainset = []
    valset = []
    testset = []
    trainsmiles = []
    valsmiles = []
    testsmiles = []
    trainidxs = []
    validxs = []
    testidxs = []
    with open(datafile, "r") as file:
        csvreader = csv.reader(file)
        print(next(csvreader))
        for row in csvreader:
            if (sampling is not None) and (random.random() > sampling):
                continue
            if row[1] == "train":
                trainsmiles.append(row[0])
                trainset.append([float(row[-1])])
            elif row[1] == "val":
                valsmiles.append(row[0])
                valset.append([float(row[-1])])
            elif row[1] == "test":
                testsmiles.append(row[0])
                testset.append([float(row[-1])])
            else:
                print("unknown file name: ", row[0])
                sys.exit(0)
    return (
        [trainsmiles, valsmiles, testsmiles],
        [torch.tensor(trainset), torch.tensor(valset), torch.tensor(testset)],
    )


## Torch Dataset for CSCE CSV format
class OGBRawDatasetFactory:
    def __init__(self, datafile, var_config, sampling=1.0, seed=43, norm_yflag=False):
        self.var_config = var_config

        ## Read full data
        smiles_sets, values_sets = ogb_datasets_load(
            datafile, sampling=sampling, seed=seed
        )
        if norm_yflag:
            ymean = var_config["ymean"]
            ystd = var_config["ystd"]

        info([len(x) for x in values_sets])
        self.dataset_lists = list()
        for idataset, (smileset, valueset) in enumerate(zip(smiles_sets, values_sets)):
            if norm_yflag:
                valueset = (valueset - torch.tensor(ymean)) / torch.tensor(ystd)
            self.dataset_lists.append((smileset, valueset))

    def get(self, label):
        ## Set only assigned label data
        labelnames = ["trainset", "valset", "testset"]
        index = labelnames.index(label)

        smileset, valueset = self.dataset_lists[index]
        return (smileset, valueset)


class OGBRawDataset(torch.utils.data.Dataset):
    def __init__(self, datasetfactory, label):
        self.smileset, self.valueset = datasetfactory.get(label)
        self.var_config = datasetfactory.var_config

    def __len__(self):
        return len(self.smileset)

    def __getitem__(self, idx):
        smilestr = self.smileset[idx]
        ytarget = self.valueset[idx]
        data = generate_graphdata_from_smilestr(
            smilestr,
            ytarget,
            ogb_node_types,
            self.var_config,
        )
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfilesubstr", help="input file substr")
    parser.add_argument("--sampling", type=float, help="sampling ratio", default=None)
    parser.add_argument(
        "--preonly",
        action="store_true",
        help="preprocess only. Adios saving and no train",
    )
    parser.add_argument("--shmem", action="store_true", help="use shmem")
    parser.add_argument("--mae", action="store_true", help="do mae calculation")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--adios",
        help="Adios dataset",
        action="store_const",
        dest="format",
        const="adios",
    )
    group.add_argument(
        "--pickle",
        help="Pickle dataset",
        action="store_const",
        dest="format",
        const="pickle",
    )
    group.add_argument(
        "--csv", help="CSV dataset", action="store_const", dest="format", const="csv"
    )
    parser.add_argument(
        "--log_postfix", 
        type=str, 
        help="postfix for log", 
        default=""
    )
    parser.set_defaults(format="adios")
    args = parser.parse_args()

    graph_feature_names = ["GAP"]
    graph_feature_dim = [1]
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset/")
    ##################################################################################################################
    inputfilesubstr = args.inputfilesubstr
    input_filename = os.path.join(dirpwd, "ogb_" + inputfilesubstr + ".json")
    ##################################################################################################################
    # Configurable run choices (JSON file that accompanies this example script).
    with open(input_filename, "r") as f:
        config = json.load(f)
    verbosity = config["Verbosity"]["level"]
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    var_config["output_names"] = [
        graph_feature_names[item]
        for ihead, item in enumerate(var_config["output_index"])
    ]
    var_config["graph_feature_names"] = graph_feature_names
    var_config["graph_feature_dims"] = graph_feature_dim
    (
        var_config["input_node_feature_names"],
        var_config["input_node_feature_dims"],
    ) = get_node_attribute_name(ogb_node_types)
    var_config["node_feature_dims"] = var_config["input_node_feature_dims"]
    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.setup_ddp()
    ##################################################################################################################

    comm = MPI.COMM_WORLD

    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

    log_name = "ogb_" + inputfilesubstr + args.log_postfix
    hydragnn.utils.setup_log(log_name)
    writer = hydragnn.utils.get_summary_writer(log_name)
    hydragnn.utils.save_config(config, log_name)

    modelname = "ogb_" + inputfilesubstr
    if args.preonly:
        norm_yflag = False  # True

        ## local data
        total = OGBDataset(
            os.path.join(datadir),
            var_config,
            dist=True,
        )
        ## This is a local split
        trainset, valset, testset = split_dataset(
            dataset=total,
            perc_train=0.9,
            stratify_splitting=False,
        )
        print("Local splitting: ", len(total), len(trainset), len(valset), len(testset))

        deg = gather_deg(trainset)
        config["pna_deg"] = deg

        setnames = ["trainset", "valset", "testset"]

        ## local data
        if args.format == "pickle":

            ## pickle
            basedir = os.path.join(
                os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
            )
            attrs = dict()
            attrs["pna_deg"] = deg
            SimplePickleWriter(
                trainset,
                basedir,
                "trainset",
                # minmax_node_feature=total.minmax_node_feature,
                # minmax_graph_feature=total.minmax_graph_feature,
                use_subdir=True,
                attrs=attrs,
            )
            SimplePickleWriter(
                valset,
                basedir,
                "valset",
                # minmax_node_feature=total.minmax_node_feature,
                # minmax_graph_feature=total.minmax_graph_feature,
                use_subdir=True,
            )
            SimplePickleWriter(
                testset,
                basedir,
                "testset",
                # minmax_node_feature=total.minmax_node_feature,
                # minmax_graph_feature=total.minmax_graph_feature,
                use_subdir=True,
            )

        if args.format == "adios":
            fname = os.path.join(os.path.dirname(__file__), "dataset", "ogb_gap.bp")
            adwriter = AdiosWriter(fname, comm)
            adwriter.add("trainset", trainset)
            adwriter.add("valset", valset)
            adwriter.add("testset", testset)
            adwriter.add_global("pna_deg", deg)
            adwriter.save()

        sys.exit(0)

    timer = Timer("load_data")
    timer.start()
    if args.format == "adios":
        opt = {"preload": True, "shmem": False}
        if args.shmem:
            opt = {"preload": False, "shmem": True}
        fname = os.path.join(os.path.dirname(__file__), "dataset", "ogb_gap.bp")
        trainset = AdiosDataset(fname, "trainset", comm, **opt)
        valset = AdiosDataset(fname, "valset", comm, **opt)
        testset = AdiosDataset(fname, "testset", comm, **opt)
    elif args.format == "csv":
        fname = os.path.join(os.path.dirname(__file__), "dataset", "pcqm4m_gap.csv")
        fact = OGBRawDatasetFactory(
            fname, var_config=var_config, sampling=args.sampling
        )
        trainset = OGBRawDataset(fact, "trainset")
        valset = OGBRawDataset(fact, "valset")
        testset = OGBRawDataset(fact, "testset")
    elif args.format == "pickle":
        info("Pickle load")
        basedir = os.path.join(
            os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
        )
        trainset = SimplePickleDataset(
            basedir=basedir, label="trainset", var_config=var_config
        )
        valset = SimplePickleDataset(
            basedir=basedir, label="valset", var_config=var_config
        )
        testset = SimplePickleDataset(
            basedir=basedir, label="testset", var_config=var_config
        )
        pna_deg = trainset.pna_deg
    else:
        raise NotImplementedError("No supported format: %s" % (args.format))

    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
    timer.stop()

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    # Use model parallel instead of the original data parallel
    # model = hydragnn.utils.get_distributed_model(model, verbosity)
    model = model_parallel(layers=model, num_stages=torch.cuda.device_count())

    if rank == 0:
        print_model(model)
    dist.barrier()

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    hydragnn.utils.load_existing_model_config(
        model, config["NeuralNetwork"]["Training"], optimizer=optimizer
    )

    ##################################################################################################################

    hydragnn.train.train_validate_test_model_parallel(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config["NeuralNetwork"],
        log_name,
        verbosity,
        create_plots=False,
    )

    hydragnn.utils.save_model(model, optimizer, log_name)
    hydragnn.utils.print_timers(verbosity)

    peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
    print_distributed(3, "Peak memory usage: {:.4f} GB".format(peak_mem))
    if writer is not None:
        writer.add_scalar("Peak memory usage", peak_mem)

    if args.mae:
        ##################################################################################################################
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        for isub, (loader, setname) in enumerate(
            zip([train_loader, val_loader, test_loader], ["train", "val", "test"])
        ):
            error, rmse_task, true_values, predicted_values = hydragnn.train.test_model_parallel(
                loader, model, verbosity
            )
            ihead = 0
            head_true = np.asarray(true_values[ihead].cpu()).squeeze()
            head_pred = np.asarray(predicted_values[ihead].cpu()).squeeze()
            ifeat = var_config["output_index"][ihead]
            outtype = var_config["type"][ihead]
            varname = graph_feature_names[ifeat]

            ax = axs[isub]
            error_mae = np.mean(np.abs(head_pred - head_true))
            error_rmse = np.sqrt(np.mean(np.abs(head_pred - head_true) ** 2))
            # print(varname, ": ev, mae=", error_mae, ", rmse= ", error_rmse)
            print_distributed(
                1, setname, varname, " : ev, mae=", error_mae, ", rmse= ", error_rmse
            )

            if writer is not None:
                writer.add_scalar(setname + "/" + varname + "/MAE", error_mae)

            ax.scatter(
                head_true,
                head_pred,
                s=7,
                linewidth=0.5,
                edgecolor="b",
                facecolor="none",
            )
            minv = np.minimum(np.amin(head_pred), np.amin(head_true))
            maxv = np.maximum(np.amax(head_pred), np.amax(head_true))
            ax.plot([minv, maxv], [minv, maxv], "r--")
            ax.set_title(setname + "; " + varname + " (eV)", fontsize=16)
            ax.text(
                minv + 0.1 * (maxv - minv),
                maxv - 0.1 * (maxv - minv),
                "MAE: {:.2f}".format(error_mae),
            )
        if rank == 0:
            fig.savefig("./logs/" + log_name + "/" + varname + "_all.png")
        plt.close()

    if args.shmem:
        trainset.unlink()

    sys.exit(0)
