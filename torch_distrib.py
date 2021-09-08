#!/usr/bin/env python3
# coding: utf-8

import os
import socket

import hostlist

# Get SLURM variables
rank = int(os.environ['SLURM_PROCID'])
local_rank = int(os.environ['SLURM_LOCALID'])
world_size = int(os.environ['SLURM_NTASKS'])
cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])

# Get node list from SLURM
hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])

# Get IDs of reserved GPUs
gpu_ids = os.environ['SLURM_STEP_GPUS'].split(',')

# Define MASTER_ADDR & MASTER_POST
os.environ['MASTER_ADDR'] = hostnames[0]

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind((hostnames[0], 0))
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    os.environ['MASTER_PORT'] = sock.getsockname()[1]
