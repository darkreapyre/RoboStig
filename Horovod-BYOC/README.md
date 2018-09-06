## Sample Hyperparameters
```python
hyperparameters = {'learning_rate': .0001,
                   'epochs': 20,
                   'batch_size': 32                  
                   }
```

## Original MPI Command
```python
mpi_command = 'mpirun --allow-run-as-root --host {}'.format(",".join(host_list)) \
                  + " -bind-to none" \
                  + " -map-by slot" \
                  + " -mca btl_tcp_if_include {}".format(resources.get('network_interface_name')) \
                  + " -mca oob_tcp_if_include {}".format(resources.get('network_interface_name')) \
                  + " -mca pml ob1" \
                  + " -mca btl ^openib" \
                  + " -x PATH" \
                  + " -x LD_LIBRARY_PATH" \
                  + " -x LD_PRELOAD={}".format(_CHANGE_HOSTNAME_LIBRARY) \
                  + " -mca orte_abort_on_non_zero_status 1" \
                  + " -x NCCL_DEBUG=INFO" \
                  + " -x NCCL_SOCKET_IFNAME={}".format(resources.get('network_interface_name')) \
                  + " -np {} ".format(num_processes)
```

## Default settings `train.py`
- augment-data = False
- Use `get_data()` to pull data from S3
