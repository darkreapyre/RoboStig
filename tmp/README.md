### Sample Hyperparameters
```python
hyperparameters = {'learning_rate': .0001,
                   'epochs': 20,
                   'batch_size': 32                  
                   }
```

### Original MPI Command
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
- Issues with `btl_tcp_if_include` and `oob_tcp_if_include` and `[algo-1]` doesn't change it's hostname. `mpirun` can't do a local connect and sees `[algo-1]` as `aws`.
- Intermitent issues with `orte_abort_on_non_zero_status` --> removed.

## Default settings `train.py`
- augment-data = False --> CPU runs 100% with no status on output for over 2 hours. Need to verify if this is `hvd.init()` on CPU OR `keras.fit_generator()` which is CPU bound OR maybye give it more then 2 hours.
- Use `get_data()` to pull data from S3
