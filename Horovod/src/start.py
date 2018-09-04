# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

# Taken from https://github.com/aws/sagemaker-chainer-container/blob/master/src/sagemaker_chainer_container/training.py

from __future__ import absolute_import

import logging
import os
import shlex
import socket
import stat
import subprocess
import sys
import textwrap
import time
import timeout
from retrying import retry

# Configure the trainer environemnt from BYOC `environment.py`
from environment import create_trainer_environment

"""
Note: Confirm if the logging code below is needed
"""
#logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
#                    level=logging.INFO)

#logging.getLogger('boto3').setLevel(logging.INFO)
#logging.getLogger('s3transfer').setLevel(logging.INFO)
#logging.getLogger('botocore').setLevel(logging.WARN)

#logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)

# Global variables required to execute `mpirun` and to change the container hostname
_MPI_SCRIPT = "/mpi_script.sh"
_MPI_IS_RUNNING = "/mpi_is_running"
_MPI_IS_FINISHED = "/mpi_is_finished"
_CHANGE_HOSTNAME_LIBRARY = "/libchangehostname.so"

# Helper Functions
def _decode(obj):  # type: (bytes or str or unicode or object) -> unicode
    """
    Decode an object to unicode.
    
    Args:
        obj (bytes or str or unicode or anything serializable): object to be decoded
    Returns:
        object decoded in unicode.
    """
    if obj is None:
        return u''
    if six.PY3 and isinstance(obj, six.binary_type):
        # transforms a byte string (b'') in unicode
        return obj.decode('latin1')
    elif six.PY3:
        # PY3 strings are unicode.
        return str(obj)
    elif isinstance(obj, six.text_type):
        # returns itself if it is unicode
        return obj
    else:
        # decodes pY2 string to unicode
        return str(obj).decode('utf-8')

def to_cmd_args(mapping):  # type: (dict) -> list
    """
    Transform a dictionary in a list of cmd arguments.
    Example:
        >>>args = mapping.to_cmd_args({'model_dir': '/opt/ml/model', 'batch_size': 25})
        >>>
        >>>print(args)
        ['--model_dir', '/opt/ml/model', '--batch_size', 25]
    
    Args:
        mapping (dict[str, object]): A Python mapping.
    Returns:
        (list): List of cmd arguments
    """

    sorted_keys = sorted(mapping.keys())

    def arg_name(obj):
        string = _decode(obj)
        if string:
            return u'--%s' % string if len(string) > 1 else u'-%s' % string
        else:
            return u''

    arg_names = [arg_name(argument) for argument in sorted_keys]

    def arg_value(value):
        if hasattr(value, 'items'):
            map_items = ['%s=%s' % (k, v) for k, v in sorted(value.items())]
            return ','.join(map_items)
        return _decode(value)

    arg_values = [arg_value(mapping[key]) for key in sorted_keys]

    items = zip(arg_names, arg_values)

    return [item for item in itertools.chain.from_iterable(items)]

# Main MPI training functions
def train(env, hyperparameters):
    """
    Runs Horovod training on a user supplied module in either a local or distributed
    SageMaker environment.
    The user supplied module and its dependencies are downloaded from S3.
    Training is invoked by calling a "train" function in the user supplied module.
    If the environment contains multiple hosts, then a distributed learning
    task is started with mpirun.
    The following is a list of other hyperparameters that can be used to change training behavior.
    * `sagemaker_use_mpi`: [REQUIRED for Horovod --> Default]
    * `sagemaker_process_slots_per_host`: the number of GPUs per host. [NOT REQUIRED since automatically calculated]
    * `sagemaker_num_processes`: the total number of processes to run. [NOT REQUIRED unless for oversubscription]
    * `sagemaker_additional_mpi_options`: a string of options to pass to mpirun. [NOT USED]
    For more on how distributed training uses these parameters, please see :func:`_get_mpi_command`.
    """
    # `sagemaker_use_mpi` by default
    current_host = env.current_host
    hosts = list(env.hosts)

    # change the container hostname to training hostname
    _change_hostname(current_host)

    # Start the SSH Daemon for workers to communicate
    _start_ssh_daemon()

    # Generate MPI script to run the training
    _create_mpi_script(env)

    # launch master script if master host
    if current_host == _get_master_host_name(hosts):
        # test connectivity to workers
        _wait_for_worker_nodes_to_start_sshd(hosts)

        # execute training mpirun
        _run_mpi_on_all_nodes(env, hyperparameters)
    else:
        _wait_for_training_to_finish(env)

def _change_hostname(current_host):
    """
    Compiles a shared library to correct the behavior of the gethostname system call,
    which OpenMPI depends on.
    
    Args:
        current_host (str): name of the current host, such as algo-1, algo-2, etc.
    """
    os.system("change-hostname.sh {}".format(current_host))

def _start_ssh_daemon():
    subprocess.Popen(["/usr/sbin/sshd", "-D"])

def _create_mpi_script(env):
    """
    Creates a MPI script with user provided information.
    For distributed training: the 'master node' runs mpirun with this script,
    '/mpi_script.sh'. 
    
    This script creates a file '/mpi_is_running' that worker nodes use to
    determine whether training # (started by MPI from the master node) is still running.
    
    Processes on worker nodes use # /mpi_is_finished file to determine when to exit.
    
    Args:
        env (TrainingEnv): an instance of the training environment.
    """
    # return list of cmd args
    hyperparameters = to_cmd_args(hyperparameters)
    channels = to_cmd_args(env.channel_dirs)
    output = to_env_vars(env.output_data_dir)

    python_cmd = [sys.executable, 'train.py']
    python_cmd.extend(hyperparameters)
    python_cmd.extend(channels)
    python_cmd.extend(output)

    content = textwrap.dedent("""#!/usr/bin/env bash
touch /mpi_is_running
%s
EXIT_CODE=$?
touch /mpi_is_finished
exit ${EXIT_CODE}
""" % ' '.join(python_cmd))

    # build MPI script
    with open(_MPI_SCRIPT, 'w') as w:
        w.write(content)
    
    # change permissions on script
    st = os.stat(_MPI_SCRIPT)
    os.chmod(_MPI_SCRIPT, st.st_mode | stat.S_IEXEC)

def _get_master_host_name(hosts):
    return sorted(hosts)[0]

def _can_connect(host, port, s):
    try:
        #logger.debug("testing connection to host %s", host)
        print("testing connection to host %s" % host)
        s.connect((host, port))
        s.close()
        #logger.debug("can connect to host %s", host)
        print("can connect to host %s" % host)
        return True
    except socket.error:
        #logger.debug("can't connect to host %s", host)
        print("can't connect to host %s" % host)
    return False

def _wait_for_worker_nodes_to_start_sshd(hosts, interval=1, timeout_in_seconds=180):
    with timeout(seconds=timeout_in_seconds):
        while hosts:
            #logger.info("hosts that aren't SSHable yet: %s", str(hosts))
            print("hosts that aren't SSHable yet: %s" % str(hosts))
            for host in hosts:
                ssh_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                if _can_connect(host, 22, ssh_socket):
                    hosts.remove(host)
            time.sleep(interval)

def _get_mpi_command(env, hyperparameters):
    """Constructs a command to run distributed training with MPI using mpirun.
    Runs /mpi_script.sh on all hosts listed in the training environment. How many
    processes in total is determined by the 'sagemaker_num_processes' hyperparameter, or one
    per GPU, or one per CPU, as applicable. The 'sagemaker_process_slots_per_host'
    hyperparameter can be used to override how many processes can be placed on each host.
    Additional MPI options can be passed (and override other MPI options) using the
    'sagemaker_additional_mpi_options' hyperparameter.
    This command passes many options to the mpirun command:
    * --host [host:slots]: A comma-delimited list of hosts and the number of process
        slots on each host.
    * -mca btl_tcp_if_include [env.network_interface_name]: Tell OpenMPI to use
        the given network interface name for byte transfer layer communication.
    * -mca oob_tcp_if_include [env.network_interface_name]: Tell OpenMPI to use
        the given network interface name for out-of-band communication.
    * -mca btl ^openib: Don't look for openib components (this just avoids a warning)
    * -x PATH: pass $PATH from the current environment to the execution environments on remote hosts
    * -x LD_LIBRARY_PATH: pass $LD_LIBRARY_PATH from the current environment to the execution
        environments on remote hosts
    * -x LD_PRELOAD=[changehostname library]: Load the changehostname library to return
        correct values from gethostname system calls.
    * -mca orte_abort_on_non_zero_status 1: Return a non-zero exit code if any process exits
        with a non-zero exit code.
    * -x NCCL_DEBUG=INFO: Enable info level logging for NCCL.
    * -x NCCL_SOCKET_IFNAME=[env.network_interface_name]: Tell NCCL to use the given
        network interface name for socket communication.
    * -np [num_processes]: total number of processes to run across all nodes.
    Args:
        env: training environment object containing environment variables,
                              training arguments and hyperparameters.
    Returns:
        str: The mpirun command to run.
    """
    is_gpu = env.available_gpus if env.available_gpus > 0 else 1

    process_slots_per_host = int(hyperparameters.get('sagemaker_process_slots_per_host', is_gpu))

    num_hosts = len(env.hosts)
    num_processes = process_slots_per_host * num_hosts
    num_processes = int(hyperparameters.get('sagemaker_num_processes', num_processes))

    # By default, use one process per GPU, or one process per node (if training with CPU).
    host_list = env.hosts if process_slots_per_host == 1 else \
        [host + ':{}'.format(process_slots_per_host) for host in env.hosts]

    additional_mpi_options = str(hyperparameters.get('sagemaker_additional_mpi_options', ''))

    #credential_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN']

    #logger.info('network interface name: %s', env.network_interface_name)
    print('network interface name: %s' % env.network_interface_name)

    mpi_command = 'mpirun --allow-run-as-root --host {}'.format(",".join(host_list)) \
                  + " -bind-to none" \
                  + " -map-by slot" \
                  + " -mca btl_tcp_if_include {}".format(env.network_interface_name) \
                  + " -mca oob_tcp_if_include {}".format(env.network_interface_name) \
                  + " -mca pml ob1" \
                  + " -mca btl ^openib" \
                  + " -x PATH" \
                  + " -x LD_LIBRARY_PATH" \
                  + " -x LD_PRELOAD={}".format(_CHANGE_HOSTNAME_LIBRARY) \
                  + " -mca orte_abort_on_non_zero_status 1" \
                  + " -x NCCL_DEBUG=INFO" \
                  + " -x NCCL_SOCKET_IFNAME={}".format(env.network_interface_name) \
                  + " -np {} ".format(num_processes)

    """
    Note: Confirm if AWS Credentials are needed
    """
    #for v in credential_vars:
    #    if v in os.environ:
    #        mpi_command += " -x {}".format(v)
    
    """
    Note: It may not be necessary to include SageMaker environment for `mpirun`,
    since the hyperparameters are passes to the training funciton in _MPI_SCRIPT
    and NOT using `mpi4py`
    """
    #for name, value in env.to_env_vars().items():
    #    mpi_command += ' -x {}="{}"'.format(name, value)

    mpi_command += " {} ".format(additional_mpi_options) + " {}".format(_MPI_SCRIPT)
    return mpi_command


def _run_mpi_on_all_nodes(env, hyperparameters):
    mpi_command = _get_mpi_command(env, hyperparameters)
    cmd = shlex.split(mpi_command)

    #framework.logging.log_script_invocation(cmd, env.to_env_vars(), logger)

    with open(_MPI_SCRIPT) as f:
        print('Running MPI script:\n\n%s', f.read())
#        logger.info('Running MPI script:\n\n%s' % f.read())
    
    subprocess.check_call(cmd)

def _retry_if_false(result):
    return result is False

@retry(stop_max_delay=30 * 1000, wait_fixed=1000, retry_on_result=_retry_if_false)
def _wait_for_mpi_to_start_running():
    return os.path.isfile(_MPI_IS_RUNNING)

@retry(wait_fixed=5000, retry_on_result=_retry_if_false)
def _wait_until_mpi_stops_running():
    return os.path.isfile(_MPI_IS_FINISHED)

def _wait_for_training_to_finish(env):
    current_host = env.current_host

    #logger.info("Worker node %s is waiting for MPI to start training process", current_host)
    print("Worker node %s is waiting for MPI to start training process" % current_host)
    _wait_for_mpi_to_start_running()

    #logger.info("MPI started training process on worker node %s", current_host)
    print("MPI started training process on worker node %s" % current_host)

    _wait_until_mpi_stops_running()
    
    #logger.info("Training process started by MPI on worker node %s stopped", current_host)
    print("Training process started by MPI on worker node %s stopped" % current_host)

def main():
    # Configure taining environment
    env = create_trainer_environment()
    print("Creating SageMaker trainer environment:\n%s" % str(env))
    
    # Get Hyperparameters
    hyperparameters = env.hyperparameters

    # Start MPI training environment
    train(env, hyperparameters)

# This branch hit by mpi_script.sh
if __name__ == '__main__':
    main()