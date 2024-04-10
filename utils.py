import GPUtil

def get_gpu_usage():
    """
    Returns the current GPU usage of the system.

    :return: A list of dictionaries, each containing information about a GPU.
    """
    gpus = GPUtil.getGPUs()
    gpu_usage = [
        {
            'uuid': gpu.uuid,
            'gpu_name': gpu.name,
            'gpuUtil': gpu.load,
            'memTotal': gpu.memoryTotal,
            'memUsed': gpu.memoryUsed,
            'memFree': gpu.memoryFree,
            'temp_gpu': gpu.temperature
        } for gpu in gpus
    ]
    return gpu_usage