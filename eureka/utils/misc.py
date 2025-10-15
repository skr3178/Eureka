import subprocess
import os
import json
import logging
import torch
import gc
import time

from utils.extract_task_code import file_to_string

def clear_gpu_memory():
    """Clear GPU memory cache to prevent out of memory errors"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logging.info("GPU memory cache cleared")

def set_freest_gpu():
    freest_gpu = get_freest_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(freest_gpu)
    logging.info(f"Using GPU {freest_gpu}")

def get_freest_gpu():
    sp = subprocess.Popen(['gpustat', '--json'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str, _ = sp.communicate()
    gpustats = json.loads(out_str.decode('utf-8'))
    # Find GPU with most free memory
    freest_gpu = min(gpustats['gpus'], key=lambda x: x['memory.used'])

    return freest_gpu['index']

def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found

def block_until_training(rl_filepath, log_status=False, iter_num=-1, response_id=-1):
    """Wait for training to start and then wait for completion"""
    # First wait for training to start
    while True:
        rl_log = file_to_string(rl_filepath)
        if "fps step:" in rl_log or "Traceback" in rl_log:
            if log_status and "fps step:" in rl_log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} successfully training!")
            if log_status and "Traceback" in rl_log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
            break
    
    # If there was an error, we're done
    if "Traceback" in rl_log:
        return
    
    # Now wait for training to complete
    # Look for completion indicators in the log
    max_wait_time = 3600  # 1 hour timeout
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        rl_log = file_to_string(rl_filepath)
        
        # Check for completion indicators
        if any(indicator in rl_log for indicator in [
            "Training completed",
            "Final evaluation",
            "Best reward",
            "Tensorboard Directory:",
            "wandb: Run finished"
        ]):
            if log_status:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} completed successfully!")
            break
            
        # Check if process is still running by looking for recent activity
        if "fps step:" in rl_log:
            # Get the last fps step line to check if it's recent
            lines = rl_log.split('\n')
            last_fps_line = None
            for line in reversed(lines):
                if "fps step:" in line:
                    last_fps_line = line
                    break
            
            if last_fps_line:
                # If we haven't seen a new fps step in the last 30 seconds, assume completion
                # This is a heuristic - you might need to adjust based on your training patterns
                pass
        
        time.sleep(5)  # Check every 5 seconds
    else:
        if log_status:
            logging.warning(f"Iteration {iter_num}: Code Run {response_id} timed out after {max_wait_time} seconds!")

if __name__ == "__main__":
    print(get_freest_gpu())