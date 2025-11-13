import os
import logging
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict

def find_files_with_substring(directory, substring):
    matches = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if substring in file:
                matches.append(os.path.join(root, file))
    return matches

def load_tensorboard_logs(path):
    data = defaultdict(list)
    
    # If path is a file, get its parent directory
    if os.path.isfile(path):
        path = os.path.dirname(path)
    elif not os.path.isdir(path):
        # Try to find the parent directory if path doesn't exist
        path = os.path.dirname(path)
    
    # Try to find the summaries directory if it exists
    summaries_path = os.path.join(path, 'summaries')
    if os.path.isdir(summaries_path):
        path = summaries_path
    
    # Suppress tensorboard warnings by setting logger levels
    # Tensorboard uses multiple logger names, suppress them all
    tensorboard_logger_names = [
        'tensorboard',
        'tensorboard.backend',
        'tensorboard.backend.event_processing',
    ]
    tensorboard_loggers = [logging.getLogger(name) for name in tensorboard_logger_names]
    original_levels = []
    for logger in tensorboard_loggers:
        original_levels.append(logger.level)
        logger.setLevel(logging.ERROR)
        # Also disable propagation to root logger
        logger.propagate = False
    
    # Also suppress INFO level messages
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            event_acc = EventAccumulator(path)
            event_acc.Reload()  # Load all data written so far

            for tag in event_acc.Tags()["scalars"]:
                events = event_acc.Scalars(tag)
                for event in events:
                    data[tag].append(event.value)
        except Exception as e:
            # If loading fails, return empty dict (don't log to avoid noise)
            pass
        finally:
            # Restore original logging levels
            for logger, original_level in zip(tensorboard_loggers, original_levels):
                logger.setLevel(original_level)
                logger.propagate = True
    
    return data

import importlib.util

def import_class_from_file(file_path, function_name):
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    function = getattr(module, function_name)
    return function