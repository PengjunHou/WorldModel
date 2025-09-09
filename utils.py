import configargparse
import importlib
import logging

LOG = logging.getLogger(__name__)
DEFAULT_LOG_LEVEL = logging.ERROR

PARSER = configargparse.ArgParser(default_config_files=['config/CarlaVehSensors.config'])

PARSER.add('-c', '--config_path', required = False, is_config_file = True, help = 'config file path')
PARSER.add('-v', '--v2x_data_path', required = False, is_config_file = True, help = 'V2X config file path')
PARSER.add('--num_vehicles', type = int, default = 5, help = 'number of vehicles')
PARSER.add('--env_name', required = True, help = 'environment name')
PARSER.add('--data_path', required = True, type = str, help = 'dataset path')
PARSER.add('--result_path', required = True, type = str, help = 'result path')
PARSER.add('--visualize', type = int, default = 0, help = 'visualize the environment')
PARSER.add('--seed', type = int, help = 'seed')
PARSER.add('--n_clusters', type = int, help = 'number of clusters')

PARSER.add('--Comm_Comp_model', required = True, type = str, default = 'Base', help = 'communication and computing model')

PARSER.add('--time_steps', required = True, type = int, help = 'number of time steps')
PARSER.add('--strategy', required = True, type = str, help = 'action selection strategy')
PARSER.add('--obs_dim', required = False, type = int, help = 'observation dimension')
PARSER.add('--action_dim', required = False, type = int, help = 'action dimension')
PARSER.add('--epoch', required = True, type = int, default = 100, help = "number of epoch")
PARSER.add('--model_checkpt', required = True, type = str, help = 'model checkpoint path')
PARSER.add('--save_model', required = True, type = int, default = 1, help = "enable save RL model")
PARSER.add('--load_model', required = True, type = int, default = 0, help = "enable load RL model")

def init_log(level_str):
    ## --------------- 日志的处理 -------------------
    # remove old log handlers (otherwise sequential simulations only log to first simulation)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # start new log file

    logging.VERBOSE = 5
    logging.addLevelName(logging.VERBOSE, "VERBOSE")
    logging.Logger.verbose = lambda inst, msg, *args, **kwargs: inst.log(logging.VERBOSE, msg, *args, **kwargs)
    logging.LoggerAdapter.verbose = lambda inst, msg, *args, **kwargs: inst.log(logging.VERBOSE, msg, *args, **kwargs)
    logging.verbose = lambda msg, *args, **kwargs: logging.log(logging.VERBOSE, msg, *args, **kwargs)

    if level_str == "verbose":
        log_level = logging.VERBOSE
    elif level_str == "debug":
        log_level = logging.DEBUG
    elif level_str == "info":
        log_level = logging.INFO
    elif level_str == "warning":
        log_level = logging.WARNING
    else:
        log_level = DEFAULT_LOG_LEVEL

    log_file = "output.log"
    if log_level < logging.INFO:
        streams = [logging.FileHandler(log_file, mode='w'), logging.StreamHandler()]
    else:
        print("Only minimum output to console -> see log-file")
        streams = [logging.FileHandler(log_file)]
    ## 这里加上日志的信息和对应的代码行数，输出格式：日期-时间-文件名：行数：日志信息
    logging.basicConfig(handlers=streams,
                        level=log_level, format='%(asctime)s %(name)40s:%(lineno)4d [%(levelname)6s]: %(message)s')


def load_module(module_dict, module_str, module_type_str):
    '''
    Load a module from a dictionary of modules, similar to the way #include works in C++
    '''
    module = None
    ctx = module_dict.get(module_str)
    if ctx is not None:
        module_name, class_name = ctx
        module = importlib.import_module(module_name)
        class_t = getattr(module, class_name)
        return class_t
    else:
        raise ValueError(f'{module_type_str} {module_str} not found')

def get_pred_model_modules():
    '''
    Get the predictive model modules, all models in dictory /prediction/
    '''
    pred_dict = {}      #str -> (module path, class name)   
    pred_dict['RNN'] = ('prediction.rnn', 'RNN')
    pred_dict['LSTM'] = ('prediction.lstm', 'LSTM')
    return pred_dict

def load_pred_model(model_name = "RNN"):
    '''
    Load a predictive model, deaults to RNN
    '''
    pred_dict = get_pred_model_modules()
    return load_module(pred_dict, model_name, 'Predictive Model')

def get_em_model_modules():
    '''
    Get the embedding model modules, all models in dictory /embedding/
    '''
    em_dict = {}        #str -> (module path, class name)
    em_dict['VAE'] = ('embedding.vae', 'VAE')
    return em_dict

def load_em_model(model_name = "VAE"):
    '''
    Load an embedding model, defaults to VAE
    '''
    em_dict = get_em_model_modules()   
    return load_module(em_dict, model_name, 'Embedding Model')