import configargparse
import importlib

PARSER = configargparse.ArgParser(default_config_files=['config/CarRacing.config'])

PARSER.add('-c', '--config_path', required = False, is_config_file = True, help = 'config file path')
PARSER.add('--env_name', required = True, help = 'environment name')
PARSER.add('--seed', type = int, help = 'seed')

PARSER.add('--em_model', required = True, type = str, help = 'embedding model')
PARSER.add('--pred_model', required = True, type = str, help = 'predictive model')
PARSER.add('--ctrl_model', required = True, type = str, help = 'controller model')

PARSER.add('--latent_dim', required = True, type = int, help = 'embedding latent model')


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