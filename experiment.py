import sys
import logging
from os.path import join
from utils import load_data, gen_observations, load_featurizer
from mem_net import run_mem_net, test_mem_network
from data_processing.data_extract import get_data_path

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)
cache_pickle = "{}.pkl"
cache_dir = ".cache-pythia"


def main(dataset_name='trec'):
    """
    controls the over-arching implmentation of the algorithms
    """
    print('starting')
    path = get_data_path(dataset_name)
    directory = {
        'full_data': [join(path, '2003_preprocessed.parquet'), join(path, '2004_preprocessed.parquet')]
    }
    algorithms = {}

    # parsing
    print("parsing json data...", file=sys.stderr)

    data = load_data(directory['full_data'])

    # featurization
    tokenizer, bert = load_featurizer()
    print("generating training data...", file=sys.stderr)
    train_data, train_target, train_ids = gen_observations(data, tokenizer, bert)
    print("generating testing data...", file=sys.stderr)
    test_data, test_target, test_ids = gen_observations(data, tokenizer, bert)

    # modeling
    print("running algorithms...", file=sys.stderr)
    mem_net_model, model_name = run_mem_net(train_data, test_data)
    predicted_labels, perform_results = test_mem_network(mem_net_model, model_name)

    # results
    perform_results = {
        "id": test_ids,
        "predicted_label": predicted_labels.tolist(),
        "novelty": test_target
    }

    return perform_results


if __name__ == '__main__':
    print("Algorithm details and Results:", file=sys.stderr)
    print(main(), file=sys.stdout)
    sys.exit(0)
