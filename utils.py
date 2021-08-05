import sys
import json
import os.path
from os.path import basename
from collections import defaultdict, namedtuple, OrderedDict
import nltk
from nltk.tokenize import word_tokenize

import numpy


def fix_escapes(line):
    '''
    Purpose - Substitutes any leaning right/left quote characters in body_text segment of JSON files.
    Input - A line of text from a Pythia-style JSON file
    Output - The input line with any leaning right/left quote characters replaced with standard quotes
    '''

    # Remove embedded special left/right leaning quote characters in body_text segment of json object
    if line.find('\\\xe2\x80\x9d'):
        spot = line.find("body_text")
        line = line[:spot + 13] + line[spot + 13:].replace('\\\xe2\x80\x9d', '\\"')
    if line.find('\\\xe2\x80\x9c'):
        spot = line.find("body_text")
        line = line[:spot + 13] + line[spot + 13:].replace('\\\xe2\x80\x9c', '\\"')
    return line


def count_vocab(text, wordcount):
    '''
    Purpose - Counts the number of times any word appears in a text string.
    Input - A line of text and a dictionary with words and their associated counts
    Output - An updated dictionary with words and their associated counts
    '''

    # Tokenize text and add words to corpus dictionary
    wordlist = word_tokenize(text)
    for word in wordlist: wordcount[word] += 1

    return wordcount


def parse_json(folder, seed=1, **kwargs):
    '''
    Purpose - Parses a folder full of JSON files containing document data.
    Input - a directory full of files with JSON data
    Output - A set of cluster IDs, a dictionary mapping to sets of tuples with document
             arrival order and indices, an array of the parsed JSON document data

    The JSON data schema for ingesting documents into Pythia is:
    corpus = name of corpus
    cluster_id = unique identifier for associated cluster
    post_id = unique identifier for element (document, post, etc)
    order = int signifying order item was received
    body_text = text of element (document, post, etc)
    novelty = boolean assessment of novelty
    '''

    data = []
    clusters = set()
    order = defaultdict(set)
    wordcount = defaultdict(int)
    i = 0

    test_data = []
    test_clusters = set()
    test_order = defaultdict(set)
    j = 0

    random_state = numpy.random.RandomState(seed)
    for file_name in os.listdir(folder):
        if file_name.endswith(".json"):
            # Read JSON file line by line and retain stats about number of clusters and order of objects
            full_file_name = os.path.join(folder, file_name)

            with open(full_file_name, 'r') as dataFile:
                if random_state.random_sample() > 0.2:
                    for line in dataFile:
                        parsedData = json.loads(fix_escapes(line))
                        clusters.add(parsedData["cluster_id"])
                        order[parsedData["cluster_id"]].add((parsedData["order"], i))
                        wordcount = count_vocab(parsedData["body_text"], wordcount)
                        data.append(parsedData)
                        i += 1
                else:
                    for line in dataFile:
                        parsedData = json.loads(fix_escapes(line))
                        test_clusters.add(parsedData["cluster_id"])
                        test_order[parsedData["cluster_id"]].add((parsedData["order"], j))
                        test_data.append(parsedData)
                        j += 1
    return clusters, order, data, test_clusters, test_order, test_data, wordcount


def data_gen(all_clusters, lookup_order, document_data, features, parameters, vocab, full_vocab, encoder_decoder, lda_model, tf_session, w2v_model, hdf5_path=None, hdf5_save_frequency=100):
    '''
    Controls the generation of observations with the specified features.

    Args:
        argv (list): contains a set of all the cluster IDs, a dictionary of the document arrival order, an array of parsed JSON documents, the filename of the corpus, the feature tuple with the specified features, the vocabluary of the dataset and the skipthoughts vectors encoder/decoder

    Returns:
        list: contains for each obeservation
    '''
    data, labels, ids = gen_observations(all_clusters, lookup_order, document_data, features, parameters, vocab, full_vocab, encoder_decoder, lda_model, tf_session, w2v_model, hdf5_path, hdf5_save_frequency)

    return data, labels, ids


def gen_observations(all_clusters, lookup_order, document_data, features, parameters, vocab, full_vocab,
                     encoder_decoder, lda_model, tf_session, w2v_model, hdf5_path=None, dtype=np.float32):
    '''
    Generates observations for each cluster found in JSON file and calculates the specified features.

    Args:
        all_clusters (set): cluster IDs
        lookup_order (dict): document arrival order
        document_data (array): parsed JSON documents
        features (dict): the specified features to be calculated
        parameters (dict): data structure with run parameters
        vocab (dict): the vocabulary of the data set
        full_vocab (dict_: to vocabulary of the data set including stop wrods and punctuation
        encoder_decoder (???): the encoder/decoder for skipthoughts vectors
        lda_model (sklearn.???): trained LDA model
        tf_session: active TensorFlow session
        w2v_model (gensim.word2vec): trained word2vec model

    Returns:
        data(list): contains for each obeservation the features of the document vs corpus which could include:
            tfidf sum, cosine similarity, bag of words vectors, skip thoughts, lda, w2v or, onehot cnn encoding
        labels(list): the labels for each document where a one is novel and zero is duplicate
    '''

    # Prepare to store results of feature assessments
    data = list()
    labels = list()
    # mem_net_features is used when the mem_net algorithm is ran
    # It consist of inputs, labels(answers), input_masks and questions for each entry
    mem_net_features = {}
    inputs = []
    input_masks = []
    questions = []
    # Sentence punctuation delimiters
    punkt = ['.', '?', '!']

    corpus_unprocessed = list()
    # HDF5-related parameters
    hdf5_save_frequency = parameters['hdf5_save_frequency']
    data_key = 'data'
    labels_key = 'labels'
    # Truncate any existing files at save location, or return early if
    # using existing files
    if hdf5_path is not None:
        if parameters['hdf5_use_existing'] and os.path.isfile(hdf5_path):
            return hdf5_path, hdf5_path
        open(hdf5_path, 'w').close()

    # Create random state
    random_state = np.random.RandomState(parameters['seed'])

    # Iterate through clusters found in JSON file, generate observations
    # pairing data and label
    for cluster in all_clusters:
        # Determine arrival order in this cluster
        sorted_entries = [x[1] for x in sorted(lookup_order[cluster], key=lambda x: x[0])]
        observations = [document_data[sorted_entries[0]]]
        for index in sorted_entries[1:]:
            next_doc = document_data[index]
            observations.append(next_doc)
            labeled_observation = {'novelty': next_doc['novelty'],
                                   'data': copy.copy(observations)}
            corpus_unprocessed.append(labeled_observation)

    # Resample if necessary
    # If oversampling +/- replacement, sample up
    # to larger class size for both classes, with replacement
    # If -oversampling, sample down to
    # smaller class size for both classes with or w/o replacement
    if 'resampling' in parameters:
        resampling_parameters = parameters['resampling']
        if resampling_parameters.get('over', False):
            desired_size = None
            resampling_parameters['replacement'] = True
        else:
            desired_size = -np.Inf
        if resampling_parameters.get('replacement', False):
            replacement = True
        else:
            replacement = False
        logger.debug("Replacement: {}, Desired size: {}".format(replacement, desired_size))
        logger.debug("Size of data: {}, Number of clusters: {}".format(len(corpus_unprocessed), len(all_clusters)))
        corpus = sampling.label_sample(corpus_unprocessed, "novelty", replacement, desired_size, random_state)
    else:
        corpus = corpus_unprocessed

    # Featurize each observation
    # Some duplication of effort here bc docs will appear multiple times
    # across observations

    clusterids = []
    postids = []
    for case in corpus:

        # Create raw and normalized document arrays
        case_docs_raw = [record['body_text'] for record in case['data']]
        case_docs_normalized = [normalize.xml_normalize(body_text) for body_text in case_docs_raw]
        case_docs_no_stop_words = [normalize.normalize_and_remove_stop_words(body_text) for body_text in case_docs_raw]
        # create ids for individual data points
        postid = [record['post_id'] for record in case['data']][-1]
        postids.append(postid)
        clusterid = [record['cluster_id'] for record in case['data']][0]
        clusterids.append(clusterid)
        # Pull out query documents
        doc_raw = case_docs_raw[-1]
        doc_normalized = case_docs_normalized[-1]
        doc_no_stop_words = case_docs_no_stop_words[-1]
        # Create lists of background documents
        bkgd_docs_raw = case_docs_raw[:-1]
        bkgd_docs_normalized = case_docs_normalized[:-1]
        bkgd_docs_no_stop_words = case_docs_no_stop_words[:-1]
        bkgd_text_raw = '\n'.join(bkgd_docs_raw)
        bkgd_text_normalized = '\n'.join(bkgd_docs_normalized)
        bkgd_text_no_stop_words = '\n'.join(bkgd_docs_no_stop_words)
        feature_vectors = list()

        if 'mem_net' in features:
            # Get all sentences for the memory network algorithm
            bkgd_sentences_full = tokenize.punkt_sentences(bkgd_text_raw)
            doc_input, doc_questions, doc_masks = gen_mem_net_observations(doc_raw, bkgd_text_raw, bkgd_sentences_full,
                                                                           features['mem_net'], vocab, full_vocab,
                                                                           w2v_model, encoder_decoder)

            # Now add all of the input docs to the primary list
            inputs.append(doc_input)
            questions.append(doc_questions)
            input_masks.append(doc_masks)

        if case["novelty"]:
            labels.append(1)
        else:
            labels.append(0)

        # save to HDF5 if desired
        if hdf5_path is not None and len(data) % hdf5_save_frequency == 0:
            with h5py.File(hdf5_path, 'a') as h5:
                data_np = np.array(data)
                labels_np = np.reshape(np.array(labels), (-1, 1))
                add_to_hdf5(h5, data_np, data_key)
                add_to_hdf5(h5, labels_np, labels_key, np.uint8)
                labels = list()
                data = list()
    # Save off any remainder
    if hdf5_path is not None and len(data) > 0:
        with h5py.File(hdf5_path, 'a') as h5:
            data_np = np.array(data)
            labels_np = np.reshape(np.array(labels), (-1, 1))
            add_to_hdf5(h5, data_np, data_key)
            add_to_hdf5(h5, labels_np, labels_key, np.uint8)

    mem_net_features['inputs'] = inputs
    mem_net_features['questions'] = questions
    mem_net_features['input_masks'] = input_masks
    mem_net_features['answers'] = labels

    ids = ["C" + str(clusterid) + "_P" + str(postid) for clusterid, postid in zip(clusterids, postids)]

    if 'mem_net' in features:
        return mem_net_features, labels, ids
    if hdf5_path is not None:
        return hdf5_path, hdf5_path, ids
    else:
        return data, labels, ids


def preprocess(features, parameters, corpus_dict, trainingdata):
    '''
    Controls the preprocessing of the corpus, including building vocabulary and model creation.

    Args:
        argv (list): contains a list of the command line features, a dictionary of all
        tokens in the corpus, an array of parsed JSON documents, a list of the command line parameters

    Returns:
        multiple: dictionary of the corpus vocabulary, skipthoughts encoder_decoder, trained LDA model
    '''

    # Look at environment variable 'PYTHIA_MODELS_PATH' for user-defined model location
    # If environment variable is not defined, use current working directory
    if os.environ.get('PYTHIA_MODELS_PATH') is not None:
        path_to_models = os.environ.get('PYTHIA_MODELS_PATH')
    else:
        path_to_models = os.path.join(os.getcwd(), 'models')
    # Make the directory for the models unless it already exists
    try:
        os.makedirs(path_to_models)
    except OSError as exception:
        if exception.errno != errno.EEXIST: raise

    encoder_decoder = None
    vocab= None
    lda_model = None
    tf_session = None
    w2v_model = None
    full_vocab = None

    if 'st' in features:
        from src.featurizers.skipthoughts import skipthoughts
        encoder_decoder = skipthoughts.load_model()

    if 'bow' in features or 'lda' in features:
        vocab = gen_vocab(corpus_dict, **parameters)

    if 'cnn' in features:
        from src.featurizers import tensorflow_cnn
        train_mode = 'train'
        cnn_params = copy.deepcopy(parameters)
        # Look for the trained Tensorflow model and if it isn't present create and save it
        # If we are loading the model, ensure the full_vocab settings are the same as the training
        if os.path.isfile(os.path.join(path_to_models, "tf_trained_session.cpt")):
            train_mode='load'
            cnn_params['full_vocab_stem']=False
            cnn_params['full_vocab_type']='character'
            cnn_params['full_char_vocab']="abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/|_@#$%^&*~`+-=<>()[]{}"
        full_vocab = gen_full_vocab(corpus_dict, **cnn_params)
        features['cnn']['vocab'] = full_vocab
        tf_session = tensorflow_cnn.tensorflow_cnn(trainingdata, mode=train_mode, model_path=path_to_models, **features['cnn'])

    if 'lda' in features:
        features['lda']['vocab'] = vocab
        lda_model = build_lda(trainingdata, vocab,
            random_state=parameters['seed'], **features['lda'])

    if 'w2v' in features: w2v_model = build_w2v(trainingdata, **features['w2v'])

    if 'wordonehot' in features: full_vocab = gen_full_vocab(corpus_dict, **parameters)

    #get the appropriate model(s) when running the memory network code
    if 'mem_net' in features:
        if features['mem_net'].get('embed_mode', False):
            embed_mode = features['mem_net']['embed_mode']
        else: embed_mode = 'word2vec'
        if embed_mode=='skip_thought' and not encoder_decoder:
            from src.featurizers.skipthoughts import skipthoughts
            encoder_decoder = skipthoughts.load_model()
        if embed_mode=="onehot" and not full_vocab:
            full_vocab = gen_full_vocab(corpus_dict, **parameters)
        if embed_mode=='word2vec' and not w2v_model:
            w2v_model = build_w2v(trainingdata, **features['mem_net'])

    return vocab, full_vocab, encoder_decoder, lda_model, tf_session, w2v_model


def dir_hash(directory, verbose=0):
    """# http://akiscode.com/articles/sha-1directoryhash.shtml
    # Copyright (c) 2009 Stephen Akiki
    # MIT License (Means you can do whatever you want with this)
    #  See http://www.opensource.org/licenses/mit-license.php
    # Error Codes:
    #   -1 -> Directory does not exist
    #   -2 -> General error (see stack traceback)"""
    SHAhash = hashlib.sha1()
    if not os.path.exists(directory):
        return -1

    try:
        for root, dirs, files in os.walk(directory):
            for names in files:
                if verbose == 1:
                    logger.info('Hashing', names)
                filepath = os.path.join(root, names)
                try:
                    f1 = open(filepath, 'rb')
                except:
                    # You can't open the file for some reason
                    f1.close()
                    continue

                while 1:
                    # Read file in as little chunks
                    buf = f1.read(4096)
                    logger.debug("Type: {}".format(type(buf)))
                    if not buf: break
                    file_hash = hashlib.sha1(buf).digest()
                    logger.debug("File hash digest type: {}".format(type(file_hash)))
                    SHAhash.update(file_hash)
                f1.close()

    except:
        import traceback
        # Print the stack traceback
        traceback.print_exc()
        return -2

    return SHAhash.hexdigest()
