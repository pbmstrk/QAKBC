import logging
import argparse
import blink.main_dense as main_dense

logger = logging.getLogger(__name__)

class EntityLinker:

    def __init__(self, logger=None):

        self.logger = logger

        self.models_path = "models/"

        self.config =  {
            "test_entities": None,
            "test_mentions": None,
            "interactive": False,
            "biencoder_model": self.models_path+"biencoder_wiki_large.bin",
            "biencoder_config": self.models_path+"biencoder_wiki_large.json",
            "entity_catalogue": self.models_path+"entity.jsonl",
            "entity_encoding": self.models_path+"all_entities_large.t7",
            "crossencoder_model": self.models_path+"crossencoder_wiki_large.bin",
            "crossencoder_config": self.models_path+"crossencoder_wiki_large.json",
            "fast": False, # set this to be true if speed is a concern
            "output_path": "logs/", # logging directory
            "faiss_index": "flat",
            "index_path": self.models_path+"faiss_flat_index.pkl",
            "top_k": 10
        }

        self.args = argparse.Namespace(**self.config)

        self.models = main_dense.load_models(self.args, logger=self.logger)


    def __call__(self, data_to_link):

        _, _, _, _, _, predictions, scores, = main_dense.run(self.args, logger=self.logger, 
            biencoder=self.models[0], biencoder_params=self.models[1], crossencoder=self.models[2], 
            crossencoder_params=self.models[3], candidate_encoding=self.models[4], 
            title2id=self.models[5], id2title=self.models[6], id2text=self.models[7], 
            wikipedia_id2local_id=self.models[8], faiss_indexer=self.models[9], 
            test_data=data_to_link)

        return predictions


