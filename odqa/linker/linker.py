import logging
import spacy

logger = logging.getLogger(__name__)

class EntityLinker:

    def __init__(self, model_path):
        self.model = spacy.load(model_path)

    def __call__(self, text):

        doc = self.model(text)
        match_dict = {(e.start_char, e.end_char): e.kb_id_ for e in doc.ents}
        return match_dict
