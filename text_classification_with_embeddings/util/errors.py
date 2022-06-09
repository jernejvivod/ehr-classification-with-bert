class EmbeddingError(Exception):
    def __init__(self, method, return_code):
        super().__init__('Computing embeddings using {0} failed with status value {1}'.format(method, return_code))
