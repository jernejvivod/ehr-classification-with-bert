import argparse

from text_classification_with_embeddings import Tasks, EntityEmbeddingMethod
from text_classification_with_embeddings.document_embedding import embed
from text_classification_with_embeddings.util.argparse import file_path, dir_path


def main(**kwargs):
    # COMPUTING DOCUMENT EMBEDDINGS FROM fastText FORMAT INPUT
    if kwargs['task'] == Tasks.GET_ENTITY_EMBEDDINGS.value:
        if kwargs['starspace_path'] is None:
            raise ValueError('path to StarSpace executable must be defined when using StarSpace')
        embed.get_starspace_entity_embeddings(kwargs['starspace_path'], kwargs['train_data_path'], kwargs['output_dir_path'], kwargs['starspace_args'])


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(prog='text-classification-with-embeddings')
    subparsers = parser.add_subparsers(required=True, dest='task', help='Task to run')

    # COMPUTING DOCUMENT EMBEDDINGS FROM fastText FORMAT INPUT
    get_entity_embeddings_parser = subparsers.add_parser(Tasks.GET_ENTITY_EMBEDDINGS.value)
    get_entity_embeddings_parser.add_argument('--method', type=str, default=EntityEmbeddingMethod.STARSPACE.value,
                                              choices=[v.value for v in EntityEmbeddingMethod], help='Method of generating entity embeddings')
    get_entity_embeddings_parser.add_argument('--train-data-path', type=file_path, required=True, help='Path to file containing the training data in fastText format')
    get_entity_embeddings_parser.add_argument('--output-dir-path', type=dir_path, default='.', help='Path to directory in which to save the embeddings')
    get_entity_embeddings_parser.add_argument('--starspace-path', type=file_path, help='Path to StarSpace executable')
    get_entity_embeddings_parser.add_argument('--starspace-args', type=str, default='', help='Arguments passed to StarSpace implementation (enclose in quotes)')

    # DOCUMENT EMBEDDING EVALUATION
    evaluate_parser = subparsers.add_parser(Tasks.EVALUATE.value)
    # TODO path to training document embeddings

    args = parser.parse_args()
    main(**vars(args))
