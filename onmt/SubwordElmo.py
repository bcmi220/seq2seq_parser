import json
import logging
from typing import Union, List, Dict, Any

import torch
from torch.autograd import Variable
from torch.nn.modules import Dropout

import numpy
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.checks import ConfigurationError
from allennlp.modules.elmo_lstm import ElmoLstm
from allennlp.modules.highway import Highway
from allennlp.modules.scalar_mix import ScalarMix

logger = logging.getLogger(__name__)


class SubwordElmo(torch.nn.Module):
    """
    Compute ELMo representations using a pre-trained bidirectional language model.
    See "Deep contextualized word representations", Peters et al. for details.
    This module takes character id input and computes ``num_output_representations`` different layers
    of ELMo representations.  Typically ``num_output_representations`` is 1 or 2.  For example, in
    the case of the SRL model in the above paper, ``num_output_representations=1`` where ELMo was included at
    the input token representation layer.  In the case of the SQuAD model, ``num_output_representations=2``
    as ELMo was also included at the GRU output layer.
    In the implementation below, we learn separate scalar weights for each output layer,
    but only run the biLM once on each input sequence for efficiency.
    Parameters
    ----------
    options_file : ``str``, required.
        ELMo JSON options file
    weight_file : ``str``, required.
        ELMo hdf5 weight file
    num_output_representations: ``int``, required.
        The number of ELMo representation layers to output.
    requires_grad: ``bool``, optional
        If True, compute gradient of ELMo parameters for fine tuning.
    do_layer_norm : ``bool``, optional, (default=False).
        Should we apply layer normalization (passed to ``ScalarMix``)?
    dropout : ``float``, optional, (default = 0.5).
        The dropout to be applied to the ELMo representations.
    module : ``torch.nn.Module``, optional, (default = None).
        If provided, then use this module instead of the pre-trained ELMo biLM.
        If using this option, then pass ``None`` for both ``options_file``
        and ``weight_file``.  The module must provide a public attribute
        ``num_layers`` with the number of internal layers and its ``forward``
        method must return a ``dict`` with ``activations`` and ``mask`` keys
        (see `_ElmoBilm`` for an example).  Note that ``requires_grad`` is also
        ignored with this option.
    """

    def __init__(self,
                 options_file: str,
                 subword_weight,
                 num_output_representations: int,
                 do_layer_norm: bool = False,
                 dropout: float = 0.5) -> None:
        super(SubwordElmo, self).__init__()

        logging.info("Initializing Subword ELMo")
        self._elmo_lstm = _ElmoBiLm(options_file, subword_weight)
        self._dropout = Dropout(p=dropout)
        self._scalar_mixes: Any = []
        for k in range(num_output_representations):
            scalar_mix = ScalarMix(self._elmo_lstm.num_layers, do_layer_norm=do_layer_norm)
            self.add_module('scalar_mix_{}'.format(k), scalar_mix)
            self._scalar_mixes.append(scalar_mix)

    def get_output_dim(self):
        return self._elmo_lstm.get_output_dim()

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Parameters
        ----------
        inputs : ``torch.autograd.Variable``
            Shape ``(batch_size, timesteps, 50)`` of character ids representing the current batch.
            We also accept tensors with additional optional dimensions:
            ``(batch_size, dim0, dim1, ..., dimn, timesteps, 50)``
        Returns
        -------
        Dict with keys:
        ``'elmo_representations'``: ``List[torch.autograd.Variable]``
            A ``num_output_representations`` list of ELMo representations for the input sequence.
            Each representation is shape ``(batch_size, timesteps, embedding_dim)``
        ``'mask'``:  ``torch.autograd.Variable``
            Shape ``(batch_size, timesteps)`` long tensor with sequence mask.
        """
        # reshape the input if needed
        original_shape = inputs.size()
        timesteps, num_characters = original_shape[-2:]
        if len(original_shape) > 3:
            reshaped_inputs = inputs.view(-1, timesteps, num_characters)
        else:
            reshaped_inputs = inputs

        # run the biLM
        bilm_output = self._elmo_lstm(reshaped_inputs)
        layer_activations = bilm_output['activations']
        mask = bilm_output['mask']

        # compute the elmo representations
        representations = []
        for i in range(len(self._scalar_mixes)):
            scalar_mix = getattr(self, 'scalar_mix_{}'.format(i))
            representation = scalar_mix(layer_activations, mask)
            representations.append(self._dropout(representation))

        elmo_representations = representations

        return {'subword_elmo_representations': elmo_representations, 'mask': mask}


class _ElmoSubwordEncoder(torch.nn.Module):
    """
    Compute context sensitive token representation using pretrained biLM.
    This embedder has input character ids of size (batch_size, sequence_length, 50)
    and returns (batch_size, sequence_length + 2, embedding_dim), where embedding_dim
    is specified in the options file (typically 512).
    We add special entries at the beginning and end of each sequence corresponding
    to <S> and </S>, the beginning and end of sentence tokens.
    Note: this is a lower level class useful for advanced usage.  Most users should
    use ``ElmoTokenEmbedder`` or ``allennlp.modules.Elmo`` instead.
    Parameters
    ----------
    options_file : ``str``
        ELMo JSON options file
    weight_file : ``str``
        ELMo hdf5 weight file
    requires_grad: ``bool``, optional
        If True, compute gradient of ELMo parameters for fine tuning.
    The relevant section of the options file is something like:
    .. example-code::
        .. code-block:: python
            {'char_cnn': {
                'activation': 'relu',
                'embedding': {'dim': 4},
                'filters': [[1, 4], [2, 8], [3, 16], [4, 32], [5, 64]],
                'max_characters_per_token': 50,
                'n_characters': 262,
                'n_highway': 2
                }
            }
    """

    def __init__(self,
                 options_file: str,
                 subword_weight) -> None:
        super(_ElmoSubwordEncoder, self).__init__()

        with open(cached_path(options_file), 'r') as fin:
            self._options = json.load(fin)
        
        self._subword_weight = subword_weight

        self.output_dim = self._options['lstm']['projection_dim']

        self._load_weights()

    def get_output_dim(self):
        return self.output_dim

    @overrides
    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute context insensitive token embeddings for ELMo representations.
        Parameters
        ----------
        inputs: ``torch.autograd.Variable``
            Shape ``(batch_size, sequence_length, 50)`` of character ids representing the
            current batch.
        Returns
        -------
        Dict with keys:
        ``'token_embedding'``: ``torch.autograd.Variable``
            Shape ``(batch_size, sequence_length, embedding_dim)`` tensor with context
            insensitive token representations.
        ``'mask'``:  ``torch.autograd.Variable``
            Shape ``(batch_size, sequence_length)`` long tensor with sequence mask.
        """

        mask = Variable(((inputs > 0).long().sum(dim=-1) > 0).long())

        # the character id embedding
        max_subwords_per_token = self._options['subword_cnn']['max_subwords_per_token']

        # (batch_size * sequence_length, max_subwords_per_token, embed_dim)
        subword_embedding = torch.nn.functional.embedding(
            inputs.view(-1, max_subwords_per_token),
            self._subword_embedding_weights
        )

        # run convolutions
        cnn_options = self._options['subword_cnn']
        if cnn_options['activation'] == 'tanh':
            activation = torch.nn.functional.tanh
        elif cnn_options['activation'] == 'relu':
            activation = torch.nn.functional.relu
        else:
            raise ConfigurationError("Unknown activation")

        # (batch_size * sequence_length, embed_dim, max_chars_per_token)
        subword_embedding = torch.transpose(subword_embedding, 1, 2)
        convs = []
        for i in range(len(self._convolutions)):
            conv = getattr(self, 'subword_conv_{}'.format(i))
            convolved = conv(subword_embedding)
            # (batch_size * sequence_length, n_filters for this width)
            convolved, _ = torch.max(convolved, dim=-1)
            convolved = activation(convolved)
            convs.append(convolved)

        # (batch_size * sequence_length, n_filters)
        token_embedding = torch.cat(convs, dim=-1)

        # apply the highway layers (batch_size * sequence_length, n_filters)
        token_embedding = self._highways(token_embedding)

        # final projection  (batch_size * sequence_length, embedding_dim)
        token_embedding = self._projection(token_embedding)

        # reshape to (batch_size, sequence_length, embedding_dim)
        batch_size, sequence_length, _ = inputs.size()

        return {
            'mask': mask,
            'token_embedding': token_embedding.view(batch_size, sequence_length, -1)
        }

    def _load_weights(self):
        self._load_subword_embedding()
        self._load_cnn_weights()
        self._load_highway()
        self._load_projection()

    def _load_subword_embedding(self):
        self._subword_embedding_weights = torch.nn.Parameter(
            torch.FloatTensor(self._subword_weight),
            requires_grad=False
        )



    def _load_cnn_weights(self):
        cnn_options = self._options['subword_cnn']
        filters = cnn_options['filters']
        subword_embed_dim = cnn_options['embedding']['dim']

        convolutions = []
        for i, (width, num) in enumerate(filters):
            conv = torch.nn.Conv1d(
                in_channels=subword_embed_dim,
                out_channels=num,
                kernel_size=width,
                bias=True
            )
            convolutions.append(conv)
            self.add_module('subword_conv_{}'.format(i), conv)

        self._convolutions = convolutions

    def _load_highway(self):
        # pylint: disable=protected-access
        # the highway layers have same dimensionality as the number of cnn filters
        cnn_options = self._options['subword_cnn']
        filters = cnn_options['filters']
        n_filters = sum(f[1] for f in filters)
        n_highway = cnn_options['n_highway']
        # create the layers
        self._highways = Highway(n_filters, n_highway, activation=torch.nn.functional.relu)

    def _load_projection(self):
        cnn_options = self._options['subword_cnn']
        filters = cnn_options['filters']
        n_filters = sum(f[1] for f in filters)
        self._projection = torch.nn.Linear(n_filters, self.output_dim, bias=True)


class _ElmoBiLm(torch.nn.Module):
    """
    Run a pre-trained bidirectional language model, outputing the activations at each
    layer for weighting together into an ELMo representation (with
    ``allennlp.modules.seq2seq_encoders.Elmo``).  This is a lower level class, useful
    for advanced uses, but most users should use ``allennlp.modules.seq2seq_encoders.Elmo``
    directly.
    Parameters
    ----------
    options_file : ``str``
        ELMo JSON options file
    weight_file : ``str``
        ELMo hdf5 weight file
    requires_grad: ``bool``, optional
        If True, compute gradient of ELMo parameters for fine tuning.
    """

    def __init__(self,
                 options_file: str,
                 subword_weight) -> None:
        super(_ElmoBiLm, self).__init__()

        self._token_embedder = _ElmoSubwordEncoder(options_file, subword_weight)

        with open(cached_path(options_file), 'r') as fin:
            options = json.load(fin)
        if not options['lstm'].get('use_skip_connections'):
            raise ConfigurationError('We only support pretrained biLMs with residual connections')
        self._elmo_lstm = ElmoLstm(input_size=options['lstm']['projection_dim'],
                                   hidden_size=options['lstm']['projection_dim'],
                                   cell_size=options['lstm']['dim'],
                                   num_layers=options['lstm']['n_layers'],
                                   memory_cell_clip_value=options['lstm']['cell_clip'],
                                   state_projection_clip_value=options['lstm']['proj_clip'])
        # Number of representation layers including context independent layer
        self.num_layers = options['lstm']['n_layers'] + 1

    def get_output_dim(self):
        return 2 * self._token_embedder.get_output_dim()

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Parameters
        ----------
        inputs: ``torch.autograd.Variable``
            Shape ``(batch_size, timesteps, 50)`` of character ids representing the current batch.
        Returns
        -------
        Dict with keys:
        ``'activations'``: ``List[torch.autograd.Variable]``
            A list of activations at each layer of the network, each of shape
            ``(batch_size, timesteps, embedding_dim)``
        ``'mask'``:  ``torch.autograd.Variable``
            Shape ``(batch_size, timesteps)`` long tensor with sequence mask.
        Note that the output tensors all include additional special begin and end of sequence
        markers.
        """
        token_embedding = self._token_embedder(inputs)
        type_representation = token_embedding['token_embedding']
        mask = token_embedding['mask']
        lstm_outputs = self._elmo_lstm(type_representation, mask)

        # Prepare the output.  The first layer is duplicated.
        output_tensors = [
            torch.cat([type_representation, type_representation], dim=-1)
        ]
        for layer_activations in torch.chunk(lstm_outputs, lstm_outputs.size(0), dim=0):
            output_tensors.append(layer_activations.squeeze(0))

        return {
            'activations': output_tensors,
            'mask': mask,
        }
