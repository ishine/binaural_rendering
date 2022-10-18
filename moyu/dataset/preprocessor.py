from typing import Callable, Dict, Tuple


class AmbisonicBinauralPreprocessor:
    r"""Batch data preprocessor for Ambisonic-Binaural dataset.
        """

    def __call__(self, batch_data_dict: Dict) -> Tuple[Dict, Dict]:
        r"""Format waveforms and targets for training.

        Args:
            batch_data_dict: dict, e.g., {
                'ambisonic': (batch_size, input_channels, segment_samples),
                'binaural': (batch_size, output_channels, segment_samples),
            }

        Returns:
            input_dict: dict, e.g., {
                'waveform': (batch_size, input_channels, segment_samples),
            }
            output_dict: dict, e.g., {
                'waveform': (batch_size, output_channels, segment_samples)
            }
        """
        input_dict = {'waveform': batch_data_dict['ambisonic']}
        target_dict = {'waveform': batch_data_dict['binaural']}

        return input_dict, target_dict
