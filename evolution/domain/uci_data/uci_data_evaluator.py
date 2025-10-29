import os

from typing import Any
from typing import Dict

import pandas as pd

from leaf_common.candidates.constants import BEHAVIOR_PREFIX, EXPERIMENTS_DIR
from leaf_common.representation.rule_based.data.rule_set import RuleSet

from esp_sdk.esp_evaluator import EspEvaluator

from domain.uci_data.sklearn_network_config_filter import SklearnNetworkConfigFilter
from domain.uci_data.rules_model import RulesModel
from domain.uci_data.uci_data_fitness import uci_data_fitness, uci_nn_fitness


class UciDataEvaluator(EspEvaluator):
    """
    Evaluates individuals in the context of a {distributed} {co-}evolution
    loop using ESP service.

    This file is an adaptation vector_optimizer_coevaluator.py for
    the leading one domain.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Creates an abstract EspEvaluator base class.

        :param config: the configuration dictionary for the instance
        """
        super().__init__(config)

        # Note: There's all kinds of getting specific values out of the config
        #       in here.  It'd be much simpler to simply store the config
        #       and get what's needed when it's needed.
        empty_config = {}

        # Output directory for this experiment
        leaf_config = self.config.get("LEAF", empty_config)
        persistence_dir = leaf_config.get("persistence_dir", ".")
        self.experiment_id = leaf_config.get("experiment_id", "unknown")
        self.output_directory = os.path.join(persistence_dir,
                                             self.experiment_id)

        # Get various incarnations of the domain-specific config
        # #1 - modern way: using domain_config key.
        #       This conforms to what ENN does too.
        domain_config = self.config.get("domain_config", None)
        if domain_config is None:
            # #2 - old way: using domain key
            # #3 - fall back to the given config itself.
            #       for now, this makes distributed evaulation work
            domain_config = self.config.get("domain", self.config)

        self.fitness_function_name = \
            domain_config.get("fitness", "uci_data_fitness")

        # Create network specification inputs and outputs based on the data
        config_filter = SklearnNetworkConfigFilter()
        self.config = config_filter.filter_config(self.config)

        # Get the data set read in during the filtering
        data_set = config_filter.get_data_set()

        # Load/convert the data once for evaluation
        data_frame = pd.DataFrame(data_set.data, columns=data_set.feature_names)
        self.input_vectors = data_frame.values.T.tolist()
        self.inputs = data_frame.values

        # Keeping labels separate makes prediction simple
        self.labels = data_set.target.tolist()


        # Check if synthetic data is used
        synthetic_data_file = domain_config.get("synthetic_data_file", None)
        if synthetic_data_file:
            synthetic_data_path = os.path.join(EXPERIMENTS_DIR, synthetic_data_file)
            if os.path.exists(synthetic_data_path):
                print(f"Using synthetic data from {synthetic_data_path}")

                data_frame = pd.read_csv(synthetic_data_path)
                data = data_frame[data_set.feature_names]
                targets = data_frame[list(data_set.target_names)]
                
                self.s_input_vectors = data.values.T.tolist()
                self.s_inputs = data.values
                self.s_labels = targets.values.tolist()
            else:
                raise FileNotFoundError(f"Synthetic data file not found: {synthetic_data_path}")
        else:
            self.s_input_vectors = []
            self.s_inputs = []
            self.s_labels = []
    
        
    

    def evaluate_candidate(self, candidate: object) -> Dict[str, object]:
        """
        Evaluates the passed candidate by making it play the game.
        :param candidate: an evolved candidate structure
        :return: the metrics of this candidate, as a dictionary
        """
        metrics = {}
        predictor = candidate
        if isinstance(candidate, RuleSet):
            predictor = RulesModel(candidate, self.config)

        fitness_functions = {
            "uci_data_fitness": uci_data_fitness,
            "uci_nn_fitness": uci_nn_fitness
        }

        input_vectors = self.s_input_vectors if len(self.s_input_vectors) > 0 else self.input_vectors
        inputs = self.s_inputs if len(self.s_inputs) > 0 else self.inputs
        targets = self.s_labels if len(self.s_labels) > 0 else self.labels

        # argmax gives us a single answer for classification sake
        # RulesModel expects feature-major input (list of feature arrays) so
        # we pass `self.input_vectors` for RuleSet-based predictors. Typical
        # ML models (e.g. Keras) expect a samples x features 2D array, so
        # pass `self.inputs` for non-RuleSet predictors.
        if isinstance(candidate, RuleSet):
            # Rule-based predictor expects feature-major list-of-arrays
            input_for_predictor = input_vectors
        else:
            # Typical ML predictors expect a single 2D array (samples x features)
            # but some Keras models are built as multiple Input layers (one per
            # feature) and therefore expect a list of input arrays. Detect that
            # and adapt accordingly.
            input_for_predictor = inputs
            try:
                # Keras Model has an `inputs` attribute which is a list of
                # Input tensors. If the model expects multiple separate inputs
                # (len > 1) and that number matches our feature count, build
                # a list of 2D arrays (samples x 1) — one per input.
                model_inputs = getattr(predictor, 'inputs', None)
                if model_inputs is not None and isinstance(model_inputs, (list, tuple)) and len(model_inputs) > 1:
                    num_model_inputs = len(model_inputs)
                    # number of features in the dataset
                    num_features = self.inputs.shape[1]
                    if num_model_inputs == num_features:
                        # Build list of column vectors (samples x 1)
                        input_for_predictor = [self.inputs[:, i:i+1] for i in range(num_features)]
            except Exception:
                # If anything goes wrong with introspection, fall back to
                # passing the 2D array; the predictor may still accept it.
                input_for_predictor = self.inputs

        predictions = predictor.predict(input_for_predictor)
        predictions = predictions.argmax(axis=1).tolist()
        accuracy = fitness_functions[self.fitness_function_name](
            inputs, predictions, targets) # we use inputs here to be consistent with predictor input

        # Using predictions array as the behavior of the classifier
        metrics[BEHAVIOR_PREFIX+"score_vector"] = predictions
        metrics["score"] = accuracy


        baseline_preds = predictor.predict(self.input_vectors)  # TODO: make robust for NN evolution, not just rulesets
        baseline_preds = baseline_preds.argmax(axis=1).tolist()
        accuracy_to_baseline = fitness_functions["uci_data_fitness"](
            self.inputs, baseline_preds, self.labels)
        metrics["accuracy_to_baseline"] = accuracy_to_baseline
        return metrics
