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
        self.output_directory = os.path.join(persistence_dir, self.experiment_id)

        # Get various incarnations of the domain-specific config
        # #1 - modern way: using domain_config key.
        #       This conforms to what ENN does too.
        domain_config = self.config.get("domain_config", None)
        if domain_config is None:
            # #2 - old way: using domain key
            # #3 - fall back to the given config itself.
            #       for now, this makes distributed evaulation work
            domain_config = self.config.get("domain", self.config)

        self.fitness_function_name = domain_config.get("fitness", "uci_data_fitness")
        self.weights_filename = domain_config.get("weights_file", "")

        # Create network specification inputs and outputs based on the data
        config_filter = SklearnNetworkConfigFilter()
        self.config = config_filter.filter_config(self.config)

        # Get the data set read in during the filtering
        data_set = config_filter.get_data_set()
        self.target_names = data_set.target_names
        self.num_classes = len(data_set.target_names)

        # Load/convert the data once for evaluation
        data_frame = pd.DataFrame(data_set.data, columns=data_set.feature_names)


        ### We need to differentiate between BASELINE data and EVOLUTION data
        ### If a synthetic data file name is provided, then we know it is the EVOLUTION data
        ### Else, we can set the BASELINE data to be the EVOLUTION data


        self.baseline_input_vectors = data_frame.values.T.tolist()
        self.baseline_inputs = data_frame.values
        self.baseline_labels = data_set.target.tolist()

        # Check if synthetic data is used
        synthetic_data_file = domain_config.get("synthetic_data_file", None)
        if synthetic_data_file:
            synthetic_data_path = os.path.join(EXPERIMENTS_DIR, synthetic_data_file)
            if os.path.exists(synthetic_data_path):
                print(f"Using synthetic data from {synthetic_data_path}")

                data_frame = pd.read_csv(synthetic_data_path)
                data = data_frame[data_set.feature_names]
                targets = data_frame[list(data_set.target_names)]
                
                self.evolution_input_vectors = data.values.T.tolist()
                self.evolution_inputs = data.values
                self.evolution_labels = targets.values.tolist()
            else:
                raise FileNotFoundError(f"Synthetic data file not found: {synthetic_data_path}")
        else:
            self.evolution_input_vectors = data_frame.values.T.tolist()
            self.evolution_inputs = data_frame.values
            self.evolution_labels = data_set.target.tolist()

        
        # Set up in-distribution and out-of-distribution baseline
        in_distribution_data = domain_config.get("in_distribution_data", None)
        out_of_distribution_data = domain_config.get("out_of_distribution_data", None)
        if in_distribution_data and out_of_distribution_data:
            synthetic_in_path = os.path.join(EXPERIMENTS_DIR, in_distribution_data)
            synthetic_out_path = os.path.join(EXPERIMENTS_DIR, out_of_distribution_data)
            if os.path.exists(synthetic_in_path) and os.path.exists(synthetic_out_path):
                print(f"Using synthetic data from {synthetic_in_path} and {synthetic_out_path}")

                in_data_frame = pd.read_csv(synthetic_in_path)
                in_data = in_data_frame[data_set.feature_names]
                in_targets = in_data_frame[list(data_set.target_names)]
                
                self.in_dist_input_vectors = in_data.values.T.tolist()
                self.in_dist_inputs = in_data.values
                self.in_dist_labels = in_targets.values.argmax(axis=1).tolist()

                out_data_frame = pd.read_csv(synthetic_out_path)
                out_data = out_data_frame[data_set.feature_names]
                out_targets = out_data_frame[list(data_set.target_names)]
                
                self.out_dist_input_vectors = out_data.values.T.tolist()
                self.out_dist_inputs = out_data.values
                self.out_dist_labels = out_targets.values.argmax(axis=1).tolist()
            else:
                raise FileNotFoundError(f"In/out data file not found: {synthetic_in_path}, {synthetic_out_path}")







        # self.input_vectors = data_frame.values.T.tolist()
        # self.inputs = data_frame.values

        # # Keeping labels separate makes prediction simple
        # self.labels = data_set.target.tolist()


        # # Check if synthetic data is used
        # synthetic_data_file = domain_config.get("synthetic_data_file", None)
        # if synthetic_data_file:
        #     synthetic_data_path = os.path.join(EXPERIMENTS_DIR, synthetic_data_file)
        #     if os.path.exists(synthetic_data_path):
        #         print(f"Using synthetic data from {synthetic_data_path}")

        #         data_frame = pd.read_csv(synthetic_data_path)
        #         data = data_frame[data_set.feature_names]
        #         targets = data_frame[list(data_set.target_names)]
                
        #         self.s_input_vectors = data.values.T.tolist()
        #         self.s_inputs = data.values
        #         self.s_labels = targets.values.tolist()
        #     else:
        #         raise FileNotFoundError(f"Synthetic data file not found: {synthetic_data_path}")
        # else:
        #     self.s_input_vectors = []
        #     self.s_inputs = []
        #     self.s_labels = []
    
        
    

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

        # input_vectors = self.s_input_vectors if len(self.s_input_vectors) > 0 else self.input_vectors
        # inputs = self.s_inputs if len(self.s_inputs) > 0 else self.inputs
        # targets = self.s_labels if len(self.s_labels) > 0 else self.labels

        # argmax gives us a single answer for classification sake
        # RulesModel expects feature-major input (list of feature arrays) so
        # we pass `self.input_vectors` for RuleSet-based predictors. Typical
        # ML models (e.g. Keras) expect a samples x features 2D array, so
        # pass `self.inputs` for non-RuleSet predictors.
        if isinstance(candidate, RuleSet):
            # Rule-based predictor expects feature-major list-of-arrays
            input_for_predictor = self.evolution_input_vectors
        else:
            # Typical ML predictors expect a single 2D array (samples x features)
            # but some Keras models are built as multiple Input layers (one per
            # feature) and therefore expect a list of input arrays. Detect that
            # and adapt accordingly.
            input_for_predictor = self.evolution_inputs
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


        ### Run through fitness functions

        def run_fitness(function_name, inputs, preds, targets):
            if function_name == "uci_data_fitness":
                return fitness_functions[function_name](preds, targets)
            elif function_name == "uci_nn_fitness":
                return fitness_functions[function_name](inputs, preds, self.weights_filename, self.num_classes)
            else:
                raise ValueError("fitness function not found")

        predictions = predictor.predict(input_for_predictor)
        predictions = predictions.argmax(axis=1).tolist()
        accuracy = run_fitness(self.fitness_function_name, self.evolution_inputs, predictions, self.evolution_labels)
        metrics[BEHAVIOR_PREFIX+"score_vector"] = predictions
        metrics["score"] = accuracy



        ## Run through baseline metrics

        def run_baseline_metrics(input_vectors, inputs, labels, metric_name):
            baseline_preds = predictor.predict(input_vectors)  # TODO: make robust for NN evolution, not just rulesets
            baseline_preds = baseline_preds.argmax(axis=1).tolist()
            accuracy_to_baseline = run_fitness("uci_data_fitness", inputs, baseline_preds, labels)
            metrics[metric_name] = accuracy_to_baseline

        # Baseline is always compared against original data
        # baseline_preds = predictor.predict(self.baseline_input_vectors)  # TODO: make robust for NN evolution, not just rulesets
        # baseline_preds = baseline_preds.argmax(axis=1).tolist()
        # accuracy_to_baseline = run_fitness("uci_data_fitness", self.baseline_inputs, baseline_preds, self.baseline_labels)
        # metrics["accuracy_to_baseline"] = accuracy_to_baseline

        run_baseline_metrics(self.baseline_input_vectors, self.baseline_inputs, self.baseline_labels, "baseline_accuracy")

        # Also calculate in-distribution and out-of-distribution accuracy
        run_baseline_metrics(self.in_dist_input_vectors, self.in_dist_inputs, self.in_dist_labels, "in_distribution_accuracy")
        run_baseline_metrics(self.out_dist_input_vectors, self.out_dist_inputs, self.out_dist_labels, "out_of_distribution_accuracy")
        

        return metrics









        # predictions = predictor.predict(input_for_predictor)
        # predictions = predictions.argmax(axis=1).tolist()
        # accuracy = fitness_functions[self.fitness_function_name](
        #     # inputs, predictions, targets) # we use inputs here to be consistent with predictor input
        #     self.evolution_inputs, predictions, self.evolution_labels) # we use inputs here to be consistent with predictor input

        # # Using predictions array as the behavior of the classifier
        # metrics[BEHAVIOR_PREFIX+"score_vector"] = predictions
        # metrics["score"] = accuracy


        # # Baseline is always compared against original data
        # baseline_preds = predictor.predict(self.baseline_input_vectors)  # TODO: make robust for NN evolution, not just rulesets
        # baseline_preds = baseline_preds.argmax(axis=1).tolist()
        # accuracy_to_baseline = fitness_functions["uci_data_fitness"](
        #     self.baseline_inputs, baseline_preds, self.baseline_labels)
        # metrics["accuracy_to_baseline"] = accuracy_to_baseline
        # return metrics
