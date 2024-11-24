import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pickle
import os

class FuzzyLogicController:
    def __init__(self, model_path="models/fuzzy_logic/fuzzy_controller.pkl"):
        self.model_path = model_path

    def train(self):
        """
        Train the fuzzy logic controller for handling ambiguous queries.
        """
        # Define fuzzy variables
        ambiguity = ctrl.Antecedent(np.arange(0, 11, 1), "ambiguity")
        confidence = ctrl.Antecedent(np.arange(0, 11, 1), "confidence")
        domain_routing = ctrl.Consequent(np.arange(0, 11, 1), "domain_routing")

        # Define membership functions
        ambiguity.automf(3)
        confidence.automf(3)
        domain_routing["low"] = fuzz.trimf(domain_routing.universe, [0, 0, 5])
        domain_routing["high"] = fuzz.trimf(domain_routing.universe, [5, 10, 10])

        # Define rules
        rule1 = ctrl.Rule(ambiguity["poor"] & confidence["good"], domain_routing["high"])
        rule2 = ctrl.Rule(ambiguity["good"] & confidence["poor"], domain_routing["low"])
        rule3 = ctrl.Rule(ambiguity["average"] | confidence["average"], domain_routing["high"])

        # Control system and simulation
        routing_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
        routing_sim = ctrl.ControlSystemSimulation(routing_ctrl)

        # Save the controller
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(routing_sim, f)

    def evaluate(self, ambiguity_score, confidence_score):
        """
        Evaluate the routing certainty based on ambiguity and confidence.
        """
        with open(self.model_path, "rb") as f:
            routing_sim = pickle.load(f)

        # Provide inputs
        routing_sim.input["ambiguity"] = ambiguity_score
        routing_sim.input["confidence"] = confidence_score
        routing_sim.compute()

        return routing_sim.output["domain_routing"]
