"""
Base class for all models.
"""


class BaseModelHandler:
    """Base handler for every model."""

    def __init__(self, name, args):
        """Initialize the model handler."""
        self.name = name
        self.args = args

        # set up the model
        self.setup_model()

    def setup_model(self):
        """Setup the model."""
        raise NotImplementedError
    
    def get_model_outputs(self, inputs):
        """Get the model output.
        
        Args:
            inputs: list of prompts to be passed to the model.
        
        Use the hyperparameters like temperature from self.args to generate the outputs on the given inputs.
        """
        raise NotImplementedError