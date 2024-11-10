"""
Handlers for the Falcon models.
"""


from models.model import BaseModelHandler


class FalconHandler(BaseModelHandler):
    """Handler for Falcon models."""

    def setup_model(self):
        """Setup the model based on the specific desirable variant. 
        
        Since multiple Falcon variants come from the same class, we can probably use the same setup_model() function for all of them.

        The desired variant can be extracted from the args object.
        """
        
        self.model = None


class Falcon7BHandler(FalconHandler):

    def get_model_outputs(self, inputs):
        """Get the model outputs.
        
        Args:
            inputs: list of prompts to be passed to the model.

        TODO - we can also shift this method to the parent class if possible.
        """

        outputs = {}
        for item in inputs:
            gen_out = f"This is a dummy output response from the Falcon7B model. The first five words of the input are: {' '.join(item.split()[:5])}."
            outputs[item] = gen_out
        
        return outputs