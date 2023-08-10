import bilby
import numpy as np
import bilby
import numpy as np
class DiscreteUniform(bilby.prior.Uniform):

    def __init__(self, minimum, maximum, name=None, latex_label=None,
                 unit=None, boundary=None):
        """Log-Uniform prior with bounds
        Parameters
        ==========
        minimum: int
            See superclass
        maximum: int
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        if not isinstance(minimum, int) or not isinstance(maximum, int):
            raise ValueError("minimum/maximum must be integers")
            
        super(DiscreteUniform, self).__init__(name=name, latex_label=latex_label, unit=unit,
                                         minimum=minimum, maximum=maximum+1, boundary=boundary)
        
        
    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the power-law prior.
        This maps to the inverse CDF. This has been analytically solved for this case.
        Parameters
        ==========
        val: Union[float, int, array_like]
            Uniform probability
        Returns
        =======
        Union[float, array_like]: Rescaled probability
        """
        value = np.floor(self.minimum + val * (self.maximum  - self.minimum))
        value = np.where(value == 0, np.floor(np.random.uniform(low=self.minimum, high=self.maximum, size=1))[0], value)
        
            
        return value
