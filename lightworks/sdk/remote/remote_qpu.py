from ..circuit import Circuit, Parameter
from ..state import State
from ..utils import unpack_circuit_spec

class RemoteQPU:
    
    def __init__(self, qpu, circuit, input_state, n_samples,
                 min_detection_number = 1) -> None:
        # Assign all values to attributes
        self.qpu = qpu
        self.circuit = circuit
        self.input_state = input_state
        self.n_samples = n_samples
        self.min_detection_number = min_detection_number
        
        return
    
    @property
    def qpu(self) -> str:
        """Store the id of the target QPU for the job."""
        return self.__qpu
    
    @qpu.setter
    def qpu(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("Provided qpu id should be a string.")
        self.__qpu = value
        
    @property
    def circuit(self) -> str:
        """Store the circuit to be used for the job."""
        return self.__circuit
    
    @circuit.setter
    def circuit(self, value: str) -> None:
        if not isinstance(value, Circuit):
            raise TypeError("Provided circuit should be a Circuit object.")
        self.__circuit = value
        
    @property
    def input_state(self) -> str:
        """Store the input state to be used for the job."""
        return self.__input_state
    
    @input_state.setter
    def input_state(self, value: str) -> None:
        if not isinstance(value, State):
            raise TypeError("Provided input should be a State object.")
        if len(value) != self.circuit.n_modes:
            msg = f"""Mismatch in mode number between provided circuit 
                      ({self.circuit.n_modes}) and input state ({len(value)}).
                      """
            raise ValueError(" ".join(msg.split()))
        for i in value:
            if i > 1:
                msg = "Inputs must have a max of 1 photons per mode."
                raise ValueError(msg)
        self.__input_state = value
        
    @property
    def n_samples(self) -> str:
        """Store the number of samples to be taken from the QPU."""
        return self.__n_samples
    
    @n_samples.setter
    def n_samples(self, value: str) -> None:
        if not isinstance(value, int):
            raise TypeError("Number of samples should be an integer.")
        if value < 1:
            raise ValueError("Number of samples should be > 0.")
        self.__n_samples = value
        
    @property
    def min_detection_number(self) -> str:
        """Store the minimum number of photons to be detected per sample."""
        return self.__min_detection_number
    
    @min_detection_number.setter
    def min_detection_number(self, value: str) -> None:
        if not isinstance(value, int):
            msg = "Minimum photon detection number should be an integer."
            raise TypeError(msg)
        if value < 1:
            raise ValueError("Minimum photon detection should be at least 1.")
        self.__min_detection_number = value
            
    def submit(self):
        """
        Prepares and packages a job for submission to a QPU.
        """
        # Re-assign input state to ensure circuit and state have the same
        # number of modes
        self.input_state = self.input_state
        # Perform final checks
        if self.min_detection_number > sum(self.input_state):
            msg = """Minimum photon detection cannot be larger than input 
                     photon number."""
            raise ValueError(" ".join(msg.split()))
        # Final check to determine if a circuit is valid before sending
        self.circuit._build()
        # Then process circuit spec before outputting, removing any parameters
        spec = self.circuit._get_circuit_spec()
        spec = unpack_circuit_spec(spec)
        for i in range(len(spec)):
            conv_params = tuple([p.get() if isinstance(p, Parameter) else p 
                                 for p in spec[i][1]])
            spec[i] = [spec[i][0], conv_params]
        # Create dictionary and return
        data = {"qpu" : self.qpu,
                "n_modes" : self.circuit.n_modes,
                "input" : self.input_state.s,
                "n_samples" : self.n_samples,
                "min_detection" : self.min_detection_number,
                "circuit_spec" : spec,
                }
        
        return data