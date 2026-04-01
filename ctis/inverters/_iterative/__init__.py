import dataclass

@dataclasses.dataclass
class AbstractIterativeMethod(
    ctis.inverters.AbstractInverter,
):
    "Interface for iterative inversion methods like MART or RL"


    def project:
        pass

    def deproject:
        pass

    def merit:
        pass