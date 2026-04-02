import dataclass

@dataclasses.dataclass
class AbstractIterativeMethod(
    ctis.inverters.AbstractInverter,
):
    "Interface for iterative inversion methods like MART or RL"


    def project(self):
        pass

    def back_project(self):
        pass

    def correction(self):
        pass

    def filter(self):
        pass
