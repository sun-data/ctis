import dataclasses
import ctis
import numpy as np

@dataclasses.dataclass
class MART(
    ctis.inverters.AbstractInverter,
):

    instrument: ctis._instruments.AbstractInstrument
    gamma: float
    niter: int

    def __call__(
            self,
            data,
            guess,
    )->ctis.results.Result:

        i=0
        while i < self.niter:
            data_prime = self.instrument.project(guess)
            data_prime[data_prime < 0] = 0

            if not self.converged(data, data_prime):
                c = self.correction(data,data_prime)
                guess *= self.backproject(c)
            i += 1

        if i == self.niter:
            convergence_message = f"Max number of {self.niter} iterations exceeded."
        else:
            convergence_message = "Achieved mean chi squared of less than 1."


        return ctis.results.Result(
            solution = guess,
            convergence_message = convergence_message,
            merit = self.mean_chi_squared(data, data_prime),
        )

    def converged(self, data, data_prime):
        """
        Return true if mean_chi_squared < 1
        """
        return self.mean_chi_squared(data,data_prime) < 1

    def mean_chi_squared(self, data, data_prime):
        """
        Evaluated mean chi_square normalized by uncertainty in each pixel.
        """
        return np.mean((data-data_prime)**2/self.instrument.uncertainty(data_prime))

    def correction(self, data, data_prime):
        """
        Compute multiplicative correction factor.
        """

        return (data_prime / data)