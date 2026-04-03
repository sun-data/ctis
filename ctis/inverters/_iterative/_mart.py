import dataclasses

@dataclasses.dataclass
class MART(
    ctis.inverters._inverters._iterative.AbstractIterativeMethod,
):

    instrument

    def __call__(
            self,
            data,
            guess,
            gamma,
            niter,
    )->ctis.results.Result:

        while i < niter:
            data_prime = self.project(guess, self.weights_forward)
            c = self.correction(data,data_prime, self.weights_backward)
            guess *= self.back_project(c)

        return ctis.results.Result(
            guess,
        )

    def merit(self):
        """

        Returns
        -------

        """

    def correction(self):
        """

        Returns
        -------

        """