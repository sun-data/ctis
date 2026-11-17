"""
The convolutional neural network trained by
:class:`~ctis.inverters.NeuralNetworkInverter`.

This module imports :mod:`torch`, which is an optional dependency of
:mod:`ctis`, so it is only imported once a network is built.
"""

import torch

__all__ = [
    "UNet",
]


class _Block(torch.nn.Sequential):
    """
    Two convolutions of a :class:`UNet` at one resolution, each followed by
    a leaky rectifier.
    """

    def __init__(
        self,
        num_input: int,
        num_output: int,
        slope: float,
    ) -> None:
        """
        Parameters
        ----------
        num_input
            The number of channels which the block takes.
        num_output
            The number of channels which the block gives.
        slope
            The slope of the leaky rectifiers for negative values.
        """
        super().__init__(
            torch.nn.Conv3d(num_input, num_output, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(slope),
            torch.nn.Conv3d(num_output, num_output, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(slope),
        )


class UNet(torch.nn.Module):
    """
    A three-dimensional U-Net :cite:p:`Ronneberger2015`, which maps a stack of
    spectral cubes onto another stack of spectral cubes of the same size.

    Each level of the encoder halves the resolution along all three axes, so
    that the deepest level sees the whole field of view of a CTIS, across
    which a channel spreads the light of one voxel when it is backprojected.
    The input is padded with zeros up to a multiple of the factor by which it
    is shrunk, and the padding is cropped from the output.

    The output passes through a softplus, so that the radiance of the
    reconstructed scene is positive.
    """

    def __init__(
        self,
        num_input: int,
        num_output: int,
        num_filter: int,
        num_level: int,
        slope: float = 0.1,
    ) -> None:
        """
        Parameters
        ----------
        num_input
            The number of cubes in the input, one for each channel of the CTIS
            and each spectral line of its scene.
        num_output
            The number of cubes in the output, one for each spectral line of
            the scene.
        num_filter
            The number of filters at the first level, which doubles at each
            level after it.
        num_level
            The number of levels of the encoder, each of which halves the
            resolution.
        slope
            The slope of the leaky rectifiers for negative values.
        """
        super().__init__()

        self.num_input = num_input
        self.num_output = num_output
        self.num_filter = num_filter
        self.num_level = num_level
        self.slope = slope

        width = [num_filter * 2**i for i in range(num_level + 1)]

        self.encoders = torch.nn.ModuleList(
            _Block(num_input if i == 0 else width[i - 1], width[i], slope)
            for i in range(num_level)
        )
        self.bottom = _Block(width[num_level - 1], width[num_level], slope)
        self.upsamplers = torch.nn.ModuleList(
            torch.nn.ConvTranspose3d(width[i + 1], width[i], kernel_size=2, stride=2)
            for i in reversed(range(num_level))
        )
        self.decoders = torch.nn.ModuleList(
            _Block(2 * width[i], width[i], slope) for i in reversed(range(num_level))
        )
        self.head = torch.nn.Conv3d(width[0], num_output, kernel_size=1)

    @property
    def config(self) -> dict[str, int | float]:
        """
        The arguments which build a network of the same shape as this one.
        """
        return dict(
            num_input=self.num_input,
            num_output=self.num_output,
            num_filter=self.num_filter,
            num_level=self.num_level,
            slope=self.slope,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map a batch of input cubes onto a batch of output cubes.

        Parameters
        ----------
        x
            The input cubes, with the axes of the batch, the cubes, and the
            three axes of each cube.
        """
        size = x.shape[2:]
        multiple = 2**self.num_level

        # torch pads the last axis first
        padding = []
        for n in reversed(size):
            padding += [0, -n % multiple]
        x = torch.nn.functional.pad(x, padding)

        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = torch.nn.functional.max_pool3d(x, kernel_size=2)

        x = self.bottom(x)

        for upsampler, decoder, skip in zip(
            self.upsamplers,
            self.decoders,
            reversed(skips),
        ):
            x = upsampler(x)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        x = self.head(x)

        x = x[..., : size[0], : size[1], : size[2]]

        return torch.nn.functional.softplus(x)
