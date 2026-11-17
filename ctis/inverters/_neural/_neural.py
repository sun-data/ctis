from typing import ClassVar, Self, TYPE_CHECKING
import dataclasses
import datetime
import importlib.metadata
import math
import pathlib
import numpy as np
import astropy.units as u
import named_arrays as na
import ctis
from .._inverters import AbstractInverter
from .._results import InversionResult

if TYPE_CHECKING:
    import torch

__all__ = [
    "NeuralNetworkInverter",
]


def _torch():
    """
    Import :mod:`torch`, which is an optional dependency of :mod:`ctis`,
    with a hint on how to install it if it is missing.
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "`NeuralNetworkInverter` needs PyTorch, "
            "which can be installed with `pip install ctis[torch]`."
        ) from e
    return torch


@dataclasses.dataclass
class _Axes:
    """
    The roles which the axes of a backprojection, or of a scene, play for
    the network of a :class:`NeuralNetworkInverter`.
    """

    batch: dict[str, int]
    """
    The axes of the independent frames, and their lengths, which are
    flattened into the batch of the network.
    """

    input: dict[str, int]
    """
    The axes of the channels and of the spectral lines, and their lengths,
    which are flattened into the cubes which the network takes.
    """

    output: dict[str, int]
    """
    The axis of the spectral lines and its length, if the scene has one,
    which is flattened into the cubes which the network gives.
    """

    cube: dict[str, int]
    """
    The axes of each cube, and their lengths: the two axes of position and
    the axis of wavelength of the scene.
    """


@dataclasses.dataclass
class NeuralNetworkInverter(
    AbstractInverter,
):
    """
    An inversion routine which reconstructs a scene with a convolutional
    neural network, trained on scenes whose truth is known.

    The network takes the backprojection of the images through each channel
    of the instrument, :meth:`~ctis.instruments.AbstractInstrument.backproject`,
    which lies on the same grid as the scene, and gives the scene itself.
    It is a three-dimensional U-Net :cite:p:`Ronneberger2015`, which convolves
    along the two axes of position and the axis of wavelength of the scene,
    and halves the resolution along all three at each level, so that its
    deepest level sees the whole field of view.

    Each frame is divided by the mean of its backprojection before it is
    given to the network, and the scene which the network gives is multiplied
    by the same mean, so that a scene which is brighter by some factor is
    reconstructed brighter by the same factor.
    The network still learns the noise of the scenes it was trained on, so
    it is only expected to work on images of similar brightness.

    The network is trained with :meth:`train`, saved with :meth:`save`,
    and loaded again with :meth:`load`.
    It needs :mod:`torch`, which is an optional dependency of :mod:`ctis`,
    and can be installed with ``pip install ctis[torch]``.
    """

    instrument: ctis.instruments.AbstractInstrument = dataclasses.MISSING
    """
    A model of a CTIS instrument which transforms the radiance of an observed
    scene to photons measured by the sensors.
    The network only works for the instrument it was trained on.
    """

    network: "torch.nn.Module" = dataclasses.MISSING
    """
    The trained network, which maps a batch of backprojections onto a batch
    of scenes, each divided by the mean of its backprojection.
    """

    unit: u.UnitBase = dataclasses.MISSING
    """
    The unit of the scenes which the network was trained on, which is the
    unit of the reconstructed scene.
    """

    loss_training: None | na.AbstractScalar = None
    """
    The mean squared error of the scenes which the network was trained on,
    after each epoch of training, along :attr:`axis_epoch`.
    """

    loss_validation: None | na.AbstractScalar = None
    """
    The mean squared error of the scenes which were set aside to validate
    the network, after each epoch of training, along :attr:`axis_epoch`.
    The network kept is the one after the epoch with the least error.
    """

    metadata: dict[str, str] = dataclasses.field(default_factory=dict)
    """
    A record of how the network was made, such as the versions of the
    packages it was trained with, which is saved along with it by
    :meth:`save`.
    """

    size_batch: int = dataclasses.field(default=8, kw_only=True)
    """
    The number of frames which are given to the network at once when a
    scene is reconstructed.
    """

    axis_epoch: ClassVar[str] = "epoch"
    """
    The axis of :attr:`loss_training` and :attr:`loss_validation`.
    """

    version_format: ClassVar[int] = 1
    """
    The version of the format of the files written by :meth:`save`.
    """

    @classmethod
    def _axes(
        cls,
        instrument: ctis.instruments.AbstractInstrument,
        shape: dict[str, int],
    ) -> _Axes:
        """
        The roles which the axes of the backprojection of some images play
        for the network, which the scenes of those images share.

        Parameters
        ----------
        instrument
            The instrument which the network is trained for.
        shape
            The axes of the backprojection and their lengths.
        """
        shape_line = instrument._shape_line

        axis_input = (*instrument._axis_channel, *shape_line)
        axis_cube = (*instrument.axis_scene_xy, instrument.axis_wavelength)

        for ax in (*axis_input, *axis_cube):
            if ax not in shape:
                raise ValueError(
                    f"the backprojection must vary along {ax!r}, "
                    f"but it only has the axes {tuple(shape)}."
                )

        return _Axes(
            batch={
                ax: shape[ax]
                for ax in shape
                if ax not in axis_input and ax not in axis_cube
            },
            input={ax: shape[ax] for ax in axis_input},
            output={ax: shape_line[ax] for ax in shape_line},
            cube={ax: shape[ax] for ax in axis_cube},
        )

    @classmethod
    def _cubes(
        cls,
        a: na.AbstractScalar,
        batch: dict[str, int],
        cubes: dict[str, int],
        cube: dict[str, int],
        unit: u.UnitBase,
    ) -> np.ndarray:
        """
        The values of a stack of spectral cubes as the plain array that the
        network takes or gives: the axes of the batch, the cubes, and the
        three axes of each cube.

        Parameters
        ----------
        a
            The values of the cubes.
        batch
            The axes which are flattened into the batch, and their lengths.
        cubes
            The axes which are flattened into the cubes, and their lengths.
        cube
            The axes of each cube, and their lengths.
        unit
            The unit which the values are given in.
        """
        shape = batch | cubes | cube

        a = na.as_named_array(a).broadcast_to(shape)
        a = a.ndarray_aligned(tuple(shape))
        a = u.Quantity(a, copy=False).to_value(unit)

        num_batch = math.prod(batch.values())
        num_cubes = math.prod(cubes.values())

        return np.ascontiguousarray(
            a.reshape(num_batch, num_cubes, *cube.values()),
            dtype=np.float32,
        )

    @classmethod
    def _backprojections(
        cls,
        instrument: ctis.instruments.AbstractInstrument,
        images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
        unit: u.UnitBase,
    ) -> tuple[_Axes, np.ndarray, np.ndarray]:
        """
        The backprojections of some images, as the network takes them.

        The results are the roles of the axes of the backprojections, the
        backprojections, each divided by its mean, and those means, which
        the scene of each frame is divided by for the network.

        Parameters
        ----------
        instrument
            The instrument which the network is trained for.
        images
            The images to backproject.
        unit
            The unit of the scenes.
        """
        position_images = images.inputs.position
        position_sensor = instrument.coordinates_sensor.position
        if not np.all(position_images == position_sensor):
            raise ValueError(
                "`images.inputs.position` and "
                "`instrument.coordinates_sensor.position` are not equal."
            )

        backprojected = instrument.backproject(images, unit=unit).outputs

        axes = cls._axes(instrument, na.shape(backprojected))

        x = cls._cubes(
            a=backprojected,
            batch=axes.batch,
            cubes=axes.input,
            cube=axes.cube,
            unit=unit,
        )

        scale = x.mean(axis=(1, 2, 3, 4), keepdims=True, dtype=np.float64)

        if np.any(~(scale > 0)):
            raise ValueError(
                "the backprojection of every frame of the images must have a "
                "positive mean, which the network is given each frame "
                "divided by."
            )

        x = (x / scale).astype(np.float32)

        return axes, x, scale

    @classmethod
    def train(
        cls,
        instrument: ctis.instruments.AbstractInstrument,
        scenes: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
        images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
        num_epoch: int = 100,
        size_batch: int = 8,
        learning_rate: float = 1e-3,
        fraction_validation: float = 0.2,
        num_filter: int = 16,
        num_level: int = 3,
        seed: None | int = None,
        device: str = "cpu",
        verbose: bool = False,
    ) -> Self:
        """
        Train a network to reconstruct the given scenes from their images.

        A fraction of the frames, chosen at random, is set aside to validate
        the network, and the network kept is the one after the epoch whose
        error on those frames is the least.

        Parameters
        ----------
        instrument
            The instrument which the network is trained for.
        scenes
            The true scenes, evaluated on the coordinates
            :attr:`~ctis.instruments.AbstractInstrument.coordinates_scene`
            of the instrument, with one or more axes of independent frames.
        images
            The images of `scenes`, as the instrument would observe them,
            for example with its :meth:`~ctis.instruments.AbstractInstrument.image`
            method, which adds noise by default.
            Must be evaluated on the coordinates
            :attr:`~ctis.instruments.AbstractInstrument.coordinates_sensor`
            of the instrument.
        num_epoch
            The number of passes over the frames set aside for training.
        size_batch
            The number of frames in each step of the optimizer, and in each
            reconstruction by the trained inverter.
        learning_rate
            The learning rate of the Adam optimizer :cite:p:`Kingma2014`.
        fraction_validation
            The fraction of the frames which are set aside to validate the
            network, which must leave at least one frame for each.
        num_filter
            The number of filters of the first level of the network, which
            doubles at each level after it.
        num_level
            The number of levels of the network, each of which halves the
            resolution of the scene.
        seed
            The seed of the random numbers which initialize the network and
            choose the frames, for a reproducible training.
            If :obj:`None`, they are different every time.
        device
            The :mod:`torch` device to train the network on,
            such as ``"cpu"`` or ``"cuda"``.
        verbose
            Whether to print the error after each epoch.
        """
        torch = _torch()
        from ._network import UNet

        if not 0 < fraction_validation < 1:
            raise ValueError(
                f"`fraction_validation` must be between 0 and 1, "
                f"got {fraction_validation}."
            )

        position_scenes = scenes.inputs.position
        position_scene = instrument.coordinates_scene.position
        if not np.all(position_scenes == position_scene):
            raise ValueError(
                "`scenes.inputs.position` and "
                "`instrument.coordinates_scene.position` are not equal."
            )

        unit = na.unit(scenes.outputs, unit_dimensionless=u.dimensionless_unscaled)

        axes, x, scale = cls._backprojections(instrument, images, unit)

        if set(na.shape(scenes.outputs)) - set(axes.batch | axes.output | axes.cube):
            raise ValueError(
                f"`scenes` has the axes {tuple(na.shape(scenes.outputs))}, "
                f"which must be among those of the backprojection of `images`, "
                f"{tuple(axes.batch | axes.output | axes.cube)}."
            )

        y = cls._cubes(
            a=scenes.outputs,
            batch=axes.batch,
            cubes=axes.output,
            cube=axes.cube,
            unit=unit,
        )
        y = (y / scale).astype(np.float32)

        num_frame = x.shape[0]
        num_validation = round(fraction_validation * num_frame)
        if not 0 < num_validation < num_frame:
            raise ValueError(
                f"`fraction_validation` of {fraction_validation} sets aside "
                f"{num_validation} of the {num_frame} frames, which must leave "
                f"at least one frame for training and one for validation."
            )

        rng = np.random.default_rng(seed)
        index = rng.permutation(num_frame)
        index_validation = index[:num_validation]
        index_training = index[num_validation:]

        # the network is initialized on the host, from its own random numbers,
        # so that a seed gives the same network on every device
        with torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.manual_seed(seed)
            network = UNet(
                num_input=x.shape[1],
                num_output=y.shape[1],
                num_filter=num_filter,
                num_level=num_level,
            )
        network = network.to(device)

        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

        def loss_mean(index: np.ndarray, step: bool) -> float:
            """
            The mean squared error of some frames, in batches, taking a step
            of the optimizer after each batch if asked to.

            Parameters
            ----------
            index
                The indices of the frames.
            step
                Whether to train the network on these frames.
            """
            network.train(step)
            total = 0.0
            for i in range(0, len(index), size_batch):
                batch = index[i : i + size_batch]
                x_batch = torch.from_numpy(x[batch]).to(device)
                y_batch = torch.from_numpy(y[batch]).to(device)
                with torch.set_grad_enabled(step):
                    loss = torch.nn.functional.mse_loss(network(x_batch), y_batch)
                if step:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total += loss.item() * len(batch)
            return total / len(index)

        loss_training = []
        loss_validation = []
        loss_best = np.inf
        state_best = None

        for epoch in range(num_epoch):
            loss_training.append(loss_mean(rng.permutation(index_training), True))
            loss_validation.append(loss_mean(index_validation, False))

            if verbose:  # pragma: nocover
                print(
                    f"epoch {epoch}: "
                    f"training loss {loss_training[-1]:.4g}, "
                    f"validation loss {loss_validation[-1]:.4g}"
                )

            if loss_validation[-1] < loss_best:
                loss_best = loss_validation[-1]
                state_best = {
                    k: v.detach().clone() for k, v in network.state_dict().items()
                }

        if state_best is not None:
            network.load_state_dict(state_best)
        network.eval()

        metadata = dict(
            date=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            ctis=importlib.metadata.version("ctis"),
            torch=str(torch.__version__),
            num_frame_training=str(len(index_training)),
            num_frame_validation=str(len(index_validation)),
            num_epoch=str(num_epoch),
            size_batch=str(size_batch),
            learning_rate=str(learning_rate),
            seed=str(seed),
        )

        return cls(
            instrument=instrument,
            network=network,
            unit=unit,
            loss_training=na.ScalarArray(np.array(loss_training), cls.axis_epoch),
            loss_validation=na.ScalarArray(np.array(loss_validation), cls.axis_epoch),
            metadata=metadata,
            size_batch=size_batch,
        )

    def __call__(
        self,
        images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
        **kwargs,
    ) -> InversionResult:
        """
        Reconstruct a scene using the observed images.

        Parameters
        ----------
        images
            The observed images used to calculate the reconstruction.
            Must be evaluated on the same position coordinates as
            :attr:`~ctis.instruments.AbstractInstrument.coordinates_sensor`
            attribute of :attr:`instrument`.
            Any axes beyond those of the channels and the sensor are those of
            independent frames, each of which is reconstructed on its own.
        kwargs
            Unused, for compatibility with the other inverters.
        """
        torch = _torch()

        instrument = self.instrument
        network = self.network
        unit = self.unit

        axes, x, scale = self._backprojections(instrument, images, unit)

        device = next(network.parameters()).device

        network.eval()
        y = []
        with torch.no_grad():
            for i in range(0, x.shape[0], self.size_batch):
                x_batch = torch.from_numpy(x[i : i + self.size_batch]).to(device)
                y.append(network(x_batch).cpu().numpy())
        y = np.concatenate(y) * scale

        shape = axes.batch | axes.output | axes.cube

        solution = na.ScalarArray(
            ndarray=y.reshape(tuple(shape.values())) << unit,
            axes=tuple(shape),
        )

        return InversionResult(
            solution=na.FunctionArray(
                inputs=instrument.coordinates_scene,
                outputs=solution,
            ),
            success=True,
            images=images,
            inverter=self,
            message="",
        )

    def save(self, path: str | pathlib.Path) -> None:
        """
        Save the trained network to a file, along with the unit of its scenes,
        its errors during training, and :attr:`metadata`, so that it can be
        loaded again with :meth:`load`.

        The instrument is not saved, and must be given to :meth:`load`.

        The file is written with :func:`torch.save`, and holds only tensors,
        numbers, and strings, so it can be loaded without running any code
        from it.

        Parameters
        ----------
        path
            The file to write.
        """
        torch = _torch()
        from ._network import UNet

        network = self.network
        if not isinstance(network, UNet):
            raise TypeError(
                f"only the network built by `train` can be saved, "
                f"got {type(network)}."
            )

        state = {k: v.detach().cpu() for k, v in network.state_dict().items()}

        # a subclass of a string, such as `torch.__version__`, could not be
        # loaded without running code from the file, so they are copied
        # into plain strings
        metadata = self.metadata
        if not all(isinstance(v, str) for v in (*metadata, *metadata.values())):
            raise TypeError(
                f"the keys and values of `metadata` must be strings, "
                f"got {metadata}."
            )
        metadata = {str(k): str(v) for k, v in metadata.items()}

        def losses(a: None | na.AbstractScalar) -> None | list[float]:
            """
            The errors after each epoch, as a list.

            Parameters
            ----------
            a
                The errors along :attr:`axis_epoch`, if there are any.
            """
            if a is None:
                return None
            return na.as_named_array(a).ndarray_aligned((self.axis_epoch,)).tolist()

        torch.save(
            dict(
                version_format=self.version_format,
                config=network.config,
                state=state,
                unit=self.unit.to_string(),
                loss_training=losses(self.loss_training),
                loss_validation=losses(self.loss_validation),
                metadata=metadata,
                size_batch=self.size_batch,
            ),
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | pathlib.Path,
        instrument: ctis.instruments.AbstractInstrument,
        device: str = "cpu",
    ) -> Self:
        """
        Load a network saved by :meth:`save`.

        Parameters
        ----------
        path
            The file to read.
        instrument
            The instrument which the network was trained for.
            The file does not hold the instrument, so it is only checked to
            have as many channels and spectral lines as the network expects.
        device
            The :mod:`torch` device to load the network onto,
            such as ``"cpu"`` or ``"cuda"``.
        """
        torch = _torch()
        from ._network import UNet

        saved = torch.load(path, map_location=device, weights_only=True)

        version_format = saved["version_format"]
        if version_format != cls.version_format:
            raise ValueError(
                f"the file was written in version {version_format} of the "
                f"format, but this version of ctis reads version "
                f"{cls.version_format}."
            )

        config = saved["config"]

        num_input = instrument.num_channel * math.prod(instrument._shape_line.values())
        num_output = math.prod(instrument._shape_line.values())
        if (config["num_input"], config["num_output"]) != (num_input, num_output):
            raise ValueError(
                f"the network takes {config['num_input']} cubes and gives "
                f"{config['num_output']}, but the instrument has "
                f"{instrument.num_channel} channels and {num_output} spectral "
                f"lines, so it needs one which takes {num_input} and gives "
                f"{num_output}."
            )

        network = UNet(**config)
        network.load_state_dict(saved["state"])
        network = network.to(device)
        network.eval()

        def losses(a: None | list[float]) -> None | na.ScalarArray:
            """
            The errors after each epoch, as an array along :attr:`axis_epoch`.

            Parameters
            ----------
            a
                The errors as a list, if there are any.
            """
            if a is None:
                return None
            return na.ScalarArray(np.array(a), cls.axis_epoch)

        return cls(
            instrument=instrument,
            network=network,
            unit=u.Unit(saved["unit"]),
            loss_training=losses(saved["loss_training"]),
            loss_validation=losses(saved["loss_validation"]),
            metadata=saved["metadata"],
            size_batch=saved["size_batch"],
        )
