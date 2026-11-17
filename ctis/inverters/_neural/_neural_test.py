import sys
import pathlib
import subprocess
import dataclasses
import numpy as np
import pytest
import astropy.units as u
import named_arrays as na
import ctis
from ctis.instruments._instruments_test import (
    _instrument_optika_lines,
    _scene_frames,
)
from .._inverters_test import AbstractTestAbstractInverter

torch = pytest.importorskip("torch")

wavelength_rest = 171 * u.AA


def _instrument(num_channel: int) -> ctis.instruments.IdealInstrument:
    """
    A small ideal instrument, which a network is quick to train for.

    Parameters
    ----------
    num_channel
        The number of channels, whose dispersion angles are evenly spaced.
    """
    velocity = na.linspace(-200, 200, axis="wavelength", num=9) * u.km / u.s
    position_scene = na.Cartesian2dVectorLinearSpace(
        start=-5 * u.arcsec,
        stop=5 * u.arcsec,
        axis=na.Cartesian2dVectorArray("scene_x", "scene_y"),
        num=17,
    )
    position_sensor = na.Cartesian2dVectorArray(
        x=na.arange(0, 32 + 1, axis="sensor_x") * u.pix,
        y=na.arange(0, 32 + 1, axis="sensor_y") * u.pix,
    )
    angle = na.linspace(0, 360, num=num_channel, axis="channel", endpoint=False)
    angle = angle * u.deg
    return ctis.instruments.IdealInstrument(
        area_effective=1 * u.cm**2,
        timedelta_exposure=20 * u.s,
        plate_scale=0.625 * u.arcsec / u.pix,
        dispersion=0.03 * u.AA / u.pix,
        angle=angle,
        wavelength_ref=wavelength_rest,
        position_ref=na.Cartesian2dVectorArray(16, 16) * u.pix,
        coordinates_scene=na.DopplerPositionalVectorArray.from_velocity(
            velocity=velocity,
            wavelength_rest=wavelength_rest,
            position=position_scene,
        ),
        coordinates_sensor=na.DopplerPositionalVectorArray.from_velocity(
            velocity=velocity,
            wavelength_rest=wavelength_rest,
            position=position_sensor,
        ),
        channel=angle,
        axis_channel="channel",
        axis_wavelength="wavelength",
        axis_scene_xy=("scene_x", "scene_y"),
        axis_sensor_xy=("sensor_x", "sensor_y"),
    )


instrument = _instrument(num_channel=4)

scene = ctis.scenes.gaussians(instrument.coordinates_scene)
scene = scene + scene.max() / 100

# the same scene, brighter or dimmer in each frame
factor = na.random.uniform(0.5, 2, shape_random=dict(sample=10), seed=0)
scenes = dataclasses.replace(scene, outputs=scene.outputs * factor)

images_training = instrument.image(scenes)

kwargs_train = dict(
    num_epoch=4,
    size_batch=4,
    num_filter=4,
    num_level=2,
    seed=1,
)

inverter = ctis.inverters.NeuralNetworkInverter.train(
    instrument=instrument,
    scenes=scenes,
    images=images_training,
    **kwargs_train,
)

images = instrument.image(scene)

# the same scene in three frames, along an axis other than that of training
images_frames = instrument.image(
    dataclasses.replace(
        scene,
        outputs=scene.outputs * na.ScalarArray(np.array([0.5, 1, 2]), "time"),
    )
)


def _values(
    result: ctis.inverters.AbstractInversionResult,
) -> np.ndarray:
    """
    The values of the solution of an inversion, in the unit of the inverter,
    with the axes of the scene in a fixed order after any others.

    Parameters
    ----------
    result
        The result of an inversion.
    """
    instrument = result.inverter.instrument
    axis = (*instrument.axis_scene_xy, instrument.axis_wavelength)
    outputs = result.solution.outputs
    axis = tuple(ax for ax in outputs.axes if ax not in axis) + axis
    return outputs.ndarray_aligned(axis).to_value(result.inverter.unit)


@pytest.mark.parametrize("a", [inverter])
class TestNeuralNetworkInverter(
    AbstractTestAbstractInverter,
):
    @pytest.mark.parametrize("images", [images, images_frames])
    def test__call__(
        self,
        a: ctis.inverters.NeuralNetworkInverter,
        images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
    ):
        result = super().test__call__(a, images)

        assert isinstance(result, ctis.inverters.InversionResult)
        assert result.success
        assert result.message == ""
        assert result.solution.outputs.unit.is_equivalent(scene.outputs.unit)
        assert np.all(result.solution.outputs >= 0)

        shape = na.shape(result.solution.outputs)
        shape_scene = na.shape(scene.outputs)
        shape_frames = {
            ax: n
            for ax, n in na.shape(images.outputs).items()
            if ax not in instrument._axis_channel
            and ax not in instrument.axis_sensor_xy
        }
        assert shape == shape_frames | shape_scene

    def test_train(self, a: ctis.inverters.NeuralNetworkInverter):
        assert a.unit == scene.outputs.unit
        assert a.size_batch == kwargs_train["size_batch"]

        num_epoch = kwargs_train["num_epoch"]
        for loss in (a.loss_training, a.loss_validation):
            assert loss.shape == {a.axis_epoch: num_epoch}
            assert np.all(np.isfinite(loss))
            assert np.all(loss > 0)

        # the network learns something about these scenes
        loss = a.loss_training
        assert loss[{a.axis_epoch: -1}] < loss[{a.axis_epoch: 0}]

        metadata = a.metadata
        assert metadata["torch"] == torch.__version__
        assert metadata["num_frame_training"] == "8"
        assert metadata["num_frame_validation"] == "2"
        assert metadata["seed"] == str(kwargs_train["seed"])
        assert all(isinstance(v, str) for v in metadata.values())

    def test_frames(self, a: ctis.inverters.NeuralNetworkInverter):
        """
        Each frame is reconstructed as it would be alone.
        """
        result = _values(a(images_frames))
        for i in range(images_frames.outputs.shape["time"]):
            expected = _values(a(images_frames[dict(time=i)]))
            atol = 1e-5 * expected.max()
            assert np.allclose(result[i], expected, rtol=1e-5, atol=atol)

    def test_scale(self, a: ctis.inverters.NeuralNetworkInverter):
        """
        Images which are brighter by some factor give a scene which is
        brighter by the same factor.
        """
        result = _values(a(2 * images))
        expected = 2 * _values(a(images))
        assert np.allclose(result, expected, rtol=1e-5, atol=1e-5 * expected.max())

    def test_save_load(
        self,
        a: ctis.inverters.NeuralNetworkInverter,
        tmp_path: pathlib.Path,
    ):
        path = tmp_path / "network.pt"
        a = dataclasses.replace(a, metadata=a.metadata | dict(scenes="gaussians"))
        a.save(path)

        b = ctis.inverters.NeuralNetworkInverter.load(path, instrument=instrument)

        assert b.instrument is instrument
        assert b.unit == a.unit
        assert b.metadata == a.metadata
        assert all(type(v) is str for v in b.metadata.values())
        assert b.size_batch == a.size_batch
        assert np.all(b.loss_training == a.loss_training)
        assert np.all(b.loss_validation == a.loss_validation)
        assert np.array_equal(_values(b(images)), _values(a(images)))

    def test_save_load_losses(
        self,
        a: ctis.inverters.NeuralNetworkInverter,
        tmp_path: pathlib.Path,
    ):
        """
        An inverter without a record of its training can be saved and loaded.
        """
        path = tmp_path / "network.pt"
        a = dataclasses.replace(a, loss_training=None, loss_validation=None)
        a.save(path)

        b = ctis.inverters.NeuralNetworkInverter.load(path, instrument=instrument)

        assert b.loss_training is None
        assert b.loss_validation is None

    def test_save_metadata(
        self,
        a: ctis.inverters.NeuralNetworkInverter,
        tmp_path: pathlib.Path,
    ):
        """
        Metadata which is not made of strings is refused, since it could not
        be loaded without running code from the file.
        """
        a = dataclasses.replace(a, metadata=a.metadata | dict(seed=1))
        with pytest.raises(TypeError, match="must be strings"):
            a.save(tmp_path / "network.pt")

    def test_save_network(
        self,
        a: ctis.inverters.NeuralNetworkInverter,
        tmp_path: pathlib.Path,
    ):
        """
        Only the network built by `train` can be saved.
        """
        a = dataclasses.replace(a, network=torch.nn.Identity())
        with pytest.raises(TypeError, match="only the network built by `train`"):
            a.save(tmp_path / "network.pt")

    def test_load_instrument(
        self,
        a: ctis.inverters.NeuralNetworkInverter,
        tmp_path: pathlib.Path,
    ):
        """
        A network cannot be loaded for an instrument with another number of
        channels.
        """
        path = tmp_path / "network.pt"
        a.save(path)
        with pytest.raises(ValueError, match="the network takes 4 cubes"):
            ctis.inverters.NeuralNetworkInverter.load(path, instrument=_instrument(3))

    def test_load_version(
        self,
        a: ctis.inverters.NeuralNetworkInverter,
        tmp_path: pathlib.Path,
    ):
        """
        A file in another version of the format is refused.
        """
        path = tmp_path / "network.pt"
        a.save(path)
        saved = torch.load(path, weights_only=True)
        saved["version_format"] = 2
        torch.save(saved, path)
        with pytest.raises(ValueError, match="version 2 of the format"):
            ctis.inverters.NeuralNetworkInverter.load(path, instrument=instrument)

    def test_call_position(self, a: ctis.inverters.NeuralNetworkInverter):
        """
        Images on another grid of the sensor are refused.
        """
        position = images.inputs.position + 1 * u.pix
        images_shifted = dataclasses.replace(
            images,
            inputs=dataclasses.replace(images.inputs, position=position),
        )
        with pytest.raises(ValueError, match="are not equal"):
            a(images_shifted)

    def test_call_dark(self, a: ctis.inverters.NeuralNetworkInverter):
        """
        A frame whose backprojection has no positive mean cannot be scaled
        for the network.
        """
        with pytest.raises(ValueError, match="positive mean"):
            a(0 * images)


@pytest.mark.parametrize("size_chunk", [1, 2])
def test_backprojections_chunks(size_chunk: int):
    """
    Images backprojected a few at a time along their first axis of frames
    give the same cubes, in the same order, as all at once.
    """
    a = ctis.inverters.NeuralNetworkInverter
    factor_time = na.ScalarArray(np.array([0.5, 1, 2]), "time")
    factor_sample = na.ScalarArray(np.array([1, 3]), "sample")
    images_2d = instrument.image(
        dataclasses.replace(
            scene,
            outputs=scene.outputs * factor_time * factor_sample,
        )
    )
    unit = scene.outputs.unit

    axes, x, scale = a._backprojections(instrument, images_2d, unit, size_chunk)
    axes_all, x_all, scale_all = a._backprojections(instrument, images_2d, unit, 100)

    assert axes.batch == dict(time=3, sample=2)
    assert axes == axes_all
    assert np.array_equal(x, x_all)
    assert np.array_equal(scale, scale_all)

    # the flattened batch is in the order of the axes of the batch
    for i, j in np.ndindex(3, 2):
        frame = images_2d[dict(time=i, sample=j)]
        _, x_frame, _ = a._backprojections(instrument, frame, unit, 100)
        assert np.array_equal(x[2 * i + j], x_frame[0])


def test_train_seed():
    """
    The same seed trains the same network, to within the rounding of the
    convolutions, whose order of summation a CPU need not repeat exactly.
    """
    a = ctis.inverters.NeuralNetworkInverter.train(
        instrument=instrument,
        scenes=scenes,
        images=images_training,
        **kwargs_train,
    )
    loss = a.loss_training.ndarray
    loss_expected = inverter.loss_training.ndarray
    assert np.allclose(loss, loss_expected, rtol=1e-5, atol=0)

    result = _values(a(images))
    expected = _values(inverter(images))
    assert np.allclose(result, expected, rtol=1e-4, atol=1e-4 * expected.max())


@pytest.mark.parametrize("fraction_validation", [0, 1, 0.01, 0.99])
def test_train_fraction_validation(fraction_validation: float):
    """
    Some frames must be left for training, and some for validation.
    """
    with pytest.raises(ValueError, match="fraction_validation"):
        ctis.inverters.NeuralNetworkInverter.train(
            instrument=instrument,
            scenes=scenes,
            images=images_training,
            fraction_validation=fraction_validation,
            **kwargs_train,
        )


def test_train_scenes_position():
    """
    Scenes on another grid of the sky are refused.
    """
    position = scenes.inputs.position + 1 * u.arcsec
    scenes_shifted = dataclasses.replace(
        scenes,
        inputs=dataclasses.replace(scenes.inputs, position=position),
    )
    with pytest.raises(ValueError, match="are not equal"):
        ctis.inverters.NeuralNetworkInverter.train(
            instrument=instrument,
            scenes=scenes_shifted,
            images=images_training,
            **kwargs_train,
        )


def test_train_scenes_axes():
    """
    Scenes with an axis which the images do not have are refused, since the
    images of those frames are missing.
    """
    scenes_other = dataclasses.replace(
        scenes,
        outputs=scenes.outputs.add_axes("other"),
    )
    with pytest.raises(ValueError, match="must be among those of the backprojection"):
        ctis.inverters.NeuralNetworkInverter.train(
            instrument=instrument,
            scenes=scenes_other,
            images=images_training,
            **kwargs_train,
        )


def test_lines():
    """
    An instrument whose scene has several spectral lines gives the network
    one cube for each channel and line, and gets back one for each line.
    """
    a = _instrument_optika_lines()
    shape_frames = dict(sample=4)
    scenes_lines = _scene_frames(a, shape_frames=shape_frames)
    images_lines = a.image(scenes_lines, noise=False)

    b = ctis.inverters.NeuralNetworkInverter.train(
        instrument=a,
        scenes=scenes_lines,
        images=images_lines,
        num_epoch=1,
        size_batch=2,
        fraction_validation=0.5,
        num_filter=2,
        num_level=1,
        seed=0,
    )

    num_line = a._shape_line[a.axis_line]
    assert b.network.num_input == a.num_channel * num_line
    assert b.network.num_output == num_line

    result = b(images_lines)

    assert na.shape(result.solution.outputs) == na.shape(scenes_lines.outputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_cuda(tmp_path: pathlib.Path):
    """
    A network trained on a GPU gives the same scenes there as on the host,
    to within the rounding of single precision, after it is saved and loaded
    onto the host.
    """
    a = ctis.inverters.NeuralNetworkInverter.train(
        instrument=instrument,
        scenes=scenes,
        images=images_training,
        device="cuda",
        **kwargs_train,
    )
    assert next(a.network.parameters()).device.type == "cuda"

    path = tmp_path / "network.pt"
    a.save(path)
    b = ctis.inverters.NeuralNetworkInverter.load(path, instrument=instrument)

    assert next(b.network.parameters()).device.type == "cpu"

    result = _values(b(images))
    expected = _values(a(images))
    assert np.allclose(result, expected, rtol=1e-4, atol=1e-4 * expected.max())


def test_import():
    """
    :mod:`ctis` can be imported without importing :mod:`torch`, which is an
    optional dependency.
    """
    code = "import sys, ctis; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_torch_missing(monkeypatch: pytest.MonkeyPatch):
    """
    Training without :mod:`torch` says how to install it.
    """
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match=r"pip install ctis\[torch\]"):
        ctis.inverters.NeuralNetworkInverter.train(
            instrument=instrument,
            scenes=scenes,
            images=images_training,
            **kwargs_train,
        )
