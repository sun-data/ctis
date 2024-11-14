from typing import ClassVar
from typing_extensions import Self
import dataclasses
import time
import pathlib
import numpy as np
import astropy.units as u
from tensorflow import keras
import named_arrays as na
import ctis

__all__ = [
    "NeuralNetworkInverter",
]


@dataclasses.dataclass(eq=False)
class NeuralNetworkInverter(
    ctis.inverters.AbstractInverter,
):
    """
    Invert CTIS observations by training a convolutional neural network (CNN)
    to solve the inversion problem.
    """

    scene: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray]
    """
    The actual physical scenes corresponding to each CTIS observation.
    This is the ground truth that the CNN model is trained to reconstruct.
    """

    observation: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray]
    """
    The CTIS observations corresponding to each physical scene.
    This is the input to the CNN model which is intended to be transformed into
    the reconstructed scene.
    These are expected to be in the form of deprojected CTIS images.
    """

    model: keras.Sequential
    """
    The CNN weights used to evaluate the model.
    """

    loss: na.AbstractScalar
    """
    The RMS difference of the original and reconstructed scenes as a function
    of training epoch.
    """

    axis_sample: str
    """
    The name of the logical axis corresponding to independent observations.
    """

    axis_channel: str
    """
    The name of the logical axis corresponding to different CTIS projections.
    """

    axis_x: str
    """
    The name of the logical axis corresponding to changing horizontal position.
    """

    axis_y: str
    """
    The name of the logical axis corresponding to changing vertical position.
    """

    axis_wavelength: str
    """
    The name of the logical axis corresponding to changing wavelength.
    """

    proportion_training: float = 0.5
    """
    The proportion of `scene` and `observation` used to train the model.
    """

    axis_epoch: ClassVar[str] = "epoch"
    """
    The name of the logical axis corresponding to changing training epoch.
    """

    def __post_init__(self):
        scene = self.scene
        observation = self.observation
        if np.any(scene.inputs.wavelength != observation.inputs.wavelength):
            raise ValueError(
                "`scene` and `observation` must share the same wavelength grid."
            )
        if np.any(scene.inputs.position != observation.inputs.position):
            raise ValueError(
                "`scene` and `observation` must share the same spatial grid."
            )
        # if self.axis_sample in self.scene.inputs.shape:
        #     raise ValueError(
        #         f"{self.scene.inputs.shape=} should not contain {self.axis_sample}"
        #     )
        # if self.axis_sample in self.observation.inputs.shape:
        #     raise ValueError(
        #         f"{self.observation.inputs.shape=} should not contain {self.axis_sample}"
        #     )
        # if self.axis_channel in self.scene.shape:
        #     raise ValueError(
        #         f"{self.scene.shape=} should not contain {self.axis_channel}"
        #     )
        # if self.axis_channel in self.observation.inputs.shape:
        #     raise ValueError(
        #         f"{self.observation.inputs.shape=} should not contain {self.axis_channel}"
        #     )

    @property
    def _axis(self) -> tuple[str, str, str, str, str]:
        """
        A list of the logical axes which controls the ordering of the arrays
        passed into the model.
        """
        return (
            self.axis_sample,
            self.axis_x,
            self.axis_y,
            self.axis_wavelength,
            self.axis_channel,
        )

    def __call__(
        self,
        observation: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
        **kwargs,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray]:

        if np.any(observation.inputs.wavelength != self.observation.inputs.wavelength):
            raise ValueError(
                "`observation` must share the same wavelength grid as `self.observation`."
            )
        if np.any(observation.inputs.position != self.observation.inputs.position):
            raise ValueError(
                "`observation` must share the same spatial grid as `self.observation`."
            )

        axis = self._axis

        observation = self._normalize_observation(observation)

        scene_outputs = self.model.predict(
            x=observation.outputs.ndarray_aligned(axis),
            batch_size=1,
        )

        scene_outputs = na.ScalarArray(scene_outputs, axes=axis)

        scene = dataclasses.replace(observation, outputs=scene_outputs)

        scene = self._normalize_scene(scene, inverse=True)

        return scene

    @classmethod
    def train(
        cls,
        scene: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
        observation: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
        axis_sample: str,
        axis_channel: str,
        axis_x: str,
        axis_y: str,
        axis_wavelength: str,
        proportion_training: float = 0.5,
        epochs: int = 1000,
    ) -> Self:
        """
        Train an instance of this inversion algorithm given an
        array of physical scenes and corresponding CTIS observations.

        Parameters
        ----------
        scene
        observation
        axis_sample
        axis_channel
        axis_x
        axis_y
        axis_wavelength
        proportion_training
        epochs

        """

        self = cls(
            scene=scene,
            observation=observation,
            model=None,
            history=None,
            axis_sample=axis_sample,
            axis_channel=axis_channel,
            axis_x=axis_x,
            axis_y=axis_y,
            axis_wavelength=axis_wavelength,
            proportion_training=proportion_training,
        )

        x_training = self.observation_training
        y_training = self.scene_training
        x_validation = self.observation_validation
        y_validation = self.scene_validation

        x_training = self._normalize_observation(x_training)
        y_training = self._normalize_scene(y_training)
        x_validation = self._normalize_observation(x_validation)
        y_validation = self._normalize_scene(y_validation)

        axis = self._axis
        x_training = x_training.ndarray_aligned(axis)
        y_training = y_training.ndarray_aligned(axis)
        x_validation = x_validation.ndarray_aligned(axis)
        y_validation = y_validation.ndarray_aligned(axis)

        model = self._model_initial(
            n_filters=32,
            kernel_size=11,
            growth_factor=1,
            dropout_rate=0.5,
        )

        tensorboard_dir = pathlib.Path(__file__).parent / "logs"
        callback_tensorboard = keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir / time.strftime("%Y%m%d-%H%M%S"),
            histogram_freq=0,
            write_graph=False,
            write_images=False,
        )

        checkpoint_filepath = pathlib.Path(__file__).parent / "checkpoints"
        callback_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        )

        callback_early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=1000,
            verbose=1,
            restore_best_weights=True,
        )

        kwargs_fit = dict(
            batch_size=4,
            epochs=epochs,
            verbose=2,
            callbacks=[
                callback_tensorboard,
                # callback_checkpoint,
                callback_early_stopping,
            ],
            shuffle=True,
        )

        history = model.fit(
            x=x_training,
            y=y_training,
            validation_data=(x_validation, y_validation),
            **kwargs_fit,
        )

        loss = na.ScalarArray(history.history["val_loss"], self.axis_epoch)

        self.model = model
        self.loss = loss

        return self

    def _model_initial(
        self,
        n_filters: int = 32,
        kernel_size: int = 7,
        growth_factor: int = 2,
        alpha: float = 0.1,
        dropout_rate: float = 0.01,
    ) -> keras.Sequential:

        axis = self._axis

        shape_observation = self.observation.outputs.shape
        shape_input = tuple(
            shape_observation[ax] if ax == self.axis_channel else None
            for ax in axis
            if ax != self.axis_sample
        )

        layers = [
            keras.layers.Conv3D(
                filters=n_filters * growth_factor**0,
                kernel_size=kernel_size * growth_factor**0,
                padding="same",
                input_shape=shape_input,
                activation=keras.layers.LeakyReLU(alpha),
            ),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Conv3D(
                filters=n_filters * growth_factor**1,
                kernel_size=kernel_size * growth_factor**1,
                dilation_rate=2,
                padding="same",
                activation=keras.layers.LeakyReLU(alpha),
            ),
            keras.layers.Dropout(dropout_rate / 2),
            keras.layers.Conv3D(
                filters=n_filters * growth_factor**1,
                kernel_size=kernel_size * growth_factor**1,
                dilation_rate=1,
                padding="same",
                activation=keras.layers.LeakyReLU(alpha),
            ),
            keras.layers.Dropout(dropout_rate / 4),
            keras.layers.Conv3DTranspose(
                filters=n_filters * growth_factor**1,
                kernel_size=kernel_size * growth_factor**1,
                dilation_rate=1,
                padding="same",
                activation=keras.layers.LeakyReLU(alpha),
            ),
            keras.layers.Dropout(dropout_rate / 8),
            keras.layers.Conv3DTranspose(
                filters=n_filters * growth_factor**1,
                kernel_size=kernel_size * growth_factor**1,
                dilation_rate=2,
                padding="same",
                activation=keras.layers.LeakyReLU(alpha),
            ),
            keras.layers.Conv3DTranspose(
                filters=1,
                kernel_size=kernel_size * growth_factor**0,
                padding="same",
                kernel_initializer="zeros",
            ),
        ]

        net = keras.Sequential(layers=layers)

        net.compile(
            optimizer=keras.optimizers.Nadam(learning_rate=1e-5, clipvalue=0.5),
            loss="mse",
        )

        return net

    @classmethod
    def _normalize(
        cls,
        a: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
        shift: float | u.Quantity | na.AbstractScalarArray,
        scale: float | u.Quantity | na.AbstractScalarArray,
        inverse: bool = False,
    ):
        """
        Normalize or denormalize a spatial-spectral cube using the
        given scale and shift.

        Parameters
        ----------
        a
            The spatial-spectral cube to transform.
        shift
            The additive shift to apply to the values.
        scale
            The multiplicative scale to apply to the values.
        inverse
            A boolean flag indicating whether to normalize or denormalize.
        """
        if not inverse:
            return dataclasses.replace(a, outputs=(a.outputs - shift) / scale)
        else:
            return dataclasses.replace(a, outputs=a.outputs * scale + shift)

    def _normalize_scene(
        self,
        scene: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
        inverse: bool = False,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray]:
        return self._normalize(
            a=scene,
            shift=0,
            scale=self.scene.outputs.std(),
            inverse=inverse,
        )

    def _normalize_observation(
        self,
        observation: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
        inverse: bool = False,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray]:
        return self._normalize(
            a=observation,
            shift=0,
            scale=self.observation.outputs.std(),
            inverse=inverse,
        )

    @classmethod
    def _training(
        cls,
        a: na.AbstractArray,
        axis: str,
        proportion_training: float,
    ) -> na.AbstractArray:
        num = a.shape[axis]
        index = int(proportion_training * num)
        return a[{axis: slice(None, index)}]

    @classmethod
    def _validation(
        cls,
        a: na.AbstractArray,
        axis: str,
        proportion_training: float,
    ) -> na.AbstractArray:
        num = a.shape[axis]
        index = int(proportion_training * num)
        return a[{axis: slice(index, None)}]

    @property
    def scene_training(
        self,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray]:
        """
        The scenes used for training the model.
        """
        return dataclasses.replace(
            self.scene,
            outputs=self._training(
                a=self.scene.outputs,
                axis=self.axis_sample,
                proportion_training=self.proportion_training,
            ),
        )

    @property
    def scene_validation(
        self,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray]:
        """
        The scenes used for validating the model.
        """
        return dataclasses.replace(
            self.scene,
            outputs=self._validation(
                a=self.scene.outputs,
                axis=self.axis_sample,
                proportion_training=self.proportion_training,
            ),
        )

    @property
    def observation_training(
        self,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray]:
        """
        The observations used for training the model.
        """
        return dataclasses.replace(
            self.observation,
            outputs=self._training(
                a=self.observation.outputs,
                axis=self.axis_sample,
                proportion_training=self.proportion_training,
            ),
        )

    @property
    def observation_validation(
        self,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray]:
        """
        The observations used for validating the model.
        """
        return dataclasses.replace(
            self.observation,
            outputs=self._validation(
                a=self.observation.outputs,
                axis=self.axis_sample,
                proportion_training=self.proportion_training,
            ),
        )
