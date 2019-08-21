from keras.layers import (
    Activation,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Input,
    DepthwiseConv2D,
    add,
    Dropout,
    AveragePooling2D,
    Concatenate,
    LeakyReLU,
)
from keras.models import Model
import keras.backend as K
from keras.engine import Layer, InputSpec
from keras.utils import conv_utils
from keras.backend.common import normalize_data_format

rate_LRelu = 0.01


class BilinearUpsampling(Layer):
    def __init__(self, upsampling=(2, 2), data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.upsampling = conv_utils.normalize_tuple(upsampling, 2, "size")
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        height = (
            self.upsampling[0] * input_shape[1] if input_shape[1] is not None else None
        )
        width = (
            self.upsampling[1] * input_shape[2] if input_shape[2] is not None else None
        )
        return (input_shape[0], height, width, input_shape[3])

    def call(self, inputs):
        return K.tf.image.resize_bilinear(
            inputs,
            (
                int(inputs.shape[1] * self.upsampling[0]),
                int(inputs.shape[2] * self.upsampling[1]),
            ),
        )

    def get_config(self):
        config = {"size": self.upsampling, "data_format": self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def xception_downsample_block(x, channels, is_top_relu=False):
    """
    Xception Downsample block??? built using TensorFlor/Keras Functional API
    :param x:
    :param channels: key parameter that determine how many POINTWISE convolution happens.
    :param is_top_relu:
    :return:
    """
    ##Original Depthwise, Separable ConvStack1:fewer connections, lighter model. I wonder if these can be swapped out for modified version.
    if is_top_relu:
        x = LeakyReLU(alpha=rate_LRelu)(x)
    # DepthwiseConv2D does Conv2D in EACH channel/depth domain (i.e. RGB in color images).
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    # Pointwise Convolution to change the dimension for let's say x*y*5 to x*y*3
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=rate_LRelu)(x)

    ##Depthwise, Separable ConvStack2
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=rate_LRelu)(x)

    ##Depthwise, Separable ConvStack3, with downsampling stride! NOTICE THE STRIDES! This is where the DOWNSAMPLING happens.
    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    return x


def res_xception_downsample_block(x, channels):
    """
    Where Residual block post convolution is COMBINED to the Xception downsample block
    :param x:
    :param channels:
    :return:
    """
    # Residual connections post regular convolution 2d WITH STRIDE (downsampled)
    res = Conv2D(channels, (1, 1), strides=(2, 2), padding="same", use_bias=False)(x)
    res = BatchNormalization()(res)

    # Xception Downsample block (also downsampled stride 2x2 by ONCE
    x = xception_downsample_block(x, channels)

    # Combination of both the residual connection block and the inception downsample block.
    x = add([x, res])

    return x


def xception_block(x, channels):
    """
    Xception block without any downsampling. Notice the LACK of strides in the 3rd separable ConvStack3
    :param x:
    :param channels:
    :return: x underwent THREE times of DepthWiseConv2D at 3x3
    """
    ##Depthwise, Separable ConvStack1
    x = LeakyReLU(alpha=rate_LRelu)(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    ##Depthwise, Separable ConvStack2
    x = LeakyReLU(alpha=rate_LRelu)(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    ##Depthwise, Separable ConvStack3
    x = LeakyReLU(alpha=rate_LRelu)(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    return x


def res_xception_block(x, channels):
    """
    Residual Xception block without downsampling.
    :param x:
    :param channels:
    :return:
    """
    res = x
    x = xception_block(x, channels)
    x = add([x, res])
    return x


def aspp(x, input_shape, out_stride):
    """
    Atrous Spatial Pyramid Pooling: aka Dilated Convolution
    :param x: the input tensor
    :param input_shape: along with out-stride determine the final returned shape.
    :param out_stride: along with input_shape determine the final returned shape.
    :return: x: the output tensor after the series of operations.
    """

    print(f"ASPP b0 Shape {K.shape(x)}")

    # B0 Block: Regular convolution? No Dilation?
    b0 = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    b0 = BatchNormalization()(b0)
    b0 = LeakyReLU(alpha=rate_LRelu)(b0)

    print(f"ASPP b0 Shape {K.shape(b0)}")

    # B1 Block: Convolution with dilation rate of 6
    b1 = DepthwiseConv2D((3, 3), dilation_rate=(6, 6), padding="same", use_bias=False)(
        x
    )
    b1 = BatchNormalization()(b1)
    b1 = LeakyReLU(alpha=rate_LRelu)(b1)
    b1 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b1)
    b1 = BatchNormalization()(b1)
    b1 = LeakyReLU(alpha=rate_LRelu)(b1)

    print(f"ASPP b1 Shape {K.shape(b1)}")

    # B2 Block: Convolution with dilation rate of 12
    b2 = DepthwiseConv2D(
        (3, 3), dilation_rate=(12, 12), padding="same", use_bias=False
    )(x)
    b2 = BatchNormalization()(b2)
    b2 = LeakyReLU(alpha=rate_LRelu)(b2)
    b2 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b2)
    b2 = BatchNormalization()(b2)
    b2 = LeakyReLU(alpha=rate_LRelu)(b2)

    print(f"ASPP b2 Shape {K.shape(b2)}")

    # B3 Block: Convolution with dilation rate of 12 again?? Why not using 18?
    b3 = DepthwiseConv2D(
        (3, 3), dilation_rate=(12, 12), padding="same", use_bias=False
    )(x)
    b3 = BatchNormalization()(b3)
    b3 = LeakyReLU(alpha=rate_LRelu)(b3)
    b3 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b3)
    b3 = BatchNormalization()(b3)
    b3 = LeakyReLU(alpha=rate_LRelu)(b3)

    print(f"ASPP b3 Shape {K.shape(b3)}")

    # B4 block: ???
    out_shape = int(input_shape[0] / out_stride)
    b4 = AveragePooling2D(pool_size=(out_shape, out_shape))(x)
    b4 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b4)
    b4 = BatchNormalization()(b4)
    b4 = LeakyReLU(alpha=rate_LRelu)(b4)
    # Special B4 layers: upsampling TO the final outershape
    b4 = BilinearUpsampling((out_shape, out_shape))(b4)

    print(f"ASPP b4 Shape {K.shape(b4)}")

    # Concatenate and return all. How can they have the same shape?
    x = Concatenate()([b4, b0, b1, b2, b3])
    print(f"ASPP Final X Shape {K.shape(x)}")
    return x


def deeplabv3_plus(input_shape=(512, 512, 3), out_stride=16, num_classes=21):
    """
    The full DeepLabV3 Architectures
    :param input_shape:
    :param out_stride:
    :param num_classes:
    :return:
    """

    #  Obtain the shape of the input (PER image??
    img_input = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), strides=(2, 2), padding="same", use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=rate_LRelu)(x)
    x = Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=rate_LRelu)(x)

    x = res_xception_downsample_block(x, 128)

    res = Conv2D(256, (1, 1), strides=(2, 2), padding="same", use_bias=False)(x)
    res = BatchNormalization()(res)
    x = LeakyReLU(alpha=rate_LRelu)(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=rate_LRelu)(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    skip = BatchNormalization()(x)
    x = LeakyReLU(alpha=rate_LRelu)(skip)
    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = add([x, res])

    x = xception_downsample_block(x, 728, is_top_relu=True)

    for i in range(16):
        x = res_xception_block(x, 728)

    res = Conv2D(1024, (1, 1), padding="same", use_bias=False)(x)
    res = BatchNormalization()(res)
    x = LeakyReLU(alpha=rate_LRelu)(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(728, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=rate_LRelu)(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(1024, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=rate_LRelu)(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(1024, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = add([x, res])

    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(1536, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=rate_LRelu)(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(1536, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=rate_LRelu)(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(2048, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=rate_LRelu)(x)

    # aspp
    x = aspp(x, input_shape, out_stride)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=rate_LRelu)(x)
    x = Dropout(0.9)(x)

    ##decoder
    x = BilinearUpsampling((4, 4))(x)
    dec_skip = Conv2D(48, (1, 1), padding="same", use_bias=False)(skip)
    dec_skip = BatchNormalization()(dec_skip)
    dec_skip = LeakyReLU(alpha=rate_LRelu)(dec_skip)
    x = Concatenate()([x, dec_skip])

    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=rate_LRelu)(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=rate_LRelu)(x)

    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=rate_LRelu)(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=rate_LRelu)(x)

    x = Conv2D(num_classes, (1, 1), padding="same")(x)
    x = BilinearUpsampling((4, 4))(x)
    model = Model(img_input, x)
    return model


if __name__ == "__main__":
    model = deeplabv3_plus(num_classes=1)
    model.summary()
