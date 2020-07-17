# Copyright © 2019 Arm Ltd. All rights reserved.
# Copyright 2020 NXP
# SPDX-License-Identifier: MIT
import os
import stat
import platform

import pytest
import pyarmnn as ann


@pytest.fixture(scope="function")
def get_runtime(shared_data_folder, network_file):
    parser= ann.ITfLiteParser()
    preferred_backends = [ann.BackendId('CpuAcc'), ann.BackendId('CpuRef')]
    network = parser.CreateNetworkFromBinaryFile(os.path.join(shared_data_folder, network_file))
    options = ann.CreationOptions()
    runtime = ann.IRuntime(options)

    yield preferred_backends, network, runtime


@pytest.mark.parametrize("network_file",
                         [
                             'inception_v3_quant.tflite',
                             'ssd_mobilenetv1.tflite'
                         ],
                         ids=['inception v3', 'mobilenetssd v1'])
def test_optimize_executes_successfully(network_file, get_runtime):
    preferred_backends = [ann.BackendId('CpuRef')]
    network = get_runtime[1]
    runtime = get_runtime[2]

    opt_network, messages = ann.Optimize(network, preferred_backends, runtime.GetDeviceSpec(), ann.OptimizerOptions())

    assert len(messages) == 0, 'With only CpuRef, there should be no warnings irrelevant of architecture.'
    assert opt_network


@pytest.mark.parametrize("network_file",
                         [
                             'inception_v3_quant.tflite',
                         ],
                         ids=['inception v3'])
def test_optimize_owned_by_python(network_file, get_runtime):
    preferred_backends = get_runtime[0]
    network = get_runtime[1]
    runtime = get_runtime[2]

    opt_network, _ = ann.Optimize(network, preferred_backends, runtime.GetDeviceSpec(), ann.OptimizerOptions())
    assert opt_network.thisown


@pytest.mark.juno
@pytest.mark.parametrize("network_file",
                         [
                             ('inception_v3_quant.tflite')
                         ],
                         ids=['inception v3'])
def test_optimize_executes_successfully_for_neon_backend_only(network_file, get_runtime):
    preferred_backends = [ann.BackendId('CpuAcc')]
    network = get_runtime[1]
    runtime = get_runtime[2]

    opt_network, messages = ann.Optimize(network, preferred_backends, runtime.GetDeviceSpec(), ann.OptimizerOptions())
    assert 0 == len(messages)
    assert opt_network


@pytest.mark.parametrize("network_file",
                         [
                             'inception_v3_quant.tflite'
                         ],
                         ids=['inception v3'])
def test_optimize_fails_for_invalid_backends(network_file, get_runtime):
    invalid_backends = [ann.BackendId('Unknown')]
    network = get_runtime[1]
    runtime = get_runtime[2]

    with pytest.raises(RuntimeError) as err:
        ann.Optimize(network, invalid_backends, runtime.GetDeviceSpec(), ann.OptimizerOptions())

    expected_error_message = "None of the preferred backends [Unknown ] are supported."
    assert expected_error_message in str(err.value)


@pytest.mark.parametrize("network_file",
                         [
                             'inception_v3_quant.tflite'
                         ],
                         ids=['inception v3'])
def test_optimize_fails_for_no_backends_specified(network_file, get_runtime):
    empty_backends = []
    network = get_runtime[1]
    runtime = get_runtime[2]

    with pytest.raises(RuntimeError) as err:
        ann.Optimize(network, empty_backends, runtime.GetDeviceSpec(), ann.OptimizerOptions())

    expected_error_message = "Invoked Optimize with no backends specified"
    assert expected_error_message in str(err.value)


@pytest.mark.parametrize("network_file",
                         [
                             'inception_v3_quant.tflite'
                         ],
                         ids=['inception v3'])
def test_serialize_to_dot(network_file, get_runtime, tmpdir):
    preferred_backends = get_runtime[0]
    network = get_runtime[1]
    runtime = get_runtime[2]
    opt_network, _ = ann.Optimize(network, preferred_backends,
                                  runtime.GetDeviceSpec(), ann.OptimizerOptions())
    dot_file_path = os.path.join(tmpdir, 'ssd.dot')
    """Check that serialized file does not exist at the start, gets created after SerializeToDot and is not empty"""
    assert not os.path.exists(dot_file_path)
    opt_network.SerializeToDot(dot_file_path)

    assert os.path.exists(dot_file_path)

    with open(dot_file_path) as res_file:
        expected_data = res_file.read()
        assert len(expected_data) > 1
        assert '[label=< [1,299,299,3] >]' in expected_data


@pytest.mark.skipif(platform.processor() != 'x86_64', reason="Platform specific test")
@pytest.mark.parametrize("network_file",
                         [
                             'inception_v3_quant.tflite'
                         ],
                         ids=['inception v3'])
def test_serialize_to_dot_mode_readonly(network_file, get_runtime, tmpdir):
    preferred_backends = get_runtime[0]
    network = get_runtime[1]
    runtime = get_runtime[2]
    opt_network, _ = ann.Optimize(network, preferred_backends,
                                  runtime.GetDeviceSpec(), ann.OptimizerOptions())
    """Create file, write to it and change mode to read-only"""
    dot_file_path = os.path.join(tmpdir, 'ssd.dot')
    f = open(dot_file_path, "w+")
    f.write("test")
    f.close()
    os.chmod(dot_file_path, stat.S_IREAD)
    assert os.path.exists(dot_file_path)

    with pytest.raises(RuntimeError) as err:
        opt_network.SerializeToDot(dot_file_path)

    expected_error_message = "Failed to open dot file"
    assert expected_error_message in str(err.value)


@pytest.mark.juno
@pytest.mark.parametrize("network_file",
                         [
                             'ssd_mobilenetv1.tflite'
                         ],
                         ids=['mobilenetssd v1'])
def test_optimize_error_tuple(network_file, get_runtime):
    preferred_backends = get_runtime[0]
    network = get_runtime[1]
    runtime = get_runtime[2]

    opt_network, error_messages = ann.Optimize(network, preferred_backends,
                                               runtime.GetDeviceSpec(), ann.OptimizerOptions())

    assert type(error_messages) == tuple
    assert 'WARNING: Layer of type DetectionPostProcess is not supported on requested backend CpuAcc for input data ' \
           'type QAsymm8' in error_messages[0]


@pytest.mark.parametrize("method", [
    'AddActivationLayer',
    'AddAdditionLayer',
    'AddBatchNormalizationLayer',
    'AddBatchToSpaceNdLayer',
    'AddConcatLayer',
    'AddConstantLayer',
    'AddConvolution2dLayer',
    'AddDepthwiseConvolution2dLayer',
    'AddDequantizeLayer',
    'AddDetectionPostProcessLayer',
    'AddDivisionLayer',
    'AddFloorLayer',
    'AddFullyConnectedLayer',
    'AddGatherLayer',
    'AddInputLayer',
    'AddL2NormalizationLayer',
    'AddLstmLayer',
    'AddMaximumLayer',
    'AddMeanLayer',
    'AddMergeLayer',
    'AddMinimumLayer',
    'AddMultiplicationLayer',
    'AddNormalizationLayer',
    'AddOutputLayer',
    'AddPadLayer',
    'AddPermuteLayer',
    'AddPooling2dLayer',
    'AddPreluLayer',
    'AddQuantizeLayer',
    'AddQuantizedLstmLayer',
    'AddReshapeLayer',
    'AddResizeLayer',
    'AddRsqrtLayer',
    'AddSoftmaxLayer',
    'AddSpaceToBatchNdLayer',
    'AddSpaceToDepthLayer',
    'AddSplitterLayer',
    'AddStackLayer',
    'AddStridedSliceLayer',
    'AddSubtractionLayer',
    'AddSwitchLayer',
    'AddTransposeConvolution2dLayer'
])
def test_network_method_exists(method):
    assert getattr(ann.INetwork, method, None)


def test_fullyconnected_layer_optional_none():
    net = ann.INetwork()
    layer = net.AddFullyConnectedLayer(fullyConnectedDescriptor=ann.FullyConnectedDescriptor(),
                               weights=ann.ConstTensor())

    assert layer


def test_fullyconnected_layer_optional_provided():
    net = ann.INetwork()
    layer = net.AddFullyConnectedLayer(fullyConnectedDescriptor=ann.FullyConnectedDescriptor(),
                               weights=ann.ConstTensor(),
                               biases=ann.ConstTensor())

    assert layer


def test_fullyconnected_layer_all_args():
    net = ann.INetwork()
    layer = net.AddFullyConnectedLayer(fullyConnectedDescriptor=ann.FullyConnectedDescriptor(),
                                       weights=ann.ConstTensor(),
                                       biases=ann.ConstTensor(),
                                       name='NAME1')

    assert layer
    assert 'NAME1' == layer.GetName()


def test_DepthwiseConvolution2d_layer_optional_none():
    net = ann.INetwork()
    layer = net.AddDepthwiseConvolution2dLayer(convolution2dDescriptor=ann.DepthwiseConvolution2dDescriptor(),
                               weights=ann.ConstTensor())

    assert layer


def test_DepthwiseConvolution2d_layer_optional_provided():
    net = ann.INetwork()
    layer = net.AddDepthwiseConvolution2dLayer(convolution2dDescriptor=ann.DepthwiseConvolution2dDescriptor(),
                               weights=ann.ConstTensor(),
                               biases=ann.ConstTensor())

    assert layer


def test_DepthwiseConvolution2d_layer_all_args():
    net = ann.INetwork()
    layer = net.AddDepthwiseConvolution2dLayer(convolution2dDescriptor=ann.DepthwiseConvolution2dDescriptor(),
                                       weights=ann.ConstTensor(),
                                       biases=ann.ConstTensor(),
                                       name='NAME1')

    assert layer
    assert 'NAME1' == layer.GetName()


def test_Convolution2d_layer_optional_none():
    net = ann.INetwork()
    layer = net.AddConvolution2dLayer(convolution2dDescriptor=ann.Convolution2dDescriptor(),
                               weights=ann.ConstTensor())

    assert layer


def test_Convolution2d_layer_optional_provided():
    net = ann.INetwork()
    layer = net.AddConvolution2dLayer(convolution2dDescriptor=ann.Convolution2dDescriptor(),
                               weights=ann.ConstTensor(),
                               biases=ann.ConstTensor())

    assert layer


def test_Convolution2d_layer_all_args():
    net = ann.INetwork()
    layer = net.AddConvolution2dLayer(convolution2dDescriptor=ann.Convolution2dDescriptor(),
                                       weights=ann.ConstTensor(),
                                       biases=ann.ConstTensor(),
                                       name='NAME1')

    assert layer
    assert 'NAME1' == layer.GetName()
