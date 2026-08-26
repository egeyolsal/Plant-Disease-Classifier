import 'dart:io';
import 'dart:convert';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class Prediction {
  final String label;
  final double confidence;

  const Prediction({required this.label, required this.confidence});
}

class ClassificationResult {
  final List<Prediction> top3;

  const ClassificationResult({required this.top3});

  Prediction get best => top3.first;
}

class TFLiteClassifier {
  static const String _modelPath = 'assets/models/mobilenetv3small_dynamic.tflite';
  static const String _labelsPath = 'assets/models/labels.json';
  static const int _inputSize = 224;

  Interpreter? _interpreter;
  List<String>? _labels;

  bool get isReady => _interpreter != null && _labels != null;

  Future<void> loadModel() async {
    _interpreter = await Interpreter.fromAsset(_modelPath);

    final labelsJson = await rootBundle.loadString(_labelsPath);
    final List<dynamic> decoded = jsonDecode(labelsJson);
    _labels = decoded.cast<String>();

    final inputShape = _interpreter!.getInputTensor(0).shape;
    final outputShape = _interpreter!.getOutputTensor(0).shape;

    if (inputShape[1] != _inputSize || inputShape[2] != _inputSize) {
      throw StateError(
        'Model input size ${inputShape[1]}x${inputShape[2]} does not match '
        'expected ${_inputSize}x$_inputSize',
      );
    }
    if (outputShape.last != _labels!.length) {
      throw StateError(
        'Model has ${outputShape.last} output classes but labels.json has '
        '${_labels!.length} entries',
      );
    }
  }

  Future<ClassificationResult> classify(File imageFile) async {
    if (!isReady) {
      throw StateError('Classifier not initialized. Call loadModel() first.');
    }

    final rawBytes = await imageFile.readAsBytes();
    final decoded = img.decodeImage(rawBytes);
    if (decoded == null) {
      throw const FormatException('Could not decode image file.');
    }

    final resized = img.copyResize(
      decoded,
      width: _inputSize,
      height: _inputSize,
      interpolation: img.Interpolation.linear,
    );

    // Raw [0,255] float32 pixels, matching training preprocessing.
    // MobileNetV3's include_preprocessing=True handles normalization
    // internally on the Python side -- do not divide by 255 here.
    final input = List.generate(
      1,
      (_) => List.generate(
        _inputSize,
        (y) => List.generate(
          _inputSize,
          (x) {
            final pixel = resized.getPixel(x, y);
            return [pixel.r.toDouble(), pixel.g.toDouble(), pixel.b.toDouble()];
          },
        ),
      ),
    );

    final outputShape = _interpreter!.getOutputTensor(0).shape;
    final output = List.generate(
      outputShape[0],
      (_) => List.filled(outputShape[1], 0.0),
    );

    _interpreter!.run(input, output);

    final scores = output[0];
    final indexed = List.generate(scores.length, (i) => MapEntry(i, scores[i]));
    indexed.sort((a, b) => b.value.compareTo(a.value));

    final top3 = indexed.take(3).map((entry) {
      return Prediction(label: _labels![entry.key], confidence: entry.value);
    }).toList();

    return ClassificationResult(top3: top3);
  }

  void dispose() {
    _interpreter?.close();
  }
}