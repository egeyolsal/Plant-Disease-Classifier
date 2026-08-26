import 'dart:io';
import 'package:flutter/material.dart';
import '../services/image_input_service.dart';
import '../../inference/services/tflite_classifier.dart';

class ScanScreen extends StatefulWidget {
  const ScanScreen({super.key});

  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen> {
  final ImageInputService _imageInputService = ImageInputService();
  final TFLiteClassifier _classifier = TFLiteClassifier();

  File? _selectedImage;
  String? _errorMessage;
  bool _isModelReady = false;
  bool _isClassifying = false;
  ClassificationResult? _result;

  @override
  void initState() {
    super.initState();
    _initModel();
  }

  Future<void> _initModel() async {
    try {
      await _classifier.loadModel();
      if (mounted) setState(() => _isModelReady = true);
    } catch (e) {
      if (mounted) {
        setState(() => _errorMessage = 'Could not load the model: $e');
      }
    }
  }

  Future<void> _handlePick(ImageSourceType source) async {
    setState(() {
      _errorMessage = null;
      _result = null;
    });

    final result = await _imageInputService.pickImage(source);
    if (!mounted) return;

    if (!result.isSuccess) {
      if (result.error != null) setState(() => _errorMessage = result.error);
      return;
    }

    setState(() {
      _selectedImage = result.file;
      _isClassifying = true;
    });

    try {
      final classification = await _classifier.classify(result.file!);
      if (mounted) {
        setState(() {
          _result = classification;
          _isClassifying = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = 'Classification failed: $e';
          _isClassifying = false;
        });
      }
    }
  }

  @override
  void dispose() {
    _classifier.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('PlantInsight')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Expanded(
              child: Container(
                width: double.infinity,
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.surfaceContainerHighest,
                  borderRadius: BorderRadius.circular(16),
                ),
                clipBehavior: Clip.antiAlias,
                child: _selectedImage == null
                    ? Center(
                        child: Icon(
                          Icons.eco_outlined,
                          size: 96,
                          color: Theme.of(context).colorScheme.outline,
                        ),
                      )
                    : Image.file(_selectedImage!, fit: BoxFit.contain),
              ),
            ),
            if (!_isModelReady && _errorMessage == null) ...[
              const SizedBox(height: 12),
              const LinearProgressIndicator(),
              const SizedBox(height: 4),
              const Text('Loading model...', style: TextStyle(fontSize: 12)),
            ],
            if (_isClassifying) ...[
              const SizedBox(height: 12),
              const CircularProgressIndicator(),
            ],
            if (_result != null) ...[
              const SizedBox(height: 12),
              _ResultCard(result: _result!),
            ],
            if (_errorMessage != null) ...[
              const SizedBox(height: 12),
              Text(
                _errorMessage!,
                style: TextStyle(color: Theme.of(context).colorScheme.error),
                textAlign: TextAlign.center,
              ),
            ],
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: FilledButton.icon(
                    onPressed: _isModelReady ? () => _handlePick(ImageSourceType.camera) : null,
                    icon: const Icon(Icons.camera_alt_outlined),
                    label: const Text('Camera'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: _isModelReady ? () => _handlePick(ImageSourceType.gallery) : null,
                    icon: const Icon(Icons.photo_library_outlined),
                    label: const Text('Gallery'),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _ResultCard extends StatelessWidget {
  final ClassificationResult result;

  const _ResultCard({required this.result});

  String _formatLabel(String label) {
  final parts = label.split('___');
  final species = parts.first.replaceAll('_', ' ');
  final condition = parts.length > 1 ? parts[1].replaceAll('_', ' ') : null;
  return condition != null ? '$species — $condition' : species;
  }
  
  @override
  Widget build(BuildContext context) {
    final best = result.best;
    final parts = best.label.split('___');
    final species = parts.first.replaceAll('_', ' ');
    final condition = parts.length > 1 ? parts[1].replaceAll('_', ' ') : null;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(species, style: Theme.of(context).textTheme.titleLarge),
            if (condition != null)
              Text(condition, style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 4),
            Text('Confidence: ${(best.confidence * 100).toStringAsFixed(1)}%'),
            const Divider(),
            ...result.top3.map((p) => Text(
                  '${_formatLabel(p.label)}: ${(p.confidence * 100).toStringAsFixed(1)}%',
                  style: const TextStyle(fontSize: 12),
                )),
          ],
        ),
      ),
    );
  }
}