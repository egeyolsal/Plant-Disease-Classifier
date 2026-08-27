import 'dart:io';
import 'package:flutter/material.dart';
import '../services/image_input_service.dart';
import '../../inference/services/tflite_classifier.dart';
import '../../plantnet/services/plantnet_service.dart';
import '../../care/services/care_repository.dart';
import '../../care/presentation/care_screen.dart';

class ScanScreen extends StatefulWidget {
  const ScanScreen({super.key});

  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen> {
  final ImageInputService _imageInputService = ImageInputService();
  final TFLiteClassifier _classifier = TFLiteClassifier();
  final PlantNetService _plantNetService = PlantNetService();

  File? _selectedImage;
  String? _errorMessage;
  bool _isModelReady = false;
  bool _isClassifying = false;
  ClassificationResult? _result;

  bool _isCheckingPlantNet = false;
  PlantNetResult? _plantNetResult;

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
      _plantNetResult = null;
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
      return;
    }

    // PlantNet comparison runs independently; its failure must not affect
    // the local result already shown to the user.
    setState(() => _isCheckingPlantNet = true);
    final plantNet = await _plantNetService.identify(result.file!);
    if (mounted) {
      setState(() {
        _plantNetResult = plantNet;
        _isCheckingPlantNet = false;
      });
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
      appBar: AppBar(
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.eco, color: Theme.of(context).colorScheme.primary),
            const SizedBox(width: 8),
            Text(
              'PlantInsight',
              style: TextStyle(
                fontWeight: FontWeight.w700,
                letterSpacing: -0.5,
                color: Theme.of(context).colorScheme.onSurface,
              ),
            ),
          ],
        ),
        centerTitle: true,
        backgroundColor: Colors.transparent,
        elevation: 0,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            AnimatedContainer(
              duration: const Duration(milliseconds: 300),
              curve: Curves.easeInOut,
              height: (_result == null && !_isClassifying)
                  ? MediaQuery.of(context).size.height * 0.68
                  : 220,
              width: double.infinity,
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.surfaceContainerHighest,
                borderRadius: BorderRadius.circular(20),
              ),
              clipBehavior: Clip.antiAlias,
              child: _selectedImage == null
                  ? Center(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(
                            Icons.eco_outlined,
                            size: 96,
                            color: Theme.of(context).colorScheme.outline,
                          ),
                          const SizedBox(height: 12),
                          Text(
                            'Take or choose a leaf photo to get started',
                            style: TextStyle(
                              color: Theme.of(context).colorScheme.outline,
                              fontSize: 13,
                            ),
                          ),
                        ],
                      ),
                    )
                  : Image.file(_selectedImage!, fit: BoxFit.contain),
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
            if (_isCheckingPlantNet) ...[
              const SizedBox(height: 8),
              const Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  SizedBox(
                    width: 14,
                    height: 14,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  ),
                  SizedBox(width: 8),
                  Text('Checking PlantNet...', style: TextStyle(fontSize: 12)),
                ],
              ),
            ],
            if (_plantNetResult != null) ...[
              const SizedBox(height: 8),
              _PlantNetCard(result: _plantNetResult!),
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
                    onPressed:
                        _isModelReady ? () => _handlePick(ImageSourceType.camera) : null,
                    icon: const Icon(Icons.camera_alt_outlined),
                    label: const Text('Camera'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed:
                        _isModelReady ? () => _handlePick(ImageSourceType.gallery) : null,
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
  final CareRepository careRepository = CareRepository();
  _ResultCard({required this.result});
 
  static const double _confidenceThreshold = 0.45;

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
    final isLowConfidence = best.confidence < _confidenceThreshold;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('PlantInsight (on-device)',
                style: Theme.of(context).textTheme.labelMedium),
            const SizedBox(height: 6),
            if (isLowConfidence)
              Container(
                padding: const EdgeInsets.all(8),
                margin: const EdgeInsets.only(bottom: 8),
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.errorContainer,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Row(
                  children: [
                    Icon(Icons.info_outline,
                        size: 18, color: Theme.of(context).colorScheme.onErrorContainer),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        'The model is not fully confident about this prediction. '
                        'The options below may help, or try a clearer photo.',
                        style: TextStyle(
                          fontSize: 12,
                          color: Theme.of(context).colorScheme.onErrorContainer,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
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
            const SizedBox(height: 8),
            FutureBuilder<CareInfo?>(
              future: careRepository.getCareInfo(best.label),
              builder: (context, snapshot) {
                if (snapshot.connectionState != ConnectionState.done) {
                  return const SizedBox.shrink();
                }
                final careInfo = snapshot.data;
                if (careInfo == null) {
                  return const Text(
                    'No care guide available yet for this class.',
                    style: TextStyle(fontSize: 11, color: Colors.grey, fontStyle: FontStyle.italic),
                  );
                }
                return Align(
                  alignment: Alignment.centerLeft,
                  child: TextButton.icon(
                    onPressed: () {
                      Navigator.of(context).push(
                        MaterialPageRoute(
                          builder: (_) => CareScreen(label: best.label, careInfo: careInfo),
                        ),
                      );
                    },
                    icon: const Icon(Icons.spa_outlined, size: 18),
                    label: const Text('Care Recommendations'),
                  ),
                );
              },
            ),
          ],
        ),
      ),
    );
  }
}

class _PlantNetCard extends StatelessWidget {
  final PlantNetResult result;

  const _PlantNetCard({required this.result});

  @override
  Widget build(BuildContext context) {
    return Card(
      color: Theme.of(context).colorScheme.secondaryContainer,
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('PlantNet (species reference)',
                style: Theme.of(context).textTheme.labelMedium),
            const SizedBox(height: 6),
            if (!result.isSuccess)
              Text(
                result.error ?? 'PlantNet comparison unavailable.',
                style: const TextStyle(fontSize: 12),
              )
            else
              ...result.predictions.map((p) => Padding(
                    padding: const EdgeInsets.only(bottom: 2),
                    child: Text(
                      '${p.displayName}: ${(p.score * 100).toStringAsFixed(1)}%',
                      style: const TextStyle(fontSize: 13),
                    ),
                  )),
            const SizedBox(height: 4),
            Text(
              'Note: PlantNet identifies species only, not disease. Scores are '
              'not directly comparable to the on-device model above.',
              style: TextStyle(
                fontSize: 10,
                fontStyle: FontStyle.italic,
                color: Theme.of(context).colorScheme.onSecondaryContainer,
              ),
            ),
          ],
        ),
      ),
    );
  }
}