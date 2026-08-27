import 'dart:io';
import 'package:flutter/material.dart';
import '../services/image_input_service.dart';
import '../../inference/services/tflite_classifier.dart';
import '../../plantnet/services/plantnet_service.dart';
import '../../care/services/care_repository.dart';
import '../../care/presentation/care_screen.dart';
import '../../../core/theme/app_theme.dart';

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
    final scheme = Theme.of(context).colorScheme;
    final bool isEmptyState = _result == null && !_isClassifying;

    return Scaffold(
      appBar: AppBar(
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.eco, color: scheme.primary),
            const SizedBox(width: 8),
            Text(
              'PlantInsight',
              style: Theme.of(context).textTheme.titleLarge?.copyWith(
                    color: scheme.onSurface,
                  ),
            ),
          ],
        ),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
        child: Column(
          children: [
            _ImagePreview(
              image: _selectedImage,
              expanded: isEmptyState,
            ),
            if (!_isModelReady && _errorMessage == null) ...[
              const SizedBox(height: 16),
              _StatusPill(
                icon: Icons.downloading_outlined,
                label: 'Loading model…',
                showSpinner: true,
              ),
            ],
            if (_isClassifying) ...[
              const SizedBox(height: 16),
              _StatusPill(
                icon: Icons.center_focus_strong_outlined,
                label: 'Analysing leaf…',
                showSpinner: true,
              ),
            ],
            if (_result != null) ...[
              const SizedBox(height: 16),
              _FadeSlideIn(child: _ResultCard(result: _result!)),
            ],
            if (_isCheckingPlantNet) ...[
              const SizedBox(height: 12),
              _StatusPill(
                icon: Icons.public_outlined,
                label: 'Checking PlantNet…',
                showSpinner: true,
              ),
            ],
            if (_plantNetResult != null) ...[
              const SizedBox(height: 12),
              _FadeSlideIn(child: _PlantNetCard(result: _plantNetResult!)),
            ],
            if (_errorMessage != null) ...[
              const SizedBox(height: 16),
              _ErrorBanner(message: _errorMessage!),
            ],
            const SizedBox(height: 24),
            Row(
              children: [
                Expanded(
                  child: FilledButton.icon(
                    onPressed: _isModelReady
                        ? () => _handlePick(ImageSourceType.camera)
                        : null,
                    icon: const Icon(Icons.camera_alt_outlined),
                    label: const Text('Camera'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: _isModelReady
                        ? () => _handlePick(ImageSourceType.gallery)
                        : null,
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

/// Animated leaf-photo area. Fills most of the screen while empty, then
/// collapses to a compact thumbnail once a result is on screen.
class _ImagePreview extends StatelessWidget {
  final File? image;
  final bool expanded;

  const _ImagePreview({required this.image, required this.expanded});

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;

    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      curve: Curves.easeInOut,
      height: expanded ? MediaQuery.of(context).size.height * 0.68 : 220,
      width: double.infinity,
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            scheme.surfaceContainerLow,
            scheme.surfaceContainerHigh,
          ],
        ),
        borderRadius: BorderRadius.circular(28),
        border: Border.all(color: scheme.outlineVariant),
      ),
      clipBehavior: Clip.antiAlias,
      child: image == null
          ? Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Container(
                    padding: const EdgeInsets.all(22),
                    decoration: BoxDecoration(
                      color: scheme.surfaceContainerLowest,
                      shape: BoxShape.circle,
                      border: Border.all(color: scheme.outlineVariant),
                    ),
                    child: Icon(
                      Icons.eco_outlined,
                      size: 56,
                      color: AppTheme.leaf,
                    ),
                  ),
                  const SizedBox(height: 18),
                  Text(
                    'Scan a plant leaf',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                          color: scheme.onSurface,
                        ),
                  ),
                  const SizedBox(height: 6),
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 32),
                    child: Text(
                      'Take or choose a leaf photo to get started',
                      textAlign: TextAlign.center,
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                            color: scheme.onSurfaceVariant,
                          ),
                    ),
                  ),
                ],
              ),
            )
          : Image.file(image!, fit: BoxFit.contain),
    );
  }
}

/// Compact inline progress indicator used for the transient loading states.
class _StatusPill extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool showSpinner;

  const _StatusPill({
    required this.icon,
    required this.label,
    this.showSpinner = false,
  });

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: scheme.surfaceContainerLowest,
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: scheme.outlineVariant),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (showSpinner)
            SizedBox(
              width: 16,
              height: 16,
              child: CircularProgressIndicator(
                strokeWidth: 2,
                color: scheme.primary,
              ),
            )
          else
            Icon(icon, size: 18, color: scheme.primary),
          const SizedBox(width: 10),
          Text(
            label,
            style: Theme.of(context).textTheme.labelLarge?.copyWith(
                  color: scheme.onSurface,
                ),
          ),
        ],
      ),
    );
  }
}

class _ErrorBanner extends StatelessWidget {
  final String message;

  const _ErrorBanner({required this.message});

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: scheme.errorContainer,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.error_outline, size: 20, color: scheme.onErrorContainer),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              message,
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: scheme.onErrorContainer,
                  ),
            ),
          ),
        ],
      ),
    );
  }
}

/// One-shot fade + upward slide, played when the widget first mounts.
/// Purely decorative — it does not gate or delay any behaviour.
class _FadeSlideIn extends StatelessWidget {
  final Widget child;

  const _FadeSlideIn({required this.child});

  @override
  Widget build(BuildContext context) {
    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0, end: 1),
      duration: const Duration(milliseconds: 260),
      curve: Curves.easeOutCubic,
      builder: (context, t, child) => Opacity(
        opacity: t.clamp(0, 1),
        child: Transform.translate(
          offset: Offset(0, (1 - t) * 12),
          child: child,
        ),
      ),
      child: child,
    );
  }
}

/// Small uppercase "eyebrow" label with a leading icon, used to tag each card.
class _SectionLabel extends StatelessWidget {
  final IconData icon;
  final String text;
  final Color color;

  const _SectionLabel({
    required this.icon,
    required this.text,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Icon(icon, size: 16, color: color),
        const SizedBox(width: 8),
        Text(
          text,
          style: Theme.of(context).textTheme.labelMedium?.copyWith(
                color: color,
              ),
        ),
      ],
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
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    final best = result.best;
    final parts = best.label.split('___');
    final species = parts.first.replaceAll('_', ' ');
    final condition = parts.length > 1 ? parts[1].replaceAll('_', ' ') : null;
    final isLowConfidence = best.confidence < _confidenceThreshold;

    return Container(
      decoration: BoxDecoration(
        color: scheme.surfaceContainerLowest,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: scheme.primary.withValues(alpha: 0.35)),
        boxShadow: [
          BoxShadow(
            color: scheme.primary.withValues(alpha: 0.10),
            blurRadius: 24,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _SectionLabel(
              icon: Icons.verified_outlined,
              text: 'ON-DEVICE DIAGNOSIS',
              color: scheme.primary,
            ),
            const SizedBox(height: 14),
            if (isLowConfidence) ...[
              _LowConfidenceNotice(),
              const SizedBox(height: 14),
            ],
            Text(
              species,
              style: theme.textTheme.headlineSmall?.copyWith(
                color: scheme.onSurface,
              ),
            ),
            if (condition != null) ...[
              const SizedBox(height: 6),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: scheme.secondaryContainer,
                  borderRadius: BorderRadius.circular(999),
                ),
                child: Text(
                  condition,
                  style: theme.textTheme.labelLarge?.copyWith(
                    color: scheme.onSecondaryContainer,
                  ),
                ),
              ),
            ],
            const SizedBox(height: 18),
            _ConfidenceBar(
              value: best.confidence,
              lowConfidence: isLowConfidence,
            ),
            const Divider(),
            Text(
              'PREDICTION BREAKDOWN',
              style: theme.textTheme.labelMedium?.copyWith(
                color: scheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 10),
            ...result.top3.map(
              (p) => Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: _BreakdownRow(
                  label: _formatLabel(p.label),
                  value: p.confidence,
                ),
              ),
            ),
            const SizedBox(height: 6),
            FutureBuilder<CareInfo?>(
              future: careRepository.getCareInfo(best.label),
              builder: (context, snapshot) {
                if (snapshot.connectionState != ConnectionState.done) {
                  return const SizedBox.shrink();
                }
                final careInfo = snapshot.data;
                if (careInfo == null) {
                  return Text(
                    'No care guide available yet for this class.',
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: scheme.onSurfaceVariant,
                      fontStyle: FontStyle.italic,
                    ),
                  );
                }
                return SizedBox(
                  width: double.infinity,
                  child: FilledButton.tonalIcon(
                    onPressed: () {
                      Navigator.of(context).push(
                        _fadeThroughRoute(
                          CareScreen(label: best.label, careInfo: careInfo),
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

/// Calm, informative notice for uncertain predictions.
/// The wording is a reviewed statement and must not change.
class _LowConfidenceNotice extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppTheme.noticeContainer,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppTheme.noticeAccent.withValues(alpha: 0.35)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.info_outline,
              size: 18, color: AppTheme.noticeAccent),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              'The model is not fully confident about this prediction. '
              'The options below may help, or try a clearer photo.',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: AppTheme.onNoticeContainer,
                    height: 1.35,
                  ),
            ),
          ),
        ],
      ),
    );
  }
}

class _ConfidenceBar extends StatelessWidget {
  final double value;
  final bool lowConfidence;

  const _ConfidenceBar({required this.value, required this.lowConfidence});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    final Color barColor =
        lowConfidence ? AppTheme.noticeAccent : scheme.primary;
    final pct = (value.clamp(0, 1) * 100);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              'Confidence',
              style: theme.textTheme.labelLarge?.copyWith(
                color: scheme.onSurfaceVariant,
              ),
            ),
            Text(
              '${pct.toStringAsFixed(1)}%',
              style: theme.textTheme.titleMedium?.copyWith(color: barColor),
            ),
          ],
        ),
        const SizedBox(height: 8),
        ClipRRect(
          borderRadius: BorderRadius.circular(999),
          child: TweenAnimationBuilder<double>(
            tween: Tween(begin: 0, end: value.clamp(0, 1).toDouble()),
            duration: const Duration(milliseconds: 500),
            curve: Curves.easeOutCubic,
            builder: (context, t, _) => LinearProgressIndicator(
              value: t,
              minHeight: 10,
              backgroundColor: scheme.surfaceContainerHigh,
              valueColor: AlwaysStoppedAnimation<Color>(barColor),
            ),
          ),
        ),
      ],
    );
  }
}

class _BreakdownRow extends StatelessWidget {
  final String label;
  final double value;

  const _BreakdownRow({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    return Row(
      children: [
        Expanded(
          child: Text(
            label,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: theme.textTheme.bodyMedium?.copyWith(
              color: scheme.onSurfaceVariant,
            ),
          ),
        ),
        const SizedBox(width: 12),
        SizedBox(
          width: 72,
          child: ClipRRect(
            borderRadius: BorderRadius.circular(999),
            child: LinearProgressIndicator(
              value: value.clamp(0, 1).toDouble(),
              minHeight: 6,
              backgroundColor: scheme.surfaceContainerHigh,
              valueColor:
                  AlwaysStoppedAnimation<Color>(scheme.primary.withValues(alpha: 0.55)),
            ),
          ),
        ),
        const SizedBox(width: 10),
        SizedBox(
          width: 48,
          child: Text(
            '${(value * 100).toStringAsFixed(1)}%',
            textAlign: TextAlign.right,
            style: theme.textTheme.bodySmall?.copyWith(
              color: scheme.onSurface,
              fontFeatures: const [FontFeature.tabularFigures()],
            ),
          ),
        ),
      ],
    );
  }
}

class _PlantNetCard extends StatelessWidget {
  final PlantNetResult result;

  const _PlantNetCard({required this.result});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;

    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        color: scheme.surfaceContainerLow,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: scheme.outlineVariant),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _SectionLabel(
              icon: Icons.public_outlined,
              text: 'PLANTNET · SPECIES REFERENCE',
              color: scheme.onSurfaceVariant,
            ),
            const SizedBox(height: 12),
            if (!result.isSuccess)
              Text(
                result.error ?? 'PlantNet comparison unavailable.',
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: scheme.onSurfaceVariant,
                ),
              )
            else
              ...result.predictions.map(
                (p) => Padding(
                  padding: const EdgeInsets.only(bottom: 6),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Expanded(
                        child: Text(
                          p.displayName,
                          style: theme.textTheme.bodyMedium?.copyWith(
                            color: scheme.onSurface,
                          ),
                        ),
                      ),
                      const SizedBox(width: 10),
                      Text(
                        '${(p.score * 100).toStringAsFixed(1)}%',
                        style: theme.textTheme.bodyMedium?.copyWith(
                          color: scheme.onSurfaceVariant,
                          fontFeatures: const [FontFeature.tabularFigures()],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            const SizedBox(height: 10),
            Text(
              'Note: PlantNet identifies species only, not disease. Scores are '
              'not directly comparable to the on-device model above.',
              style: theme.textTheme.bodySmall?.copyWith(
                fontStyle: FontStyle.italic,
                color: scheme.onSurfaceVariant,
                height: 1.35,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Gentle fade-through page transition for pushing detail screens.
Route<T> _fadeThroughRoute<T>(Widget page) {
  return PageRouteBuilder<T>(
    transitionDuration: const Duration(milliseconds: 280),
    reverseTransitionDuration: const Duration(milliseconds: 220),
    pageBuilder: (context, animation, secondaryAnimation) => page,
    transitionsBuilder: (context, animation, secondaryAnimation, child) {
      final curved =
          CurvedAnimation(parent: animation, curve: Curves.easeOutCubic);
      return FadeTransition(
        opacity: curved,
        child: SlideTransition(
          position: Tween<Offset>(
            begin: const Offset(0, 0.03),
            end: Offset.zero,
          ).animate(curved),
          child: child,
        ),
      );
    },
  );
}
