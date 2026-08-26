import 'dart:io';
import 'package:flutter/material.dart';
import '../services/image_input_service.dart';

class ScanScreen extends StatefulWidget {
  const ScanScreen({super.key});

  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen> {
  final ImageInputService _imageInputService = ImageInputService();
  File? _selectedImage;
  String? _errorMessage;

  Future<void> _handlePick(ImageSourceType source) async {
    setState(() => _errorMessage = null);
    final result = await _imageInputService.pickImage(source);

    if (!mounted) return;

    if (result.isSuccess) {
      setState(() => _selectedImage = result.file);
    } else if (result.error != null) {
      setState(() => _errorMessage = result.error);
    }
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
                    onPressed: () => _handlePick(ImageSourceType.camera),
                    icon: const Icon(Icons.camera_alt_outlined),
                    label: const Text('Camera'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () => _handlePick(ImageSourceType.gallery),
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