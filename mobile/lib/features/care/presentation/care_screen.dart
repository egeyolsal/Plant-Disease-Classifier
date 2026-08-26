import 'package:flutter/material.dart';
import '../services/care_repository.dart';

class CareScreen extends StatelessWidget {
  final String label;
  final CareInfo careInfo;

  const CareScreen({super.key, required this.label, required this.careInfo});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(careInfo.title)),
      body: ListView(
        padding: const EdgeInsets.all(16.0),
        children: [
          Text(careInfo.summary, style: Theme.of(context).textTheme.bodyLarge),
          const SizedBox(height: 20),
          Text('Care Tips', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          ...careInfo.careTips.map(
            (tip) => Padding(
              padding: const EdgeInsets.only(bottom: 10),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Icon(Icons.circle, size: 6),
                  const SizedBox(width: 10),
                  Expanded(child: Text(tip)),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.surfaceContainerHighest,
              borderRadius: BorderRadius.circular(8),
            ),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Icon(Icons.info_outline, size: 18),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    careInfo.disclaimer,
                    style: const TextStyle(fontSize: 12, fontStyle: FontStyle.italic),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 8),
          const Text(
            'This app is an educational prototype and does not replace professional agricultural diagnosis.',
            style: TextStyle(fontSize: 11, color: Colors.grey),
          ),
        ],
      ),
    );
  }
}