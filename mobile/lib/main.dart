import 'package:flutter/material.dart';
import 'core/theme/app_theme.dart';
import 'features/scan/presentation/scan_screen.dart';

void main() {
  runApp(const PlantDoctorApp());
}

class PlantDoctorApp extends StatelessWidget {
  const PlantDoctorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'PlantInsight',
      theme: AppTheme.light,
      debugShowCheckedModeBanner: false,
      home: const ScanScreen(),
    );
  }
}