import 'dart:convert';
import 'package:flutter/services.dart';

class CareInfo {
  final String title;
  final String summary;
  final List<String> careTips;
  final String disclaimer;

  const CareInfo({
    required this.title,
    required this.summary,
    required this.careTips,
    required this.disclaimer,
  });

  factory CareInfo.fromJson(Map<String, dynamic> json) {
    return CareInfo(
      title: json['title'] as String,
      summary: json['summary'] as String,
      careTips: List<String>.from(json['care_tips'] as List),
      disclaimer: json['disclaimer'] as String,
    );
  }
}

class CareRepository {
  static const String _dataPath = 'assets/data/care_recommendations.json';

  Map<String, CareInfo>? _cache;

  Future<void> _ensureLoaded() async {
    if (_cache != null) return;
    final jsonString = await rootBundle.loadString(_dataPath);
    final Map<String, dynamic> decoded = jsonDecode(jsonString);
    _cache = decoded.map(
      (key, value) => MapEntry(key, CareInfo.fromJson(value as Map<String, dynamic>)),
    );
  }

  /// Returns care info for the given combined label (e.g. 'Tomato___Early_blight'),
  /// or null if no entry exists for that class.
  Future<CareInfo?> getCareInfo(String label) async {
    await _ensureLoaded();
    return _cache![label];
  }
}