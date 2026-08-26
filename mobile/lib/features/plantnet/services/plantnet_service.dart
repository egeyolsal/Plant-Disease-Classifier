import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class PlantNetPrediction {
  final String scientificName;
  final String? commonName;
  final double score;

  const PlantNetPrediction({
    required this.scientificName,
    this.commonName,
    required this.score,
  });

  String get displayName =>
      commonName != null ? '$scientificName ($commonName)' : scientificName;
}

class PlantNetResult {
  final List<PlantNetPrediction> predictions;
  final String? error;

  const PlantNetResult({this.predictions = const [], this.error});

  bool get isSuccess => error == null && predictions.isNotEmpty;
}

class PlantNetService {
  static const String _apiKey = String.fromEnvironment('PLANTNET_API_KEY');
  static const String _baseUrl = 'https://my-api.plantnet.org/v2/identify/all';

  Future<PlantNetResult> identify(File imageFile) async {
    if (_apiKey.isEmpty) {
      return const PlantNetResult(
        error: 'PlantNet API key not configured.',
      );
    }

    try {
      final uri = Uri.parse('$_baseUrl?api-key=$_apiKey');
      final request = http.MultipartRequest('POST', uri);
      request.files.add(
        await http.MultipartFile.fromPath('images', imageFile.path),
      );
      request.fields['organs'] = 'leaf';

      final streamedResponse =
          await request.send().timeout(const Duration(seconds: 15));
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 404) {
        return const PlantNetResult(
          error: 'PlantNet could not detect a plant in this image.',
        );
      }


      if (response.statusCode != 200) {
        return PlantNetResult(
          error: 'PlantNet API error (${response.statusCode}).',
        );
      }

      final decoded = jsonDecode(response.body) as Map<String, dynamic>;
      final results = decoded['results'] as List<dynamic>?;

      if (results == null || results.isEmpty) {
        return const PlantNetResult(error: 'No match found by PlantNet.');
      }

      final predictions = results.take(3).map((r) {
        final species = r['species'] as Map<String, dynamic>;
        final commonNames = species['commonNames'] as List<dynamic>?;
        return PlantNetPrediction(
          scientificName:
              species['scientificNameWithoutAuthor'] as String? ?? 'Unknown',
          commonName: (commonNames != null && commonNames.isNotEmpty)
              ? commonNames.first as String
              : null,
          score: (r['score'] as num).toDouble(),
        );
      }).toList();

      return PlantNetResult(predictions: predictions);
    } on SocketException {
      return const PlantNetResult(error: 'No internet connection.');
    } catch (e) {
      return PlantNetResult(error: 'PlantNet request failed: $e');
    }
  }
}