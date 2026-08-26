import 'dart:io';
import 'package:image_picker/image_picker.dart';

enum ImageSourceType { camera, gallery }

class ImageInputResult {
  final File? file;
  final String? error;

  const ImageInputResult({this.file, this.error});

  bool get isSuccess => file != null;
}

class ImageInputService {
  final ImagePicker _picker = ImagePicker();

  Future<ImageInputResult> pickImage(ImageSourceType source) async {
    try {
      final XFile? picked = await _picker.pickImage(
        source: source == ImageSourceType.camera
            ? ImageSource.camera
            : ImageSource.gallery,
        maxWidth: 1024,
        imageQuality: 90,
      );

      if (picked == null) {
        return const ImageInputResult(error: null); // user cancelled, not an error
      }

      final file = File(picked.path);
      if (!await file.exists()) {
        return const ImageInputResult(error: 'Selected file could not be read.');
      }

      return ImageInputResult(file: file);
    } catch (e) {
      return ImageInputResult(error: 'Could not access camera/gallery: $e');
    }
  }
}