import 'package:flutter/material.dart';

/// Central design system for PlantInsight.
///
/// Material 3 based, built around a confident deep-green palette with
/// high-contrast text so results stay readable outdoors in bright light.
class AppTheme {
  AppTheme._();

  // Brand + semantic colors that sit outside the generated ColorScheme.
  static const Color _seed = Color(0xFF2E7D32); // green, fits a plant app
  static const Color brandGreen = Color(0xFF1E6B2E);
  static const Color leaf = Color(0xFF4C9A51);

  /// Calm, informative "attention" palette for the low-confidence notice.
  /// Deliberately not red — it flags uncertainty, it does not report failure.
  static const Color noticeContainer = Color(0xFFFFF3D6);
  static const Color onNoticeContainer = Color(0xFF4A3300);
  static const Color noticeAccent = Color(0xFF9A6B00);

  static const double _radius = 20;

  static ThemeData get light {
    final scheme = ColorScheme.fromSeed(
      seedColor: _seed,
      brightness: Brightness.light,
    ).copyWith(
      primary: brandGreen,
      surface: const Color(0xFFFBFDF7),
      surfaceContainerLowest: Colors.white,
      surfaceContainerLow: const Color(0xFFF3F6EE),
      surfaceContainer: const Color(0xFFEDF2E7),
      surfaceContainerHigh: const Color(0xFFE7EEE0),
      surfaceContainerHighest: const Color(0xFFE1EAD9),
      secondaryContainer: const Color(0xFFDCEBD8),
      onSecondaryContainer: const Color(0xFF243424),
      outlineVariant: const Color(0xFFC7D4C1),
    );

    final baseText = Typography.material2021(colorScheme: scheme).black;
    final textTheme = baseText.copyWith(
      displaySmall: baseText.displaySmall?.copyWith(
        fontWeight: FontWeight.w700,
        letterSpacing: -0.5,
      ),
      headlineSmall: baseText.headlineSmall?.copyWith(
        fontWeight: FontWeight.w700,
        letterSpacing: -0.4,
      ),
      titleLarge: baseText.titleLarge?.copyWith(
        fontWeight: FontWeight.w700,
        letterSpacing: -0.3,
      ),
      titleMedium: baseText.titleMedium?.copyWith(
        fontWeight: FontWeight.w600,
        letterSpacing: -0.2,
      ),
      labelLarge: baseText.labelLarge?.copyWith(
        fontWeight: FontWeight.w600,
        letterSpacing: 0.1,
      ),
      labelMedium: baseText.labelMedium?.copyWith(
        fontWeight: FontWeight.w700,
        letterSpacing: 0.8,
      ),
    );

    return ThemeData(
      useMaterial3: true,
      colorScheme: scheme,
      scaffoldBackgroundColor: const Color(0xFFF6F8F1),
      textTheme: textTheme,
      appBarTheme: const AppBarTheme(
        centerTitle: true,
        elevation: 0,
        scrolledUnderElevation: 0,
        backgroundColor: Colors.transparent,
        surfaceTintColor: Colors.transparent,
      ),
      cardTheme: CardThemeData(
        elevation: 0,
        margin: EdgeInsets.zero,
        clipBehavior: Clip.antiAlias,
        color: scheme.surfaceContainerLowest,
        surfaceTintColor: Colors.transparent,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(_radius),
        ),
      ),
      dividerTheme: DividerThemeData(
        color: scheme.outlineVariant,
        thickness: 1,
        space: 24,
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          minimumSize: const Size(0, 54),
          padding: const EdgeInsets.symmetric(horizontal: 20),
          textStyle: textTheme.labelLarge,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          minimumSize: const Size(0, 54),
          padding: const EdgeInsets.symmetric(horizontal: 20),
          textStyle: textTheme.labelLarge,
          foregroundColor: brandGreen,
          side: BorderSide(color: scheme.outlineVariant),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
        ),
      ),
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(
          foregroundColor: brandGreen,
          textStyle: textTheme.labelLarge,
        ),
      ),
      progressIndicatorTheme: ProgressIndicatorThemeData(
        color: brandGreen,
        linearTrackColor: scheme.surfaceContainerHigh,
        linearMinHeight: 6,
      ),
      snackBarTheme: const SnackBarThemeData(
        behavior: SnackBarBehavior.floating,
      ),
    );
  }
}
