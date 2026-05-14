import 'package:flutter_test/flutter_test.dart';
import '../lib/main.dart';

void main() {
  testWidgets('Sonar Retina smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const SonarRetinaApp());

    // Verify that the title 'Sonar Retina' is present on the screen.
    expect(find.text('SONAR RETINA'), findsOneWidget);
    
    // Verify that the 'LISTEN' button exists.
    expect(find.text('LISTEN'), findsOneWidget);
  });
}
