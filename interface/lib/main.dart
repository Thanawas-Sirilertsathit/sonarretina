import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:record/record.dart';

// ── Config ────────────────────────────────────────────────────────────────
const String _apiBase = 'http://10.0.2.2:8000'; // Change for your env
const int _sampleRate = 16000;
const int _chunkMs = 400; // <500ms latency requirement
const int _modelChannels = 4;

void main() => runApp(const SonarRetinaApp());

// ── Models ────────────────────────────────────────────────────────────────
class DetectionEvent {
  final String id;
  final double distance;
  final double confidence;
  final DateTime timestamp;
  int? feedback; // null: none, 1: good, -1: bad

  // Persist text input state across widget rebuilds
  final TextEditingController textController = TextEditingController();

  DetectionEvent({
    required this.id,
    required this.distance,
    required this.confidence,
    required this.timestamp,
    this.feedback,
  });

  String get rangeLabel {
    if (distance <= 2.5) return 'near';
    if (distance <= 5.0) return 'mid-range';
    return 'far';
  }

  void dispose() {
    textController.dispose();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
class SonarRetinaApp extends StatelessWidget {
  const SonarRetinaApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'SonarRetina',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          brightness: Brightness.light,
          scaffoldBackgroundColor: Colors.white,
          fontFamily: 'Montserrat', // Clean sans-serif
          colorScheme: const ColorScheme.light(primary: Color(0xFF00D4FF)),
          useMaterial3: true,
        ),
        home: const SonarScreen(),
      );
}

// ─────────────────────────────────────────────────────────────────────────────
class SonarScreen extends StatefulWidget {
  const SonarScreen({super.key});
  @override
  State<SonarScreen> createState() => _SonarScreenState();
}

class _SonarScreenState extends State<SonarScreen>
    with TickerProviderStateMixin {
  final AudioRecorder _recorder = AudioRecorder();

  bool _isListening = false;
  bool _isSending = false;
  String _error = '';

  // List to stack multiple detections
  List<DetectionEvent> _detections = [];

  late AnimationController _sweepCtrl;

  @override
  void initState() {
    super.initState();
    _sweepCtrl =
        AnimationController(vsync: this, duration: const Duration(seconds: 2));
  }

  @override
  void dispose() {
    _isListening = false;
    _sweepCtrl.dispose();
    _recorder.dispose();
    for (var d in _detections) {
      d.dispose();
    }
    super.dispose();
  }

  // ── WAV helpers ──────────────────────────────────────────────────────────

  Uint8List _expandChannels(Uint8List mono, int n) {
    if (n == 1) return mono;
    final samples = mono.length ~/ 2;
    final out = ByteData(samples * 2 * n);
    final inView = mono.buffer.asByteData();
    for (var i = 0; i < samples; i++) {
      final s = inView.getInt16(i * 2, Endian.little);
      for (var ch = 0; ch < n; ch++) {
        out.setInt16((i * n + ch) * 2, s, Endian.little);
      }
    }
    return out.buffer.asUint8List();
  }

  Uint8List _buildWav(Uint8List pcm, int sr, int ch) {
    const bps = 16;
    final byteRate = sr * ch * bps ~/ 8;
    final blockAlign = ch * bps ~/ 8;
    final hdr = ByteData(44);
    void str(int o, String s) {
      for (var i = 0; i < s.length; i++) hdr.setUint8(o + i, s.codeUnitAt(i));
    }

    str(0, 'RIFF');
    hdr.setUint32(4, 36 + pcm.length, Endian.little);
    str(8, 'WAVE');
    str(12, 'fmt ');
    hdr.setUint32(16, 16, Endian.little);
    hdr.setUint16(20, 1, Endian.little); // PCM
    hdr.setUint16(22, ch, Endian.little);
    hdr.setUint32(24, sr, Endian.little);
    hdr.setUint32(28, byteRate, Endian.little);
    hdr.setUint16(32, blockAlign, Endian.little);
    hdr.setUint16(34, bps, Endian.little);
    str(36, 'data');
    hdr.setUint32(40, pcm.length, Endian.little);
    return Uint8List.fromList([...hdr.buffer.asUint8List(), ...pcm]);
  }

  // ── Recording loop ───────────────────────────────────────────────────────

  Future<void> _startListening() async {
    final ok = await _recorder.hasPermission();
    if (!ok) {
      setState(() => _error = 'Microphone permission denied.');
      return;
    }
    setState(() {
      _isListening = true;
      _error = '';
    });
    _sweepCtrl.repeat();
    _loop(); // fire-and-forget
  }

  Future<void> _loop() async {
    while (_isListening) {
      await _runChunk();
    }
  }

  Future<void> _runChunk() async {
    if (!_isListening) return;
    try {
      final stream = await _recorder.startStream(const RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: _sampleRate,
        numChannels: 1,
      ));

      final bytes = <int>[];
      final done = Completer<void>();
      final sub = stream.listen(
        bytes.addAll,
        onDone: done.complete,
        onError: (e) => done.completeError(e),
      );

      await Future.delayed(const Duration(milliseconds: _chunkMs));
      await _recorder.stop();
      await done.future.timeout(const Duration(seconds: 3));
      await sub.cancel();

      if (!_isListening || bytes.isEmpty) return;

      final mono = Uint8List.fromList(bytes);
      final multi = _expandChannels(mono, _modelChannels);
      final wav = _buildWav(multi, _sampleRate, _modelChannels);

      setState(() => _isSending = true);
      await _predict(wav);
      setState(() => _isSending = false);
    } catch (e) {
      setState(() {
        _error = 'Recording error: $e';
        _isSending = false;
      });
    }
  }

  void _stopListening() {
    _isListening = false;
    _recorder.stop();
    _sweepCtrl.stop();
    setState(() {
      _isSending = false;
    });
  }

  void _toggle() => _isListening ? _stopListening() : _startListening();

  void _cleanupOldDetections() {
    final now = DateTime.now();
    setState(() {
      _detections.removeWhere((d) {
        final shouldRemove =
            d.feedback != -1 && now.difference(d.timestamp).inSeconds >= 3;
        if (shouldRemove) {
          d.dispose();
        }
        return shouldRemove;
      });
    });
  }

  // ── API call ─────────────────────────────────────────────────────────────

  Future<void> _predict(Uint8List wav) async {
    try {
      final req = http.MultipartRequest('POST', Uri.parse('$_apiBase/predict'))
        ..files.add(http.MultipartFile.fromBytes(
          'audio_file',
          wav,
          filename: 'chunk.wav',
        ));
      final res = await req.send().timeout(const Duration(seconds: 10));
      final body = await res.stream.bytesToString();

      if (res.statusCode == 200) {
        final d = jsonDecode(body) as Map<String, dynamic>;
        final dist = (d['predicted_distance'] as num).toDouble();
        final conf = (d['confidence_score'] as num?)?.toDouble() ?? 0.0;

        final newDetection = DetectionEvent(
          id: UniqueKey().toString(),
          distance: dist,
          confidence: conf,
          timestamp: DateTime.now(),
        );

        setState(() {
          _detections.insert(0, newDetection); // Stack newest at the top
          _error = '';
        });

        // Trigger cleanup timer for this specific event
        Timer(const Duration(seconds: 3), () {
          if (mounted) _cleanupOldDetections();
        });
      } else {
        setState(() => _error = 'API ${res.statusCode}: $body');
      }
    } on TimeoutException {
      setState(() => _error = 'Request timed out');
    } catch (e) {
      setState(() => _error = 'Send error: $e');
    }
  }

  // ── UI ───────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            _buildHeader(),
            const SizedBox(height: 10),
            Expanded(
              child: SingleChildScrollView(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    _buildRadar(),
                    const SizedBox(height: 30),
                    _buildListenButton(),
                    _buildDetectedSoundsList(),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Padding(
      padding: const EdgeInsets.only(top: 24, bottom: 16),
      child: Center(
        child: Column(
          children: [
            RichText(
              text: const TextSpan(
                style: TextStyle(
                  fontSize: 34,
                  fontWeight: FontWeight.w900,
                  letterSpacing: 1.5,
                ),
                children: [
                  TextSpan(
                    text: 'SONAR ',
                    style: TextStyle(color: Color(0xFF0F4C68)),
                  ),
                  TextSpan(
                    text: 'RETINA',
                    style: TextStyle(color: Color(0xFF00D4FF)),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 4),
            const Text(
              'Hear the world around you',
              style: TextStyle(
                color: Color(0xFF0F4C68),
                fontSize: 14,
                fontWeight: FontWeight.w700,
                letterSpacing: 0.5,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildRadar() {
    final double radarSize = MediaQuery.of(context).size.width * 0.8;
    final double centerOffset = radarSize / 2;

    return Stack(
      alignment: Alignment.center,
      clipBehavior: Clip.none,
      children: [
        // Radar Rings and Cross
        SizedBox(
          width: radarSize,
          height: radarSize,
          child: CustomPaint(
            painter: _RadarPainter(),
          ),
        ),

        // N, E, S, W Labels
        Positioned(top: -28, child: _radarLabel('N')),
        Positioned(bottom: -28, child: _radarLabel('S')),
        Positioned(right: -28, child: _radarLabel('E')),
        Positioned(left: -28, child: _radarLabel('W')),

        // Sweep animation
        if (_isListening)
          SizedBox(
            width: radarSize,
            height: radarSize,
            child: RotationTransition(
              turns: _sweepCtrl,
              child: CustomPaint(
                painter: _RadarSweepPainter(),
              ),
            ),
          ),

        // Detected Object Circles (Draw all active ones)
        ..._detections.map((det) {
          final double maxDist = 10.0;
          final double distRatio = (det.distance / maxDist).clamp(0.15, 1.0);
          final double radius = (radarSize / 2) * distRatio;

          return Positioned(
            left: centerOffset - radius,
            top: centerOffset - radius,
            child: Container(
              width: radius * 2,
              height: radius * 2,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                border: Border.all(
                  color: const Color(0xFF425664)
                      .withOpacity(0.6), // Slightly transparent dark grey
                  width: 3.0,
                ),
              ),
            ),
          );
        }),

        // Detected Object Distance Labels
        ..._detections.map((det) {
          final double maxDist = 10.0;
          final double distRatio = (det.distance / maxDist).clamp(0.15, 1.0);
          final double radius = (radarSize / 2) * distRatio;

          return Positioned(
            top: centerOffset - radius - 10,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 2),
              decoration: BoxDecoration(
                color: const Color(0xFF0D1B2A),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Text(
                '${det.distance.toStringAsFixed(1)}m',
                style: const TextStyle(
                    color: Colors.white,
                    fontSize: 10,
                    fontWeight: FontWeight.bold),
              ),
            ),
          );
        }),

        // Center "You" Icon
        Container(
          width: 60,
          height: 60,
          decoration: BoxDecoration(
            color: const Color(0xFFE6F9FC), // Very light cyan
            shape: BoxShape.circle,
            border: Border.all(color: const Color(0xFF00D4FF), width: 2),
            boxShadow: [
              BoxShadow(
                color: const Color(0xFF00D4FF).withOpacity(0.3),
                blurRadius: 10,
                spreadRadius: 2,
              )
            ],
          ),
          child: const Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.person_outline, color: Color(0xFF00D4FF), size: 30),
              Text('You',
                  style: TextStyle(
                      color: Color(0xFF00D4FF),
                      fontSize: 11,
                      fontWeight: FontWeight.bold)),
            ],
          ),
        ),
      ],
    );
  }

  Widget _radarLabel(String text) {
    return Text(
      text,
      style: const TextStyle(
        color: Color(0xFF00D4FF),
        fontWeight: FontWeight.bold,
        fontSize: 16,
      ),
    );
  }

  Widget _buildListenButton() {
    return GestureDetector(
      onTap: _toggle,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 300),
        width: 220,
        height: 56,
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(28),
          gradient: LinearGradient(
            colors: _isListening
                ? [const Color(0xFF007A99), const Color(0xFF005F80)]
                : [const Color(0xFF007A99), const Color(0xFF00D4FF)],
          ),
          boxShadow: [
            BoxShadow(
              color: const Color(0xFF00D4FF).withOpacity(0.5),
              blurRadius: 12,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              _isListening ? Icons.stop : Icons.mic,
              color: Colors.white,
            ),
            const SizedBox(width: 8),
            Text(
              _isListening ? 'STOP' : 'LISTEN',
              style: const TextStyle(
                color: Colors.white,
                fontSize: 16,
                fontWeight: FontWeight.bold,
                letterSpacing: 1.2,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDetectedSoundsList() {
    // Separate into items being edited and normal items
    final editingList = _detections.where((d) => d.feedback == -1).toList();
    final normalList = _detections.where((d) => d.feedback != -1).toList();
    final displayList = [...editingList, ...normalList];

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'DETECTED SOUNDS',
            style: TextStyle(
              color: Color(0xFF0F4C68),
              fontWeight: FontWeight.w800,
              fontSize: 14,
              letterSpacing: 1.0,
            ),
          ),
          const SizedBox(height: 16),
          if (_error.isNotEmpty)
            Text(
              _error,
              style: const TextStyle(color: Colors.redAccent, fontSize: 12),
            )
          else if (displayList.isNotEmpty)
            // Render the stack of detections
            ...displayList.map((det) => Padding(
                  key: ValueKey(det.id),
                  padding: const EdgeInsets.only(bottom: 12),
                  child: _buildSoundCard(det),
                ))
          else
            const Text(
              'No sounds detected yet.',
              style: TextStyle(color: Colors.grey, fontSize: 14),
            ),
          if (_isSending) ...[
            const SizedBox(height: 8),
            const Row(
              children: [
                SizedBox(
                    width: 12,
                    height: 12,
                    child: CircularProgressIndicator(strokeWidth: 2)),
                SizedBox(width: 8),
                Text('Processing audio...',
                    style: TextStyle(color: Colors.grey, fontSize: 12)),
              ],
            )
          ]
        ],
      ),
    );
  }

  Widget _buildSoundCard(DetectionEvent det) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [Color(0xFF006C88), Color(0xFF00D4FF)],
          begin: Alignment.centerLeft,
          end: Alignment.centerRight,
        ),
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: const Color(0xFF00D4FF).withOpacity(0.4),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: [
                  Text(
                    det.distance.toStringAsFixed(1),
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                      fontSize: 16,
                    ),
                  ),
                  const SizedBox(width: 16),
                  Text(
                    det.rangeLabel,
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w600,
                      fontSize: 16,
                    ),
                  ),
                ],
              ),
              const Text(
                'Source',
                style: TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  fontSize: 18,
                  letterSpacing: 1.0,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          // Intelligence Experience & Feedback Section
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              // Confidence Score
              Row(
                children: [
                  const Icon(Icons.analytics_outlined,
                      color: Colors.white70, size: 16),
                  const SizedBox(width: 4),
                  Text(
                    'Confidence: ${(det.confidence * 100).toStringAsFixed(0)}%',
                    style: const TextStyle(color: Colors.white70, fontSize: 12),
                  ),
                ],
              ),
              // Feedback Buttons for Continuous Improvement
              Row(
                children: [
                  const Text('Accurate?',
                      style: TextStyle(color: Colors.white70, fontSize: 12)),
                  const SizedBox(width: 8),
                  InkWell(
                    onTap: () {
                      setState(() {
                        det.feedback = det.feedback == 1 ? null : 1;
                        _cleanupOldDetections();
                      });
                    },
                    child: Icon(
                      det.feedback == 1
                          ? Icons.thumb_up_alt
                          : Icons.thumb_up_alt_outlined,
                      color:
                          det.feedback == 1 ? Colors.greenAccent : Colors.white,
                      size: 18,
                    ),
                  ),
                  const SizedBox(width: 12),
                  InkWell(
                    onTap: () {
                      setState(() {
                        det.feedback = det.feedback == -1 ? null : -1;
                        _cleanupOldDetections();
                      });
                    },
                    child: Icon(
                      det.feedback == -1
                          ? Icons.thumb_down_alt
                          : Icons.thumb_down_alt_outlined,
                      color:
                          det.feedback == -1 ? Colors.redAccent : Colors.white,
                      size: 18,
                    ),
                  ),
                ],
              ),
            ],
          ),
          // Correction Input Field (Visible if feedback is negative)
          if (det.feedback == -1) ...[
            const SizedBox(height: 12),
            TextField(
              controller: det.textController,
              style: const TextStyle(color: Colors.white, fontSize: 12),
              decoration: InputDecoration(
                isDense: true,
                hintText: 'What is the correct sound distance?',
                hintStyle: const TextStyle(color: Colors.white38, fontSize: 12),
                filled: true,
                fillColor: Colors.black26,
                contentPadding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(8),
                  borderSide: BorderSide.none,
                ),
                suffixIcon: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    // Cancel Icon
                    IconButton(
                      padding: EdgeInsets.zero,
                      constraints: const BoxConstraints(minWidth: 32),
                      icon: const Icon(Icons.close,
                          color: Colors.white54, size: 16),
                      onPressed: () {
                        setState(() {
                          det.feedback = null; // Untoggle feedback
                          _cleanupOldDetections(); // Let it be removed if it's > 3s old
                        });
                      },
                    ),
                    // Send Icon
                    IconButton(
                      padding: EdgeInsets.zero,
                      constraints: const BoxConstraints(minWidth: 32),
                      icon: const Icon(Icons.send,
                          color: Colors.white70, size: 16),
                      onPressed: () {
                        if (det.textController.text.trim().isEmpty) return;

                        ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(
                                content: Text('Correction submitted!')));
                        setState(() {
                          _detections
                              .remove(det); // Remove it immediately upon submit
                          det.dispose();
                        });
                      },
                    ),
                  ],
                ),
              ),
            ),
          ]
        ],
      ),
    );
  }
}

// ── Custom Painters ────────────────────────────────────────────────────────

class _RadarPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final maxRadius = size.width / 2;

    // Glowing background for the whole radar
    final glowPaint = Paint()
      ..color = const Color(0xFF00D4FF).withOpacity(0.05)
      ..style = PaintingStyle.fill;
    canvas.drawCircle(center, maxRadius, glowPaint);

    // Outline stroke with glow
    final strokePaint = Paint()
      ..color = const Color(0xFF00D4FF).withOpacity(0.3)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.0;

    final thickStrokePaint = Paint()
      ..color = const Color(0xFF00D4FF).withOpacity(0.5)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    // Draw 4 concentric rings
    for (int i = 1; i <= 4; i++) {
      canvas.drawCircle(
          center, maxRadius * (i / 4), i == 4 ? thickStrokePaint : strokePaint);
    }

    // Draw cross lines
    canvas.drawLine(
        Offset(center.dx, 0), Offset(center.dx, size.height), strokePaint);
    canvas.drawLine(
        Offset(0, center.dy), Offset(size.width, center.dy), strokePaint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

class _RadarSweepPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = size.width / 2;

    final rect = Rect.fromCircle(center: center, radius: radius);

    final sweepGradient = SweepGradient(
      colors: [
        const Color(0xFF00D4FF).withOpacity(0.0),
        const Color(0xFF00D4FF).withOpacity(0.4),
      ],
      startAngle: 0.0,
      endAngle: math.pi / 2,
    );

    final paint = Paint()
      ..shader = sweepGradient.createShader(rect)
      ..style = PaintingStyle.fill;

    // We draw an arc that represents the sweeping part
    canvas.drawArc(rect, -math.pi / 2, math.pi / 2, true, paint);

    // Leading edge line
    final linePaint = Paint()
      ..color = const Color(0xFF00D4FF).withOpacity(0.8)
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;

    canvas.drawLine(center, Offset(size.width, center.dy), linePaint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
