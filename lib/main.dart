import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'package:flutter/services.dart' show rootBundle;

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    WidgetsFlutterBinding.ensureInitialized();
    return MaterialApp(
      home: ModelTestScreen(),
    );
  }
}

class ModelTestScreen extends StatefulWidget {
  @override
  _ModelTestScreenState createState() => _ModelTestScreenState();
}

class _ModelTestScreenState extends State<ModelTestScreen> {
  late Interpreter _interpreter;
  List<String> _labels = [];
  List<double> _output = [];
  File? _image;
  String _predictedLabel = ' ';
  bool _isModelLoaded = false;

  @override
  void initState() {
    super.initState();
    _loadModel();
    _loadLabels();
  }

  void _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/model_unquant.tflite');
      print('Model loaded successfully');
      setState(() {
        _isModelLoaded = true;
      });

      var inputShape = _interpreter.getInputTensor(0).shape;
      var outputShape = _interpreter.getOutputTensor(0).shape;
      print('Input shape: $inputShape');
      print('Output shape: $outputShape');
    } catch (e) {
      print('Failed to load model: $e');
    }
  }

  void _loadLabels() async {
    try {
      print('Attempting to load labels...');
      final labelData = await rootBundle.loadString('assets/label.txt');
      print('Raw label data: $labelData'); // Show raw data

      setState(() {
        _labels = labelData
            .split('\n')
            .map((s) => s.trim())
            .where((s) => s.isNotEmpty)
            .toList();
      });
      print('Parsed labels: $_labels');
      print('Labels loaded successfully');
    } catch (e) {
      print('Failed to load labels: $e');
    }
  }

  void _pickImage() async {
    final pickedFile =
        await ImagePicker().pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
      });
      _runInference();
    }
  }

  Uint8List _imageToByteListFloat32(img.Image image) {
    var convertedBytes = Float32List(224 * 224 * 3); // Correct the size
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;

    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {
        var pixel = image.getPixel(x, y);
        buffer[pixelIndex++] =
            (img.getRed(pixel) / 255.0 - 0.5) * 2; // Normalize to [-1, 1]
        buffer[pixelIndex++] =
            (img.getGreen(pixel) / 255.0 - 0.5) * 2; // Normalize to [-1, 1]
        buffer[pixelIndex++] =
            (img.getBlue(pixel) / 255.0 - 0.5) * 2; // Normalize to [-1, 1]
      }
    }
    return convertedBytes.buffer.asUint8List(); // Correct return type
  }

  void _runInference() async {
    if (_image == null || !_isModelLoaded) return;

    try {
      var imageBytes = await _image!.readAsBytes();
      img.Image? image = img.decodeImage(imageBytes);

      if (image == null) throw Exception('Failed to decode image');

      // Resize the image
      var resizedImage = img.copyResize(image, width: 224, height: 224);

      // Convert the image to model input format
      var input = _imageToByteListFloat32(resizedImage);

      // Create output buffer with correct shape
      var output = List.filled(5, 0.0).reshape([1, 5]); // Correct output buffer

      // Run inference
      _interpreter.run(input, output);

      // Process output (e.g., find highest probability class)
      int predictedClass = _processOutput(output[0]);
      String predictedLabel = _labels[predictedClass];
      setState(() {
        _output = output[0];
        _predictedLabel = _labels[predictedClass];
      });

      print('Predicted class: $predictedClass');
      print('Predicted label: $predictedLabel');
    } catch (e) {
      print('Error during inference: $e');
    }
  }

  int _processOutput(List<double> output) {
    double maxProb = output[0];
    int maxIndex = 0;

    for (int i = 1; i < output.length; i++) {
      if (output[i] > maxProb) {
        maxProb = output[i];
        maxIndex = i;
      }
    }
    return maxIndex;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Animal Finder',
          style: TextStyle(fontFamily: 'Roboto', fontWeight: FontWeight.bold),
        ),
        backgroundColor: Colors.deepPurple,
      ),
      body: Center(
        child: SingleChildScrollView(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              AnimatedSwitcher(
                duration: Duration(seconds: 1),
                transitionBuilder: (Widget child, Animation<double> animation) {
                  return ScaleTransition(scale: animation, child: child);
                },
                child: _image != null
                    ? ClipRRect(
                        borderRadius: BorderRadius.circular(15),
                        child: Image.file(
                          _image!,
                          key: ValueKey<File>(_image!),
                          height: 300,
                          width: 300,
                          fit: BoxFit.cover,
                        ),
                      )
                    : Container(
                        key: ValueKey<int>(0),
                        height: 300,
                        width: 300,
                        decoration: BoxDecoration(
                          color: Colors.grey.shade200,
                          borderRadius: BorderRadius.circular(15),
                          border:
                              Border.all(color: Colors.deepPurple, width: 2),
                        ),
                        child: Center(
                          child: Text(
                            'No Image Selected',
                            style: TextStyle(
                                color: Colors.deepPurple, fontSize: 18),
                          ),
                        ),
                      ),
              ),
              SizedBox(height: 20),
              AnimatedDefaultTextStyle(
                duration: Duration(milliseconds: 300),
                style: TextStyle(
                  fontSize: 24,
                  color: Colors.deepPurple,
                  fontWeight: FontWeight.bold,
                ),
                child: Text('Model Output: $_predictedLabel'),
              ),
              SizedBox(height: 20),
              ElevatedButton(
                onPressed: _pickImage,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.deepPurple,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  padding: EdgeInsets.symmetric(horizontal: 20, vertical: 15),
                ),
                child: Text(
                  'Pick Image and Run Inference',
                  style: TextStyle(fontSize: 16, color: Colors.white),
                ),
              ),
              SizedBox(height: 40),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20),
                child: Divider(color: Colors.deepPurple.shade200, thickness: 1),
              ),
              SizedBox(height: 10),
              Text(
                'Powered by TFLite',
                style: TextStyle(
                  color: Colors.deepPurple.shade400,
                  fontStyle: FontStyle.italic,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
