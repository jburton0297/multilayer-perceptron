import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class UAMLP {

	public static final int IN_SIZE = 28*28;
	public static final int HIDDEN_SIZE = 300;
	public static final int OUT_SIZE = 10;
	
	public static final boolean CSV = false;
	
	public static final float TRAIN_AMT = 0.6f;
	public static final float VAL_AMT = 0.2f;
	public static final float TEST_AMT = 0.2f;
	
	public static int maxEpochs;
	public static float lr;
	public static int hiddenLayers;
	public static int currentEpoch;
	public static double[] weights;
	
	public static double valLoss;
	public static double valLossTarget = 0.05D;

	public static void main(String[] args) {
		try {
			if(args.length != 5) {
				throw new Exception("Invalid arguments.");
			}
			
			String inputDir = args[0];
			String testDir = args[1];
			maxEpochs = Integer.parseInt(args[2]);
			lr = Float.parseFloat(args[3]);
			hiddenLayers = Integer.parseInt(args[4]);
			
			// Initialize neural network
			NeuralNetwork nn = initNeuralNetwork();
			
			// Get train data
			File input = new File(inputDir);
			File[] inputFiles = input.listFiles();
			List<double[]> images = new ArrayList<>();
			if(CSV) {
				for(File f : inputFiles) {
					BufferedReader br = new BufferedReader(new FileReader(f));
					String line;
					String[] values;
					while((line = br.readLine()) != null) {
						double[] currentImage = new double[IN_SIZE + 1];
						values = line.split(",");
						for(int i = 0; i < values.length; i++) {
							double value = Double.parseDouble(values[i]);
							if(i == 0) {
								currentImage[i] = value;
							} else {
								currentImage[i] = value / 255;
							}
						}
						images.add(currentImage);
					}
					br.close();
				}
			} else {
				File trainImages = null;
				File trainLabels = null;
				File testImages = null;
				File testLabels = null;
				for(File f : inputFiles) {
					String fileName = f.getName();
					if(fileName.contains("train")) {
						if(fileName.contains("images")) {
							trainImages = f;
						} else if(fileName.contains("labels")) {
							trainLabels = f;
						}
					} else if(fileName.contains("t10k")) {
						if(fileName.contains("images")) {
							testImages = f;
						} else if(fileName.contains("labels")) {
							testLabels = f;
						}
					}
				}
				List<File> imageFiles = new ArrayList<>();
				List<File> labelFiles = new ArrayList<>();
				imageFiles.add(trainImages);
				imageFiles.add(testImages);
				labelFiles.add(trainLabels);
				labelFiles.add(testLabels);
				for(int i = 0; i < imageFiles.size(); i++) {
					
					File imageFile = imageFiles.get(i);
					File labelFile = labelFiles.get(i);
					
					BufferedInputStream mbis = new BufferedInputStream(new FileInputStream(imageFile));
					
					byte[] magic = new byte[Integer.BYTES];
					mbis.read(magic);
					byte[] imgCountBytes = new byte[Integer.BYTES];
					mbis.read(imgCountBytes);
					byte[] rowCountBytes = new byte[Integer.BYTES];
					mbis.read(rowCountBytes);
					byte[] colCountBytes = new byte[Integer.BYTES];
					mbis.read(colCountBytes);
					
					BufferedInputStream lbis = new BufferedInputStream(new FileInputStream(labelFile));
					
					magic = new byte[Integer.BYTES];
					lbis.read(magic);
					imgCountBytes = new byte[Integer.BYTES];
					lbis.read(imgCountBytes);
					
					int imgCount = ByteBuffer.wrap(imgCountBytes).getInt();
					int rowCount = ByteBuffer.wrap(rowCountBytes).getInt();
					int colCount = ByteBuffer.wrap(colCountBytes).getInt();
					for(int j = 0; j < imgCount; j++) {
						double[] image = new double[(rowCount * colCount) + 1];
						image[0] = lbis.read();
						for(int k = 1; k <= rowCount * colCount; k++) {
							image[k] = mbis.read();
						}
						images.add(image);
					}
					
					mbis.close();
					lbis.close();
				}
			}
			Collections.shuffle(images);
			
			// Split to train val test
			int trainCount = (int)Math.floor(images.size() * TRAIN_AMT);
			int testCount = (int)Math.floor(images.size() * TEST_AMT);
			int valCount = images.size() - trainCount - testCount;
			System.out.println("Training on " + trainCount + " images");
			System.out.println("Validating on " + valCount + " images");
			System.out.println("Testing on " + testCount + " images");
			System.out.println();
			
			int currentIndex = 0;
			List<double[]> trainData = new ArrayList<>();
			for(int i = 0; i < trainCount; i++) {
				trainData.add(images.get(currentIndex++));
			}
			
			List<double[]> valData = new ArrayList<>();
			for(int i = 0; i < valCount; i++) {
				valData.add(images.get(currentIndex++));
			}
			
			List<double[]> testData = new ArrayList<>();
			for(int i = 0; i < testCount; i++) {
				testData.add(images.get(currentIndex++));
			}
			
			// Train nn
			for(int i = 1; i <= maxEpochs; i++) {
				System.out.println("EPOCH " + i);
				nn = backpropLearning(trainData, valData, nn);
				Collections.shuffle(trainData);
				
				if(valLoss < 0.05D) {
					System.out.println("Stopping early.");
					break;
				}
			}
			nn.writeWeightsToFile(new File("weights.txt"));
			
			// Testing
			int total = 0;
			int correct = 0;
			int[][] confusionMatrix = new int[10][10];
			for(int i = 0; i < testData.size(); i++) {
				double[] testImg = testData.get(i);
				double[] pred = predict(testImg, nn);
				int decodedPred = oneHotDecodePrediction(pred);
				total++;
				if(decodedPred == testImg[0]) {
					correct++;
					confusionMatrix[decodedPred][decodedPred]++;
				} else {
					confusionMatrix[(int)testImg[0]][decodedPred]++;
				}
			}
			System.out.printf("\nAccuracy: %2.3f%% \n", ((float)correct / total) * 100f);
			System.out.printf("%d / %d (correct / total) \n \n", correct, total);
			
			// Print confusion matrix
			System.out.println("Confusion Matrix (Actual x Predicted):");
			System.out.println("|---|-------------------------------------------------|");
			System.out.println("|   | 0  | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  |");
			System.out.println("|---|----|----|----|----|----|----|----|----|----|----|");
			for(int i = 0; i < 10; i++) {
				System.out.printf("| " + i + " |");
				for(int j = 0; j < 10; j++) {
					System.out.printf("%4d|", confusionMatrix[i][j]);				
				}
				System.out.println("\n|---|----|----|----|----|----|----|----|----|----|----|");
			}	
		} catch(IOException e) {
			e.printStackTrace();
		} catch(NumberFormatException e) {
			e.printStackTrace();
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	public static double[] predict(double[] img, NeuralNetwork nn) throws Exception {
		nn.feedforward(img);
		return nn.getOutputLayerG();
	}
	
	public static NeuralNetwork backpropLearning(List<double[]> images, List<double[]> val, NeuralNetwork nn) throws Exception {
		
		double[] error = null;
		double sumLoss = 0D;
		double loss = 0D;
		int step = 0;
		for(double[] img : images) {
			double label = img[0];
			double[] input = Arrays.copyOfRange(img, 1, img.length);
			
			InputLayer il = nn.getInputLayer();
			List<HiddenLayer> hls = nn.getHiddenLayers();
			OutputLayer ol = nn.getOutputLayer();
			
			nn.feedforward(input);			
			
			double[] inputLayerG = nn.getInputLayerG();
			List<double[]> hiddenLayerGs = nn.getHiddenLayerGs();
			double[] outputLayerG = nn.getOutputLayerG();
			
			// Calculate error
			double[] encodedLabel = oneHotEncodeLabel(label);
			error = calculateError(encodedLabel, outputLayerG);
			for(int i = 0; i < error.length; i++) {
				loss += error[i];
			}
			loss /= error.length;
			sumLoss += loss;
			
			step++;
			
			// Back propagate output layer
			double[] outputLayerDelta = new double[OUT_SIZE];
			for(int i = 0; i < outputLayerDelta.length; i++) {
				outputLayerDelta[i] = (outputLayerG[i] - encodedLabel[i]) * outputLayerG[i] * (1 - outputLayerG[i]);  
			}
			
			// Back propagate hidden layers
			List<double[]> hiddenLayerDeltas = new ArrayList<>();
			double[] prevHiddenLayerDelta = null;
			for(int l = hls.size()-1; l >= 0; l--) {
				HiddenLayer hl = hls.get(l);
				double[] hiddenLayerDelta = new double[HIDDEN_SIZE];
				for(int i = 0; i < hl.getNeuronCount(); i++) {
					Neuron ni = hl.getNeurons().get(i);
					double nextDeltaSum = 0D;
					for(int j = 0; j < ni.getOutputCount(); j++) {
						double currentWeight = 0D;
						if(l == hls.size() - 1) {
							currentWeight = ol.getNeurons().get(j).getInputWeights().get(i);
							nextDeltaSum += currentWeight * outputLayerDelta[j];
						} else {
							currentWeight = hls.get(l+1).getNeurons().get(j).getInputWeights().get(i);
							nextDeltaSum += currentWeight * prevHiddenLayerDelta[j];
						}
					}
					double hiddenLayerG = hiddenLayerGs.get(l)[i];
					double previousOutput = 0D;
					if(l == 0) {
						previousOutput = inputLayerG[i];
					} else {
						previousOutput = hiddenLayerGs.get(l-1)[i];
					}
					hiddenLayerDelta[i] = nextDeltaSum * hiddenLayerG * (1 - hiddenLayerG) * previousOutput;  
				}
				hiddenLayerDeltas.add(hiddenLayerDelta);
				prevHiddenLayerDelta = hiddenLayerDelta;
			}
			Collections.reverse(hiddenLayerDeltas);
			
			// Back propagate input layer
			double[] inputLayerDelta = new double[IN_SIZE];
			for(int i = 0; i < il.getNeuronCount(); i++) {
				Neuron ni = il.getNeurons().get(i);
				double nextDeltaSum = 0D;
				for(int j = 0; j < ni.getOutputCount(); j++) {
					double currentWeight = hls.get(0).getNeurons().get(j).getInputWeights().get(i);
					nextDeltaSum += currentWeight * hiddenLayerDeltas.get(0)[j];
				}
				inputLayerDelta[i] = nextDeltaSum * input[i] * (1 - input[i]);
			}
			
			// Update input layer weights
			List<Double> nodeWeights = null;
			for(int i = 0; i < il.getNeuronCount(); i++) {
				Neuron n = il.getNeurons().get(i);
				nodeWeights = n.getInputWeights();
				List<Double> newWeights = new ArrayList<>();
				for(double w : nodeWeights) {
					double nw = w - (lr * inputLayerDelta[i]);
					newWeights.add(nw);
				}
				n.setInputWeights(newWeights);
				double bw = n.getBiasWeight();
				double nbw = bw - (lr * inputLayerDelta[i]);
				n.setBiasWeight(nbw);
			}
			
			// Update hidden layer weights
			for(int l = 0; l < hls.size(); l++) {
				HiddenLayer hl = hls.get(l);
				double[] hiddenLayerDelta = hiddenLayerDeltas.get(l);
				for(int i = 0; i < hl.getNeuronCount(); i++) {
					Neuron n = hl.getNeurons().get(i);
					nodeWeights = n.getInputWeights();
					List<Double> newWeights = new ArrayList<>();
					for(double w : nodeWeights) {
						double nw = 0D;
						if(l == 0) {
							nw = w - (lr * hiddenLayerDelta[i]);
						} else {
							nw = w - (lr * hiddenLayerDelta[i]);
						}
						newWeights.add(nw);
					}
					n.setInputWeights(newWeights);
					double bw = n.getBiasWeight();
					double nbw = bw - (lr * hiddenLayerDelta[i]);
					n.setBiasWeight(nbw);
				}
			}
			
			// Update output layer weights
			for(int i = 0; i < ol.getNeuronCount(); i++) {
				Neuron n = ol.getNeurons().get(i);
				nodeWeights = n.getInputWeights();
				List<Double> newWeights = new ArrayList<>();
				for(double w : nodeWeights) {
					double nw = w - (lr * outputLayerDelta[i]);
					newWeights.add(nw);
				}
				n.setInputWeights(newWeights);
				double bw = n.getBiasWeight();
				double nbw = bw - (lr * outputLayerDelta[i]);
				n.setBiasWeight(nbw);
			}	
		}
		sumLoss /= step;
		
		System.out.print("END OF EPOCH ");
		System.out.printf("loss=%.3f ", sumLoss);
		
		// Validation
		double sumValLoss = 0D;
		for(double[] img : val) {

			double label = img[0];
			double[] input = Arrays.copyOfRange(img, 1, img.length);

			nn.feedforward(input);			
			double[] outputLayerG = nn.getOutputLayerG();

			// Calculate loss
			double[] encodedLabel = oneHotEncodeLabel(label);
			error = calculateError(encodedLabel, outputLayerG);
			double sumError = 0D;
			for(int i = 0; i < error.length; i++) {
				sumError += error[i];
			}
			
			sumValLoss += (sumError / (double)error.length);
		}

		double vLoss = sumValLoss / (double)val.size();
		System.out.printf("val_loss=%.3f\n\n", vLoss);
		valLoss = vLoss;
		
		return nn;
	}

	public static double[] calculateError(double[] encodedLabel, double[] output) {
		double[] error = new double[output.length];
		for(int i = 0; i < output.length; i++) {
			error[i] = Math.pow(encodedLabel[i] - output[i], 2);
		}
		return error;
	}
	
	public static void printError(double[] error) {
		double sum = 0D;
		for(int i = 0; i < error.length; i++) {
			sum += error[i];
		}
		System.out.printf("%.3f", sum/(double)error.length);
	}
	
	public static double[] oneHotEncodeLabel(double label) {
		double[] result = new double[10];
		for(int i = 0; i < result.length; i++) {
			if(i == label) {
				result[i] = 1;
			} else {
				result[i] = 0;
			}
		}
		return result;
	}
	
	public static int oneHotDecodePrediction(double[] pred) {
		int index = 0;
		double max = 0D;
		for(int i = 0; i < pred.length; i++) {
			if(pred[i] > max) {
				max = pred[i];
				index = i;
			}
		}
		return index;
	}
	
	public static NeuralNetwork initNeuralNetwork() {
		
		// Initialize input layer
		InputLayer il;
		{
			Neuron n;
			List<Neuron> ns = new ArrayList<>();
			for(int i = 0; i < IN_SIZE; i++) {
				n = new Neuron(1, HIDDEN_SIZE);
				ns.add(n);
			}
			il = new InputLayer(ns, ns.size());
		}
		
		// Initialize hidden layers
		HiddenLayer hl;
		List<HiddenLayer> hls = new ArrayList<>();
		for(int i = 0; i < hiddenLayers; i++) {
			Neuron n;
			List<Neuron> ns = new ArrayList<>();
			for(int j = 0; j < HIDDEN_SIZE; j++) {
				if(hiddenLayers == 1) {
					n = new Neuron(IN_SIZE, OUT_SIZE);
				} else {
					if(i == 0) {
						n = new Neuron(IN_SIZE, HIDDEN_SIZE);
					} else if(i < hiddenLayers - 1) {
						n = new Neuron(HIDDEN_SIZE, HIDDEN_SIZE);
					} else {
						n = new Neuron(HIDDEN_SIZE, OUT_SIZE); 
					}
				}
				ns.add(n);
			}
			hl = new HiddenLayer(ns, ns.size());
			hls.add(hl);
		}
		
		// Initialize output layer
		OutputLayer ol;
		{
			Neuron n;
			List<Neuron> ns = new ArrayList<>();
			for(int j = 0; j < OUT_SIZE; j++) {
				n = new Neuron(HIDDEN_SIZE, 1);
				ns.add(n);
			}
			ol = new OutputLayer(ns, ns.size());
		}
		
		// Initialize neural network
		NeuralNetwork nn = new NeuralNetwork(il, hls, ol);
		
		return nn;
	}
}

class NeuralNetwork {
	
	private int IN_SIZE;
	private int HIDDEN_SIZE;
	private int OUT_SIZE;
	
	private InputLayer inputLayer;
	private List<HiddenLayer> hiddenLayers;
	private OutputLayer outputLayer;
	
	private double[] inputLayerG;
	private double[] inputLayerGPrime;
	private List<double[]> hiddenLayerGs;
	private List<double[]> hiddenLayerGPrimes;
	private double[] outputLayerG;
	private double[] outputLayerGPrime;
	
	public NeuralNetwork(InputLayer inputLayer, List<HiddenLayer> hiddenLayers, OutputLayer outputLayer) {
		this.inputLayer = inputLayer;
		this.hiddenLayers = hiddenLayers;
		this.outputLayer = outputLayer;
		this.IN_SIZE = inputLayer.getNeuronCount();
		this.HIDDEN_SIZE = hiddenLayers.get(0).getNeuronCount();
		this.OUT_SIZE = outputLayer.getNeuronCount();
	}
	
	public void feedforward(double[] img) throws Exception {
		
		// Feed input forward
		double[] inputLayerG = new double[IN_SIZE];
		double[] inputLayerGPrime = new double[IN_SIZE];
		InputLayer il = this.getInputLayer();
		{
			int i = 0;
			for(Neuron n : il.getNeurons()) {
				double weightedSum = n.sumInputs(new double[] { img[i]});
				double gOut = n.activate(weightedSum);
				double gPrimeOut = n.deriv(weightedSum);
				inputLayerG[i] = gOut;
				inputLayerGPrime[i++] = gPrimeOut;
			}
		}
		this.setInputLayerG(inputLayerG);
		this.setInputLayerGPrime(inputLayerGPrime);
		
		List<double[]> hiddenLayerGs = new ArrayList<>();
		List<double[]> hiddenLayerGPrimes = new ArrayList<>();
		List<HiddenLayer> hls = this.getHiddenLayers();
		double[] hiddenLayerInput = null;
		{
			HiddenLayer hl;
			for(int i = 0; i < hls.size(); i++) {
				hl = hls.get(i);
				if(i == 0) {
					hiddenLayerInput = inputLayerG;
				} else {
					hiddenLayerInput = hiddenLayerGs.get(i-1);
				}
				int j = 0;
				double[] g = new double[HIDDEN_SIZE];
				double[] gPrime = new double[HIDDEN_SIZE];
				for(Neuron n : hl.getNeurons()) {
					double weightedSum = n.sumInputs(hiddenLayerInput);
					double gOut = n.activate(weightedSum);
					double gPrimeOut = n.deriv(weightedSum);
					g[j] = gOut;
					gPrime[j++] = gPrimeOut;
				} 
				hiddenLayerGs.add(g);
				hiddenLayerGPrimes.add(gPrime);
			}
		}
		this.setHiddenLayerGs(hiddenLayerGs);
		this.setHiddenLayerGPrimes(hiddenLayerGPrimes);
	
		double[] outputLayerG = new double[OUT_SIZE];
		double[] outputLayerGPrime = new double[OUT_SIZE];
		OutputLayer ol = this.getOutputLayer();
		List<Neuron> neurons = ol.getNeurons();
		for(int i = 0; i < ol.getNeuronCount(); i++) {
			Neuron n = neurons.get(i);
			double weightedSum = n.sumInputs(hiddenLayerGs.get(hls.size()-1));
			double gOut = n.activate(weightedSum);
			double gPrimeOut = n.deriv(weightedSum);
			outputLayerG[i] = gOut;
			outputLayerGPrime[i] = gPrimeOut;
		}
		this.setOutputLayerG(outputLayerG);
		this.setOutputLayerGPrime(outputLayerGPrime);
	}
	
	public void writeWeightsToFile(File file) {
		System.out.println("Writing weights to file...");
		
		BufferedWriter bw = null;
		try {
			if(!file.exists()) {
				file.createNewFile();
			}
			bw = new BufferedWriter(new FileWriter(file));
			List<Double> nWeights = null;
			bw.write(this.inputLayer.getNeuronCount() + " ");
			for(Neuron n : this.inputLayer.getNeurons()) {
				nWeights = n.getInputWeights();
				for(int i = 0; i < nWeights.size(); i++) {
					double w = nWeights.get(i);
					bw.write(w + " ");
				}
			}
			bw.write("\n");
			
			for(HiddenLayer hl : this.hiddenLayers) {
				bw.write(hl.getNeuronCount() + " ");
				for(Neuron n : hl.getNeurons()) {
					nWeights = n.getInputWeights();
					for(int i = 0; i < nWeights.size(); i++) {
						double w = nWeights.get(i);
						bw.write(w + " ");
					}
				}
				bw.write("\n");
			}
			
			bw.write(this.outputLayer.getNeuronCount() + " ");
			for(Neuron n : this.outputLayer.getNeurons()) {
				nWeights = n.getInputWeights();
				for(int i = 0; i < nWeights.size(); i++) {
					double w = nWeights.get(i);
					bw.write(w + " ");
				}
			}
		} catch(IOException e) {
			e.printStackTrace();
		} finally {
			try {
				if(bw != null) bw.close();
			} catch(IOException e) {
				e.printStackTrace();
			}
		}
		
		System.out.println("Weights written to file.");
	}
	
	public static NeuralNetwork buildFromWeightsFile(File file) {
		BufferedReader br = null;
		NeuralNetwork nn = null;
		try {
			
			br = new BufferedReader(new FileReader(file));
			InputLayer il = null;
			List<HiddenLayer> hls = new ArrayList<>();
			OutputLayer ol = null;
			
			// Load weights
			String line;
			String[] values;
			List<String[]> layers = new ArrayList<>();
			while((line = br.readLine()) != null) {
				values = line.split(" ");
				layers.add(values);
			}

			int weightsPerNeuron = 0;
			int neuronCount = 0;
			String[] layer;
			for(int i = 0; i < layers.size(); i++) {
				layer = layers.get(i);
				neuronCount = Integer.parseInt(layer[0]);
				weightsPerNeuron = (layer.length - 1) / neuronCount;
				int weightIndex = 0;
				List<Double> inputWeights = null;
				List<Neuron> neurons = new ArrayList<>();
				if(i == 0) {
					int nextLayerNeuronCount = Integer.parseInt(layers.get(i+1)[0]);
					for(int n = 0; n < neuronCount; n++) {
						inputWeights = new ArrayList<>();
						for(int w = 0; w < weightsPerNeuron; w++) {
							double weight = Double.parseDouble(layer[weightIndex++]);
							inputWeights.add(weight);
						}
						Neuron neuron = new Neuron(inputWeights, nextLayerNeuronCount);
						neurons.add(neuron);
					}
					il = new InputLayer(neurons, neurons.size());
				} else if(i > 0 && i < layers.size() - 1) {
					int nextLayerNeuronCount = Integer.parseInt(layers.get(i+1)[0]);
					for(int n = 0; n < neuronCount; n++) {
						inputWeights = new ArrayList<>();
						for(int w = 0; w < weightsPerNeuron; w++) {
							double weight = Double.parseDouble(layer[weightIndex++]);
							inputWeights.add(weight);
						}
						Neuron neuron = new Neuron(inputWeights, nextLayerNeuronCount);
						neurons.add(neuron);
					}
					HiddenLayer hl = new HiddenLayer(neurons, neurons.size());
					hls.add(hl);
				} else {
					for(int n = 0; n < neuronCount; n++) {
						inputWeights = new ArrayList<>();
						for(int w = 0; w < weightsPerNeuron; w++) {
							double weight = Double.parseDouble(layer[weightIndex++]);
							inputWeights.add(weight);
						}
						Neuron neuron = new Neuron(inputWeights, 1);
						neurons.add(neuron);
					}
					ol = new OutputLayer(neurons, neurons.size());
				}			
			}
			
			nn = new NeuralNetwork(il, hls, ol);
			
		} catch(IOException e) {
			e.printStackTrace();
		} finally {
			try {
				if(br != null) br.close();
			} catch(IOException e) {
				e.printStackTrace();
			}
		}
		return nn;
	}

	public InputLayer getInputLayer() {
		return inputLayer;
	}

	public void setInputLayer(InputLayer inputLayer) {
		this.inputLayer = inputLayer;
	}

	public List<HiddenLayer> getHiddenLayers() {
		return hiddenLayers;
	}

	public void setHiddenLayers(List<HiddenLayer> hiddenLayers) {
		this.hiddenLayers = hiddenLayers;
	}

	public OutputLayer getOutputLayer() {
		return outputLayer;
	}

	public void setOutputLayer(OutputLayer outputLayer) {
		this.outputLayer = outputLayer;
	}

	public double[] getInputLayerG() {
		return inputLayerG;
	}

	public void setInputLayerG(double[] inputLayerG) {
		this.inputLayerG = inputLayerG;
	}

	public double[] getInputLayerGPrime() {
		return inputLayerGPrime;
	}

	public void setInputLayerGPrime(double[] inputLayerGPrime) {
		this.inputLayerGPrime = inputLayerGPrime;
	}

	public List<double[]> getHiddenLayerGs() {
		return hiddenLayerGs;
	}

	public void setHiddenLayerGs(List<double[]> hiddenLayerGs) {
		this.hiddenLayerGs = hiddenLayerGs;
	}

	public List<double[]> getHiddenLayerGPrimes() {
		return hiddenLayerGPrimes;
	}

	public void setHiddenLayerGPrimes(List<double[]> hiddenLayerGPrimes) {
		this.hiddenLayerGPrimes = hiddenLayerGPrimes;
	}

	public double[] getOutputLayerG() {
		return outputLayerG;
	}

	public void setOutputLayerG(double[] outputLayerG) {
		this.outputLayerG = outputLayerG;
	}

	public double[] getOutputLayerGPrime() {
		return outputLayerGPrime;
	}

	public void setOutputLayerGPrime(double[] outputLayerGPrime) {
		this.outputLayerGPrime = outputLayerGPrime;
	}
	
}

class InputLayer extends Layer {

	public InputLayer(List<Neuron> neurons, int neuronCount) {
		super(neurons, neuronCount);
	}
	
}

class OutputLayer extends Layer {

	public OutputLayer(List<Neuron> neurons, int neuronCount) {
		super(neurons, neuronCount);
	}
	
}

class HiddenLayer extends Layer {

	public HiddenLayer(List<Neuron> neurons, int neuronCount) {
		super(neurons, neuronCount);
	}
	
}

abstract class Layer {
	
	private List<Neuron> neurons;
	private int neuronCount;
	
	public Layer(List<Neuron> neurons, int neuronCount) {
		this.neurons = neurons;
		this.neuronCount = neuronCount;
	}
	public List<Neuron> getNeurons() {
		return neurons;
	}
	public void setNeurons(List<Neuron> neurons) {
		this.neurons = neurons;
	}
	public int getNeuronCount() {
		return neuronCount;
	}
	public void setNeuronCount(int neuronCount) {
		this.neuronCount = neuronCount;
	}
	
}

class Neuron {
	
	public static int allWeights = 0;
	
	private List<Double> inputWeights;
	private double biasWeight;
	private int outputCount;
	private double output;
	
	public Neuron(List<Double> inputWeights, int outputCount) {
		this.inputWeights = inputWeights;
		this.outputCount = outputCount;
	}
	
	public Neuron(int inputWeightsCount, int outputCount) {
		this.inputWeights = new ArrayList<>();
		this.outputCount = outputCount;
		initWeights(inputWeightsCount);
	}
	
	public double activate(double input) {
		return 1 / (1 + Math.pow(Math.E, -1 * input));
	}
	
	public double deriv(double input) {
		return input * (1 - input);
	}
	
	public double sumInputs(double[] inputs) throws Exception {
		if(inputs.length != inputWeights.size()) {
			throw new Exception("Invalid image size.");
		}
		
		double sum = 0D;
		for(int i = 0; i < inputWeights.size(); i++) {
			sum += inputs[i] * inputWeights.get(i);
		}
		sum += -1 * this.biasWeight;
		
		return sum;
	}
	
	private void initWeights(int inputWeightsCount) {
		for(int i = 0; i < inputWeightsCount; i++) {
			double w = new Random().nextDouble() * 2D - 1D;
			this.inputWeights.add(w);
		}
		this.biasWeight = new Random().nextDouble() * 2D - 1D;
	}

	public List<Double> getInputWeights() {
		return inputWeights;
	}

	public void setInputWeights(List<Double> inputWeights) {
		this.inputWeights = inputWeights;
	}

	public double getBiasWeight() {
		return biasWeight;
	}

	public void setBiasWeight(double biasWeight) {
		this.biasWeight = biasWeight;
	}

	public int getOutputCount() {
		return outputCount;
	}

	public void setOutputCount(int outputCount) {
		this.outputCount = outputCount;
	}

	public double getOutput() {
		return output;
	}

	public void setOutput(double output) {
		this.output = output;
	}
	
}