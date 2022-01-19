import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

public class UAMLPClassifier {

	public static void main(String[] args) {
		try {
			if(args.length != 3) {
				throw new Exception("Invalid arguments.");
			}
			
			File weights = new File(args[0]);
			NeuralNetwork nn = NeuralNetwork.buildFromWeightsFile(weights);
			
			File testDir = new File(args[1]);
			File outputFile = new File(args[2]);
			
			if(!testDir.exists()) {
				throw new IOException("No test directory found.");
			}
			if(!outputFile.exists()) {
				outputFile.createNewFile();
			}
			
			// Make predictions
			System.out.printf("File Name\tClassification Result\n");
			System.out.printf("---------\t---------------------\n");
			BufferedReader br = null;
			BufferedWriter bw = new BufferedWriter(new FileWriter(outputFile));
			bw.write(String.format("File Name\tClassification Result\n"));
			bw.write(String.format("---------\t---------------------\n"));
			for(File file : testDir.listFiles()) {
				
				br = new BufferedReader(new FileReader(file));
				String line;
				String[] values = null;
				while((line = br.readLine()) != null) {
					values = line.split(" ");
					String fileName = file.getName();
					double[] image = Arrays.stream(values).mapToDouble(Double::parseDouble).toArray();
					
					// Get prediction
					double[] yhat = UAMLP.predict(image, nn);
					int pred = UAMLP.oneHotDecodePrediction(yhat);
					
					System.out.printf("%-9s\t%-21s%n", fileName, ("Class:  " + pred));
					bw.write(String.format("%-9s\t%-21s%n", fileName, ("Class:  " + pred)));
				}
				
				br.close();
			}
			bw.close();
			
		} catch(IOException e) {
			e.printStackTrace();
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
	
}
