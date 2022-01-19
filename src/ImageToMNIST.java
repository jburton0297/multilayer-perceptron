import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import javax.imageio.ImageIO;

/**
 * Convert a 28x28 pixel image file into a formatted text file to be used as input into the neural network.
 * @author Jacob Burton
 *
 */
public class ImageToMNIST {

	public static void main(String[] args) {
		
		try {
			
			File file = new File(args[0]);
			BufferedImage image = ImageIO.read(file);
			Raster raster = image.getData();
			int imageWidth = image.getWidth();
			int imageHeight = image.getHeight();
			int[] data = new int[imageWidth * imageHeight];
			for(int x = 0; x < imageWidth; x++) {
				for(int y = 0; y < imageHeight; y++) {
					data[(y * imageWidth) + x] = raster.getSample(x, y, 0);
				}
			}
			File outFile = new File(file.getParent() + "/" + file.getName().replaceFirst(".[^.]+$", ".txt"));
			BufferedWriter bw = new BufferedWriter(new FileWriter(outFile));
			for(int i = 0; i < data.length; i++) {
				if(i < data.length - 1) {
					bw.write(data[i] + " ");
				} else {
					bw.write(data[i] + "");
				}
			}
			bw.close();
			
		} catch(IOException e) {
			e.printStackTrace();
		}
	}

}
